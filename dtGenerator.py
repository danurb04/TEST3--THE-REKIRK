# ============================================================
# GENERAR dt.cc DESDE WAVEFORMS PARA GROWCLUST
# Guarda el archivo en:
#   /data/murbina/seismo/results/growclust/IN/dt.cc
# ============================================================

import os
import re
import glob
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from obspy import read, UTCDateTime
from obspy.signal.cross_correlation import correlate, xcorr_max

warnings.filterwarnings("ignore")

# ============================================================
# RUTAS
# ============================================================
CATALOG_CSV   = "/data/murbina/seismo/inputs/catalogo_unido.csv"
STATION_CSV   = "/data/murbina/seismo/inputs/XML_Cartago_Nodes.csv"
WAVEFORM_DIR  = "/data/murbina/seismo/rawdata"
OUTPUT_DTCC   = "/data/murbina/seismo/results/growclust/IN/dt.cc"

# ============================================================
# PARÁMETROS DE CORRELACIÓN
# ============================================================
CC_FREQMIN = 1.0
CC_FREQMAX = 15.0
CC_WIN_P   = (-0.5, 2.5)   # segundos
CC_WIN_S   = (-0.5, 4.0)   # segundos
CC_MIN_R   = 0.5
CC_MAX_DIST_KM = 120.0
MAX_PAIRS_PER_EVENT = 50
N_JOBS = 20

# velocidades promedio simples, igual idea que tu script original
VP_AVG = 5.5
VS_AVG = 3.15

# ============================================================
# HELPERS
# ============================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def parse_utc(value):
    try:
        return UTCDateTime(str(value))
    except Exception:
        return None


def jday_key(origin_utc, jday_field):
    if pd.isna(jday_field):
        jday_str = ""
    else:
        try:
            jday_str = str(int(jday_field))
        except Exception:
            jday_str = str(jday_field).strip()

    if len(jday_str) == 7:
        return jday_str

    return f"{origin_utc.year}{jday_str.zfill(3)}"


def parse_sta_from_file(fpath):
    # esperado: i4.C0001.HHE.2024190_0+
    parts = Path(fpath).name.split(".")
    if len(parts) >= 3:
        return parts[1], parts[2]
    return None, None


def build_day_file_map(root_dir):
    all_files = [
        f for f in glob.glob(os.path.join(root_dir, "**", "*"), recursive=True)
        if os.path.isfile(f)
    ]

    day_map = {}
    for f in all_files:
        name = Path(f).name
        m = re.search(r"(\d{7})", name)
        if m:
            day_map.setdefault(m.group(1), []).append(f)

    return all_files, day_map


def get_event_files(ev, day_file_map):
    key = jday_key(ev["_utc"], ev.get("jday", ""))
    return day_file_map.get(key, [])


def group_by_station(files):
    grouped = {}
    for f in files:
        sta_code, cha = parse_sta_from_file(f)
        if sta_code:
            grouped.setdefault(sta_code, []).append((cha, f))
    return grouped


def load_trace(file_list, t_center, half_win, pref_cha_ends=("Z", "E", "N", "1", "2")):
    for end in pref_cha_ends:
        for cha, fpath in file_list:
            if not cha.endswith(end):
                continue
            try:
                st = read(
                    fpath,
                    starttime=t_center - half_win - 2,
                    endtime=t_center + half_win + 2,
                )
                tr = st.select(channel=cha)
                if len(tr) == 0:
                    continue

                tr = tr[0].copy()
                tr.detrend("demean")
                tr.taper(0.05)
                tr.filter(
                    "bandpass",
                    freqmin=CC_FREQMIN,
                    freqmax=CC_FREQMAX,
                    corners=4,
                    zerophase=True,
                )
                return tr
            except Exception:
                continue
    return None


# ============================================================
# CARGA DE DATOS
# ============================================================
print("Cargando catálogo y estaciones...")
cat = pd.read_csv(CATALOG_CSV)
sta = pd.read_csv(STATION_CSV)

cat.columns = [c.strip().lower() for c in cat.columns]
sta.columns = [c.strip().lower() for c in sta.columns]

ot_col = "origin_time_utc" if "origin_time_utc" in cat.columns else "origin_time"
if ot_col not in cat.columns:
    raise ValueError("No encontré 'origin_time_utc' ni 'origin_time' en el catálogo.")

required_cat = ["lat", "lon", "depth_km"]
for col in required_cat:
    if col not in cat.columns:
        raise ValueError(f"Falta la columna '{col}' en el catálogo.")

required_sta = ["code", "latitude", "longitude"]
for col in required_sta:
    if col not in sta.columns:
        raise ValueError(f"Falta la columna '{col}' en la tabla de estaciones.")

cat["_ot"] = cat[ot_col]
cat["_utc"] = [parse_utc(v) for v in cat["_ot"]]

cat = cat.reset_index(drop=True)
cat["evid"] = cat.index + 1

cat_valid = cat[cat["_utc"].notna()].copy()
ev_list = cat_valid.to_dict("records")

sta_dict = {}
for _, r in sta.iterrows():
    code = str(r["code"]).strip()
    lat = float(r["latitude"])
    lon = float(r["longitude"])
    elev = float(r["elevation"]) if "elevation" in r and not pd.isna(r["elevation"]) else 0.0
    sta_dict[code] = (lat, lon, elev)

print(f"  Eventos válidos: {len(ev_list)}")
print(f"  Estaciones: {len(sta_dict)}")

# ============================================================
# INDEXAR WAVEFORMS
# ============================================================
print("Indexando waveforms...")
all_files, day_file_map = build_day_file_map(WAVEFORM_DIR)
print(f"  Archivos encontrados: {len(all_files)}")
print(f"  Day-keys encontradas: {len(day_file_map)}")

# ============================================================
# CONSTRUIR PARES DE EVENTOS
# ============================================================
print("Construyendo pares candidatos...")
pair_set = set()
ev_pair_count = {}

n_ev = len(ev_list)
for i in range(n_ev):
    dists = []
    for j in range(i + 1, n_ev):
        d = haversine_km(
            float(ev_list[i]["lat"]), float(ev_list[i]["lon"]),
            float(ev_list[j]["lat"]), float(ev_list[j]["lon"])
        )
        if d <= CC_MAX_DIST_KM:
            dists.append((d, j))

    dists.sort()
    count_i = 0

    for d, j in dists:
        if count_i >= MAX_PAIRS_PER_EVENT:
            break
        if ev_pair_count.get(j, 0) >= MAX_PAIRS_PER_EVENT:
            continue

        pair_set.add((i, j))
        ev_pair_count[i] = ev_pair_count.get(i, 0) + 1
        ev_pair_count[j] = ev_pair_count.get(j, 0) + 1
        count_i += 1

pairs = list(pair_set)
print(f"  Pares candidatos: {len(pairs)}")

# ============================================================
# CORRELACIÓN POR PAR
# ============================================================
def xcorr_pair(pair_idx):
    i, j = pair_idx
    ev_i = ev_list[i]
    ev_j = ev_list[j]

    t_i = ev_i["_utc"]
    t_j = ev_j["_utc"]

    files_i = get_event_files(ev_i, day_file_map)
    files_j = get_event_files(ev_j, day_file_map)

    if not files_i or not files_j:
        return []

    gi = group_by_station(files_i)
    gj = group_by_station(files_j)

    common_stas = set(gi.keys()) & set(gj.keys()) & set(sta_dict.keys())
    if not common_stas:
        return []

    evid_i = int(ev_i["evid"])
    evid_j = int(ev_j["evid"])
    lines = []

    for sta_code in common_stas:
        sta_lat, sta_lon, sta_elev = sta_dict[sta_code]

        dep_i = float(ev_i["depth_km"])
        dep_j = float(ev_j["depth_km"])

        epi_i = haversine_km(float(ev_i["lat"]), float(ev_i["lon"]), sta_lat, sta_lon)
        epi_j = haversine_km(float(ev_j["lat"]), float(ev_j["lon"]), sta_lat, sta_lon)

        r_i = math.sqrt(epi_i**2 + (dep_i + sta_elev / 1000.0) ** 2)
        r_j = math.sqrt(epi_j**2 + (dep_j + sta_elev / 1000.0) ** 2)

        tp_i = r_i / VP_AVG
        ts_i = r_i / VS_AVG
        tp_j = r_j / VP_AVG
        ts_j = r_j / VS_AVG

        for phase, (pre, post), tphase_i, tphase_j in [
            ("P", CC_WIN_P, tp_i, tp_j),
            ("S", CC_WIN_S, ts_i, ts_j),
        ]:
            cha_pref = ("Z",) if phase == "P" else ("E", "N", "1", "2", "Z")
            half = (post - pre) / 2.0

            t_pick_i = t_i + tphase_i + pre + half
            t_pick_j = t_j + tphase_j + pre + half

            tr_i = load_trace(gi[sta_code], t_pick_i, half + 1.0, cha_pref)
            tr_j = load_trace(gj[sta_code], t_pick_j, half + 1.0, cha_pref)

            if tr_i is None or tr_j is None:
                continue

            try:
                tr_i.trim(t_pick_i - half, t_pick_i + half)
                tr_j.trim(t_pick_j - half, t_pick_j + half)

                if len(tr_i.data) < 5 or len(tr_j.data) < 5:
                    continue

                if tr_i.stats.sampling_rate != tr_j.stats.sampling_rate:
                    tr_j.resample(tr_i.stats.sampling_rate)

                max_shift = int(tr_i.stats.sampling_rate * 0.5)
                cc = correlate(tr_i, tr_j, max_shift)
                shift, coeff = xcorr_max(cc)

                if abs(coeff) < CC_MIN_R:
                    continue

                dt = shift / tr_i.stats.sampling_rate
                lines.append(f"  {sta_code:<5s} {dt:8.5f} {abs(coeff):.4f} {phase}\n")

            except Exception:
                continue

    if not lines:
        return []

    header = f"# {evid_i:9d} {evid_j:9d}  0.000\n"
    return [header] + lines


# ============================================================
# EJECUCIÓN PARALELA
# ============================================================
print(f"Corriendo correlaciones con {N_JOBS} workers...")
results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
    delayed(xcorr_pair)(p) for p in pairs
)

# ============================================================
# ESCRIBIR dt.cc
# ============================================================
output_path = Path(OUTPUT_DTCC)
output_path.parent.mkdir(parents=True, exist_ok=True)

n_pairs_written = 0
n_obs_written = 0

with open(output_path, "w") as fh:
    for block in results:
        if block and len(block) > 1:
            fh.writelines(block)
            n_pairs_written += 1
            n_obs_written += len(block) - 1

print("\n" + "=" * 60)
print("RESUMEN dt.cc")
print("=" * 60)
print(f"Pares candidatos evaluados : {len(pairs)}")
print(f"Pares escritos en dt.cc    : {n_pairs_written}")
print(f"Observaciones escritas     : {n_obs_written}")
print(f"Archivo guardado en        : {output_path}")
print("=" * 60)