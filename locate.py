import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import heapq
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import DBSCAN

# =====================
# Configuracion temporal
# =====================
YEAR = 2024
JDAYS = [189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209]   # <- aquí pones 1 o varios días

BASE_IN = Path("/data/murbina/seismo/inputs")

# Solo picks y detections salen de results/picks
BASE_PICKS = Path("/data/murbina/seismo/results/picks")

NODES_CSV = BASE_IN / "XML_Cartago_Nodes.csv"

BASE_OUT = Path("/data/murbina/seismo/results/catalogs/Test2")
BASE_OUT.mkdir(parents=True, exist_ok=True)


# Tolerancia para match entre eventos IA y oficiales
T_TOL = 1.0
D_TOL_KM = 15.0
Z_TOL_KM = 10.0

# Parámetros para construir eventos
BIN_SEC = 11.666460891503132  # agrupa picks en este tiempo (en una misma estacion) para quedarse con el más fuerte 
WIN_SEC = 33.13564416676824 # tamaño de ventana para agrupar picks P entre estaciones 
MIN_STATIONS = 3 # mínimo de estaciones con picks P para considerar un evento
DEAD_SEC = 9.21132846069997 # tiempo mínimo entre eventos 

# Parametros para filtrar eventos 
STRONG_THR = 0.75 # umbral de probabilidad para considerar un pick como “fuerte” 
F_MIN_STATIONS = 8 # mínimo de estaciones con picks P 
F_MAXPROB = 0.8632901892596616 # probabilidad máxima necesaria de una estación para considerar un evento 
F_MIN_STRONG = 3 # mínimo de estaciones con picks P fuertes para un mismo evento
F_MIN_DET_SUPPORT = 22  # mínimo de detecciones (no picks) dentro de un evento para considerarlo 
F_MIN_S_SUPPORT = 11 # mínimo de picks S dentro de un evento para considerarlo

# Ventana para asignar picks al evento
P_WIN = (-4.066576938188511, 7.337208231670156)
S_WIN = (-3.0864797724218374, 34.5997751302812)

# Grid
LAT_MIN, LAT_MAX, DLAT = 9.25, 10.12, 0.01
LON_MIN, LON_MAX, DLON = -84.7, -83.16, 0.01
DEPTHS_KM = [2.771141348959358, 3.004608919681081, 8.02720600691255, 12.969199708783881, 17.658516853015016, 21.964726516510684]

# Modelo 1D por capas
VEL_LAYERS = [
    (4.0,  4.45, 2.50),
    (2.0,  5.50, 3.00),
    (2.0,  5.60, 3.15),
    (3.0,  6.00, 3.37),
    (3.0,  6.15, 3.45),
    (7.0,  6.25, 3.51),
    (7.0,  6.50, 3.65),
    (6.0,  6.80, 3.82),
    (10.0, 7.00, 3.93),
    (10.0, 7.30, 4.10),
    (20.0, 7.90, 4.44),
    (30.0, 8.20, 4.60),
    (20.0, 8.30, 4.66),
    (10.0, 8.35, 4.69),
    (50.0, 8.40, 4.72),
]

# Coarse grid
DLAT_C = 0.05
DLON_C = 0.05
DEPTHS_COARSE = [2.2497505686685226, 10.69111740332143, 22.04728751688021]
COARSE_TOPK = 12

# Fine-1
N_FINE1_FROM_COARSE = 4
REFINE1_HALFSPAN_DEG =  0.11769345949091765
DLAT_F1 = 0.01
DLON_F1 = 0.01
DEPTH_FINE_HALFSPAN_KM = 13.323975939643066
DEPTH_FINE_DZ_KM = 1.3293445212523969

# Fine-2
REFINE2_HALFSPAN_DEG = 0.01671788552306872
DLAT_F2 = 0.002
DLON_F2 = 0.002
DEPTH2_HALFSPAN_KM = 3.818546726757031
DEPTH2_DZ_KM = 0.32708685650203473

# Progreso
PRINT_EVERY = 999999
SHOW_GRID_PROGRESS = True
GRID_PROGRESS_EVERY = 20000


# =====================
# Helpers de rutas/fechas
# =====================
def jday_to_yyyymmdd(year, jday):
    dt = datetime(year, 1, 1) + timedelta(days=jday - 1)
    return int(dt.strftime("%Y%m%d"))

def official_csv_for_day(jday):
    # Si tu oficial es uno por día:
    # Si en realidad usas un solo catálogo grande, cambia por:
    return BASE_IN / "catalogo_oficial.csv"

def picks_csv_for_day(year, jday):
    jstr = f"{year}{jday:03d}"
    return BASE_PICKS / f"picks_day_{jstr}_THP0.70_THS0.55_seisbench.csv"

def detections_csv_for_day(year, jday):
    jstr = f"{year}{jday:03d}"
    return BASE_PICKS / f"detections_day_{jstr}_THP0.70_THS0.55_seisbench.csv"

def out_events_loc_csv_for_day(year, jday):
    jstr = f"{year}{jday:03d}"
    return BASE_OUT / f"events_loc_{jstr}.csv"


# =====================
# Carga catalogo oficial
# =====================
def load_official_day(csv_path, day_yyyymmdd):
    df = pd.read_csv(csv_path)

    df = df[df["date"] == int(day_yyyymmdd)].copy()

    ts = df["time"].astype("Int64").astype(str).str.zfill(8)
    hh = ts.str.slice(0, 2).astype(int)
    mm = ts.str.slice(2, 4).astype(int)
    ss = ts.str.slice(4, 6).astype(int)
    cc = ts.str.slice(6, 8).astype(int)

    date_str = df["date"].astype("Int64").astype(str)
    base = pd.to_datetime(
        date_str + " " + hh.astype(str).str.zfill(2) + ":" +
        mm.astype(str).str.zfill(2) + ":" +
        ss.astype(str).str.zfill(2),
        format="%Y%m%d %H:%M:%S",
        errors="raise"
    )
    df["t0_utc"] = base + pd.to_timedelta(cc * 10, unit="ms")
    df = df.sort_values("t0_utc").reset_index(drop=True)
    return df


# =====================
# Event building
# =====================
def normalize_eqt_picks(picks_df: pd.DataFrame) -> pd.DataFrame:
    df = picks_df.copy()
    df["prob"] = df["probability"].astype(float)
    df["station"] = df["station_code"].astype(str)
    df["t_pick"] = pd.to_datetime(df["time"])
    df["time_utc"] = df["t_pick"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    df["t"] = df["t_pick"].astype("int64") / 1e9
    out = df[["station", "phase", "prob", "time_utc", "t"]].copy()
    return out.sort_values("t").reset_index(drop=True)

def bin_best_pick_per_station(df, bin_sec, phase):
    d = df[df.phase == phase].copy()
    if d.empty:
        return d
    d["bin"] = (d["t"] // bin_sec).astype(int)
    d = d.sort_values("prob", ascending=False).drop_duplicates(["station", "bin"])
    return d.drop(columns=["bin"]).sort_values("t").reset_index(drop=True)

def build_events_sliding_window(picksP, win_sec, min_stations, dead_sec, strong_thr):
    if picksP.empty:
        return pd.DataFrame(columns=["t0","t0_utc","n_stations","n_picks","maxprob","meanprob","n_strong"])
    p = picksP.sort_values("t").reset_index(drop=True)
    events = []
    i = 0
    n = len(p)
    while i < n:
        t_start = float(p.loc[i, "t"])
        t_end = t_start + win_sec
        w = p[(p["t"] >= t_start) & (p["t"] <= t_end)]
        nsta = w["station"].nunique()
        if nsta >= min_stations:
            per_sta = w.sort_values("t").drop_duplicates("station", keep="first")

            maxprob = float(per_sta["prob"].max())
            meanprob = float(per_sta["prob"].mean())
            n_strong = int((per_sta["prob"] >= strong_thr).sum())
            t0 = float(np.median(per_sta["t"].values))

            events.append({
                "t0": t0,
                "t0_utc": pd.to_datetime(t0, unit="s"),
                "n_stations": int(nsta),
                "n_picks": int(len(w)),
                "maxprob": maxprob,
                "meanprob": meanprob,
                "n_strong": n_strong,
            })

            block_until = t0 + dead_sec
            while i < n and float(p.loc[i, "t"]) <= block_until:
                i += 1
        else:
            i += 1

    return pd.DataFrame(events).sort_values("t0").reset_index(drop=True)


# =====================
# Funciones para filtrado
# =====================
def normalize_seisbench_detections(det_raw: pd.DataFrame) -> pd.DataFrame:
    d = det_raw.copy()
    d["start_time"] = pd.to_datetime(d["start_time"], errors="coerce")
    d["end_time"]   = pd.to_datetime(d["end_time"], errors="coerce")
    d = d.dropna(subset=["start_time", "end_time"])
    d["station"] = d["station_code"].astype(str)
    d["start_t"] = d["start_time"].astype("int64") / 1e9
    d["end_t"]   = d["end_time"].astype("int64") / 1e9
    return d[["station", "start_t", "end_t"]].sort_values(["station", "start_t"]).reset_index(drop=True)

def build_det_index(det_norm: pd.DataFrame):
    idx = {}
    for sta, g in det_norm.groupby("station"):
        idx[sta] = (g["start_t"].to_numpy(), g["end_t"].to_numpy())
    return idx

def det_support_count(t0, det_idx, slack_sec) -> int:
    tL = t0 - slack_sec
    tR = t0 + slack_sec
    n = 0
    for sta, (starts, ends) in det_idx.items():
        j = np.searchsorted(starts, tR, side="right") - 1
        if j >= 0 and ends[j] >= tL:
            n += 1
    return n

def count_phase_support(picks, t0, phase, w0, w1):
    w = picks[(picks["phase"] == phase) & (picks["t"] >= t0 + w0) & (picks["t"] <= t0 + w1)]
    return w["station"].nunique()


# =====================
# Funciones para Localización por grid search
# =====================
def load_station_nodes(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "code": "station",
        "longitude": "lon",
        "latitude": "lat",
        "elevation": "elev_m"
    })

    df["station"] = df["station"].astype(str).str.strip()
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["elev_m"] = pd.to_numeric(df["elev_m"], errors="coerce")

    df = df.dropna(subset=["station", "lat", "lon"]).reset_index(drop=True)
    df["elev_km"] = df["elev_m"].fillna(0.0) / 1000.0
    return df[["station", "lat", "lon", "elev_km"]]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def picks_for_event(picks, t0, phase, w, topk_per_sta=2, min_prob=0.0):
    w0, w1 = w
    d = picks[(picks["phase"] == phase) &
              (picks["t"] >= t0 + w0) &
              (picks["t"] <= t0 + w1) &
              (picks["prob"] >= min_prob)].copy()
    if d.empty:
        return d
    d = d.sort_values(["station","prob"], ascending=[True, False])
    d["rk"] = d.groupby("station").cumcount()
    d = d[d["rk"] < topk_per_sta].drop(columns=["rk"])
    return d

def travel_time_layered(dist_km: float, depth_km: float, phase: str) -> float:
    depth_km = float(max(depth_km, 0.0))
    L = float(np.sqrt(dist_km**2 + depth_km**2))
    if L <= 0:
        return 0.0

    tt = 0.0
    z1 = depth_km

    for i, (th, vp, vs) in enumerate(VEL_LAYERS):
        top = LAYER_TOPS[i]
        bot = LAYER_TOPS[i+1]

        a = max(top, 0.0)
        b = min(bot, z1)
        if b <= a:
            if top >= z1:
                break
            continue

        dz = b - a
        ds = dz * (L / depth_km) if depth_km > 0 else 0.0

        v = vp if phase == "P" else vs
        tt += ds / v

        if bot >= z1:
            break

    return float(tt)

def locate_event_grid(pP, pS, stations,
                      lat_min=LAT_MIN, lat_max=LAT_MAX, dlat=DLAT,
                      lon_min=LON_MIN, lon_max=LON_MAX, dlon=DLON,
                      depths_km=DEPTHS_KM):

    obs = []
    for _, r in pP.iterrows():
        if r["station"] in stations.index:
            obs.append((r["station"], "P", float(r["t"])))
    for _, r in pS.iterrows():
        if r["station"] in stations.index:
            obs.append((r["station"], "S", float(r["t"])))

    if len(obs) < 4:
        return None

    stas    = [o[0] for o in obs]
    st_lats = stations.loc[stas, "lat"].to_numpy()
    st_lons = stations.loc[stas, "lon"].to_numpy()
    st_elev = stations.loc[stas, "elev_km"].to_numpy()
    phases  = np.array([o[1] for o in obs])
    t_obs   = np.array([o[2] for o in obs])

    def eval_grid(lats, lons, depths, label="grid"):
        bests = []
        total = len(depths) * len(lats) * len(lons)
        k = 0

        for dep in depths:
            dep = float(dep)
            for lat in lats:
                for lon in lons:
                    dist = haversine_km(lat, lon, st_lats, st_lons)

                    tt = np.array([
                        travel_time_layered(float(dist[i]), float(dep + st_elev[i]), phases[i])
                        for i in range(len(obs))
                    ], dtype=float)

                    t_origin = np.median(t_obs - tt)
                    res = (t_obs - (t_origin + tt))
                    score = np.median(np.abs(res))

                    cand = (score, float(lat), float(lon), dep, float(t_origin))

                    if len(bests) < COARSE_TOPK:
                        heapq.heappush(bests, (-cand[0], cand))
                    else:
                        if cand[0] < (-bests[0][0]):
                            heapq.heapreplace(bests, (-cand[0], cand))

                    k += 1
                    if SHOW_GRID_PROGRESS and (k % GRID_PROGRESS_EVERY) == 0:
                        best_rms = min(c[0] for _, c in bests) if bests else np.inf
                        print(f"[{label}] {k}/{total} ({100.0*k/total:.1f}%) best_rms={best_rms:.2f}s", end="\r")

        if SHOW_GRID_PROGRESS:
            print(" " * 80, end="\r")

        top = [c for _, c in bests]
        top.sort(key=lambda x: x[0])
        return top

    lats_c = np.arange(lat_min, lat_max + 1e-12, DLAT_C)
    lons_c = np.arange(lon_min, lon_max + 1e-12, DLON_C)

    cands = eval_grid(lats_c, lons_c, DEPTHS_COARSE, label="coarse")
    if not cands:
        return None

    cands = cands[:N_FINE1_FROM_COARSE]
    best_f1 = None

    for ic, (rms_c, lat_c, lon_c, dep_c, torig_c) in enumerate(cands, start=1):
        lat0 = max(lat_min, lat_c - REFINE1_HALFSPAN_DEG)
        lat1 = min(lat_max, lat_c + REFINE1_HALFSPAN_DEG)
        lon0 = max(lon_min, lon_c - REFINE1_HALFSPAN_DEG)
        lon1 = min(lon_max, lon_c + REFINE1_HALFSPAN_DEG)

        z0 = max(0.5, float(dep_c) - DEPTH_FINE_HALFSPAN_KM)
        z1 = float(dep_c) + DEPTH_FINE_HALFSPAN_KM
        depths_f1 = np.arange(z0, z1 + 1e-12, DEPTH_FINE_DZ_KM)

        lats_f1 = np.arange(lat0, lat1 + 1e-12, DLAT_F1)
        lons_f1 = np.arange(lon0, lon1 + 1e-12, DLON_F1)

        fine1 = eval_grid(lats_f1, lons_f1, depths_f1, label=f"fine1_{ic}")
        cand_f1 = fine1[0] if fine1 else (rms_c, lat_c, lon_c, dep_c, torig_c)

        if (best_f1 is None) or (cand_f1[0] < best_f1[0]):
            best_f1 = cand_f1

    if best_f1 is None:
        return None

    s1, lat_b, lon_b, dep_b, tor_b = best_f1

    lat0 = max(lat_min, lat_b - REFINE2_HALFSPAN_DEG)
    lat1 = min(lat_max, lat_b + REFINE2_HALFSPAN_DEG)
    lon0 = max(lon_min, lon_b - REFINE2_HALFSPAN_DEG)
    lon1 = min(lon_max, lon_b + REFINE2_HALFSPAN_DEG)

    lats_f2 = np.arange(lat0, lat1 + 1e-12, DLAT_F2)
    lons_f2 = np.arange(lon0, lon1 + 1e-12, DLON_F2)

    z0 = max(0.5, float(dep_b) - DEPTH2_HALFSPAN_KM)
    z1 = float(dep_b) + DEPTH2_HALFSPAN_KM
    depths_f2 = np.arange(z0, z1 + 1e-12, DEPTH2_DZ_KM)

    fine2 = eval_grid(lats_f2, lons_f2, depths_f2, label="fine2")

    if fine2:
        return fine2[0]
    else:
        return best_f1

def build_layer_tops(layers):
    tops = [0.0]
    z = 0.0
    for th, _, _ in layers:
        z += float(th)
        tops.append(z)
    return np.array(tops)

LAYER_TOPS = build_layer_tops(VEL_LAYERS)


# =====================
# Match con catálogo oficial
# =====================
def match_loc_to_official(official_df: pd.DataFrame, loc_df: pd.DataFrame,
                          t_tol=T_TOL, d_tol_km=D_TOL_KM, z_tol_km=Z_TOL_KM):
    off = official_df.copy()
    loc = loc_df[loc_df["ok"] == True].copy()

    off["t0"] = pd.to_datetime(off["t0_utc"])
    loc["t0"] = pd.to_datetime(loc["origin_time_utc"])

    loc_t = (loc["t0"].astype("int64") / 1e9).to_numpy()
    out_rows = []

    for i, r in off.iterrows():
        t_off = float(pd.to_datetime(r["t0"]).value / 1e9)

        dt = np.abs(loc_t - t_off)
        cand_idx = np.where(dt <= t_tol)[0]
        if cand_idx.size == 0:
            if len(loc_t) > 0:
                j_near = int(np.argmin(np.abs(loc_t - t_off)))
                rrn = loc.iloc[j_near]
                dt_near = float(loc_t[j_near] - t_off)
                dkm_near = float(haversine_km(r["lat"], r["lon"], rrn["lat"], rrn["lon"]))
                dz_near  = float(abs(float(r["depth_km"]) - float(rrn["depth_km"])))
                t0_loc_near = rrn["t0"]
                lat_loc_near = float(rrn["lat"])
                lon_loc_near = float(rrn["lon"])
                dep_loc_near = float(rrn["depth_km"])
                event_idx_near = int(rrn["event_idx"])
                rms_near = float(rrn.get("rms_sec", np.nan))
            else:
                dt_near = np.nan
                dkm_near = np.nan
                dz_near = np.nan
                t0_loc_near = pd.NaT
                lat_loc_near = np.nan
                lon_loc_near = np.nan
                dep_loc_near = np.nan
                event_idx_near = np.nan
                rms_near = np.nan

            out_rows.append({
                "idx_official": i,
                "t0_official": r["t0"],
                "lat_off": float(r["lat"]),
                "lon_off": float(r["lon"]),
                "dep_off": float(r["depth_km"]),
                "is_match": False,
                "score": np.nan,
                "event_idx": event_idx_near,
                "t0_loc": t0_loc_near,
                "lat_loc": lat_loc_near,
                "lon_loc": lon_loc_near,
                "dep_loc": dep_loc_near,
                "dt_sec": dt_near,
                "dist_km": dkm_near,
                "ddep_km": dz_near,
                "rms_sec": rms_near,
            })
            continue

        best = None
        for j in cand_idx:
            rr = loc.iloc[j]
            dkm = haversine_km(r["lat"], r["lon"], rr["lat"], rr["lon"])
            dz  = abs(float(r["depth_km"]) - float(rr["depth_km"]))
            dtj = float(loc_t[j] - t_off)

            ok = (dkm <= d_tol_km) and (dz <= z_tol_km)
            score = abs(dtj) + 0.2*dkm + 0.1*dz

            if ok and ((best is None) or (score < best["score"])):
                best = {
                    "idx_official": i,
                    "t0_official": r["t0"],
                    "lat_off": float(r["lat"]),
                    "lon_off": float(r["lon"]),
                    "dep_off": float(r["depth_km"]),
                    "event_idx": int(rr["event_idx"]),
                    "t0_loc": rr["t0"],
                    "lat_loc": float(rr["lat"]),
                    "lon_loc": float(rr["lon"]),
                    "dep_loc": float(rr["depth_km"]),
                    "dt_sec": dtj,
                    "dist_km": float(dkm),
                    "ddep_km": float(dz),
                    "rms_sec": float(rr.get("rms_sec", np.nan)),
                    "score": float(score),
                    "is_match": True,
                }

        if best is None:
            j0 = cand_idx[np.argmin(dt[cand_idx])]
            rr0 = loc.iloc[j0]
            dkm0 = haversine_km(r["lat"], r["lon"], rr0["lat"], rr0["lon"])
            dz0  = abs(float(r["depth_km"]) - float(rr0["depth_km"]))
            dt0  = float(loc_t[j0] - t_off)

            out_rows.append({
                "idx_official": i,
                "is_match": False,
                "dt_sec": dt0,
                "dist_km": float(dkm0),
                "ddep_km": float(dz0),
                "event_idx": int(rr0["event_idx"]),
                "score": np.nan,
                "t0_official": r["t0"],
                "lat_off": float(r["lat"]),
                "lon_off": float(r["lon"]),
                "dep_off": float(r["depth_km"]),
            })
        else:
            out_rows.append(best)

    return pd.DataFrame(out_rows)


def _locate_one_event(idx, ev_row):
    global _G_PICKS, _G_STATIONS

    t0 = float(ev_row["t0"])
    picks = _G_PICKS
    stations = _G_STATIONS

    pP = picks_for_event(picks, t0, "P", P_WIN)
    pS = picks_for_event(picks, t0, "S", S_WIN)

    nsta = int(ev_row["n_stations"])

    sol = locate_event_grid(pP, pS, stations)

    if sol is None:
        return {
            "event_idx": int(idx),
            "t0_seed": pd.to_datetime(t0, unit="s"),
            "ok": False,
            "nP": int(len(pP)),
            "nS": int(len(pS)),
            "nsta_event": nsta,
            "det_support": int(ev_row.get("det_support", 0)),
            "S_support": int(ev_row.get("S_support", 0)),
        }

    rms, lat, lon, dep, torigin = sol
    return {
        "event_idx": int(idx),
        "t0_seed": pd.to_datetime(t0, unit="s"),
        "origin_time_utc": pd.to_datetime(torigin, unit="s"),
        "lat": float(lat),
        "lon": float(lon),
        "depth_km": float(dep),
        "rms_sec": float(rms),
        "nP": int(len(pP)),
        "nS": int(len(pS)),
        "nsta_event": nsta,
        "maxprob": float(ev_row["maxprob"]),
        "det_support": int(ev_row["det_support"]),
        "S_support": int(ev_row["S_support"]),
        "ok": True,
    }
# =====================
# Nueva función — agregar después de tag_clusters()
# =====================
def tag_clusters(loc_df: pd.DataFrame,
                 eps_km: float = 8.511292323882095,
                 min_samples: int = 3) -> pd.DataFrame:
    """
    Asigna cluster_id a cada evento localizado usando DBSCAN espacial.
    cluster_id = -1 → ruido (evento aislado)
    cluster_id >= 0  → cluster coherente
    """
    df = loc_df.copy()
    df["cluster_id"] = -1  # default: ruido

    mask = df["ok"] == True
    coords = df.loc[mask, ["lat", "lon"]].dropna()

    if len(coords) < min_samples:
        print(f"[DBSCAN] Muy pocos eventos ({len(coords)}) para clusterizar.")
        return df

    coords_rad = np.deg2rad(coords.values)
    eps_rad = eps_km / 6371.0

    db = DBSCAN(
        eps=eps_rad,
        min_samples=min_samples,
        algorithm="ball_tree",
        metric="haversine"
    ).fit(coords_rad)

    df.loc[coords.index, "cluster_id"] = db.labels_

    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise    = int((db.labels_ == -1).sum())
    print(f"[DBSCAN] eps={eps_km} km | min_samples={min_samples} → "
          f"{n_clusters} clusters, {n_noise} eventos ruido "
          f"({100*n_noise/len(coords):.1f}%)")

    return df

# Globals
_G_PICKS = None
_G_STATIONS = None

def _init_worker(picks, stations):
    global _G_PICKS, _G_STATIONS
    _G_PICKS = picks
    _G_STATIONS = stations
    
def build_run_summary(all_results: dict) -> pd.DataFrame:
    rows = []

    for jday, res in sorted(all_results.items()):
        loc_df = res["loc_df"]      # eventos automáticos finales de ese día
        m_ok   = res["m_ok"]        # matches de ese día
        m_bad  = res["m_bad"]       # no matches de ese día

        n_auto = len(loc_df)
        n_match = len(m_ok)
        n_official = len(m_ok) + len(m_bad)

        precision_pct = (100.0 * n_match / n_auto) if n_auto > 0 else np.nan
        match_pct = (100.0 * n_match / n_official) if n_official > 0 else np.nan
        mean_dist_km = float(m_ok["dist_km"].mean()) if not m_ok.empty else np.nan

        rows.append({
            "jday": jday,
            "n_auto_final": n_auto,
            "n_official": n_official,
            "n_match": n_match,
            "precision_pct": precision_pct,
            "match_pct": match_pct,
            "mean_dist_km": mean_dist_km,
        })

    return pd.DataFrame(rows)

def main(year, jday):
    t_start = time.time()

    official_day = jday_to_yyyymmdd(year, jday)
    jstr = f"{year}{jday:03d}"

    OFFICIAL_CSV = official_csv_for_day(jday)
    PICKS_CSV = picks_csv_for_day(year, jday)
    DETECTIONS_CSV = detections_csv_for_day(year, jday)
    OUT_EVENTS_LOC_CSV = out_events_loc_csv_for_day(year, jday)

    print("\n" + "=" * 70)
    print(f"Procesando YEAR={year}, JDAY={jday}, DATE={official_day}")
    print(f"Official   : {OFFICIAL_CSV}")
    print(f"Picks      : {PICKS_CSV}")
    print(f"Detections : {DETECTIONS_CSV}")
    print("=" * 70)

    official = load_official_day(OFFICIAL_CSV, official_day)
    print(f"Oficial {official_day}: {len(official)} eventos")

    raw_picks = pd.read_csv(PICKS_CSV)
    print(f"Picks SeisBench cargados: {len(raw_picks)} ({PICKS_CSV})")

    picks = normalize_eqt_picks(raw_picks)
    print(f"Picks normalizados: {len(picks)}  columnas={list(picks.columns)}")

    picksP = bin_best_pick_per_station(picks, bin_sec=BIN_SEC, phase="P")
    print(f"\nP-picks tras bin ({BIN_SEC:.0f}s): {len(picksP)}")

    events_det = build_events_sliding_window(
        picksP, win_sec=WIN_SEC, min_stations=MIN_STATIONS, dead_sec=DEAD_SEC, strong_thr=STRONG_THR
    )
    print(f"Eventos sin filtro: {len(events_det)}")

    det_raw = pd.read_csv(DETECTIONS_CSV)
    det_norm = normalize_seisbench_detections(det_raw)
    det_idx = build_det_index(det_norm)

    events_det["det_support"] = events_det["t0"].apply(
        lambda t0: det_support_count(t0, det_idx, slack_sec=2.0)
    )
    events_det["S_support"] = events_det["t0"].apply(
        lambda t0: count_phase_support(picks, t0, "S", -3, 20.0)
    )

    events_loc = events_det[
        (events_det["n_stations"] >= F_MIN_STATIONS) &
        (events_det["maxprob"] >= F_MAXPROB) &
        (events_det["n_strong"] >= F_MIN_STRONG) &
        (events_det["det_support"] >= F_MIN_DET_SUPPORT) &
        (events_det["S_support"] >= F_MIN_S_SUPPORT)
    ].reset_index(drop=True)
    print(f"\nEventos tras filtrado: {len(events_loc)}")

    stations = load_station_nodes(NODES_CSV).set_index("station")

    n_workers = 20
    print("workers:", n_workers)

    rows = []

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(picks, stations)
    ) as ex:
        futs = [ex.submit(_locate_one_event, idx, ev) for idx, ev in events_loc.iterrows()]

        for k, f in enumerate(as_completed(futs), start=1):
            rows.append(f.result())
            if k % 5 == 0:
                print(f"locados {k}/{len(futs)}", end="\r")
                
    loc_df = pd.DataFrame(rows)
    loc_df = loc_df[
        (loc_df["ok"] == True) &
        (loc_df["rms_sec"] <= 1.0622906587137428)   # o 0.4s para ser más estricto
    ].copy()
    print(f"\nEventos tras filtrado rms: {len(loc_df)}")
    # --- Después de ensamblar loc_df ---
    if not loc_df.empty:
        loc_df = loc_df.sort_values("origin_time_utc", na_position="last")
    
    # ← AGREGA AQUÍ
    loc_df = tag_clusters(
        loc_df,
        eps_km=8.511292323882095,    # radio en km para definir vecindad
        min_samples= 3   # mínimo de eventos para formar cluster
    )

    # Opcional: exportar solo los clusters coherentes
    loc_df_clustered = loc_df[
        (loc_df["ok"] == True) &
        (loc_df["cluster_id"] >= 0)
    ].copy()
    print(f"Eventos en clusters: {len(loc_df_clustered)} / "
          f"{(loc_df['ok']==True).sum()} localizados")
    
    # Usa loc_df_clustered para el match si quieres ser estricto,
    # o usa loc_df completo (con cluster_id como columna informativa)
    
    m = match_loc_to_official(official, loc_df_clustered)   # o loc_df_clustered
    m_ok = m[m["is_match"] == True].copy()
    m_bad = m[m["is_match"] == False].copy()
    
    if not m_ok.empty:
        m_ok["event_idx"] = m_ok["event_idx"].astype(int)

    print("\n=== MATCH LOCALIZADO vs OFICIAL (solo matches) ===")
    print(m_ok.sort_values("score").to_string(index=False) if not m_ok.empty else "Sin matches")

    print("\n=== NO MATCH (detalle) ===")
    print(m_bad.to_string(index=False) if not m_bad.empty else "Sin no-matches")

    if not m_ok.empty:
        print("median dist km:", m_ok["dist_km"].median())
        print("p90 dist km:", m_ok["dist_km"].quantile(0.9))

    print(f"\nTotal oficiales: {len(m)}")
    print(f"Eventos con match: {len(m_ok)}")
    print(f"Eventos sin match: {len(m_bad)}")
    print(f"\nEventos finales: {len(loc_df_clustered)}")
    n_auto_final = len(loc_df_clustered)
    n_match = len(m_ok)
    n_official = len(m_ok) + len(m_bad)
    
    precision_day = (100.0 * n_match / n_auto_final) if n_auto_final > 0 else np.nan
    match_pct_day = (100.0 * n_match / n_official) if n_official > 0 else np.nan
    mean_dist_day = float(m_ok["dist_km"].mean()) if not m_ok.empty else np.nan
    
    print("\n=== RESUMEN DEL DÍA ===")
    print(f"Eventos automáticos finales : {n_auto_final}")
    print(f"Eventos oficiales           : {n_official}")
    print(f"Matches                     : {n_match}")
    print(f"Precisión del workflow      : {precision_day:.2f}%")
    print(f"Porcentaje de matches       : {match_pct_day:.2f}%")
    print(f"Distancia promedio          : {mean_dist_day:.2f} km")
    loc_df_clustered.to_csv(OUT_EVENTS_LOC_CSV, index=False)
    print("Guardado:", OUT_EVENTS_LOC_CSV)

    print(f"\n⏱ Total {jstr}: {time.time() - t_start:.1f}s")

    return loc_df_clustered, m_ok, m_bad


# =====================
# Ejecutar varios días secuencialmente
# =====================
all_results = {}

for jday in JDAYS:
    try:
        loc_df, m_ok, m_bad = main(YEAR, jday)
        all_results[jday] = {
            "loc_df": loc_df,
            "m_ok": m_ok,
            "m_bad": m_bad,
        }
    except FileNotFoundError as e:
        print(f"\n[ERROR] Falta archivo para JDAY {jday}: {e}")
    except Exception as e:
        print(f"\n[ERROR] Falló JDAY {jday}: {e}")

print("\nDías procesados:", list(all_results.keys()))
print("\n" + "=" * 70)
print("RESUMEN GENERAL DEL WORKFLOW")
print("=" * 70)

if all_results:
    summary_df = build_run_summary(all_results)

    total_auto = int(summary_df["n_auto_final"].sum())
    total_official = int(summary_df["n_official"].sum())
    total_matches = int(summary_df["n_match"].sum())

    precision_global = (100.0 * total_matches / total_auto) if total_auto > 0 else np.nan
    match_pct_global = (100.0 * total_matches / total_official) if total_official > 0 else np.nan

    all_distances = pd.concat(
        [res["m_ok"]["dist_km"] for res in all_results.values() if not res["m_ok"].empty],
        ignore_index=True
    )
    mean_dist_global = float(all_distances.mean()) if not all_distances.empty else np.nan

    print("\n=== RESUMEN POR DÍA ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\n=== RESUMEN GLOBAL ===")
    print(f"Eventos automáticos finales : {total_auto}")
    print(f"Eventos oficiales           : {total_official}")
    print(f"Matches totales             : {total_matches}")
    print(f"Precisión global            : {precision_global:.2f}%")
    print(f"Porcentaje de matches       : {match_pct_global:.2f}%")
    print(f"Distancia promedio global   : {mean_dist_global:.2f} km")

    summary_csv = BASE_OUT / "workflow_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nResumen guardado en: {summary_csv}")

else:
    print("No hubo días procesados correctamente.")