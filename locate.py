import time
from pathlib import Path
import numpy as np
import pandas as pd
import heapq
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuracion temporal
YEAR = 2024
JDAY = 189
OFFICIAL_DAY = 20240707  # AJUSTAR al día real (yyyymmdd)
JSTR = f"{YEAR}{JDAY:03d}"


BASE_IN = Path("/data/murbina/seismo/inputs")

#OFFICIAL_CSV = BASE_IN / "catalogo_oficial.csv"
OFFICIAL_CSV = BASE_IN / "catalogo_nacional_189.csv"
NODES_CSV    = BASE_IN / "XML_Cartago_Nodes.csv"

PICKS_CSV      = BASE_IN / f"picks_day_{JSTR}_THP0.70_THS0.55_seisbench.csv"
DETECTIONS_CSV = BASE_IN / f"detections_day_{JSTR}_THP0.70_THS0.55_seisbench.csv"

BASE_OUT = Path("/data/murbina/seismo/results")
BASE_OUT.mkdir(parents=True, exist_ok=True)

OUT_EVENTS_LOC_CSV = BASE_OUT / f"events_loc_{JSTR}.csv"
# Tolerancia para match entre eventos IA y oficiales 
T_TOL = 15.0      # segundos
D_TOL_KM = 15.0   # km (ajústalo)
Z_TOL_KM = 10.0   # km (ajústalo)

# Parámetros para construir eventos
BIN_SEC = 3.0 # agrupa picks en este tiempo (en una misma estacion) para quedarse con el más fuerte 
WIN_SEC = 30.0 # tamaño de ventana para agrupar picks P entre estaciones 
MIN_STATIONS = 3 # mínimo de estaciones con picks P para considerar un evento
DEAD_SEC = 5.0 # tiempo mínimo entre eventos 

# Parametros para filtrar eventos 
STRONG_THR = 0.75 # umbral de probabilidad para considerar un pick como “fuerte” 
F_MIN_STATIONS = 5 # mínimo de estaciones con picks P 
F_MAXPROB = 0.85 # probabilidad máxima necesaria de una estación para considerar un evento 
F_MIN_STRONG = 4 # mínimo de estaciones con picks P fuertes para un mismo evento
F_MIN_DET_SUPPORT = 47  # mínimo de detecciones (no picks) dentro de un evento para considerarlo 
F_MIN_S_SUPPORT = 8 # mínimo de picks S dentro de un evento para considerarlo

# Ventana para asignar picks al evento
P_WIN = (-5.0, 10.0)   # seconds relative to t0
S_WIN = (-3.0, 25.0)

# Grid (ajústalo a Cartago / tu red)
LAT_MIN, LAT_MAX, DLAT = 9.25, 10.12, 0.01 
LON_MIN, LON_MAX, DLON = -84.7, -83.16, 0.01
DEPTHS_KM = [1, 3, 5, 8, 10, 15, 20] 

# Modelo 1D por capas (desde superficie, en km) ---
# thickness_km, Vp, Vs
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

# Coarse grid (rápido)
DLAT_C = 0.05
DLON_C = 0.05
DEPTHS_COARSE = [3, 8, 15]
COARSE_TOPK = 8

# --- NUEVO: cuántos coarse candidates refinamos en Fine-1 ---
N_FINE1_FROM_COARSE = 2   # <-- clave: de 8 fines bajas a 2

# Fine-1: resolución "normal" en una zona mediana
REFINE1_HALFSPAN_DEG = 0.06
DLAT_F1 = 0.01
DLON_F1 = 0.01

# Fine depth alrededor del dep coarse (para Fine-1)
DEPTH_FINE_HALFSPAN_KM = 8.0
DEPTH_FINE_DZ_KM = 0.5

# Fine-2: alta resolución en zona pequeña (1 sola vez)
REFINE2_HALFSPAN_DEG = 0.01
DLAT_F2 = 0.002
DLON_F2 = 0.002

# Fine-2 depth más apretado
DEPTH2_HALFSPAN_KM = 2.0
DEPTH2_DZ_KM = 0.25


# Parámetros para imprimir progreso 
PRINT_EVERY = 999999         # imprime cada N eventos (1 = todos)
SHOW_GRID_PROGRESS = True
GRID_PROGRESS_EVERY = 20000   # cada cuántas celdas del grid imprime avance (fine/coarse)

# --- Doble Diferencia settings ---

DD_PHASE = "P"
DD_MAX_PAIR_DIST_KM = 30.0     # eventos vecinos
DD_MAX_PAIR_DT_SEC  = 900.0    # opcional: 5 min
DD_MIN_COMMON_STA    = 3       # estaciones P en común
DD_K_NEIGHBORS       = 30      # vecinos por evento (reduce costo)

DD_ITERS = 5
DD_DAMP = 0.3                  # damping (regularización) para estabilidad
DD_STEP_LAT = 0.002            # deg (para derivadas finitas)
DD_STEP_LON = 0.002            # deg
DD_STEP_Z   = 0.5              # km

DD_ADJUST_DEPTH = False        # <-- por ahora NO ajustar z
# Carga catalogo oficial
def load_official_day(csv_path, day_yyyymmdd):
    df = pd.read_csv(csv_path)

    # filtra por día
    df = df[df["date"] == int(day_yyyymmdd)].copy()

    # HHMMSScc (HH=hora, MM=minuto, SS=segundo, cc=centisegundo)
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
            # 1 pick por estación (el primero en tiempo dentro de la ventana)
            per_sta = w.sort_values("t").drop_duplicates("station", keep="first")

            maxprob = float(per_sta["prob"].max())
            meanprob = float(per_sta["prob"].mean())
            n_strong = int((per_sta["prob"] >= strong_thr).sum())

            # tiempo representativo del evento: mediana 0.25 de picks por estación
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
# Normaliza formato de detections
def normalize_seisbench_detections(det_raw: pd.DataFrame) -> pd.DataFrame:
    d = det_raw.copy()
    d["start_time"] = pd.to_datetime(d["start_time"], errors="coerce")
    d["end_time"]   = pd.to_datetime(d["end_time"], errors="coerce")
    d = d.dropna(subset=["start_time", "end_time"])
    d["station"] = d["station_code"].astype(str)
    # a epoch seconds para comparar con t0
    d["start_t"] = d["start_time"].astype("int64") / 1e9
    d["end_t"]   = d["end_time"].astype("int64") / 1e9
    return d[["station", "start_t", "end_t"]].sort_values(["station", "start_t"]).reset_index(drop=True)
# Extrae código de estación
def build_det_index(det_norm: pd.DataFrame):
    idx = {}
    for sta, g in det_norm.groupby("station"):
        idx[sta] = (g["start_t"].to_numpy(), g["end_t"].to_numpy())
    return idx
# Cuenta detections que apoyan un evento en t0 
def det_support_count(t0, det_idx, slack_sec) -> int:
    tL = t0 - slack_sec
    tR = t0 + slack_sec
    n = 0
    for sta, (starts, ends) in det_idx.items():
        # Encontrar detections cuyo start <= tR
        # y verificar si alguna tiene end >= tL
        # (búsqueda rápida con searchsorted)
        j = np.searchsorted(starts, tR, side="right") - 1
        if j >= 0 and ends[j] >= tL:
            n += 1
    return n
# Cuenta picks de fase dentro de ventana de evento
def count_phase_support(picks, t0, phase, w0, w1):
    # picks: tu df normalizado (station, phase, t, prob)
    w = picks[(picks["phase"] == phase) & (picks["t"] >= t0 + w0) & (picks["t"] <= t0 + w1)]
    return w["station"].nunique()

#====================
# Funciones para Localización por grid search
#====================
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
    # distancia aproximada en km
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
    """
    Tiempo de viaje con rayo recto en un medio estratificado por capas horizontales.
    dist_km: distancia horizontal
    depth_km: profundidad fuente (positiva)
    phase: "P" o "S"
    """
    depth_km = float(max(depth_km, 0.0))
    # longitud total del rayo (recto)
    L = float(np.sqrt(dist_km**2 + depth_km**2))
    if L <= 0:
        return 0.0

    # proyección vertical del rayo: z(s) = (depth/L)*s
    # => fracción de camino dentro de cada intervalo de profundidad
    tt = 0.0
    z0 = 0.0
    z1 = depth_km

    # recorre capas desde z=0 hasta z=depth_km
    for i, (th, vp, vs) in enumerate(VEL_LAYERS):
        top = LAYER_TOPS[i]
        bot = LAYER_TOPS[i+1]

        # segmento del rayo que cae dentro de [top, bot] intersectado con [0, depth]
        a = max(top, 0.0)
        b = min(bot, z1)
        if b <= a:
            if top >= z1:
                break
            continue

        # convertir tramo vertical (dz) a tramo a lo largo del rayo (ds)
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

    # ---- arma observaciones ----
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
        bests = []  # heap de tamaño COARSE_TOPK: (-rms, cand)
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
                    score = np.median(np.abs(res))   # L1 robusto

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



    # ---- COARSE ----
    lats_c = np.arange(lat_min, lat_max + 1e-12, DLAT_C)
    lons_c = np.arange(lon_min, lon_max + 1e-12, DLON_C)

    cands = eval_grid(lats_c, lons_c, DEPTHS_COARSE, label="coarse")
    if not cands:
        return None

    # --- SOLO LOS N MEJORES COARSE para Fine-1 ---
    cands = cands[:N_FINE1_FROM_COARSE]

    best_f1 = None  # mejor candidato después de Fine-1

    # =========================
    # Fine-1: solo N candidatos
    # =========================
    for ic, (rms_c, lat_c, lon_c, dep_c, torig_c) in enumerate(cands, start=1):

        lat0 = max(lat_min, lat_c - REFINE1_HALFSPAN_DEG)
        lat1 = min(lat_max, lat_c + REFINE1_HALFSPAN_DEG)
        lon0 = max(lon_min, lon_c - REFINE1_HALFSPAN_DEG)
        lon1 = min(lon_max, lon_c + REFINE1_HALFSPAN_DEG)

        # depths fine alrededor del depth coarse
        z0 = max(0.5, float(dep_c) - DEPTH_FINE_HALFSPAN_KM)
        z1 = float(dep_c) + DEPTH_FINE_HALFSPAN_KM
        depths_f1 = np.arange(z0, z1 + 1e-12, DEPTH_FINE_DZ_KM)

        lats_f1 = np.arange(lat0, lat1 + 1e-12, DLAT_F1)
        lons_f1 = np.arange(lon0, lon1 + 1e-12, DLON_F1)

        fine1 = eval_grid(lats_f1, lons_f1, depths_f1, label=f"fine1_{ic}")

        cand_f1 = fine1[0] if fine1 else (rms_c, lat_c, lon_c, dep_c, torig_c)

        if (best_f1 is None) or (cand_f1[0] < best_f1[0]):
            best_f1 = cand_f1

    # =========================
    # Fine-2: 1 sola vez (alta res)
    # =========================
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
    # Devuelve límites acumulados: [0, z1, z2, ...]
    tops = [0.0]
    z = 0.0
    for th, _, _ in layers:
        z += float(th)
        tops.append(z)
    return np.array(tops)

LAYER_TOPS = build_layer_tops(VEL_LAYERS)



#====================
# Funciones para comparar eventos localizados con catálogo oficial
#====================

def match_loc_to_official(official_df: pd.DataFrame, loc_df: pd.DataFrame,
                          t_tol=T_TOL, d_tol_km=D_TOL_KM, z_tol_km=Z_TOL_KM):
    """
    Retorna tabla por cada oficial con el mejor match (si existe).
    Requiere:
      official_df: t0_utc, lat, lon, depth_km
      loc_df: origin_time_utc, lat, lon, depth_km
    """
    off = official_df.copy()
    loc = loc_df[loc_df["ok"] == True].copy()

    off["t0"] = pd.to_datetime(off["t0_utc"])
    loc["t0"] = pd.to_datetime(loc["origin_time_utc"])

    loc_t = (loc["t0"].astype("int64") / 1e9).to_numpy()
    out_rows = []

    for i, r in off.iterrows():
        t_off = float(pd.to_datetime(r["t0"]).value / 1e9)

        # candidatos por tiempo
        dt = np.abs(loc_t - t_off)
        cand_idx = np.where(dt <= t_tol)[0]
        if cand_idx.size == 0:
            # Diagnóstico: evento localizado más cercano en tiempo (aunque esté fuera de tolerancia)
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
                j_near = None
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

                # info del localizado más cercano (para debug)
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
            # score: prioriza tiempo, luego distancia, luego profundidad
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
            # si no hubo ok por dist/prof, guarda el más cercano en tiempo para diagnosticar
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
    nP = int(pP["station"].nunique()) if not pP.empty else 0
    nS = int(pS["station"].nunique()) if not pS.empty else 0

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
def build_event_P_obs(picks: pd.DataFrame, loc_df: pd.DataFrame, p_win=(-5.0, 10.0), min_prob=0.0):
    """
    Devuelve dict: obs[event_idx][station] = t_pick_epoch_seconds
    Usa ventana alrededor de t0_seed (si existe) o origin_time_utc.
    """
    obs = {}
    for _, ev in loc_df.iterrows():
        eid = int(ev["event_idx"])
        if pd.notna(ev.get("t0_seed", pd.NaT)):
            t0 = pd.to_datetime(ev["t0_seed"]).value / 1e9
        else:
            t0 = pd.to_datetime(ev["origin_time_utc"]).value / 1e9

        w0, w1 = p_win
        d = picks[(picks["phase"] == "P") &
                  (picks["t"] >= t0 + w0) &
                  (picks["t"] <= t0 + w1) &
                  (picks["prob"] >= min_prob)].copy()
        if d.empty:
            obs[eid] = {}
            continue

        # 1 pick por estación: mayor prob
        d = d.sort_values("prob", ascending=False).drop_duplicates("station", keep="first")
        obs[eid] = dict(zip(d["station"].astype(str), d["t"].astype(float)))
    return obs

def build_event_pairs(loc_df: pd.DataFrame, max_dist_km=15.0, max_dt_sec=300.0, k_neighbors=10):
    """
    Retorna lista de pares (eid_i, eid_j).
    """
    ev = loc_df.copy()
    ev = ev[ev["ok"] == True].copy()

    ev["eid"] = ev["event_idx"].astype(int)
    ev["t0"] = pd.to_datetime(ev["origin_time_utc"]).astype("int64") / 1e9

    eids = ev["eid"].to_numpy()
    lat = ev["lat"].to_numpy(dtype=float)
    lon = ev["lon"].to_numpy(dtype=float)
    t0  = ev["t0"].to_numpy(dtype=float)

    pairs = set()
    for a in range(len(eids)):
        # distancia a todos (barato para ~70)
        dkm = haversine_km(lat[a], lon[a], lat, lon)
        dt  = np.abs(t0[a] - t0)

        mask = (dkm <= max_dist_km) & (dt <= max_dt_sec) & (eids != eids[a])
        cand_idx = np.where(mask)[0]
        if cand_idx.size == 0:
            continue

        # toma los K más cercanos
        cand_sorted = cand_idx[np.argsort(dkm[cand_idx])][:k_neighbors]
        for b in cand_sorted:
            i, j = int(eids[a]), int(eids[b])
            if i == j:
                continue
            if i < j:
                pairs.add((i, j))
            else:
                pairs.add((j, i))

    return sorted(pairs)

def tt_for_event_station(lat_e, lon_e, z_e, sta_lat, sta_lon, sta_elev_km, phase="P"):
    dist = float(haversine_km(lat_e, lon_e, sta_lat, sta_lon))
    return travel_time_layered(dist, float(z_e + sta_elev_km), phase)

def finite_diff_partials(lat_e, lon_e, z_e, sta_lat, sta_lon, sta_elev_km, phase="P",
                         dlat=0.002, dlon=0.002, dz=0.5):
    tt0 = tt_for_event_station(lat_e, lon_e, z_e, sta_lat, sta_lon, sta_elev_km, phase)

    tt_lat = tt_for_event_station(lat_e + dlat, lon_e, z_e, sta_lat, sta_lon, sta_elev_km, phase)
    tt_lon = tt_for_event_station(lat_e, lon_e + dlon, z_e, sta_lat, sta_lon, sta_elev_km, phase)
    dTT_dlat = (tt_lat - tt0) / dlat
    dTT_dlon = (tt_lon - tt0) / dlon

    tt_z = tt_for_event_station(lat_e, lon_e, z_e + dz, sta_lat, sta_lon, sta_elev_km, phase)
    dTT_dz = (tt_z - tt0) / dz

    return tt0, dTT_dlat, dTT_dlon, dTT_dz

def run_dd_reloc(loc_df, obsP, stations, pairs,
                 adjust_depth=False, min_common=6, iters=5, damp=0.3):

    # Solo eventos ok
    ev = loc_df[loc_df["ok"] == True].copy()
    ev = ev.set_index("event_idx", drop=False)
    eids = ev["event_idx"].astype(int).to_list()

    ref_eid = eids[0]  # ancla
    # mapea eid -> bloque de columnas
    # params por evento: dlat, dlon, dt0 (+ dz opcional)
    npar = 3 + (1 if adjust_depth else 0)

    col0 = {}
    c = 0
    for eid in eids:
        if eid == ref_eid:
            continue
        col0[eid] = c
        c += npar
    n_unknowns = c

    # estado actual
    lat = ev["lat"].astype(float).to_dict()
    lon = ev["lon"].astype(float).to_dict()
    z   = ev["depth_km"].astype(float).to_dict()
    t0  = (pd.to_datetime(ev["origin_time_utc"]).astype("int64") / 1e9).to_dict()

    # estaciones indexadas
    st = stations.copy()
    if "elev_km" not in st.columns:
        st["elev_km"] = 0.0

    for it in range(1, iters + 1):
        rows = []
        rhs  = []

        used_eq = 0
        for (i, j) in pairs:
            oi = obsP.get(i, {})
            oj = obsP.get(j, {})
            common = set(oi.keys()) & set(oj.keys())
            if len(common) < min_common:
                continue

            for sta in common:
                if sta not in st.index:
                    continue

                # observado
                dt_obs = float(oi[sta] - oj[sta])

                # predicho
                sta_lat = float(st.loc[sta, "lat"])
                sta_lon = float(st.loc[sta, "lon"])
                sta_el  = float(st.loc[sta, "elev_km"])

                # TT + derivadas
                tti, dti_lat, dti_lon, dti_z = finite_diff_partials(
                    lat[i], lon[i], z[i], sta_lat, sta_lon, sta_el, "P",
                    DD_STEP_LAT, DD_STEP_LON, DD_STEP_Z
                )
                ttj, dtj_lat, dtj_lon, dtj_z = finite_diff_partials(
                    lat[j], lon[j], z[j], sta_lat, sta_lon, sta_el, "P",
                    DD_STEP_LAT, DD_STEP_LON, DD_STEP_Z
                )

                dt_pred = (t0[i] - t0[j]) + (tti - ttj)
                r = dt_obs - dt_pred  # queremos A x = r

                # arma fila sparse en formato denso (para ~70 eventos sirve)
                a = np.zeros(n_unknowns, dtype=float)

                def add_event(eid, sgn, dTTlat, dTTlon, dTTz):
                    # sgn = +1 para i, -1 para j (porque dd_pred usa +t0_i -t0_j, +TT_i - TT_j)
                    if eid == ref_eid:
                        return
                    base = col0[eid]
                    # dd_pred deriv wrt lat = dTT/dlat, etc.
                    a[base + 0] += sgn * dTTlat
                    a[base + 1] += sgn * dTTlon
                    # dt0
                    a[base + 2] += sgn * 1.0
                    if adjust_depth:
                        a[base + 3] += sgn * dTTz

                add_event(i, +1.0, dti_lat, dti_lon, dti_z)
                add_event(j, -1.0, dtj_lat, dtj_lon, dtj_z)

                rows.append(a)
                rhs.append(r)
                used_eq += 1

        if used_eq == 0:
            print("[DD] sin ecuaciones (revisa min_common / pairs)")
            break

        A = np.vstack(rows)
        b = np.array(rhs, dtype=float)

        # damping: agrega pseudo-observaciones para mantener updates pequeños
        # (damp * I) x = 0
        A_d = np.sqrt(damp) * np.eye(n_unknowns)
        b_d = np.zeros(n_unknowns)
        A2 = np.vstack([A, A_d])
        b2 = np.concatenate([b, b_d])

        x, *_ = np.linalg.lstsq(A2, b2, rcond=None)

        # aplica updates
        max_d = 0.0
        for eid in eids:
            if eid == ref_eid:
                continue
            base = col0[eid]
            dlat = x[base + 0]
            dlon = x[base + 1]
            dt0s = x[base + 2]
            dz   = x[base + 3] if adjust_depth else 0.0

            lat[eid] += dlat
            lon[eid] += dlon
            t0[eid]  += dt0s
            if adjust_depth:
                z[eid]   += dz

            max_d = max(max_d, abs(dlat), abs(dlon), abs(dt0s), abs(dz))

        # log simple
        res = A.dot(x) - b
        mad = np.median(np.abs(res - np.median(res)))
        print(f"[DD] iter {it}/{iters}  eq={used_eq}  med|res|={np.median(np.abs(res)):.3f}s  MAD={mad:.3f}s  max_step={max_d:.3g}")

    # arma salida
    out = loc_df.copy()
    mask = out["ok"] == True
    out.loc[mask, "lat_dd"] = out.loc[mask, "event_idx"].map(lat)
    out.loc[mask, "lon_dd"] = out.loc[mask, "event_idx"].map(lon)
    out.loc[mask, "depth_km_dd"] = out.loc[mask, "event_idx"].map(z)
    out.loc[mask, "origin_time_utc_dd"] = out.loc[mask, "event_idx"].map(lambda eid: pd.to_datetime(t0[int(eid)], unit="s"))

    return out
    
# Globals (se llenan 1 vez por proceso)
_G_PICKS = None
_G_STATIONS = None

def _init_worker(picks, stations):
    global _G_PICKS, _G_STATIONS
    _G_PICKS = picks
    _G_STATIONS = stations


def main():
    t_start = time.time() # inicio medicion tiempo de ejecucion

    official = load_official_day(OFFICIAL_CSV, OFFICIAL_DAY)
    print(f"Oficial {OFFICIAL_DAY}: {len(official)} eventos")

    raw_picks = pd.read_csv(PICKS_CSV) # carga picks
    print(f"Picks SeisBench cargados: {len(raw_picks)}({PICKS_CSV})")

    picks = normalize_eqt_picks(raw_picks) # normaliza formato picks
    print(f"Picks normalizados: {len(picks)}  columnas={list(picks.columns)}")

    # ---- Binning P ----
    picksP = bin_best_pick_per_station(picks, bin_sec=BIN_SEC, phase="P") 
    print(f"\nP-picks tras bin ({BIN_SEC:.0f}s): {len(picksP)}")

    # ---- Construccion de eventos con ventana ----
    events_det = build_events_sliding_window( 
        picksP, win_sec=WIN_SEC, min_stations=MIN_STATIONS, dead_sec=DEAD_SEC, strong_thr=STRONG_THR
    )
    print(f"Eventos sin filtro: {len(events_det)}")

    # cargar detections si existe
    
    det_raw = pd.read_csv(DETECTIONS_CSV)
    det_norm = normalize_seisbench_detections(det_raw)
    det_idx = build_det_index(det_norm)

    # calcula soporte para cada evento
    events_det["det_support"] = events_det["t0"].apply(lambda t0: det_support_count(t0, det_idx, slack_sec=2.0))
    events_det["S_support"] = events_det["t0"].apply(lambda t0: count_phase_support(picks, t0, "S", -3, 20.0))

    # ---- Filtrado de eventos ----
    events_loc = events_det[
        (events_det["n_stations"] >= F_MIN_STATIONS) &
        (events_det["maxprob"] >= F_MAXPROB) &
        (events_det["n_strong"] >= F_MIN_STRONG) &
        (events_det["det_support"] >= F_MIN_DET_SUPPORT) &
        (events_det["S_support"] >= F_MIN_S_SUPPORT)
    ].reset_index(drop=True)


    print(
        f"\nEventos tras filtrado: {len(events_loc)} "
    )
    
    # stations indexado (como ya lo tenías)
    stations = load_station_nodes(NODES_CSV).set_index("station")

    # workers: usa SLURM_CPUS_PER_TASK si existe; si no, poné un número razonable
    n_workers = 5
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
    loc_df = loc_df.sort_values("origin_time_utc", na_position="last")
    obsP = build_event_P_obs(picks, loc_df, p_win=P_WIN, min_prob=0.0)
    # DEBUG: conteo de P por evento
    pcounts = np.array([len(obsP.get(int(eid), {})) for eid in loc_df.loc[loc_df["ok"]==True, "event_idx"]], dtype=int)
    print("[DD] P por evento (p25/med/p75):", np.percentile(pcounts, [25,50,75]))
    print("[DD] eventos con <3 P:", int((pcounts < 3).sum()), "/", len(pcounts))
    print("[DD] eventos con <6 P:", int((pcounts < 6).sum()), "/", len(pcounts))
    pairs = build_event_pairs(
        loc_df,
        max_dist_km=DD_MAX_PAIR_DIST_KM,
        max_dt_sec=DD_MAX_PAIR_DT_SEC,
        k_neighbors=DD_K_NEIGHBORS
    )
    print("[DD] pares:", len(pairs))
    commons = []
    for (i, j) in pairs:
        ci = set(obsP.get(i, {}).keys())
        cj = set(obsP.get(j, {}).keys())
        commons.append(len(ci & cj))
    commons = np.array(commons, dtype=int) if commons else np.array([], dtype=int)
    print("[DD] common P por par (min/p25/med/p75/max):",
          (commons.min() if len(commons) else None,
           np.percentile(commons, 25) if len(commons) else None,
           np.percentile(commons, 50) if len(commons) else None,
           np.percentile(commons, 75) if len(commons) else None,
           commons.max() if len(commons) else None))
    print("[DD] pares con >=6 comunes:", int((commons >= 6).sum()), "/", len(commons))
    print("[DD] pares con >=3 comunes:", int((commons >= 3).sum()), "/", len(commons))
    # --- DD relocation (P only) ---
    obsP = build_event_P_obs(picks, loc_df, p_win=P_WIN, min_prob=0.0)
    pairs = build_event_pairs(loc_df, DD_MAX_PAIR_DIST_KM, DD_MAX_PAIR_DT_SEC, DD_K_NEIGHBORS)
    
    print(f"[DD] pares: {len(pairs)}")
    loc_dd = run_dd_reloc(
        loc_df, obsP, stations,
        pairs,
        adjust_depth=DD_ADJUST_DEPTH,      # False por ahora
        min_common=DD_MIN_COMMON_STA,
        iters=DD_ITERS,
        damp=DD_DAMP
    )
    
    # guarda
    out_dd = BASE_OUT / f"events_loc_dd_{JSTR}.csv"
    loc_dd.to_csv(out_dd, index=False)
    print("Guardado DD:", out_dd)
    # comparar con oficial si existe
    m = match_loc_to_official(official, loc_df)
    m_ok = m[m["is_match"] == True].copy()
    m_bad = m[m["is_match"] == False].copy()
    m_ok["event_idx"] = m_ok["event_idx"].astype(int)

    print("\n=== MATCH LOCALIZADO vs OFICIAL (solo matches) ===")
    print(m_ok.sort_values("score").to_string(index=False))

    print("\n=== NO MATCH (detalle) ===")
    print(m_bad.to_string(index=False))
    print("shape:", m_bad.shape)
    print("cols:", list(m_bad.columns))

    print("median dist km:", m_ok["dist_km"].median())
    print("p90 dist km:", m_ok["dist_km"].quantile(0.9))
    print(f"\nTotal oficiales: {len(m)}")
    print(f"Eventos con match: {len(m_ok)}")
    print(f"Eventos sin match: {len(m_bad)}")

    loc_df.to_csv(OUT_EVENTS_LOC_CSV, index=False)
    print("Guardado:", OUT_EVENTS_LOC_CSV)
    #print(f"💾 Localizaciones guardadas: {OUT_EVENTS_LOC_CSV}  (ok={loc_df['ok'].sum()}/{len(loc_df)})")
    print(f"\n⏱ Total: {time.time() - t_start:.1f}s")

loc_df, m_ok, m_bad = main()