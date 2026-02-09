import time
from pathlib import Path
import numpy as np
import pandas as pd

# Configuracion temporal
YEAR = 2024
JDAY = 189
OFFICIAL_DAY = 20240707  # AJUSTAR al día real (yyyymmdd)
JSTR = f"{YEAR}{JDAY:03d}"

# Entradas 
PICKS_CSV = f"picks_day_{JSTR}_THP0.70_THS0.55_seisbench.csv"
DETECTIONS_CSV = f"detections_day_{JSTR}_THP0.70_THS0.55_seisbench.csv"  # opcional (si existe)

# Catalogo Oficial
OFFICIAL_XLSX = "catalogo_oficial.xlsx"
OFFICIAL_SHEET = 0

# Tolerancia para match entre eventos IA y oficiales 
MATCH_TOL_SEC = 15.0

# Parámetros para construir eventos
BIN_SEC = 8.0 # agrupa picks en este tiempo (en una misma estacion) para quedarse con el más fuerte 
WIN_SEC = 30.0 # tamaño de ventana para agrupar picks P entre estaciones 
MIN_STATIONS = 4 # mínimo de estaciones con picks P para considerar un evento
DEAD_SEC = 18.0 # tiempo mínimo entre eventos 

# Parametros para filtrar eventos 
STRONG_THR = 0.75 # umbral de probabilidad para considerar un pick como “fuerte” 
F_MIN_STATIONS = 7 # mínimo de estaciones con picks P 
F_MAXPROB = 0.85 # probabilidad máxima necesaria de una estación para considerar un evento 
F_MIN_STRONG = 5 # mínimo de estaciones con picks P fuertes para un mismo evento
F_MIN_DET_SUPPORT = 50  # mínimo de detecciones (no picks) dentro de un evento para considerarlo 
F_MIN_S_SUPPORT = 10 # mínimo de picks S dentro de un evento para considerarlo

# Carga catalogo oficial
def load_official_day(xlsx_path, sheet, day_yyyymmdd):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    # "date", "time", "lat", "lon", "depth_km", "mag"
    df = df[df["date"] == int(day_yyyymmdd)].copy()
    # HHMMSScc (HH=hora, MM=minuto, SS=segundo, cc=centisegundo)
    ts = df["time"].astype("Int64").astype(str).str.zfill(8)
    hh = ts.str.slice(0, 2).astype(int)
    mm = ts.str.slice(2, 4).astype(int)
    ss = ts.str.slice(4, 6).astype(int)
    cc = ts.str.slice(6, 8).astype(int)
    # construye base datetime (a segundos) + centiseconds*10ms
    date_str = df["date"].astype("Int64").astype(str)
    base = pd.to_datetime(date_str + " " + hh.astype(str).str.zfill(2) + ":" +
                          mm.astype(str).str.zfill(2) + ":" +
                          ss.astype(str).str.zfill(2),
                          format="%Y%m%d %H:%M:%S", errors="raise")
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

            # tiempo representativo del evento: cuantil 0.25 de picks por estación
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

def main():
    t_start = time.time()

    official = load_official_day(OFFICIAL_XLSX, OFFICIAL_SHEET, OFFICIAL_DAY)
    print(f"Oficial {OFFICIAL_DAY}: {len(official)} eventos")

    raw_picks = pd.read_csv(PICKS_CSV)
    print(f"Picks SeisBench cargados: {len(raw_picks)}({PICKS_CSV})")

    picks = normalize_eqt_picks(raw_picks)
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
    
    
    print(f"\n⏱ Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
