#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import numpy as np
import pandas as pd

# =====================
# CONFIG
# =====================
YEAR = 2024
JDAY = 191
JSTR = f"{YEAR}{JDAY:03d}"

# Entradas SeisBench (de tu 01)
PICKS_CSV = f"picks_day_{JSTR}_THP0.70_THS0.55_seisbench.csv"
DETECTIONS_CSV = f"detections_day_{JSTR}_THP0.70_THS0.55_seisbench.csv"  # opcional (si existe)

# Oficial (Excel)
OFFICIAL_XLSX = "catalogo_oficial.xlsx"
OFFICIAL_SHEET = 0
OFFICIAL_DAY = 20240709  # AJUSTA al día real (yyyymmdd)

MATCH_TOL_SEC = 15.0

# Parámetros para construir “eventos IA” (DET) con picks P
BIN_SEC = 8.0
WIN_SEC = 30.0
MIN_STATIONS = 4
DEAD_SEC = 18.0
STRONG_THR = 0.75

# Filtros “LOC” (para localizar, no para recall)
F_MIN_STATIONS = 7
F_MAXPROB = 0.85
F_MIN_STRONG = 5
F_MIN_DET_SUPPORT = 50  # prueba 6–10, tú tienes 73 estaciones
F_MIN_S_SUPPORT = 10

# =====================
# Helpers: oficial
# =====================
def parse_official_datetime_utc(date_yyyymmdd, time_hhmmsscc):
    ds = str(int(date_yyyymmdd))
    ts = str(int(time_hhmmsscc)).zfill(8)  # HHMMSScc
    hh = int(ts[0:2]); mm = int(ts[2:4]); ss = int(ts[4:6]); cc = int(ts[6:8])
    # cc = centiseconds -> 10 ms
    return pd.Timestamp(f"{ds[0:4]}-{ds[4:6]}-{ds[6:8]} {hh:02d}:{mm:02d}:{ss:02d}") + pd.Timedelta(milliseconds=10*cc)

def load_official_day(xlsx_path, sheet, day_yyyymmdd):
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    required = {"date", "time", "lat", "lon", "depth_km", "mag"}
    missing = required - set(df.columns.astype(str))
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}. Columnas encontradas: {list(df.columns)}")

    df = df[df["date"] == int(day_yyyymmdd)].copy()
    df["t0_utc"] = df.apply(lambda r: parse_official_datetime_utc(r["date"], r["time"]), axis=1)
    df = df.sort_values("t0_utc").reset_index(drop=True)
    return df


# =====================
# Normalización: SeisBench picks CSV -> formato tuyo
# =====================
def _extract_station_code_from_trace_id(s: str) -> str:
    # trace_id típico: NET.STA.LOC.CHAN
    parts = str(s).split(".")
    if len(parts) >= 2:
        return parts[1]
    return str(s)

def normalize_seisbench_picks(picks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte el CSV de PickList.to_dataframe() a columnas:
      station, phase, prob, time_utc, t
    """
    df = picks_df.copy()

    # Validaciones mínimas
    if "time" not in df.columns:
        raise ValueError(f"El CSV no trae columna 'time'. Columnas: {list(df.columns)}")

    # SeisBench suele usar 'probability'
    if "probability" in df.columns:
        df["prob"] = df["probability"].astype(float)
    elif "prob" in df.columns:
        df["prob"] = df["prob"].astype(float)
    else:
        # a veces aparece 'peak_value' u otras; si no existe, te lo marcamos
        raise ValueError(f"El CSV no trae 'probability' (ni 'prob'). Columnas: {list(df.columns)}")

    # Phase: suele venir como 'phase'
    if "phase" not in df.columns:
        raise ValueError(f"El CSV no trae columna 'phase'. Columnas: {list(df.columns)}")
    df["phase"] = df["phase"].astype(str).str.upper()

    # station: en SeisBench normalmente es trace_id o 'station_code' que ya añadiste
    if "station_code" in df.columns:
        df["station"] = df["station_code"].astype(str)
    elif "station" in df.columns:
        # si viene como trace_id, extraemos STA
        df["station"] = df["station"].apply(_extract_station_code_from_trace_id)
    else:
        raise ValueError(f"El CSV no trae 'station' ni 'station_code'. Columnas: {list(df.columns)}")

    # time -> datetime
    df["t_pick"] = pd.to_datetime(df["time"])
    # time_utc como string ISO (ms)
    df["time_utc"] = df["t_pick"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    # epoch seconds
    df["t"] = df["t_pick"].astype("int64") / 1e9

    # columnas finales (más cualquier otra extra que quieras conservar)
    out = df[["station", "phase", "prob", "time_utc", "t"]].copy()
    out = out.sort_values("t").reset_index(drop=True)
    return out


# =====================
# Picks -> bin
# =====================
def bin_best_pick_per_station(df, bin_sec=5.0, phase="P"):
    d = df[df.phase == phase].copy()
    if d.empty:
        return d
    d["bin"] = (d["t"] // bin_sec).astype(int)
    d = d.sort_values("prob", ascending=False).drop_duplicates(["station", "bin"])
    return d.drop(columns=["bin"]).sort_values("t").reset_index(drop=True)


# =====================
# Sliding window events (DET)
# =====================
def build_events_sliding_window(picksP, win_sec=25.0, min_stations=3, dead_sec=12.0, strong_thr=0.83):
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

            # tiempo “representativo” del evento: cuantil 0.25 de picks por estación
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
# Matching oficial <-> picks / eventos
# =====================
def report_official_pick_details(official, picks_df, tol_sec=15.0, phase="P", limit_sta=30):
    if official.empty:
        print("No hay eventos oficiales.")
        return

    d = picks_df.copy()
    d["t_pick"] = pd.to_datetime(d["time_utc"], utc=False)

    print("\n==============================")
    print(f"DETALLE PICKS vs OFICIAL ({phase}, ±{tol_sec:.0f}s)")
    print("==============================")

    for i, ev in official.iterrows():
        t0 = ev["t0_utc"]
        w0 = t0 - pd.Timedelta(seconds=tol_sec)
        w1 = t0 + pd.Timedelta(seconds=tol_sec)

        near = d[(d["phase"] == phase) & (d["t_pick"] >= w0) & (d["t_pick"] <= w1)].copy()

        print(f"\n--- Oficial idx={i}  t0={t0} ---")
        if near.empty:
            print("  (sin picks en ventana)")
            continue

        # 1 pick por estación: mayor prob
        per_sta = near.sort_values("prob", ascending=False).drop_duplicates("station", keep="first")
        per_sta["dt_sec"] = (per_sta["t_pick"] - t0).dt.total_seconds()
        per_sta = per_sta.sort_values("dt_sec")

        nsta = per_sta["station"].nunique()
        maxprob = float(per_sta["prob"].max())
        meanprob = float(per_sta["prob"].mean())
        spread_sec = float((per_sta["t_pick"].max() - per_sta["t_pick"].min()).total_seconds())

        print(f"  n_stations={nsta} | maxprob={maxprob:.3f} | meanprob={meanprob:.3f} | spread={spread_sec:.1f}s")
        show = per_sta[["station","time_utc","dt_sec","prob"]].head(limit_sta)
        print(show.to_string(index=False))


def match_official_with_picks_count(official_df, picks_df, tol_sec=MATCH_TOL_SEC, phase="P"):
    picks_df = picks_df.copy()
    picks_df["t_pick"] = pd.to_datetime(picks_df["time_utc"], utc=False)

    results = []
    for i, ev in official_df.iterrows():
        t0 = ev["t0_utc"]
        w0 = t0 - pd.Timedelta(seconds=tol_sec)
        w1 = t0 + pd.Timedelta(seconds=tol_sec)

        near = picks_df[
            (picks_df["t_pick"] >= w0) & (picks_df["t_pick"] <= w1) & (picks_df["phase"] == phase)
        ]
        n_sta = near["station"].nunique()
        results.append({"idx_official": i, "t0_utc": t0, f"n_stations_{phase}": int(n_sta)})

    return pd.DataFrame(results)


def match_events_to_official(official, events, tol_sec=15.0):
    if events.empty:
        return pd.DataFrame(columns=["idx_official","t0_official","t0_ia","dt_sec","ia_n_stations","is_match"])

    out = []
    for i, r in official.iterrows():
        t0 = r["t0_utc"].timestamp()

        cand = events[(events["t0"] >= t0 - tol_sec) & (events["t0"] <= t0 + tol_sec)].copy()

        if cand.empty:
            out.append({
                "idx_official": i,
                "t0_official": r["t0_utc"],
                "t0_ia": pd.NaT,
                "dt_sec": np.nan,
                "ia_n_stations": 0,
                "is_match": False
            })
            continue

        cand["dt_sec"] = cand["t0"] - t0
        cand["abs_dt"] = cand["dt_sec"].abs()

        # ranking: más estaciones, luego maxprob, luego menor abs_dt
        cand = cand.sort_values(["n_stations", "maxprob", "abs_dt"], ascending=[False, False, True])
        best = cand.iloc[0]

        out.append({
            "idx_official": i,
            "t0_official": r["t0_utc"],
            "t0_ia": best["t0_utc"],
            "dt_sec": float(best["dt_sec"]),
            "ia_n_stations": int(best["n_stations"]),
            "is_match": True
        })

    return pd.DataFrame(out)


def debug_near_misses(official, events, tol_sec=15.0, near_sec=60.0):
    if events.empty:
        return
    print("\n==============================")
    print(f"NEAR MISSES (tol={tol_sec}s, mirando hasta {near_sec}s)")
    print("==============================")
    ev_t = events["t0"].values
    for i, r in official.iterrows():
        t0 = r["t0_utc"].timestamp()
        dt = np.abs(ev_t - t0)
        j = int(np.argmin(dt))
        best_dt = float(ev_t[j] - t0)
        if abs(best_dt) > tol_sec and abs(best_dt) <= near_sec:
            row = events.iloc[j]
            print(f"idx={i} t0={r['t0_utc']}  nearest_dt={best_dt:.2f}s  nsta={row['n_stations']}  maxprob={row.get('maxprob', np.nan):.3f}")

# =====================
# Normalización: SeisBench detections CSV -> formato tuyo
# =====================
def normalize_seisbench_detections(det_raw: pd.DataFrame) -> pd.DataFrame:
    d = det_raw.copy()

    # nombres típicos de SeisBench DetectionList.to_dataframe()
    # start_time / end_time deberían existir
    if "start_time" not in d.columns or "end_time" not in d.columns:
        raise ValueError(f"Detections sin start_time/end_time. Columnas: {list(d.columns)}")

    d["start_time"] = pd.to_datetime(d["start_time"], errors="coerce")
    d["end_time"]   = pd.to_datetime(d["end_time"], errors="coerce")
    d = d.dropna(subset=["start_time", "end_time"])

    # station_code si existe, si no extrae de trace_id
    if "station_code" in d.columns:
        d["station"] = d["station_code"].astype(str)
    elif "station" in d.columns:
        d["station"] = d["station"].astype(str).apply(lambda x: str(x).split(".")[1] if "." in str(x) else str(x))
    else:
        raise ValueError(f"Detections sin station/station_code. Columnas: {list(d.columns)}")

    # a epoch seconds para comparar con tu t0
    d["start_t"] = d["start_time"].astype("int64") / 1e9
    d["end_t"]   = d["end_time"].astype("int64") / 1e9

    return d[["station", "start_t", "end_t"]].sort_values(["station", "start_t"]).reset_index(drop=True)


def build_det_index(det_norm: pd.DataFrame):
    """
    Devuelve dict station -> (start_array, end_array) ordenados por start.
    """
    idx = {}
    for sta, g in det_norm.groupby("station"):
        idx[sta] = (g["start_t"].to_numpy(), g["end_t"].to_numpy())
    return idx

# =====================
# Detections index -> count support
# =====================
def det_support_count(t0: float, det_idx: dict, slack_sec: float = 0.0) -> int:
    """
    slack_sec permite tolerancia: cuenta detections que cubren [t0-slack, t0+slack]
    """
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
def count_phase_support(picks, t0, phase, w0, w1):
    # picks: tu df normalizado (station, phase, t, prob)
    w = picks[(picks["phase"] == phase) & (picks["t"] >= t0 + w0) & (picks["t"] <= t0 + w1)]
    return w["station"].nunique()
# =====================
# Requisito mínimo de picks S según características del evento
# =====================
#def min_s_required(row):
    nsta = row["n_stations"]
    maxp = row["maxprob"]
    nstr = row["n_strong"]
    dets = row["det_support"]

    # "rescate": evento súper fuerte en P + alto soporte de detections
    if (nsta >= 13) and (nstr >= 10) and (maxp >= 0.925) and (dets >= 50):
        return 4
    return 10



# =====================
# MAIN
# =====================
def main():
    t_start = time.time()

    # ---- Oficial ----
    official = load_official_day(OFFICIAL_XLSX, OFFICIAL_SHEET, OFFICIAL_DAY)
    print(f"Oficial {OFFICIAL_DAY}: {len(official)} eventos")

    # ---- Picks SeisBench ----
    if not Path(PICKS_CSV).exists():
        raise FileNotFoundError(f"No existe {PICKS_CSV}. Revisa JDAY / thresholds / nombre.")

    raw_picks = pd.read_csv(PICKS_CSV)
    print(f"✅ Picks SeisBench cargados: {len(raw_picks)}  ({PICKS_CSV})")

    picks = normalize_seisbench_picks(raw_picks)
    print(f"✅ Picks normalizados: {len(picks)}  columnas={list(picks.columns)}")

    # ---- Diagnóstico P vs oficial ----
    reportP = match_official_with_picks_count(official, picks, tol_sec=MATCH_TOL_SEC, phase="P")
    ge1 = (reportP["n_stations_P"] >= 1).sum()
    ge2 = (reportP["n_stations_P"] >= 2).sum()
    ge3 = (reportP["n_stations_P"] >= 3).sum()

    report_official_pick_details(official, picks, tol_sec=MATCH_TOL_SEC, phase="P")

    print("\n=== MATCH (solo P, ±15s) ===")
    print(f"Oficiales con ≥1 estación con P-pick: {ge1}/{len(reportP)}")
    print(f"Oficiales con ≥2 estaciones con P-pick: {ge2}/{len(reportP)}")
    print(f"Oficiales con ≥3 estaciones con P-pick: {ge3}/{len(reportP)}")

    # ---- Binning P ----
    picksP = bin_best_pick_per_station(picks, bin_sec=BIN_SEC, phase="P")
    print(f"\nP-picks tras bin ({BIN_SEC:.0f}s): {len(picksP)}")

    # ---- Eventos IA DET (ventana) ----
    events_det = build_events_sliding_window(
        picksP, win_sec=WIN_SEC, min_stations=MIN_STATIONS, dead_sec=DEAD_SEC, strong_thr=STRONG_THR
    )
    print(f"Eventos IA DET (sin filtro): {len(events_det)}")

    m_det = match_events_to_official(official, events_det, tol_sec=MATCH_TOL_SEC)
    print("\n=== MATCH EVENTOS DET vs OFICIAL (±15s) ===")
    print(f"Matches DET: {m_det['is_match'].sum()}/{len(m_det)}")
    print(m_det.sort_values("dt_sec").to_string(index=False))
    # cargar detections si existe
    if Path(DETECTIONS_CSV).exists():
        det_raw = pd.read_csv(DETECTIONS_CSV)
        det_norm = normalize_seisbench_detections(det_raw)
        det_idx = build_det_index(det_norm)

        # calcula soporte para cada evento
        events_det["det_support"] = events_det["t0"].apply(lambda t0: det_support_count(t0, det_idx, slack_sec=2.0))
        events_det["S_support"] = events_det["t0"].apply(lambda t0: count_phase_support(picks, t0, "S", 0.5, 20.0))

        print("Det-support stats (p25/med/p75):",
            np.percentile(events_det["det_support"], [25,50,75]))
    else:
        events_det["det_support"] = 0

    # ---- Eventos IA LOC (filtrado) ----
    #events_det["S_req"] = events_det.apply(min_s_required, axis=1)

    events_loc = events_det[
        (events_det["n_stations"] >= F_MIN_STATIONS) &
        (events_det["maxprob"] >= F_MAXPROB) &
        (events_det["n_strong"] >= F_MIN_STRONG) &
        (events_det["det_support"] >= F_MIN_DET_SUPPORT) &
        (events_det["S_support"] >= F_MIN_S_SUPPORT)
    ].reset_index(drop=True)


    print(
        f"\nEventos IA LOC tras filtrado: {len(events_loc)} "
        f"(nsta>={F_MIN_STATIONS}, maxprob>={F_MAXPROB}, n_strong>={F_MIN_STRONG}, det_support>={F_MIN_DET_SUPPORT}, S_support>={F_MIN_S_SUPPORT})"
    )
    

    m_loc = match_events_to_official(official, events_loc, tol_sec=MATCH_TOL_SEC)
    debug_near_misses(official, events_loc, tol_sec=MATCH_TOL_SEC, near_sec=90.0)

    print("\n=== MATCH EVENTOS LOC vs OFICIAL (±15s) ===")
    print(f"Matches LOC: {m_loc['is_match'].sum()}/{len(m_loc)}")
    print(m_loc.sort_values("dt_sec").to_string(index=False))
    

    # ---- (Opcional) Si existe detections CSV, imprime resumen ----
    if Path(DETECTIONS_CSV).exists():
        det = pd.read_csv(DETECTIONS_CSV)
        print(f"\n✅ Detections cargadas: {len(det)}  ({DETECTIONS_CSV})")
        # Intento de normalización de tiempos típicos
        for c in ["start_time", "end_time"]:
            if c in det.columns:
                det[c] = pd.to_datetime(det[c], errors="coerce")
        if "station_code" in det.columns:
            # ok
            pass
        elif "station" in det.columns:
            det["station_code"] = det["station"].apply(_extract_station_code_from_trace_id)
        # resumen rápido
        if "start_time" in det.columns:
            print("Detections por estación (top 10):")
            print(det["station_code"].value_counts().head(10).to_string())

    print(f"\n⏱ Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
