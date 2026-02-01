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
JDAY = 190
JSTR = f"{YEAR}{JDAY:03d}"

# Entrada picks
PICKS_CSV = f"picks_day_{JSTR}_THP0.70_THS0.55.csv"  # ajusta si cambiaste nombre

# Oficial (Excel)
OFFICIAL_XLSX = "catalogo_oficial.xlsx"
OFFICIAL_SHEET = 0
OFFICIAL_DAY = 20240708  # cambia según el día real

MATCH_TOL_SEC = 15.0

# Parámetros de debug (rápidos de cambiar)
BIN_SEC = 8.0
WIN_SEC = 25.0
MIN_STATIONS = 4
DEAD_SEC = 18.0

# Filtros de "catálogo para localizar" (NO para medir recall)
F_MAXPROB = 0.78
F_MEANPROB = 0.77

# Para diagnóstico
DEBUG_OFFICIAL_IDX_1 = 1
DEBUG_OFFICIAL_IDX_2 = 2


# =====================
# Oficial: parse
# =====================
def parse_official_datetime_utc(date_yyyymmdd, time_hhmmsscc):
    ds = str(int(date_yyyymmdd))
    ts = str(int(time_hhmmsscc)).zfill(8)  # HHMMSScc
    hh = int(ts[0:2]); mm = int(ts[2:4]); ss = int(ts[4:6]); cc = int(ts[6:8])
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
# Sliding window events
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
            per_sta = w.sort_values("t").drop_duplicates("station", keep="first")
            probs = np.sort(per_sta["prob"].values)[::-1]  # desc
            top1 = float(probs[0])
            top2 = float(probs[1]) if len(probs) >= 2 else np.nan
            top3 = float(probs[2]) if len(probs) >= 3 else np.nan
            med_top3 = float(np.nanmedian([top1, top2, top3]))

            maxprob = float(per_sta["prob"].max())
            meanprob = float(per_sta["prob"].mean())
            n_strong = int((per_sta["prob"] >= strong_thr).sum())
            t0 = float(np.quantile(per_sta["t"].values, 0.25))

            events.append({
                "t0": t0,
                "t0_utc": pd.to_datetime(t0, unit="s"),
                "n_stations": int(nsta),
                "n_picks": int(len(w)),
                "maxprob": maxprob,
                "meanprob": meanprob,
                "n_strong": n_strong,
                "top1": top1,
                "top2": top2,
                "top3": top3,
                "med_top3": med_top3
            })

            block_until = t0 + dead_sec
            while i < n and float(p.loc[i, "t"]) <= block_until:
                i += 1
        else:
            i += 1

    return pd.DataFrame(events).sort_values("t0").reset_index(drop=True)
# =====================
# Matching
# =====================
def match_official_with_picks(official_df, picks_df, tol_sec=MATCH_TOL_SEC):
    if official_df.empty:
        return None

    picks_df = picks_df.copy()
    picks_df["t_pick"] = pd.to_datetime(picks_df["time_utc"], utc=False)

    results = []
    for i, ev in official_df.iterrows():
        t0 = ev["t0_utc"]
        w0 = t0 - pd.Timedelta(seconds=tol_sec)
        w1 = t0 + pd.Timedelta(seconds=tol_sec)

        near = picks_df[(picks_df["t_pick"] >= w0) & (picks_df["t_pick"] <= w1) & (picks_df["phase"] == "P")]
        n_sta = near["station"].nunique()
        results.append({"idx_official": i, "t0_utc": t0, "n_stations_P": int(n_sta)})

    return pd.DataFrame(results)
def report_official_pick_details(official, picks_df, tol_sec=15.0, phase="P", limit_sta=30):
    """
    Para cada evento oficial:
      - lista picks por estación en ±tol_sec
      - imprime (t_pick, dt, prob)
      - resumen: n_stations, maxprob, meanprob, spread_sec
    """
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

        # 1 pick por estación: el de mayor prob (para saber el “techo” de filtro)
        per_sta = near.sort_values("prob", ascending=False).drop_duplicates("station", keep="first")

        # Calcular dt y ordenar por dt (más intuitivo)
        per_sta["dt_sec"] = (per_sta["t_pick"] - t0).dt.total_seconds()
        per_sta = per_sta.sort_values("dt_sec")

        # resumen
        nsta = per_sta["station"].nunique()
        maxprob = float(per_sta["prob"].max())
        meanprob = float(per_sta["prob"].mean())
        spread_sec = float((per_sta["t_pick"].max() - per_sta["t_pick"].min()).total_seconds())

        print(f"  n_stations={nsta} | maxprob={maxprob:.3f} | meanprob={meanprob:.3f} | spread={spread_sec:.1f}s")

        show = per_sta[["station","time_utc","dt_sec","prob"]].head(limit_sta)
        print(show.to_string(index=False))
def match_events_to_official(official, events, tol_sec=15.0):
    """
    Matching correcto:
    - Busca TODOS los eventos IA dentro de ±tol_sec del oficial
    - Si hay varios, elige el "mejor" (más estaciones, luego mayor maxprob, luego menor |dt|)
    """
    if events.empty:
        return pd.DataFrame(columns=["idx_official","t0_official","t0_ia","dt_sec","ia_n_stations","is_match"])

    out = []

    # Asegura columnas opcionales
    has_maxprob = "maxprob" in events.columns

    for i, r in official.iterrows():
        t0 = r["t0_utc"].timestamp()

        # candidatos dentro de ventana
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

        # calcula dt y ranking
        cand["dt_sec"] = cand["t0"] - t0
        cand["abs_dt"] = cand["dt_sec"].abs()

        # score: más estaciones, luego maxprob, luego menor abs_dt
        if has_maxprob:
            cand = cand.sort_values(["n_stations", "maxprob", "abs_dt"], ascending=[False, False, True])
        else:
            cand = cand.sort_values(["n_stations", "abs_dt"], ascending=[False, True])

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
# MAIN
# =====================
def main():
    t_start = time.time()

    official = load_official_day(OFFICIAL_XLSX, OFFICIAL_SHEET, OFFICIAL_DAY)
    print(f"Oficial {OFFICIAL_DAY}: {len(official)} eventos")

    if not Path(PICKS_CSV).exists():
        raise FileNotFoundError(f"No existe {PICKS_CSV}. Corre primero 01_make_picks_day.py")

    picks = pd.read_csv(PICKS_CSV)
    print(f"✅ Picks cargados: {len(picks)}  ({PICKS_CSV})")

    # ===== Diagnóstico picks vs oficial (P-only) =====
    report = match_official_with_picks(official, picks, tol_sec=MATCH_TOL_SEC)
    ge1 = (report["n_stations_P"] >= 1).sum()
    ge2 = (report["n_stations_P"] >= 2).sum()
    ge3 = (report["n_stations_P"] >= 3).sum()
    # Detalle de picks por evento oficial (para decidir filtros)
    report_official_pick_details(official, picks, tol_sec=MATCH_TOL_SEC, phase="P")
    print("\n=== MATCH (solo P, ±15s) ===")
    print(f"Oficiales con ≥1 estación con P-pick: {ge1}/{len(report)}")
    print(f"Oficiales con ≥2 estaciones con P-pick: {ge2}/{len(report)}")
    print(f"Oficiales con ≥3 estaciones con P-pick: {ge3}/{len(report)}")
    print("\nTop 10 oficiales por # estaciones con P:")
    print(report.sort_values("n_stations_P", ascending=False).head(10).to_string(index=False))

    # ===== Binning P =====
    picksP = bin_best_pick_per_station(picks, bin_sec=BIN_SEC, phase="P")
    print(f"\nP-picks tras bin ({BIN_SEC:.0f}s): {len(picksP)}")

    # ===== Eventos IA (DET) =====
    events_det = build_events_sliding_window(picksP, win_sec=WIN_SEC, min_stations=MIN_STATIONS, dead_sec=DEAD_SEC, strong_thr=0.83)
    print(f"Eventos IA DET (sin filtro): {len(events_det)}")

    m_det = match_events_to_official(official, events_det, tol_sec=MATCH_TOL_SEC)
    print("\n=== MATCH EVENTOS DET vs OFICIAL (±15s) ===")
    print(f"Matches DET: {m_det['is_match'].sum()}/{len(m_det)}")
    print(m_det.sort_values("dt_sec").to_string(index=False))

    # ===== Eventos IA (LOC) - filtrado =====
    events_loc = events_det[
        (events_det["n_stations"] >= 3) &
        (events_det["maxprob"] >= 0.78) &
        (events_det["n_strong"] >= 1)
    ].reset_index(drop=True)

    print(f"\nEventos IA LOC tras filtrado: {len(events_loc)} (maxprob>={F_MAXPROB}, meanprob>={F_MEANPROB})")

    m_loc = match_events_to_official(official, events_loc, tol_sec=MATCH_TOL_SEC)
    debug_near_misses(official, events_loc, tol_sec=MATCH_TOL_SEC, near_sec=90.0)
    print("\n=== MATCH EVENTOS LOC vs OFICIAL (±15s) ===")
    print(f"Matches LOC: {m_loc['is_match'].sum()}/{len(m_loc)}")
    print(m_loc.sort_values("dt_sec").to_string(index=False))

    print(f"\n⏱ Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
