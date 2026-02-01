#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import numpy as np
import pandas as pd
import obspy as ob
from obspy import Stream

import seisbench as sb
sb.use_backup_repository()
from seisbench.models import EQTransformer

# =====================
# CONFIG
# =====================
YEAR = 2024
JDAY = 192  # 2024-07-08 si YEAR=2024 (ojo)
JSTR = f"{YEAR}{JDAY:03d}"

STATIONS = ["C0001", "C0003", "C0005", "C0018", "C0020", "C0023", "C0026",
            "C0035", "C0040", "C0047", "C0054", "C0073", "C0063", "C0070"]

CHANNELS = ["HHZ", "HHN", "HHE"]

TH_P = 0.70
TH_S = 0.55
MIN_SEP_SAME_PHASE = 0.35

# Salida
OUT_PICKS_CSV = f"picks_day_{JSTR}_THP{TH_P:.2f}_THS{TH_S:.2f}.csv"


# =====================
# I/O + utilidades
# =====================
def read_day(root: Path, sta: str) -> Stream | None:
    st = Stream()
    for ch in CHANNELS:
        for f in root.glob(f"*{sta}*{ch}*.{JSTR}*"):
            try:
                st += ob.read(str(f), format="MSEED")
            except Exception:
                print(f"⚠️ No se pudo leer: {f.name}")
                continue

    if not st:
        return None

    st.merge(method=1, fill_value="interpolate")
    st.detrend("demean")
    st.detrend("linear")
    st.taper(0.01)
    return st


def thin_by_phase(df: pd.DataFrame, min_dt=MIN_SEP_SAME_PHASE) -> pd.DataFrame:
    if df.empty:
        return df
    keep = []
    last_t = {}
    for _, r in df.sort_values(["phase", "t"]).iterrows():
        ph = r["phase"]
        if ph not in last_t or (r["t"] - last_t[ph]) >= min_dt:
            keep.append(r)
            last_t[ph] = r["t"]
    return pd.DataFrame(keep).reset_index(drop=True)


def picks_from_annotation(ann_stream, sta: str) -> pd.DataFrame:
    rows = []
    for tr in ann_stream:
        ch = tr.stats.channel.upper()
        if ch.endswith("_P"):
            phase = "P"; thr = TH_P
        elif ch.endswith("_S"):
            phase = "S"; thr = TH_S
        else:
            continue

        data = np.clip(tr.data.astype(float), 0.0, 1.0)
        fs = tr.stats.sampling_rate
        t0 = tr.stats.starttime

        above = np.where(data >= thr)[0]
        if above.size == 0:
            continue

        splits = np.where(np.diff(above) > 1)[0] + 1
        groups = np.split(above, splits)

        for g in groups:
            i_peak = g[np.argmax(data[g])]
            t_pick = t0 + float(i_peak) / fs
            rows.append({
                "station": sta,
                "phase": phase,
                "prob": float(data[i_peak]),
                "time_utc": t_pick.datetime.isoformat(timespec="milliseconds"),
                "t": float(t_pick.timestamp),
            })

    if not rows:
        return pd.DataFrame(columns=["station","phase","prob","time_utc","t"])

    df = pd.DataFrame(rows)
    df = thin_by_phase(df)
    return df


def load_model():
    # OJO: aquí sigues usando el "original_nonconservative" como tu modelo actual.
    return EQTransformer.from_pretrained("original_nonconservative")


def main():
    t_start = time.time()
    root = Path.cwd()

    print(f"📌 Día juliano: {JSTR}")
    print("Cargando EQTransformer...")
    model = load_model()
    print("✅ Modelo listo\n")

    all_picks = []

    for sta in STATIONS:
        st = read_day(root, sta)
        if not st:
            print(f"⚠️ {sta}: sin datos para {JSTR}")
            continue

        try:
            ann = model.annotate(st)
        except Exception as e:
            print(f"⚠️ {sta}: annotate falló: {e}")
            continue

        dfp = picks_from_annotation(ann, sta)
        print(f"{sta}: picks = {len(dfp)}")
        if not dfp.empty:
            all_picks.append(dfp)

    if not all_picks:
        print("❌ No hubo picks en ninguna estación.")
        return

    picks = pd.concat(all_picks, ignore_index=True)
    picks.to_csv(OUT_PICKS_CSV, index=False)

    print(f"\nTOTAL picks (todas estaciones): {len(picks)}")
    print(f"💾 Guardado: {OUT_PICKS_CSV}")
    print(f"⏱ Total: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
