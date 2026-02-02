#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from pathlib import Path
import pandas as pd
import obspy as ob
from obspy import Stream

import seisbench as sb
import seisbench.models as sbm

# =====================
# CONFIG
# =====================
YEAR = 2024
JDAY = 190  # 2024-07-08 si YEAR=2024 (ojo)
JSTR = f"{YEAR}{JDAY:03d}"

STATIONS = ["C0001", "C0002", "C0003","C0004", "C0005","C0007","C0008","C0009","C0010","C0011","C0012","C0013","C0014","C0015","C0016","C0017","C0018","C0019","C0020","C0021","C0022","C0023","C0024","C0025","C0026","C0027","C0028","C0029","C0030","C0031","C0032","C0033","C0034",
            "C0035", "C0036", "C0037", "C0038", "C0039", "C0040", "C0041", "C0042", "C0043", "C0044", "C0045", "C0046", "C0047", "C0048", "C0049", "C0051", "C0052", "C0053", "C0054", "C0055", "C0057", "C0058", "C0059", "C0060", "C0061", "C0062", "C0063", "C0066", "C0067", "C0070", "C0071", "C0072", "C0073"]

CHANNELS = ["HHZ", "HHN", "HHE"]

TH_P = 0.70
TH_S = 0.55

# Salidas
OUT_PICKS_CSV = f"picks_day_{JSTR}_THP{TH_P:.2f}_THS{TH_S:.2f}_seisbench.csv"
OUT_DET_CSV   = f"detections_day_{JSTR}_THP{TH_P:.2f}_THS{TH_S:.2f}_seisbench.csv"

# =====================
# I/O
# =====================
def read_day(root: Path, sta: str) -> Stream | None:
    st = Stream()

    for ch in CHANNELS:
        for f in root.glob(f"*{sta}*{ch}*.{JSTR}*"):
            try:
                st += ob.read(str(f), format="MSEED")
            except Exception:
                print(f"⚠️ No se pudo leer: {f.name}")

    if not st:
        return None

    # Merge/split: en continuo suele ser mejor NO interpolar huecos largos.
    # 'method=1' + interpolate puede inventar señal en gaps grandes.
    st.merge(method=1, fill_value=None)

    # Opcional: limpieza ligera
    st.detrend("demean")
    st.detrend("linear")
    st.taper(0.01)

    return st


def load_model():
    # Mantén tu peso actual
    # Tip útil: sbm.EQTransformer.list_pretrained()
    return sbm.EQTransformer.from_pretrained("original_nonconservative")


def extract_outputs(classify_out):
    """
    Compatibilidad:
      - nuevo: classify_out.picks / classify_out.detections  (ClassifyOutput)
      - viejo: (picks, detections) como tupla
    """
    if hasattr(classify_out, "picks"):
        picks = classify_out.picks
        detections = getattr(classify_out, "detections", None)
        return picks, detections
    if isinstance(classify_out, (tuple, list)) and len(classify_out) >= 1:
        picks = classify_out[0]
        detections = classify_out[1] if len(classify_out) > 1 else None
        return picks, detections
    raise TypeError(f"No entiendo salida de classify(): {type(classify_out)}")


def add_station_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    SeisBench pone 'station' = trace_id (ej: NET.STA.LOC.CHAN).
    Aquí extraemos STA si el formato coincide.
    """
    if df.empty or "station" not in df.columns:
        return df
    parts = df["station"].astype(str).str.split(".")
    # Si hay 4 partes, la 2da suele ser STA
    df["station_code"] = parts.str[1].where(parts.str.len() >= 2, df["station"])
    return df


def main():
    t0 = time.time()
    root = Path.cwd()

    print(f"📌 Día juliano: {JSTR}")
    model = load_model()
    print("✅ Modelo listo")

    all_pick_dfs = []
    all_det_dfs = []

    for sta in STATIONS:
        st = read_day(root, sta)
        if not st:
            print(f"⚠️ {sta}: sin datos para {JSTR}")
            continue

        try:
            out = model.classify(
                st,
                P_threshold=TH_P,
                S_threshold=TH_S,
                # batch_size=256,  # útil si tienes GPU/CPU fuerte
                # parallelism=1,   # algunos ejemplos externos lo usan
            )
            picks, dets = extract_outputs(out)
        except Exception as e:
            print(f"⚠️ {sta}: classify falló: {e}")
            continue

        # Export “oficial” SeisBench
        pick_df = picks.to_dataframe() if hasattr(picks, "to_dataframe") else pd.DataFrame()
        det_df  = dets.to_dataframe()  if (dets is not None and hasattr(dets, "to_dataframe")) else pd.DataFrame()

        pick_df = add_station_code(pick_df)
        det_df  = add_station_code(det_df)

        print(f"{sta}: picks={len(pick_df)} detections={len(det_df)}")

        if not pick_df.empty:
            all_pick_dfs.append(pick_df)
        if not det_df.empty:
            all_det_dfs.append(det_df)

    if not all_pick_dfs:
        print("❌ No hubo picks en ninguna estación.")
        return

    picks_day = pd.concat(all_pick_dfs, ignore_index=True).sort_values("time")
    picks_day.to_csv(OUT_PICKS_CSV, index=False)

    if all_det_dfs:
        dets_day = pd.concat(all_det_dfs, ignore_index=True).sort_values("start_time")
        dets_day.to_csv(OUT_DET_CSV, index=False)
        print(f"💾 Detections: {OUT_DET_CSV}")

    print(f"💾 Picks: {OUT_PICKS_CSV}")
    print(f"TOTAL picks: {len(picks_day)}")
    print(f"⏱ Total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
