"""
Empirical magnitude estimator for Costa Rica seismicity catalog. v2
NOT true ML (no instrument response/Wood-Anderson correction).
M_emp = a*log10(A) + b*log10(R) + c*R + intercept + station_terms
Calibrated against official catalog with known magnitudes.
Run as single Jupyter cell on Kabre HPC with 20 cores.
"""

# ============================================================
# USER-EDITABLE PARAMETERS
# ============================================================
CATALOG_CSV   = "/data/murbina/seismo/inputs/catalogo_unido.csv"
OFFICIAL_CSV  = "/data/murbina/seismo/inputs/catalogo_oficial.csv"
STATION_CSV   = "/data/murbina/seismo/inputs/XML_Cartago_Nodes.csv"
WAVEFORM_DIR  = "/data/murbina/seismo/rawdata"
OUTPUT_DIR    = "/data/murbina/seismo/results/catalogs/wMag"
OUTPUT_CSV    = "catalog_with_magnitude.csv"

MATCH_DT_SEC  = 15.0   # max |Δt| seconds for event matching
MATCH_DIST_KM = 35.0   # max epicentral km for event matching
SW_PRE        = 0.5    # seconds before estimated S arrival to start window
SW_POST       = 12.0   # seconds after S arrival to end window
MIN_STA       = 2      # minimum stations for valid magnitude
FMIN          = 1.0
FMAX          = 20.0
N_JOBS        = 20

VEL_LAYERS = [          # (thickness_km, Vp, Vs)
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

# ============================================================
# IMPORTS
# ============================================================
import warnings, os, glob, math, re
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from obspy import read, UTCDateTime
from sklearn.linear_model import HuberRegressor
from joblib import Parallel, delayed

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2*R*math.asin(math.sqrt(a))

def hypo_km(lat1, lon1, dep_km, lat2, lon2, elev_m):
    epi = haversine_km(lat1, lon1, lat2, lon2)
    dz  = dep_km + elev_m / 1000.0
    return math.sqrt(epi**2 + dz**2)

def s_travel_time(dist_km, depth_km):
    """1-D flat-layer S travel time via intercept-time method."""
    layers = VEL_LAYERS
    # find fastest (head-wave) S velocity at depth
    cum = 0.0
    v_deep = layers[0][2]
    for (h, vp, vs) in layers:
        cum += h
        v_deep = vs
        if cum >= depth_km:
            break
    # intercept time sum
    t_int = 0.0
    cum = 0.0
    for (h, vp, vs) in layers:
        dz = min(h, max(0, depth_km - cum))
        cum += h
        if dz <= 0:
            break
        if vs < v_deep:
            cos_i = math.sqrt(max(0.0, 1.0 - (vs/v_deep)**2))
            t_int += 2.0 * dz / vs * cos_i
    t_head = dist_km / v_deep + t_int if dist_km > 0 else t_int
    # straight-ray fallback
    path = math.sqrt(dist_km**2 + depth_km**2)
    # average S velocity over depth
    cum, v_avg_num, v_avg_den = 0.0, 0.0, 0.0
    for (h, vp, vs) in layers:
        dz = min(h, max(0, depth_km - cum))
        cum += h
        if dz <= 0:
            break
        v_avg_num += dz * vs
        v_avg_den += dz
    v_avg = v_avg_num / v_avg_den if v_avg_den > 0 else v_deep
    t_straight = path / v_avg
    return min(t_head, t_straight)

def parse_utc(val):
    """Parse any reasonable time string to UTCDateTime."""
    try:
        return UTCDateTime(str(val))
    except Exception:
        return None

def parse_official_utc(row):
    """Parse official catalog date (YYYYMMDD) + time (HHMMSSHH) columns."""
    try:
        date = str(int(row['date']))           # e.g. 20240704
        time = str(int(row['time'])).zfill(8)  # e.g. 13123296 → 8 chars
        # time format: HHMMSSCC where CC = centiseconds
        hh = time[0:2]
        mm = time[2:4]
        ss = time[4:6]
        cc = time[6:8]   # centiseconds
        dt_str = f"{date}T{hh}:{mm}:{ss}.{cc}"
        return UTCDateTime(dt_str)
    except Exception:
        # fallback: try generic parse on any available column
        for col in ['origin_time', 'origin_time_utc', 'datetime', 'time_utc']:
            if col in row.index:
                try:
                    return UTCDateTime(str(row[col]))
                except Exception:
                    pass
        return None

def jday_key(origin_utc, jday_field):
    """
    Return the YYYYJJJ key used in the file index.
    jday_field may already be YYYYJJJ (7 digits) or just JJJ (3 digits).
    """
    jday_str = str(int(jday_field)) if not pd.isna(jday_field) else ""
    if len(jday_str) == 7:          # already YYYYJJJ
        return jday_str
    elif len(jday_str) <= 3:        # just JJJ
        return f"{origin_utc.year}{jday_str.zfill(3)}"
    else:
        return jday_str             # unexpected, return as-is

def measure_amp(tr, t0, t1):
    """Bandpass-filter, trim to window, return peak abs amplitude."""
    try:
        tr2 = tr.copy()
        tr2.detrend('demean')
        tr2.taper(max_percentage=0.05, max_length=2.0)
        tr2.filter('bandpass', freqmin=FMIN, freqmax=FMAX, corners=4, zerophase=True)
        tr2.trim(t0, t1)
        if len(tr2.data) < 5:
            return None
        return float(np.max(np.abs(tr2.data)))
    except Exception:
        return None

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
cat = pd.read_csv(CATALOG_CSV)
off = pd.read_csv(OFFICIAL_CSV)
sta = pd.read_csv(STATION_CSV)

# Normalize column names
cat.columns = [c.strip().lower() for c in cat.columns]
off.columns = [c.strip().lower() for c in off.columns]
sta.columns = [c.strip().lower() for c in sta.columns]

# Identify origin_time column in my catalog
if 'origin_time_utc' in cat.columns:
    cat['_ot'] = cat['origin_time_utc']
elif 'origin_time' in cat.columns:
    cat['_ot'] = cat['origin_time']
else:
    raise ValueError(f"No origin time column found. Columns: {list(cat.columns)}")

print(f"  My catalog   : {len(cat)} events  | origin col: {cat['_ot'].name}")
print(f"  Official     : {len(off)} events")
print(f"  Stations     : {len(sta)}")
print(f"  My cat cols  : {list(cat.columns)}")
print(f"  Off cat cols : {list(off.columns)}")

# ============================================================
# INDEX WAVEFORM FILES
# ============================================================
print("\nIndexing waveform files...")
all_files = [f for f in glob.glob(os.path.join(WAVEFORM_DIR, "**", "*"), recursive=True)
             if os.path.isfile(f)]

day_file_map = {}  # YYYYJJJ -> [filepath, ...]
for f in all_files:
    name = Path(f).name
    # Primary: match YYYYJJJ 7-digit block
    m = re.search(r'(\d{7})', name)
    if m:
        day_file_map.setdefault(m.group(1), []).append(f)
    else:
        # fallback: YYYY + JJJ separated by non-digit or boundary
        m2 = re.search(r'(\d{4})(\d{3})(?:[_\.\+\-]|$)', name)
        if m2:
            day_file_map.setdefault(m2.group(1)+m2.group(2), []).append(f)

print(f"  Indexed {len(all_files)} files | day-keys: {sorted(day_file_map.keys())[:10]} ...")

# ============================================================
# STATION LOOKUP
# ============================================================
# Build dict: station_code -> row
sta_dict = {str(r['code']): r for _, r in sta.iterrows()}

# ============================================================
# PER-EVENT AMPLITUDE EXTRACTION
# ============================================================

def process_event(ev):
    """Returns list of dicts with {sta, log10A, R_km} for one event."""
    results = []
    origin = parse_utc(ev['_ot'])
    if origin is None:
        return results

    key = jday_key(origin, ev.get('jday', ''))
    files = day_file_map.get(key, [])
    if not files:
        return results

    # Group files by station code
    sta_files = {}
    for f in files:
        name = Path(f).name
        parts = name.split('.')
        if len(parts) >= 3:
            sta_code = parts[1]
            cha = parts[2]
            sta_files.setdefault(sta_code, []).append((cha, f))

    ev_lat  = float(ev['lat'])
    ev_lon  = float(ev['lon'])
    ev_dep  = float(ev['depth_km'])

    for sta_code, chan_files in sta_files.items():
        if sta_code not in sta_dict:
            continue
        sr = sta_dict[sta_code]
        R = hypo_km(ev_lat, ev_lon, ev_dep,
                    float(sr['latitude']), float(sr['longitude']),
                    float(sr.get('elevation', 0)))
        R = max(R, 0.5)

        ts  = s_travel_time(haversine_km(ev_lat, ev_lon,
                                          float(sr['latitude']),
                                          float(sr['longitude'])),
                             ev_dep)
        t0  = origin + ts - SW_PRE
        t1  = origin + ts + SW_POST

        # prefer horizontals E/N/1/2, fallback Z
        horiz = [(c, f) for c, f in chan_files if c[-1] in ('E','N','1','2')]
        vert  = [(c, f) for c, f in chan_files if c[-1] == 'Z']
        chosen = horiz if horiz else vert
        if not chosen:
            continue

        amps = []
        for cha, fpath in chosen:
            try:
                st = read(fpath, starttime=t0 - 5, endtime=t1 + 5)
                sel = st.select(channel=cha)
                if not sel:
                    sel = st  # take whatever is there
                for tr in sel:
                    a = measure_amp(tr, t0, t1)
                    if a and a > 0:
                        amps.append(a)
            except Exception:
                continue

        if amps:
            A = np.max(amps)
            results.append({'sta': sta_code, 'log10A': math.log10(A), 'R_km': R})

    return results

print("\nExtracting amplitudes (parallel)...")
ev_rows = [row for _, row in cat.iterrows()]
per_event = Parallel(n_jobs=N_JOBS, backend='loky', verbose=5)(
    delayed(process_event)(ev) for ev in ev_rows
)

n_with_amp = sum(1 for x in per_event if x)
print(f"  Events with ≥1 station amplitude: {n_with_amp} / {len(cat)}")

# ============================================================
# MATCH MY EVENTS TO OFFICIAL CATALOG
# ============================================================
print("\nParsing official catalog times...")
off_utcs = [parse_official_utc(r) for _, r in off.iterrows()]
my_utcs  = [parse_utc(ev['_ot']) for ev in ev_rows]

# Diagnostics: show first few parsed times
valid_off = [(i,t) for i,t in enumerate(off_utcs) if t is not None]
valid_my  = [(i,t) for i,t in enumerate(my_utcs)  if t is not None]
print(f"  Official times parsed: {len(valid_off)} / {len(off_utcs)}")
print(f"  My times parsed      : {len(valid_my)}  / {len(my_utcs)}")
if valid_off:
    print(f"  Official time sample : {valid_off[0][1]}")
if valid_my:
    print(f"  My time sample       : {valid_my[0][1]}")

print("Matching events...")
matched = []   # (my_idx, off_mag, sta_data_list)

for i, (t_my, sta_data) in enumerate(zip(my_utcs, per_event)):
    if t_my is None or not sta_data:
        continue
    ev = ev_rows[i]
    best_j, best_dt = None, MATCH_DT_SEC + 1
    for j, t_off in enumerate(off_utcs):
        if t_off is None:
            continue
        dt = abs(t_my - t_off)
        if dt >= best_dt:
            continue
        # distance check
        try:
            epi = haversine_km(float(ev['lat']), float(ev['lon']),
                               float(off.iloc[j]['lat']), float(off.iloc[j]['lon']))
        except Exception:
            epi = 0.0
        if dt < MATCH_DT_SEC and epi < MATCH_DIST_KM:
            best_dt = dt
            best_j = j

    if best_j is not None:
        off_row = off.iloc[best_j]
        # find magnitude column
        for mc in ['mag', 'magnitude', 'ml', 'mw', 'md']:
            if mc in off_row.index:
                try:
                    off_mag = float(off_row[mc])
                    matched.append((i, off_mag, sta_data))
                    break
                except Exception:
                    pass

print(f"  Matched events: {len(matched)}")

# ============================================================
# CALIBRATION
# ============================================================
sta_list = sorted(sta_dict.keys())
sta_idx  = {s: i for i, s in enumerate(sta_list)}
n_sta    = len(sta_list)

X_rows, y_rows = [], []
for (_, off_mag, sta_data) in matched:
    for d in sta_data:
        R = max(d['R_km'], 0.5)
        st_vec = [0.0] * n_sta
        si = sta_idx.get(d['sta'])
        if si is not None:
            st_vec[si] = 1.0
        X_rows.append([d['log10A'], math.log10(R), R] + st_vec)
        y_rows.append(off_mag)

print(f"  Regression observations: {len(X_rows)}")

use_default = False
if len(X_rows) >= 20:
    X_mat = np.array(X_rows)
    y_arr = np.array(y_rows)
    model = HuberRegressor(max_iter=1000, epsilon=1.5)
    model.fit(X_mat, y_arr)
    coef       = model.coef_
    a_coef     = coef[0]   # log10A
    b_coef     = coef[1]   # log10R
    c_coef     = coef[2]   # R
    sta_coefs  = coef[3:]
    intercept  = model.intercept_
    y_pred_cal = model.predict(X_mat)
    mae  = np.mean(np.abs(y_pred_cal - y_arr))
    rmse = np.sqrt(np.mean((y_pred_cal - y_arr)**2))
    print(f"  Calibration MAE={mae:.3f}  RMSE={rmse:.3f}")
elif len(X_rows) >= 5:
    # simplified model without station terms
    X_simple = np.array([[r[0], r[1], r[2]] for r in X_rows])
    y_arr    = np.array(y_rows)
    model_s  = HuberRegressor(max_iter=1000)
    model_s.fit(X_simple, y_arr)
    a_coef, b_coef, c_coef = model_s.coef_
    sta_coefs = np.zeros(n_sta)
    intercept = model_s.intercept_
    y_pred_cal = model_s.predict(X_simple)
    mae  = np.mean(np.abs(y_pred_cal - y_arr))
    rmse = np.sqrt(np.mean((y_pred_cal - y_arr)**2))
    print(f"  Simplified calibration (no sta terms) MAE={mae:.3f}  RMSE={rmse:.3f}")
else:
    print("  WARNING: Insufficient matched obs. Using Hutton & Boore (1987) ML defaults.")
    a_coef, b_coef, c_coef, intercept = 1.0, 1.110, 0.00189, -2.09
    sta_coefs = np.zeros(n_sta)
    mae = rmse = float('nan')
    use_default = True

# ============================================================
# COMPUTE FINAL MAGNITUDES
# ============================================================
print("\nComputing magnitudes...")
mags, mag_stds, mag_nstas, mag_methods, mag_oks = [], [], [], [], []

method_label = ('M_emp_default' if use_default
                else 'M_emp_calibrated_no_sta_terms' if len(X_rows) < 20
                else 'M_emp_calibrated')

for sta_data in per_event:
    if not sta_data:
        mags.append(np.nan); mag_stds.append(np.nan)
        mag_nstas.append(0); mag_methods.append('none'); mag_oks.append(False)
        continue

    sta_mags = []
    for d in sta_data:
        R  = max(d['R_km'], 0.5)
        si = sta_idx.get(d['sta'])
        st = float(sta_coefs[si]) if si is not None and si < len(sta_coefs) else 0.0
        M  = (a_coef * d['log10A']
              + b_coef * math.log10(R)
              + c_coef * R
              + st
              + intercept)
        sta_mags.append(M)

    if len(sta_mags) >= MIN_STA:
        mags.append(float(np.median(sta_mags)))
        mag_stds.append(float(np.std(sta_mags)))
        mag_nstas.append(len(sta_mags))
        mag_methods.append(method_label)
        mag_oks.append(True)
    else:
        mags.append(np.nan); mag_stds.append(np.nan)
        mag_nstas.append(len(sta_mags)); mag_methods.append('insufficient_sta')
        mag_oks.append(False)

# ============================================================
# OUTPUT
# ============================================================
out = cat.drop(columns=['_ot'], errors='ignore').copy()
# Remove old mag columns if re-running
for c in ['mag','mag_std','mag_nsta','mag_method','mag_ok']:
    if c in out.columns:
        out.drop(columns=[c], inplace=True)

out['mag']        = mags
out['mag_std']    = mag_stds
out['mag_nsta']   = mag_nstas
out['mag_method'] = mag_methods
out['mag_ok']     = mag_oks

out_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"Events with valid magnitude: {sum(mag_oks)} / {len(out)}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("CALIBRATION SUMMARY")
print("="*60)
print(f"  Matched events for calibration   : {len(matched)}")
print(f"  Station-obs in regression        : {len(X_rows)}")
print(f"  MAE  vs official                 : {mae:.3f}" if not math.isnan(mae) else "  MAE : N/A")
print(f"  RMSE vs official                 : {rmse:.3f}" if not math.isnan(rmse) else "  RMSE: N/A")
print(f"  Model coefficients               : log10A={a_coef:.3f}  log10R={b_coef:.3f}  R={c_coef:.5f}  c={intercept:.3f}")
print(f"  Valid magnitudes in output       : {sum(mag_oks)} / {len(out)}")
print()
print("  NOTE: This is M_emp, NOT true ML.")
print("        No Wood-Anderson correction applied (no instrument response).")
print("        Amplitudes are raw bandpass-filtered waveform counts.")
print("="*60)