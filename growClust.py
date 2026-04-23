"""
GrowClust relocation pipeline for Costa Rica automatic catalog.
Workflow:
  1. Convert catalog -> GrowClust phase-format event list
  2. Convert station CSV -> station list
  3. Compute waveform cross-correlations -> dt.cc
  4. Write velocity model and .inp control file
  5. Compile and run GrowClust
  6. Parse output, match to official catalog, evaluate metrics
  7. Tune rmsmax / rmincut and pick best run
  8. Save final relocated catalog CSV

NOTE: dt.cc is computed here from miniSEED waveforms using ObsPy.
      This is the only input GrowClust needs that is NOT in the CSVs.
"""

# ============================================================
# USER-EDITABLE PATHS AND PARAMETERS
# ============================================================
CATALOG_CSV    = "/data/murbina/seismo/inputs/catalogo_unido.csv"
OFFICIAL_CSV   = "/data/murbina/seismo/inputs/catalogo_oficial.csv"
STATION_CSV    = "/data/murbina/seismo/inputs/XML_Cartago_Nodes.csv"
WAVEFORM_DIR   = "/data/murbina/seismo/rawdata"
GROWCLUST_SRC  = "/data/murbina/seismo/tools/GrowClust-master/SRC"   # path to SRC/ with Makefile
WORK_DIR       = "/data/murbina/seismo/results/growclust"
OUTPUT_CSV     = "/data/murbina/seismo/results/catalogs/growclust/catalog_relocated.csv"

# Cross-correlation parameters
CC_FREQMIN     = 1.0     # bandpass low (Hz)
CC_FREQMAX     = 15.0    # bandpass high (Hz)
CC_WIN_P       = (-0.5, 2.5)   # window around P pick (s): pre, post
CC_WIN_S       = (-0.5, 4.0)   # window around S pick (s)
CC_MIN_R       = 0.5     # minimum CC coefficient to keep
CC_MAX_DIST_KM = 120.0   # max inter-event distance to attempt xcorr
MAX_PAIRS_PER_EVENT = 50  # cap pairs per event for efficiency
N_JOBS         = 20       # parallel workers

# GrowClust tuning grid (rmsmax, rmincut)
TUNE_RMSMAX  = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TUNE_RMINCUT = [0.0, 0.2, 0.3] 

# Match tolerance for evaluation
MATCH_DT_SEC   = 15.0
MATCH_DIST_KM  = 35.0

# Velocity model (depth_km, Vp, Vs) — layer-cake format for GrowClust
# From Costa Rica model provided by user
VEL_MODEL = [
    (0.0,  4.45, 2.50),
    (4.0,  4.45, 2.50),
    (4.0,  5.50, 3.00),
    (6.0,  5.50, 3.00),
    (6.0,  5.60, 3.15),
    (8.0,  5.60, 3.15),
    (8.0,  6.00, 3.37),
    (11.0, 6.00, 3.37),
    (11.0, 6.15, 3.45),
    (14.0, 6.15, 3.45),
    (14.0, 6.25, 3.51),
    (21.0, 6.25, 3.51),
    (21.0, 6.50, 3.65),
    (28.0, 6.50, 3.65),
    (28.0, 6.80, 3.82),
    (34.0, 6.80, 3.82),
    (34.0, 7.00, 3.93),
    (44.0, 7.00, 3.93),
    (44.0, 7.30, 4.10),
    (54.0, 7.30, 4.10),
    (54.0, 7.90, 4.44),
    (74.0, 7.90, 4.44),
    (74.0, 8.20, 4.60),
]

# ============================================================
# IMPORTS
# ============================================================
import warnings, os, glob, re, math, subprocess, shutil
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from itertools import combinations
from joblib import Parallel, delayed

from obspy import read, UTCDateTime, Stream
from obspy.signal.cross_correlation import correlate, xcorr_max

# ============================================================
# SETUP DIRECTORIES
# ============================================================
WORK_DIR = Path(WORK_DIR)
for d in ['IN', 'OUT', 'TT']:
    (WORK_DIR / d).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_CSV).parent.mkdir(parents=True, exist_ok=True)
# ============================================================
# dt.cc YA EXISTENTE
# ============================================================
dtcc_path = Path("/data/murbina/seismo/results/growclust/IN/dt.cc")

if not dtcc_path.exists():
    raise FileNotFoundError(f"No existe el dt.cc esperado: {dtcc_path}")

print(f"Usando dt.cc existente: {dtcc_path}")
# ============================================================
# HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    d1, d2 = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(d1/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(d2/2)**2
    return 2*R*math.asin(math.sqrt(a))

def parse_utc(s):
    try: return UTCDateTime(str(s))
    except: return None

def parse_off_utc(row):
    try:
        date = str(int(row['date']))
        t = str(int(row['time'])).zfill(8)
        return UTCDateTime(f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{t[0:2]}:{t[2:4]}:{t[4:6]}.{t[6:8]}")
    except: return None

def jday_key(origin_utc, jday_field):
    jday_str = str(int(jday_field)) if not pd.isna(jday_field) else ""
    if len(jday_str) == 7: return jday_str
    return f"{origin_utc.year}{jday_str.zfill(3)}"

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
cat = pd.read_csv(CATALOG_CSV)
off = pd.read_csv(OFFICIAL_CSV).dropna(subset=['date','time','lat','lon'])
sta = pd.read_csv(STATION_CSV)

cat.columns = [c.strip().lower() for c in cat.columns]
off.columns = [c.strip().lower() for c in off.columns]
sta.columns = [c.strip().lower() for c in sta.columns]

# Identify origin time column
ot_col = 'origin_time_utc' if 'origin_time_utc' in cat.columns else 'origin_time'
cat['_ot'] = cat[ot_col]
cat['_utc'] = [parse_utc(v) for v in cat['_ot']]

# Assign integer evid (1-based)
cat = cat.reset_index(drop=True)
cat['evid'] = cat.index + 1

print(f"  Catalog: {len(cat)}  Official: {len(off)}  Stations: {len(sta)}")

# ============================================================
# INDEX WAVEFORM FILES
# ============================================================
print("Indexing waveforms...")
all_files = [f for f in glob.glob(os.path.join(WAVEFORM_DIR, "**", "*"), recursive=True) if os.path.isfile(f)]
day_file_map = {}
for f in all_files:
    name = Path(f).name
    m = re.search(r'(\d{7})', name)
    if m: day_file_map.setdefault(m.group(1), []).append(f)
print(f"  {len(all_files)} files, {len(day_file_map)} day-keys")

def get_event_files(ev):
    key = jday_key(ev['_utc'], ev.get('jday', ''))
    return day_file_map.get(key, [])

def parse_sta_from_file(fpath):
    parts = Path(fpath).name.split('.')
    if len(parts) >= 3: return parts[1], parts[2]
    return None, None

# ============================================================
# WRITE GROWCLUST INPUT FILES
# ============================================================

# --- 1. Event list (phase format) ---
# yr mon day hr min sec lat lon dep mag eh ez rms evid
evlist_path = WORK_DIR / 'IN' / 'evlist.txt'
with open(evlist_path, 'w') as fh:
    for _, row in cat.iterrows():
        t = row['_utc']
        if t is None: continue
        mag = float(row['mag']) if not pd.isna(row.get('mag', float('nan'))) else 0.0
        fh.write(f"{t.year:4d} {t.month:2d} {t.day:2d} {t.hour:2d} {t.minute:2d} {t.second + t.microsecond/1e6:7.3f} "
                 f"{float(row['lat']):9.5f} {float(row['lon']):10.5f} {float(row['depth_km']):8.3f} "
                 f"{mag:6.2f}  0.000  0.000  0.000 {int(row['evid']):10d}\n")
print(f"  Wrote {evlist_path}")

# --- 2. Station list (name lat lon elev, format 2) ---
stlist_path = WORK_DIR / 'IN' / 'stlist.txt'
sta_dict = {}
with open(stlist_path, 'w') as fh:
    for _, r in sta.iterrows():
        code = str(r['code'])
        lat  = float(r['latitude'])
        lon  = float(r['longitude'])
        elev = float(r.get('elevation', 0))
        fh.write(f"{code:<5s} {lat:9.5f} {lon:10.5f} {elev:8.1f}\n")
        sta_dict[code] = (lat, lon, elev)
print(f"  Wrote {stlist_path}")

# --- 3. Velocity model (layer-cake) ---
vzmodel_path = WORK_DIR / 'IN' / 'vzmodel.txt'
with open(vzmodel_path, 'w') as fh:
    for (dep, vp, vs) in VEL_MODEL:
        fh.write(f"{dep:8.3f} {vp:6.3f} {vs:6.3f}\n")
print(f"  Wrote {vzmodel_path}")


# ============================================================
# COMPILE GROWCLUST
# ============================================================
growclust_exe = Path(GROWCLUST_SRC) / 'growclust'
if not growclust_exe.exists():
    print("Compiling GrowClust...")
    result = subprocess.run(['make', 'all'], cwd=GROWCLUST_SRC, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compile error:", result.stderr)
        raise RuntimeError("GrowClust compilation failed")
    print("  Compiled OK")
else:
    print("  GrowClust already compiled")

# ============================================================
# WRITE CONTROL FILE AND RUN FOR EACH PARAMETER COMBO
# ============================================================

# Compute travel time table limits from data
all_lats  = cat['lat'].astype(float).values
all_lons  = cat['lon'].astype(float).values
all_deps  = cat['depth_km'].astype(float).values
sta_lats  = [v[0] for v in sta_dict.values()]
sta_lons  = [v[1] for v in sta_dict.values()]

max_dist = 0.0
for i in range(len(all_lats)):
    for sl, so in zip(sta_lats, sta_lons):
        d = haversine_km(all_lats[i], all_lons[i], sl, so)
        if d > max_dist: max_dist = d
max_dist = min(max_dist * 1.2, 300.0)
max_dep  = float(all_deps.max()) * 1.3 + 10.0

def write_inp(rmsmax, rmincut, run_tag):
    out_dir = WORK_DIR / f'run_{run_tag}'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'TT').mkdir(exist_ok=True)
    inp_path = out_dir / 'growclust.inp'
    with open(inp_path, 'w') as f:
        f.write(f"""* GrowClust control file — rmsmax={rmsmax} rmincut={rmincut}
* evlist_fmt (1=phase)
1
* fin_evlist
{evlist_path}
* stlist_fmt (2=name+elev)
2
* fin_stlist
{stlist_path}
* xcordat_fmt tdif_fmt (1=text, 12=t1-t2)
1  12
* fin_xcordat
{dtcc_path}
* fin_vzmdl
{vzmodel_path}
* fout_vzfine
{out_dir}/TT/vzfine.txt
* fout_pTT
{out_dir}/TT/tt.pg
* fout_sTT
{out_dir}/TT/tt.sg
* vpvs_factor  rayparam_min
  1.732   0.0
* tt_dep0  tt_dep1  tt_ddep
  0.0  {max_dep:.1f}  1.0
* tt_del0  tt_del1  tt_ddel
  0.0  {max_dist:.1f}  2.0
* rmin  delmax  rmsmax
  0.5   {max_dist:.1f}  {rmsmax}
* rpsavgmin  rmincut  ngoodmin  iponly
  0.0   {rmincut}   2   0
* nboot  nbranch_min
  0   1
* fout_cat
{out_dir}/out.cat
* fout_clust
{out_dir}/out.clust
* fout_log
{out_dir}/out.log
* fout_boot
NONE
""")
    return inp_path, out_dir

def run_growclust(inp_path, out_dir, run_tag):
    log_stdout = out_dir / 'stdout.txt'
    try:
        result = subprocess.run(
            [str(growclust_exe), str(inp_path)],
            capture_output=True, text=True, timeout=600
        )
        (out_dir / 'stdout.txt').write_text(result.stdout + result.stderr)
        if result.returncode != 0:
            print(f"  [{run_tag}] GrowClust error — see {log_stdout}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  [{run_tag}] TIMEOUT")
        return False

# ============================================================
# PARSE GROWCLUST OUTPUT CATALOG
# ============================================================
def parse_growclust_cat(cat_path):
    """Parse GrowClust relocated catalog file (format per Section 4.1 of user guide).
    Columns: yr mon day hr min sec evid lat lon dep mag qID cID nbranch
             qnpair qndiffP qndiffS rmsP rmsS eh ez et latC lonC depC
    """
    rows = []
    with open(cat_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('*'): continue
            try:
                p = line.split()
                yr,mo,dy,hr,mi = int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[4])
                sec  = float(p[5])
                evid = int(p[6])
                lat, lon, dep = float(p[7]), float(p[8]), float(p[9])
                mag  = float(p[10])
                nbranch = int(p[13])
                eh, ez = float(p[19]), float(p[20])
                rows.append({'evid': evid, 'lat_r': lat, 'lon_r': lon, 'dep_r': dep,
                             'nbranch': nbranch, 'eh': eh, 'ez': ez,
                             'yr': yr, 'mo': mo, 'dy': dy, 'hr': hr, 'mi': mi, 'sec': sec})
            except: continue
    return pd.DataFrame(rows)

# ============================================================
# EVALUATE AGAINST OFFICIAL CATALOG
# ============================================================
off_utcs = [parse_off_utc(r) for _, r in off.iterrows()]
valid_my = [t for t in cat['_utc'] if t is not None]
if not valid_my:
    raise ValueError("No hay tiempos válidos en cat['_utc'].")

t_min = min(valid_my)
t_max = max(valid_my)

tmin_sec = t_min.timestamp - 86400
tmax_sec = t_max.timestamp + 86400

off_in_range = [
    (i, t, off.iloc[i]) for i, t in enumerate(off_utcs)
    if t and (t_min.timestamp - 86400 <= t.timestamp <= t_max.timestamp + 86400)
]
def evaluate(rel_df, run_tag):
    """Match relocated events to official catalog, return metrics dict."""
    if rel_df is None or len(rel_df) == 0:
        return {'run': run_tag, 'n_match': 0, 'dt_med': 999, 'depi_med': 999, 'ddep_med': 999}

    # Build lookup: evid -> relocated UTCDateTime + coords
    ev_lookup = {}
    for _, r in rel_df.iterrows():
        try:
            sec_int = int(r['sec'])
            usec    = int(round((r['sec'] - sec_int) * 1e6))
            t = UTCDateTime(int(r['yr']), int(r['mo']), int(r['dy']),
                            int(r['hr']), int(r['mi']), sec_int, usec)
            ev_lookup[int(r['evid'])] = (t, float(r['lat_r']), float(r['lon_r']), float(r['dep_r']))
        except: pass

    # For each official event, find best match in relocated catalog
    dt_list, depi_list, ddep_list = [], [], []
    matched = 0
    for (oi, t_off, off_row) in off_in_range:
        best_dt, best_epi, best_ddep = MATCH_DT_SEC+1, None, None
        for evid, (t_r, lat_r, lon_r, dep_r) in ev_lookup.items():
            dt = abs(t_r - t_off)
            if dt >= MATCH_DT_SEC: continue
            epi = haversine_km(float(off_row['lat']), float(off_row['lon']), lat_r, lon_r)
            if epi < MATCH_DIST_KM and dt < best_dt:
                best_dt   = dt
                best_epi  = epi
                best_ddep = abs(dep_r - float(off_row['depth_km']))
        if best_epi is not None:
            matched += 1
            dt_list.append(best_dt)
            depi_list.append(best_epi)
            ddep_list.append(best_ddep)

    return {
        'run':       run_tag,
        'n_match':   matched,
        'n_off':     len(off_in_range),
        'pct_match': round(100*matched/max(len(off_in_range),1), 1),
        'dt_mean':   round(float(np.mean(dt_list)),   3) if dt_list   else 999,
        'dt_med':    round(float(np.median(dt_list)), 3) if dt_list   else 999,
        'depi_mean': round(float(np.mean(depi_list)), 3) if depi_list else 999,
        'depi_med':  round(float(np.median(depi_list)),3) if depi_list else 999,
        'ddep_mean': round(float(np.mean(ddep_list)), 3) if ddep_list else 999,
        'ddep_med':  round(float(np.median(ddep_list)),3) if ddep_list else 999,
    }

# ============================================================
# TUNING LOOP
# ============================================================
print("\nRunning GrowClust parameter tuning...")
all_metrics = []
best_score  = 1e9
best_rel_df = None
best_tag    = None

for rmsmax in TUNE_RMSMAX:
    for rmincut in TUNE_RMINCUT:
        tag = f"rms{rmsmax}_rmin{rmincut}".replace('.','p')
        print(f"  Running {tag}...")
        inp_path, out_dir = write_inp(rmsmax, rmincut, tag)
        ok = run_growclust(inp_path, out_dir, tag)
        if not ok:
            all_metrics.append({'run': tag, 'n_match': 0})
            continue
        cat_out = out_dir / 'out.cat'
        if not cat_out.exists():
            all_metrics.append({'run': tag, 'n_match': 0})
            continue
        rel_df = parse_growclust_cat(str(cat_out))
        metrics = evaluate(rel_df, tag)
        all_metrics.append(metrics)
        print(f"    matched={metrics['n_match']}/{metrics.get('n_off','?')}  "
              f"depi_med={metrics['depi_med']:.2f}km  ddep_med={metrics['ddep_med']:.2f}km  "
              f"dt_med={metrics['dt_med']:.2f}s")
        # Score = weighted sum of median epicentral + depth error (lower=better)
        score = metrics['depi_med'] + 0.5 * metrics['ddep_med']
        if score < best_score and metrics['n_match'] > 0:
            best_score  = score
            best_rel_df = rel_df
            best_tag    = tag
            best_rmsmax, best_rmincut = rmsmax, rmincut

metrics_df = pd.DataFrame(all_metrics)
metrics_path = WORK_DIR / 'tuning_metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"\nTuning results saved: {metrics_path}")
print(f"Best run: {best_tag}  (rmsmax={best_rmsmax}, rmincut={best_rmincut})")
print(metrics_df.to_string(index=False))

# ============================================================
# BUILD FINAL OUTPUT CATALOG
# ============================================================
print("\nBuilding final output catalog...")

if best_rel_df is not None and len(best_rel_df) > 0:
    # Merge relocated positions back onto original catalog by evid
    rel_merge = best_rel_df[['evid','lat_r','lon_r','dep_r','nbranch','eh','ez']].copy()
    out = cat.drop(columns=['_ot','_utc','evid'], errors='ignore').copy()
    out['evid'] = cat['evid']
    out = out.merge(rel_merge, on='evid', how='left')

    # Rename for clarity
    out.rename(columns={
        'lat':   'lat_orig',
        'lon':   'lon_orig',
        'depth_km': 'depth_orig',
        'lat_r': 'lat',
        'lon_r': 'lon',
        'dep_r': 'depth_km',
    }, inplace=True)

    # Fill non-relocated events with original positions
    out['lat']      = out['lat'].fillna(out['lat_orig'])
    out['lon']      = out['lon'].fillna(out['lon_orig'])
    out['depth_km'] = out['depth_km'].fillna(out['depth_orig'])
    out['relocated'] = out['nbranch'].notna() & (out['nbranch'] > 1)
    out['growclust_nbranch'] = out['nbranch']
    out['growclust_eh_km']   = out['eh']
    out['growclust_ez_km']   = out['ez']
    out.drop(columns=['nbranch','eh','ez','evid'], inplace=True, errors='ignore')

    out.to_csv(OUTPUT_CSV, index=False)
    n_rel = out['relocated'].sum()
    print(f"  Relocated: {n_rel}/{len(out)} events")
    print(f"  Saved: {OUTPUT_CSV}")
else:
    print("  WARNING: No successful relocation. Check GrowClust logs in", WORK_DIR)
    cat.drop(columns=['_ot','_utc','evid'], errors='ignore').to_csv(OUTPUT_CSV, index=False)

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*60)
print("RELOCATION SUMMARY")
print("="*60)
print(f"  Best params : rmsmax={best_rmsmax}  rmincut={best_rmincut}")
if best_rel_df is not None:
    bm = [m for m in all_metrics if m['run'] == best_tag][0]
    print(f"  Matched official events : {bm['n_match']} / {bm.get('n_off','?')}  ({bm.get('pct_match','?')}%)")
    print(f"  Median epicentral error : {bm['depi_med']:.3f} km")
    print(f"  Median depth error      : {bm['ddep_med']:.3f} km")
    print(f"  Median time error       : {bm['dt_med']:.3f} s")
print(f"  Output catalog          : {OUTPUT_CSV}")
print("="*60)