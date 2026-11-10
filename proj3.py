"""
Query + clean CMIP6 tas for: "Which regions warm least by 2050, and how does it vary by scenario?"
Output: docs/data/warming.json (read by docs/index.html)

    pip install xarray zarr gcsfs dask regionmask pandas numpy
"""
import json, os, warnings
import numpy as np, pandas as pd, xarray as xr, gcsfs, regionmask
warnings.filterwarnings("ignore")

EXPS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
SCEN = EXPS[1:]
BASE = (1995, 2014)
DECADES = list(range(2020, 2100, 10))          # 2020s ... 2090s
MAX_MODELS = 5   # use 3 to test quickly; raise to 10 for the final run
LAT = np.arange(-87.5, 90, 5.0)                # common 5 degree grid
LON = np.arange(2.5, 360, 5.0)

# ---------- 1. QUERY ----------
gcs = gcsfs.GCSFileSystem(token="anon")
df = pd.read_csv("https://storage.googleapis.com/cmip6/cmip6-zarr-consolidated-stores.csv")

sel = df.query("table_id == 'Amon' & variable_id == 'tas' & experiment_id in @EXPS")
sel = sel[sel.member_id == "r1i1p1f1"].drop_duplicates(["source_id", "experiment_id"])
n = sel.groupby("source_id")["experiment_id"].nunique()
sel = sel[sel.source_id.isin(n[n == len(EXPS)].index)]
models = sorted(sel.source_id.unique())[:MAX_MODELS]
print(len(models), "models:", models)


# ---------- 2. CLEAN ----------
def annual(zstore):
    """zarr store -> annual-mean tas (year, lat, lon) on the common grid, lon in -180..180."""
    ds = xr.open_zarr(gcs.get_mapper(zstore), consolidated=True)
    tas = ds["tas"].squeeze(drop=True)                      # drop singleton 'height'
    if not tas.indexes["time"].is_monotonic_increasing:
        tas = tas.sortby("time")   # fixes models with an unsorted time axis (e.g. AWI-CM-1-1-MR)
    tas = tas.sel(time=slice("1995", "2100"))               # only what we need
    ann = tas.groupby("time.year").mean("time").compute().reset_coords(drop=True)   # monthly -> annual (handles any calendar)
    ann = ann.interp(lat=LAT, lon=LON, kwargs={"fill_value": "extrapolate"})  # common grid
    ann = ann.assign_coords(lon=((ann.lon + 180) % 360) - 180).sortby("lon")  # 0-360 -> -180..180
    return ann


good, res = [], []
import time
os.makedirs("cache", exist_ok=True)
for m in models:
    cf = f"cache/{m}.nc"
    if os.path.exists(cf):                      # reuse finished models on re-runs
        res.append(xr.open_dataarray(cf).load().reset_coords(drop=True)); good.append(m); print("cached", m); continue
    t0 = time.time()
    try:
        z = dict(zip(sel[sel.source_id == m].experiment_id, sel[sel.source_id == m].zstore))
        base = annual(z["historical"]).sel(year=slice(*BASE)).mean("year")   # 1995-2014 baseline
        per = []
        for s in SCEN:
            a = annual(z[s])
            per.append(xr.concat([a.sel(year=slice(d, d + 9)).mean("year") - base   # anomaly vs baseline
                                  for d in DECADES], dim=xr.Variable("decade", DECADES)))
        r = xr.concat(per, dim=xr.Variable("scenario", SCEN)).rename("warming")
        r.to_netcdf(f"cache/{m}.nc"); res.append(r)
        good.append(m)                    # keep a model only if every scenario worked
        print("ok", m, f"{time.time() - t0:.0f}s")
    except Exception as e:
        print("skipped", m, "->", e)

W = xr.concat(res, dim=xr.Variable("model", good)).transpose(
    "scenario", "model", "decade", "lat", "lon")

# ---------- 3. REGIONS (IPCC AR6) ----------
regs = regionmask.defined_regions.ar6.all
mask = regs.mask(W.lon, W.lat)                    # (lat, lon) region number per cell
m3 = regs.mask_3D(W.lon, W.lat)
wts = np.cos(np.deg2rad(W.lat))                   # area weighting
rm = W.weighted(m3 * wts).mean(("lat", "lon"))    # scenario, model, decade, region

# ---------- 4. EXPORT ----------
L = lambda a: np.where(np.isnan(a), None, np.round(a, 2)).tolist()
mean = W.mean("model")
out = dict(
    years=DECADES, scen=SCEN, models=good, baseline=f"{BASE[0]}-{BASE[1]}",
    lat=LAT.tolist(), lon=W.lon.values.tolist(),
    reg=np.nan_to_num(mask.values, nan=-1).astype(int).flatten().tolist(),
    regions=[dict(n=n_, a=a_) for n_, a_ in zip(regs.names, regs.abbrevs)],
    grid={s: L(mean.sel(scenario=s).values.reshape(len(DECADES), -1)) for s in SCEN},
    regmean={s: L(rm.mean("model").sel(scenario=s).values) for s in SCEN},
    regsd={s: L(rm.std("model").sel(scenario=s).values) for s in SCEN},
)
os.makedirs("docs/data", exist_ok=True)
with open("docs/data/warming.json", "w") as f:
    json.dump(out, f, separators=(",", ":"))

# ---------- 5. SANITY CHECKS ----------
gm = (mean * wts).sum(("lat", "lon")) / (wts.sum() * len(W.lon))
print("\nGlobal-mean warming (deg C) by decade:\n", gm.to_pandas().round(2))
print("NaN cells:", int(np.isnan(mean.values).sum()), "| models:", len(good))