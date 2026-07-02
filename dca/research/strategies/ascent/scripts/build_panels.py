"""Build price panels from committed Tiingo chunks.
Outputs (scratchpad):
  _me_monthly.pkl   monthly adjClose (stocks + SPY/QQQ), month-start index (like exp35/68)
  _px_weekly.pkl    weekly (Fri) adjClose for finer stop checks / biweekly cadence
  _dv_monthly.pkl   monthly median daily dollar volume (liquidity)
"""
import glob, os, time
import numpy as np, pandas as pd

t0 = time.time()
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
DATA = os.path.join(REPO, "dca/research/data/tiingo")
uni = pd.read_parquet(f"{DATA}/tiingo_universe_pit.parquet")
keep_stock = set(uni[uni.assetType == "Stock"].ticker)
keep = keep_stock | {"SPY", "QQQ"}

ac_files = sorted(glob.glob(f"{DATA}/prices/ac_*.parquet"))
vol_files = sorted(glob.glob(f"{DATA}/prices/vol_*.parquet"))

mes, wks, dvs = [], [], []
for i, f in enumerate(ac_files):
    d = pd.read_parquet(f)
    d = d[[c for c in d.columns if c in keep]]
    d.index = pd.to_datetime(d.index)
    d = d.sort_index()
    me = d.resample("ME").last()
    wk = d.resample("W-FRI").last()
    mes.append(me); wks.append(wk)
    # matching volume chunk (same suffix)
    vf = f.replace("/ac_", "/vol_")
    if os.path.exists(vf):
        v = pd.read_parquet(vf)
        v = v[[c for c in v.columns if c in d.columns]]
        v.index = pd.to_datetime(v.index); v = v.sort_index()
        dollar = (d.reindex(v.index) * v)
        dv = dollar.resample("ME").median()
        dvs.append(dv)
    print(f"chunk {i} {os.path.basename(f)} {d.shape} t={time.time()-t0:.0f}s", flush=True)

def cat(parts):
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated()]
    return out.astype(np.float32)

ME = cat(mes); WK = cat(wks); DV = cat(dvs)
# month-start timestamps to match exp68 convention
ME.index = ME.index.to_period("M").to_timestamp()
DV.index = DV.index.to_period("M").to_timestamp()
ME.to_pickle(os.path.join(HERE, "_me_monthly.pkl"))
WK.to_pickle(os.path.join(HERE, "_px_weekly.pkl"))
DV.to_pickle(os.path.join(HERE, "_dv_monthly.pkl"))
print("ME", ME.shape, "WK", WK.shape, "DV", DV.shape, f"t={time.time()-t0:.0f}s", flush=True)
print("QQQ tail:", ME["QQQ"].dropna().tail(3).to_dict(), flush=True)
