"""Build a filtered broad-universe panel from the Tiingo chunks.

Keeps US common stocks (assetType Stock, no preferreds/warrants/units — ticker
sanity filtered), 1998+, that ever reach $2M 20d-median dollar volume.
Output: sharpe3/cache/broad_close.parquet, broad_dollarvol.parquet (float32).

PIT safety: delisting-inclusive (delisted names stay until endDate); universe
eligibility each day is decided by trailing dollar volume only.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datalib

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE, exist_ok=True)

uni = datalib.load_tiingo_universe()
stocks = uni[(uni.assetType == "Stock")]
ok = stocks.ticker[~stocks.ticker.str.contains(r"-P-|-WS|-U$|-R$|-CL$|-W$", regex=True, na=True)]
keep = set(ok)
print("candidate stock tickers:", len(keep))

pdir = os.path.join(datalib.TIINGO, "prices")
acs, vols = [], []
for fn in sorted(os.listdir(pdir)):
    fp = os.path.join(pdir, fn)
    if fn.startswith("ac_"):
        df = pd.read_parquet(fp)
        df = df[[c for c in df.columns if c in keep]].loc["1998":]
        acs.append(df.astype(np.float32))
    elif fn.startswith("vol_"):
        df = pd.read_parquet(fp)
        df = df[[c for c in df.columns if c in keep]].loc["1998":]
        vols.append(df.astype(np.float32))
    print(fn, "done", flush=True)

ac = pd.concat(acs, axis=1)
ac = ac.loc[:, ~ac.columns.duplicated()]
vol = pd.concat(vols, axis=1)
vol = vol.loc[:, ~vol.columns.duplicated()]
vol = vol.reindex(index=ac.index, columns=ac.columns)
print("raw panel:", ac.shape)

dv = (ac * vol).rolling(20, min_periods=10).median()
ever = (dv > 2e6).any()
cols = ever[ever].index
ac, dv = ac[cols], dv[cols]
print("after $2M ADV filter:", ac.shape)

ac.to_parquet(os.path.join(CACHE, "broad_close.parquet"))
dv.astype(np.float32).to_parquet(os.path.join(CACHE, "broad_dollarvol.parquet"))
print("saved to cache/")
