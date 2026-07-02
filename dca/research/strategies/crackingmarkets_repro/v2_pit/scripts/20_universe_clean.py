"""Build the PIT liquidity universe from the survivorship-clean Tiingo panel.

Universe rule (fully point-in-time, no index membership needed):
  at each month-end, rank all names by trailing 63-day mean dollar volume,
  require last price > $5 and >= 252 days of history, keep the top 500.
  The membership applies to the FOLLOWING month.

Saves:
  out/univ_mask_monthly.parquet   bool, month-end x ticker (top-500 mask)
  out/clean_close.parquet         float32 adjClose for every name ever in mask
  out/clean_dadv.parquet          float32 63d dollar-ADV for the same names
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_tiingo, OUT

t0 = time.time()
ac, vol = load_tiingo()
ac = ac.loc["1995":].astype("float32")
vol = vol.loc["1995":].astype("float32")

# --- hygiene: NASDAQ test symbols, lowercase dupes -------------------------
drop = [c for c in ac.columns
        if c != c.upper() or c.endswith("ZZT") or c in ("ZTEST", "ZEXIT", "ZIEXT")]
ac = ac.drop(columns=drop)
vol = vol.drop(columns=drop, errors="ignore")
print(f"dropped {len(drop)} test/dupe tickers; panel {ac.shape}  "
      f"t={time.time()-t0:.0f}s")

# --- bad-tick repair: price >4x or <0.25x its 11-day centered median is a
# recording error -> NaN (same defense as dca/data.py). Iterate twice for
# clustered errors.
for _ in range(2):
    med = ac.rolling(11, center=True, min_periods=3).median()
    ratio = ac / med
    bad = (ratio > 4) | (ratio < 0.25)
    n = int(bad.sum().sum())
    print(f"bad ticks removed: {n}  t={time.time()-t0:.0f}s")
    if n == 0:
        break
    ac = ac.mask(bad)
print(f"panel {ac.shape}  t={time.time()-t0:.0f}s")

dv = (ac * vol).fillna(0).astype("float64")
cs = dv.cumsum()
count = ac.notna().astype("int32").cumsum()

month_ends = ac.groupby(ac.index.to_period("M")).tail(1).index
me_pos = ac.index.get_indexer(month_ends)

TOP, MINPX, MINHIST, LOOKBACK = 500, 5.0, 252, 63
mask = np.zeros((len(month_ends), ac.shape[1]), dtype=bool)
csv_, cnt_ = cs.values, count.values
px_ = ac.values
for k, p in enumerate(me_pos):
    if p < MINHIST:
        continue
    dadv63 = (csv_[p] - csv_[p - LOOKBACK]) / LOOKBACK
    hist = cnt_[p]
    px = px_[p]
    elig = (hist >= MINHIST) & np.isfinite(px) & (px > MINPX) & (dadv63 > 0)
    if elig.sum() < 50:
        continue
    d = np.where(elig, dadv63, -1)
    top = np.argpartition(d, -TOP)[-TOP:]
    top = top[d[top] > 0]
    mask[k, top] = True

univ = pd.DataFrame(mask, index=month_ends, columns=ac.columns)
ever = univ.any(axis=0)
cols = ac.columns[ever]
print(f"names ever in top-{TOP}: {len(cols)}  t={time.time()-t0:.0f}s")

cs_sub = cs[cols]
dadv63_sub = ((cs_sub - cs_sub.shift(LOOKBACK)) / LOOKBACK).astype("float32")
univ.loc[:, cols].to_parquet(os.path.join(OUT, "univ_mask_monthly.parquet"))
ac[cols].to_parquet(os.path.join(OUT, "clean_close.parquet"))
dadv63_sub.to_parquet(os.path.join(OUT, "clean_dadv.parquet"))
print(f"saved. mask per month (last 5):")
print(univ.sum(axis=1).tail(5).to_string())
print(f"DONE t={time.time()-t0:.0f}s")
