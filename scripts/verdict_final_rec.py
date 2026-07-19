"""Provenance script for §2.11 (the final recommendation) of docs/verdict.html.

Reproduces the four computations behind the signed recommendation:
  1. The equal-belief Kelly weight of QQQ vs SPY (=0) and the variance-drag hurdle.
  2. The exclusion audit: share of 2016-26 large-winner gains from QQQ-ineligible
     (non-NASDAQ) listings, and the largest ineligible winners.
  3. Rolling 36m QQQ-SPY correlation (convergence check).
  4. Top-10 concentration share of large-cap dollar volume, 2005-2025 (the U-curve).

Run:  python3 scripts/verdict_final_rec.py
Requires the monthly panels built by dca/research/strategies/ascent/scripts/build_panels.py.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

A = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "dca/research/strategies/ascent/scripts")
ME = pd.read_pickle(f"{A}/_me_monthly.pkl")
DV = pd.read_pickle(f"{A}/_dv_monthly.pkl")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
uni = pd.read_parquet(f"{ROOT}/dca/research/data/tiingo/tiingo_universe_pit.parquet").drop_duplicates("ticker")
uni = uni[uni.ticker.apply(lambda x: isinstance(x, str))].set_index("ticker")

# 1) equal-belief Kelly + drag
q = ME["QQQ"].pct_change(); sp = ME["SPY"].pct_change()
sQ = float(q.std()*np.sqrt(12)); sS = float(sp.dropna().std()*np.sqrt(12)); rho = float(q.corr(sp))
num = sS**2 - rho*sS*sQ
drag = (sQ**2 - sS**2)/2
print(f"sigma QQQ {sQ:.3f}, SPY {sS:.3f}, rho {rho:.2f}")
print(f"equal-belief Kelly numerator for QQQ: {num:+.4f} (<0 => w*=0)")
print(f"variance-drag hurdle: {drag*100:.1f} pp/yr")

# 2) exclusion audit
_st = [t for t in uni[uni.assetType == "Stock"].index if isinstance(t, str) and "-" not in t]
cols = [c for c in ME.columns if c in set(_st)]
S = ME[cols]
i0 = ME.index.get_indexer([pd.Timestamp("2016-06-01")], method="nearest")[0]
i1 = ME.index.get_indexer([pd.Timestamp("2026-06-01")], method="nearest")[0]
p0 = S.iloc[i0]; p1 = S.iloc[i0:i1+1].ffill().iloc[-1]; dv0 = DV[cols].iloc[i0]
elig = (dv0 >= 2e6) & (p0 >= 3.0) & p0.notna()
ret = (p1/p0 - 1)[elig[elig].index].dropna()
rk = dv0.rank(ascending=False)
def exch(t):
    try: return str(uni.loc[t, "exchange"]).upper()
    except Exception: return "?"
sub = ret.reindex(rk[rk <= 500].index).dropna()
top25 = sub.sort_values(ascending=False).head(25).index
nas25 = sum(1 for t in top25 if "NASDAQ" in exch(t))
gains = sub[sub > 0]
by = {}
for t, g in gains.items():
    v = "NASDAQ" if "NASDAQ" in exch(t) else "NYSE/other"
    by[v] = by.get(v, 0) + g
tot = sum(by.values())
print(f"top-25 large winners: NASDAQ {nas25}/25")
print("share of large-winner gains:", {k: f"{v/tot*100:.0f}%" for k, v in by.items()})
print("largest QQQ-ineligible winners:",
      [(t, f"+{ret[t]*100:.0f}%") for t in sub.sort_values(ascending=False).index if "NYSE" in exch(t)][:8])

# 3) convergence
rc = q.rolling(36).corr(sp).dropna()
print("rolling 36m corr:", {y: round(float(rc[rc.index.year == y].mean()), 2)
                            for y in [2003, 2008, 2013, 2018, 2023, 2026] if len(rc[rc.index.year == y])})

# 4) concentration U
print("top-10 share of top-500 dollar volume:")
for y in [2005, 2010, 2015, 2020, 2025]:
    i = ME.index.get_indexer([pd.Timestamp(f"{y}-06-01")], method="nearest")[0]
    d = DV[cols].iloc[i].dropna().sort_values(ascending=False)
    print(f"  {y}: {float(d.head(10).sum()/d.head(500).sum())*100:.0f}%")
