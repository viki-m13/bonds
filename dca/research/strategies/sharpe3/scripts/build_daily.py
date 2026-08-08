"""Build the daily PIT research panel for SHARPE3.

Outputs to $SHARPE3_WORK (default /tmp/sharpe3_work):
  _px_daily.pkl   daily adjClose (float32) for every common stock that is EVER
                  liquid (price>=5, 63d median $vol >= $10M at some month-end)
                  + SPY/QQQ, 1990-2026
  _dv_daily.pkl   daily dollar volume for the same columns (float32)
  _elig.pkl       month-end boolean eligibility matrix (PIT liquidity filter,
                  computed from trailing data only)

Universe hygiene: assetType == Stock, no '-' tickers (preferreds/warrants/units),
exchange in NYSE/NASDAQ/AMEX family.
"""
import glob, os, time
import numpy as np, pandas as pd

t0 = time.time()
HERE = os.environ.get("SHARPE3_WORK", "/tmp/sharpe3_work"); os.makedirs(HERE, exist_ok=True)
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
DATA = os.path.join(REPO, "dca/research/data/tiingo")

uni = pd.read_parquet(f"{DATA}/tiingo_universe_pit.parquet").drop_duplicates("ticker")
uni = uni[uni.ticker.apply(lambda x: isinstance(x, str))]
ok_exch = uni.exchange.astype(str).str.upper().str.contains("NYSE|NASDAQ|AMEX|BATS", na=False)
keep = set(uni[(uni.assetType == "Stock") & ok_exch].ticker) - {t for t in uni.ticker if "-" in str(t)}
keep |= {"SPY", "QQQ"}
print(f"universe candidates: {len(keep)}", flush=True)

px_parts, dv_parts = [], []
for i, f in enumerate(sorted(glob.glob(f"{DATA}/prices/ac_*.parquet"))):
    d = pd.read_parquet(f)
    cols = [c for c in d.columns if c in keep]
    d = d[cols]; d.index = pd.to_datetime(d.index); d = d.sort_index()
    vf = f.replace("/ac_", "/vol_")
    v = pd.read_parquet(vf)
    vcols = [c for c in cols if c in v.columns]
    v = v[vcols]; v.index = pd.to_datetime(v.index); v = v.sort_index()
    dollar = (d[vcols].reindex(v.index) * v).astype(np.float32)
    # liquidity pre-screen: ever has 63d median dollar volume >= $10M and price >= 5
    med = dollar.rolling(63, min_periods=40).median()
    liq_ever = ((med >= 1e7) & (d[vcols].reindex(v.index) >= 5.0)).any()
    liq_cols = list(liq_ever[liq_ever].index) + [c for c in ("SPY", "QQQ") if c in cols]
    liq_cols = sorted(set(liq_cols))
    px_parts.append(d[liq_cols].astype(np.float32))
    dv_parts.append(dollar[[c for c in liq_cols if c in dollar.columns]])
    print(f"chunk {i}: {len(cols)} -> liquid-ever {len(liq_cols)}  t={time.time()-t0:.0f}s", flush=True)

def cat(parts):
    out = pd.concat(parts, axis=1)
    return out.loc[:, ~out.columns.duplicated()].sort_index()

PX = cat(px_parts); DVD = cat(dv_parts)
print("PX", PX.shape, "DV", DVD.shape, flush=True)

# month-end PIT eligibility: trailing 63d median $vol >= $10M AND price >= 5
med = DVD.rolling(63, min_periods=40).median()
elig_daily = (med >= 1e7) & (PX[DVD.columns] >= 5.0)
ELIG = elig_daily.resample("ME").last().fillna(False)
print("ELIG", ELIG.shape, "avg eligible/month:", float(ELIG.sum(axis=1).mean()), flush=True)

PX.to_pickle(f"{HERE}/_px_daily.pkl")
DVD.to_pickle(f"{HERE}/_dv_daily.pkl")
ELIG.to_pickle(f"{HERE}/_elig.pkl")
print(f"done t={time.time()-t0:.0f}s", flush=True)
