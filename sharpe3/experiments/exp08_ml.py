"""Exp08: walk-forward cross-sectional ML on the S&P 500 PIT panel.

Target: forward 5d open-to-open return (execution-aligned: open(d+1)->open(d+6)),
cross-sectionally demeaned, vol-scaled, clipped. Features through close d only.
Refit annually (expanding window, min 4y). Ridge + LightGBM.
Outputs prediction matrices to cache/ and a scorecard.
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import datalib, bt

t0 = time.time()
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = datalib.load_summit()
close, open_, volp, member = P["close"], P["open"], P["volume"], P["member"]
r1 = close.pct_change(fill_method=None)
overnight = open_ / close.shift(1) - 1
intraday = close / open_ - 1
vol20 = r1.rolling(20).std()
dv = (close * volp).rolling(20, min_periods=5).median()
mkt = r1.where(member).mean(axis=1)

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

feats = {}
for k in (1, 2, 3, 5, 10, 21, 63, 126, 252):
    feats[f"ret{k}"] = zs(close.pct_change(k, fill_method=None) / (vol20 * np.sqrt(k)))
feats["mom12_1"] = zs(close.shift(21).pct_change(231, fill_method=None))
for k in (5, 21, 63):
    feats[f"on{k}"] = zs(overnight.rolling(k).sum())
    feats[f"in{k}"] = zs(intraday.rolling(k).sum())
feats["vol21"] = zs(r1.rolling(21).std())
feats["vol63"] = zs(r1.rolling(63).std())
feats["volofvol"] = zs(vol20.pct_change(21, fill_method=None))
feats["dvrank"] = dv.rank(axis=1, pct=True)
feats["vspike"] = zs(volp / volp.rolling(20).median())
feats["vtrend21"] = zs(volp.rolling(5).mean() / volp.rolling(63).mean())
feats["maxret21"] = zs(r1.rolling(21).max())
feats["minret21"] = zs(r1.rolling(21).min())
feats["skew63"] = zs(r1.rolling(63).skew())
feats["d20ma"] = zs(close / close.rolling(20).mean() - 1)
feats["d50ma"] = zs(close / close.rolling(50).mean() - 1)
feats["d200ma"] = zs(close / close.rolling(200).mean() - 1)
cov = r1.rolling(60).cov(mkt); var = mkt.rolling(60).var()
feats["beta60"] = zs(cov.div(var, axis=0))
feats["corr60"] = zs(r1.rolling(60).corr(mkt))
# market-state features (same value across names -> interactions for trees)
mv20 = (mkt.rolling(20).std() * np.sqrt(252))
feats["mktvol"] = pd.DataFrame(np.tile(mv20.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
mtrend = (mkt.rolling(200).mean() > 0).astype(float)
feats["mkttrend"] = pd.DataFrame(np.tile(mtrend.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
# 8-K recency
ek = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "8k_items.parquet"))
ek["date"] = pd.to_datetime(ek["date"])
ek = ek[ek.tk.isin(close.columns)]
evm = pd.DataFrame(False, index=close.index, columns=close.columns)
for tk, dts in ek.groupby("tk")["date"]:
    idx = close.index.searchsorted(dts.values)
    idx = idx[idx < len(close.index)]
    evm.iloc[idx, evm.columns.get_loc(tk)] = True
feats["news5"] = evm.rolling(5, min_periods=1).sum().clip(0, 3)
feats["news21"] = evm.rolling(21, min_periods=1).sum().clip(0, 6)

# target: fwd 5d open-to-open, demeaned, vol-scaled, clipped
o = open_
fwd = (o.shift(-6) / o.shift(-1) - 1)
y = fwd.sub(fwd.where(member).mean(axis=1), axis=0).div(vol20 * np.sqrt(5))
y = y.clip(-3, 3)

names = list(feats)
print(f"{len(names)} features; stacking...", flush=True)
mask = member & y.notna()
for f in feats.values():
    mask &= f.notna()
rows = []
didx = close.index
X_all = np.stack([feats[n].values for n in names], axis=2)  # days x tickers x F
Y_all = y.values
M_all = mask.values
print("stacked", X_all.shape, f"{time.time()-t0:.0f}s", flush=True)

years = range(2008, 2027)
pred_ridge = pd.DataFrame(np.nan, index=didx, columns=close.columns, dtype=np.float32)
pred_lgb = pd.DataFrame(np.nan, index=didx, columns=close.columns, dtype=np.float32)

from sklearn.linear_model import Ridge
import lightgbm as lgb

for yr in years:
    tr_end = didx.searchsorted(pd.Timestamp(f"{yr-1}-12-15"))  # 5d embargo before predict year
    te_lo = didx.searchsorted(pd.Timestamp(f"{yr}-01-01"))
    te_hi = didx.searchsorted(pd.Timestamp(f"{yr+1}-01-01"))
    if te_lo >= len(didx):
        break
    tr_m = M_all[:tr_end]
    Xtr = X_all[:tr_end][tr_m]
    Ytr = Y_all[:tr_end][tr_m]
    if len(Xtr) < 50000:
        continue
    rid = Ridge(alpha=1e4).fit(Xtr, Ytr)
    gbm = lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, num_leaves=31,
                            min_child_samples=1000, subsample=0.7, subsample_freq=1,
                            colsample_bytree=0.7, reg_lambda=10.0, n_jobs=4, verbose=-1)
    gbm.fit(Xtr, Ytr)
    te_m = M_all[te_lo:te_hi]
    Xte = X_all[te_lo:te_hi]
    pr = np.full(te_m.shape, np.nan, dtype=np.float32)
    pg = np.full(te_m.shape, np.nan, dtype=np.float32)
    pr[te_m] = rid.predict(Xte[te_m])
    pg[te_m] = gbm.predict(Xte[te_m])
    pred_ridge.iloc[te_lo:te_hi] = pr
    pred_lgb.iloc[te_lo:te_hi] = pg
    print(f"{yr}: train {len(Xtr)} rows  {time.time()-t0:.0f}s", flush=True)

pred_ridge.to_parquet(os.path.join(ROOT, "cache", "exp08_pred_ridge.parquet"))
pred_lgb.to_parquet(os.path.join(ROOT, "cache", "exp08_pred_lgb.parquet"))

out = {}
def ev(name, w, cost=5.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"]))
    m["yearly"] = bt.yearly_sharpes(res["net"])
    out[name] = m
    print(f"{name:24s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} IS={m['IS']:6.2f} OOS={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}", flush=True)

for nm, pr in (("ridge", pred_ridge), ("lgb", pred_lgb)):
    # IC
    ic = pr.corrwith(y, axis=1, method="spearman")
    print(f"{nm} IC: mean={ic.mean():.4f} t={ic.mean()/ic.std()*np.sqrt(ic.notna().sum()):.1f}", flush=True)
    ev(f"ml_{nm}_d10", bt.norm_ls(pr, member, 0.1, 0.1, 2.0))
    ev(f"ml_{nm}_d10_sm3", bt.norm_ls(pr.rolling(3).mean(), member, 0.1, 0.1, 2.0))
ev("ml_combo_sm3", bt.norm_ls(zs(pred_ridge).add(zs(pred_lgb)).rolling(3).mean(), member, 0.1, 0.1, 2.0))

with open(os.path.join(ROOT, "results", "exp08_ml.json"), "w") as f:
    json.dump(out, f, indent=1)
print("DONE", time.time() - t0, flush=True)
