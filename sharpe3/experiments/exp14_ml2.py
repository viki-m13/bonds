"""Exp14: ML v2 — the serious attempt.

Changes vs exp08:
  - train only on 2010+ (avoid pre-2010 microstructure-artifact alpha)
  - two horizons (1d, 5d fwd open-to-open residual, vol-scaled, clipped)
  - rank-transformed targets (cross-sectional pct) as alternative
  - larger LightGBM, annual refit, 5d embargo
  - portfolio: quintile LS with turnover buffer (hysteresis) + signal blend
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
for k in (1, 5, 21, 63):
    feats[f"on{k}"] = zs(overnight.rolling(k).sum())
    feats[f"in{k}"] = zs(intraday.rolling(k).sum())
feats["vol21"] = zs(r1.rolling(21).std())
feats["vol63"] = zs(r1.rolling(63).std())
feats["volratio"] = zs(r1.rolling(5).std() / r1.rolling(63).std())
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
mv20 = (mkt.rolling(20).std() * np.sqrt(252))
feats["mktvol"] = pd.DataFrame(np.tile(mv20.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
mret5 = mkt.rolling(5).sum()
feats["mktret5"] = pd.DataFrame(np.tile(mret5.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
mtrend = (close.where(member).mean(axis=1).pct_change(200, fill_method=None) > 0).astype(float)
feats["mkttrend"] = pd.DataFrame(np.tile(mtrend.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
dow = pd.Series(close.index.dayofweek, index=close.index).astype(float)
feats["dow"] = pd.DataFrame(np.tile(dow.values[:, None], (1, close.shape[1])), index=close.index, columns=close.columns)
ek = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "8k_items.parquet"))
ek["date"] = pd.to_datetime(ek["date"])
ek = ek[ek.tk.isin(close.columns)]
evm = pd.DataFrame(False, index=close.index, columns=close.columns)
earnm = pd.DataFrame(False, index=close.index, columns=close.columns)
for tk, g in ek.groupby("tk"):
    idx = close.index.searchsorted(g["date"].values)
    idx = idx[idx < len(close.index)]
    evm.iloc[idx, evm.columns.get_loc(tk)] = True
e2 = ek[ek["items"].str.contains("2.02", na=False)]
for tk, g in e2.groupby("tk"):
    idx = close.index.searchsorted(g["date"].values)
    idx = idx[idx < len(close.index)]
    earnm.iloc[idx, earnm.columns.get_loc(tk)] = True
feats["news5"] = evm.rolling(5, min_periods=1).sum().clip(0, 3)
feats["earn5"] = earnm.rolling(5, min_periods=1).sum().clip(0, 2)
feats["dsince_earn"] = (~earnm).astype(float).groupby(earnm.astype(int).cumsum().values.argmax(axis=0) if False else None) if False else None
del feats["dsince_earn"]

o = open_
tgt = {}
for h in (1, 5):
    fwd = (o.shift(-(h + 1)) / o.shift(-1) - 1)
    yv = fwd.sub(fwd.where(member).mean(axis=1), axis=0).div(vol20 * np.sqrt(h))
    tgt[h] = yv.clip(-3, 3)

names = list(feats)
print(len(names), "features", flush=True)
mask0 = member.copy()
for f in feats.values():
    mask0 &= f.notna()
X_all = np.stack([feats[n].values.astype(np.float32) for n in names], axis=2)
didx = close.index
import lightgbm as lgb

preds = {}
for h in (1, 5):
    y = tgt[h]
    mask = (mask0 & y.notna()).values
    Y_all = y.values.astype(np.float32)
    pred = pd.DataFrame(np.nan, index=didx, columns=close.columns, dtype=np.float32)
    for yr in range(2014, 2027):
        tr_lo = didx.searchsorted(pd.Timestamp("2010-01-01"))
        tr_end = didx.searchsorted(pd.Timestamp(f"{yr-1}-12-15"))
        te_lo = didx.searchsorted(pd.Timestamp(f"{yr}-01-01"))
        te_hi = didx.searchsorted(pd.Timestamp(f"{yr+1}-01-01"))
        if te_lo >= len(didx): break
        tr_m = mask[tr_lo:tr_end]
        Xtr = X_all[tr_lo:tr_end][tr_m]; Ytr = Y_all[tr_lo:tr_end][tr_m]
        if len(Xtr) < 50000: continue
        gbm = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                min_child_samples=2000, subsample=0.7, subsample_freq=1,
                                colsample_bytree=0.6, reg_lambda=20.0, n_jobs=4, verbose=-1)
        gbm.fit(Xtr, Ytr)
        te_m = mask[te_lo:te_hi]
        pg = np.full(te_m.shape, np.nan, dtype=np.float32)
        pg[te_m] = gbm.predict(X_all[te_lo:te_hi][te_m])
        pred.iloc[te_lo:te_hi] = pg
        print(f"h{h} {yr}: {len(Xtr)} rows {time.time()-t0:.0f}s", flush=True)
    preds[h] = pred
    pred.to_parquet(os.path.join(ROOT, "cache", f"exp14_pred_h{h}.parquet"))
    ic = pred.corrwith(y, axis=1, method="spearman")
    icd = ic.dropna()
    print(f"h{h} IC mean={icd.mean():.4f} t={icd.mean()/icd.std()*np.sqrt(len(icd)):.1f}", flush=True)

out = {}
def ev(name, w, cost=5.0):
    res = bt.run(w, P, mode="open", cost_bps=cost)
    m = bt.metrics(res["net"])
    m["gross_sharpe"] = bt.sharpe(res["gross"])
    m["avg_turnover"] = round(float(res["turnover"].mean()), 3)
    m.update(bt.is_oos(res["net"], split="2020-01-01"))
    m["yearly"] = bt.yearly_sharpes(res["net"])
    out[name] = m
    print(f"{name:26s} netSR={m['sharpe']:6.2f} grossSR={m['gross_sharpe']:6.2f} 14-19={m['IS']:6.2f} 20+={m['OOS']:6.2f} to={m['avg_turnover']:5.2f}", flush=True)

blend = zs(preds[1]).add(zs(preds[5]))
for nm, pr in (("h1", preds[1]), ("h5", preds[5]), ("blend", blend)):
    ev(f"ml2_{nm}_q20", bt.norm_ls(pr, member, 0.2, 0.2, 2.0))
    ev(f"ml2_{nm}_q20_sm3", bt.norm_ls(pr.rolling(3).mean(), member, 0.2, 0.2, 2.0))

# turnover-buffered top/bottom-quintile: enter at q20, exit only past q35
rk = blend.where(member).rank(axis=1, pct=True)
W = pd.DataFrame(0.0, index=didx, columns=close.columns)
state = np.zeros(close.shape[1])
rkv = rk.values
for i in range(rkv.shape[0]):
    row = rkv[i]
    newstate = state.copy()
    newstate[(state == 0) & (row >= 0.8)] = 1
    newstate[(state == 0) & (row <= 0.2)] = -1
    newstate[(state == 1) & (row < 0.65)] = 0
    newstate[(state == -1) & (row > 0.35)] = 0
    newstate[np.isnan(row)] = 0
    state = newstate
    W.iloc[i] = state
gl = W.clip(lower=0).sum(axis=1); gs = (-W.clip(upper=0)).sum(axis=1)
Wn = W.clip(lower=0).div(gl.replace(0, np.nan), axis=0).fillna(0) - (-W.clip(upper=0)).div(gs.replace(0, np.nan), axis=0).fillna(0)
ev("ml2_blend_buffered", Wn)

json.dump(out, open(os.path.join(ROOT, "results", "exp14_ml2.json"), "w"), indent=1)
print("DONE", time.time() - t0, flush=True)
