"""Exp 29 — faithful ESSENCE of ReCAP (regime-adaptive portfolio), tested honestly.
Detect market regimes (KMeans on macro/market features), learn best long-only
allocation PER REGIME on TRAIN only (the 'policy library'), apply OOS by current
regime ('regime gate'). Walk-forward by year. CONTROLS: static equal-weight,
QQQ, 60/40, and RANDOM-REGIME (shuffle labels) — if real regimes don't beat
shuffled, the regime machinery is noise."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from sklearn.cluster import KMeans
t0 = time.time()
U = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "DBC", "VNQ"]
aux = ["HYG"]
px = yf.download(U + aux, start="2006-01-01", auto_adjust=True, progress=False)["Close"].dropna()
me = px.resample("ME").last(); me.index = me.index.to_period("M").to_timestamp()
ret = me[U].pct_change()
# market-state features (all causal, known at month t)
spy = me["SPY"]
feat = pd.DataFrame(index=me.index)
feat["mom3"] = spy / spy.shift(3) - 1
feat["mom12"] = spy / spy.shift(12) - 1
feat["vol"] = ret["SPY"].rolling(6).std()
feat["credit"] = (me["HYG"] / me["IEF"]) / (me["HYG"] / me["IEF"]).shift(3) - 1
feat["rates"] = me["TLT"] / me["TLT"].shift(3) - 1
feat["disp"] = ret[U].rolling(3).std().mean(axis=1)
feat = feat.dropna()
print(f"data {me.index[0].date()}->{me.index[-1].date()}, {len(feat)} months  t={time.time()-t0:.0f}s",
      flush=True)
K = 3


def regime_alloc(train_feat, train_ret, labels):
    """per-regime long-only max-Sharpe-ish weights from training months."""
    alloc = {}
    for g in range(K):
        idx = train_feat.index[labels == g]
        rr = train_ret.loc[idx]
        if len(rr) < 6:
            alloc[g] = pd.Series(1.0 / len(U), index=U); continue
        score = (rr.mean() / (rr.var() + 1e-9)).clip(lower=0)
        alloc[g] = (score / score.sum()) if score.sum() > 0 else pd.Series(1.0 / len(U), index=U)
    return alloc


def run(shuffle=False):
    oos = {}
    for Y in range(2013, 2026):
        tr_f = feat[feat.index < pd.Timestamp(Y, 1, 1)]
        te_f = feat[(feat.index >= pd.Timestamp(Y, 1, 1)) & (feat.index < pd.Timestamp(Y + 1, 1, 1))]
        if len(tr_f) < 36 or len(te_f) == 0:
            continue
        sc = (tr_f - tr_f.mean()) / (tr_f.std() + 1e-9)
        km = KMeans(K, n_init=5, random_state=0).fit(sc)
        labels = km.labels_.copy()
        if shuffle:
            rng = np.random.default_rng(Y); rng.shuffle(labels)   # break regime->month link
        alloc = regime_alloc(tr_f, ret.reindex(tr_f.index), labels)
        for d in te_f.index:
            g = int(km.predict(((te_f.loc[[d]] - tr_f.mean()) / (tr_f.std() + 1e-9)).values)[0])
            nxt = me.index[me.index.get_loc(d) + 1] if me.index.get_loc(d) + 1 < len(me.index) else None
            if nxt is None:
                continue
            w = alloc[g]
            oos[nxt] = float((w * ret.loc[nxt]).sum())
    return pd.Series(oos).dropna()


def stats(s, name):
    eq = (1 + s).cumprod(); yrs = len(s) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    sh = float(s.mean() / (s.std() + 1e-12) * np.sqrt(12))
    print(f"   {name:30s} Sharpe {sh:.2f}  CAGR {cagr*100:5.1f}%  maxDD {mdd*100:4.0f}%", flush=True)
    return sh


print("\nReCAP-essence vs controls (OOS 2013-2025, walk-forward):", flush=True)
reg = run(shuffle=False)
idx = reg.index
stats(ret["QQQ"].reindex(idx).dropna(), "QQQ")
stats((0.6 * ret["SPY"] + 0.4 * ret["IEF"]).reindex(idx).dropna(), "60/40")
stats(ret[U].reindex(idx).mean(axis=1), "static equal-weight")
sh_reg = stats(reg, "REGIME-ADAPTIVE (ReCAP-ess)")
shuf = [run(shuffle=True) for _ in range(5)]
sh_shuf = np.mean([float(s.mean() / (s.std() + 1e-12) * np.sqrt(12)) for s in shuf])
cg_shuf = np.mean([(1 + s).cumprod().iloc[-1] ** (12 / len(s)) - 1 for s in shuf])
print(f"   {'RANDOM-REGIME control (avg5)':30s} Sharpe {sh_shuf:.2f}  CAGR {cg_shuf*100:5.1f}%", flush=True)
print(f"\n   -> regime-adaptive Sharpe {sh_reg:.2f} vs random-regime {sh_shuf:.2f}: "
      f"{'real regimes add value' if sh_reg > sh_shuf + 0.15 else 'NO better than shuffling labels'}",
      flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
