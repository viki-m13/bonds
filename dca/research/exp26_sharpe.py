"""Experiment 26 — how high can a tactical ETF Sharpe REALISTICALLY go (OOS)?
Legit Sharpe-improvers: diversification + inverse-vol (risk parity) + trend
filter + NO-LEVERAGE vol-targeting. vs QQQ / 60-40. THEN demonstrate the mirage:
in-sample max-Sharpe MVO (walk-forward) -> high IS Sharpe that collapses OOS.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
U = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "LQD", "GLD", "DBC", "VNQ"]
px = yf.download(U, start="2006-01-01", auto_adjust=True, progress=False)["Close"].dropna()
r = px.pct_change().dropna()
ma200 = px.rolling(200).mean()
print(f"{len(U)} ETFs {px.index[0].date()}->{px.index[-1].date()}  t={time.time()-t0:.0f}s",
      flush=True)


def sharpe(x):
    x = x.dropna()
    return float(x.mean() / (x.std() + 1e-12) * np.sqrt(252))


def stats(daily, name):
    d = daily.dropna()
    eq = (1 + d).cumprod(); yrs = len(d) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    vol = d.std() * np.sqrt(252)
    print(f"   {name:30s} Sharpe {sharpe(d):.2f}  CAGR {cagr*100:5.1f}%  "
          f"vol {vol*100:4.0f}%  maxDD {mdd*100:4.0f}%", flush=True)
    return sharpe(d)


# rebalance monthly; weights known at prior month-end (no lookahead)
me = r.resample("ME").last().index
vol20 = r.rolling(60).std()                        # ~3m vol for inverse-vol


def backtest(weight_fn, voltarget=None):
    w_prev = None; rets = []
    daily_w = pd.DataFrame(index=r.index, columns=U, dtype=float)
    cur = None
    for i, dt in enumerate(r.index):
        if cur is None or (i > 0 and dt.month != r.index[i - 1].month):
            cur = weight_fn(dt)                    # rebalance at month change
        daily_w.loc[dt] = cur
    port = (daily_w.shift(1) * r).sum(axis=1)
    if voltarget is not None:                      # NO-LEVERAGE vol target (delever only)
        rv = port.rolling(20).std() * np.sqrt(252)
        scale = (voltarget / rv).clip(upper=1.0).shift(1).fillna(1.0)
        port = port * scale
    return port


def w_equal(dt):
    return pd.Series(1.0 / len(U), index=U)


def w_invvol(dt):
    v = vol20.loc[:dt].iloc[-1]
    iv = (1 / v).replace([np.inf], np.nan).dropna()
    return (iv / iv.sum()).reindex(U).fillna(0)


def w_invvol_trend(dt):
    up = (px.loc[:dt].iloc[-1] > ma200.loc[:dt].iloc[-1])
    v = vol20.loc[:dt].iloc[-1]
    iv = (1 / v).where(up, 0).replace([np.inf], np.nan).fillna(0)
    s = iv.sum()
    return (iv / s).reindex(U).fillna(0) if s > 0 else pd.Series(0.0, index=U)


print("\nLegit tactical ETF portfolios (full period, monthly rebal, OOS-honest):",
      flush=True)
stats(r["QQQ"], "QQQ (reference)")
stats(0.6 * r["SPY"] + 0.4 * r["IEF"], "60/40 SPY-IEF (reference)")
stats(backtest(w_equal), "equal-weight 10 ETFs")
stats(backtest(w_invvol), "inverse-vol (risk parity)")
stats(backtest(w_invvol_trend), "inv-vol + 200d trend filter")
stats(backtest(w_invvol_trend, voltarget=0.10), "inv-vol+trend+vol-target 10%")

# ---- the MIRAGE: in-sample max-Sharpe MVO, walk-forward ----
print("\nMIRAGE demo — in-sample max-Sharpe optimization:", flush=True)
rm = r.resample("ME").last().dropna()
is_sh, oos = [], []
oos_rets = []
for i in range(36, len(rm) - 1):
    train = r[(r.index > rm.index[i - 36]) & (r.index <= rm.index[i])]
    mu = train.mean().values * 252
    cov = train.cov().values * 252
    try:
        w = np.linalg.solve(cov + np.eye(len(U)) * 1e-4, mu)   # max-Sharpe (unconstrained)
        w = np.clip(w, 0, None); w = w / w.sum() if w.sum() > 0 else None
    except Exception:
        continue
    if w is None:
        continue
    is_sh.append(sharpe((train * w).sum(axis=1)))
    nxt = r[(r.index > rm.index[i]) & (r.index <= rm.index[i + 1])]
    oos_rets.append((nxt * w).sum(axis=1))
oos_series = pd.concat(oos_rets)
print(f"   in-sample Sharpe (avg over windows): {np.mean(is_sh):.2f}", flush=True)
print(f"   OUT-OF-SAMPLE Sharpe (stitched):     {sharpe(oos_series):.2f}", flush=True)
print(f"   -> the IS Sharpe is a mirage; OOS is what you actually get.", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
