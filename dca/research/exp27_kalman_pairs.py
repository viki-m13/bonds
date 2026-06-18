"""Experiment 27 — Ernie Chan's Kalman-filter PAIRS trading (adaptive stat-arb).
Dynamic hedge ratio via Kalman filter (Chan 'Algorithmic Trading' Ex 3.3):
state=[beta,alpha] random walk; observe y=beta*x+alpha. Trade the forecast
error e vs its sqrt(Q) band (mean-reversion). Market-neutral, dollar-hedged.
Test classic + sector ETF pairs, net of cost, IN-SAMPLE(<2018) vs OOS(2018+).
NOTE: requires shorting one leg (margin)."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
PAIRS = [("GLD", "GDX"), ("EWA", "EWC"), ("XLE", "XOP"), ("XLK", "VGT"),
         ("QQQ", "SPY"), ("XLF", "KRE"), ("IYR", "VNQ"), ("XLP", "XLU"),
         ("EWA", "EWC"), ("GDX", "GDXJ")]
allt = sorted(set([t for p in PAIRS for t in p]))
px = yf.download(allt, start="2008-01-01", auto_adjust=True, progress=False)["Close"]
COST = 0.0005                                       # 5 bps per leg per trade


def kalman_pairs(y, x):
    """returns daily strategy returns series (market-neutral spread)."""
    n = len(y)
    delta = 1e-4
    Vw = delta / (1 - delta) * np.eye(2)
    Ve = 0.001
    beta = np.zeros((2, n))                          # [hedge, intercept]
    R = np.zeros((2, 2)); P = np.zeros((2, 2))
    e = np.zeros(n); Q = np.zeros(n)
    bb = np.zeros(2)
    for t in range(n):
        Fx = np.array([x[t], 1.0])
        if t > 0:
            R = P + Vw
        yhat = Fx @ bb
        et = y[t] - yhat
        Qt = Fx @ R @ Fx + Ve
        K = R @ Fx / Qt
        bb = bb + K * et
        P = R - np.outer(K, Fx) @ R
        beta[:, t] = bb; e[t] = et; Q[t] = Qt
    sq = np.sqrt(Q)
    # state machine: +1 long spread when e<-sqrt(Q), -1 short when e>sqrt(Q), exit on revert
    pos = np.zeros(n); cur = 0
    for t in range(1, n):
        if cur == 0:
            if e[t] < -sq[t]:
                cur = 1
            elif e[t] > sq[t]:
                cur = -1
        elif cur == 1 and e[t] > 0:
            cur = 0
        elif cur == -1 and e[t] < 0:
            cur = 0
        pos[t] = cur
    # spread daily pnl: position * (dy - beta*dx), capital = |y|+|beta*x|
    dy = np.diff(y, prepend=y[0]); dx = np.diff(x, prepend=x[0])
    hedge = beta[0]
    pos_l = np.roll(pos, 1); pos_l[0] = 0           # yesterday's position (no lookahead)
    hedge_l = np.roll(hedge, 1); hedge_l[0] = hedge[0]
    pnl = pos_l * (dy - hedge_l * dx)
    cap = np.abs(y) + np.abs(hedge_l * x) + 1e-9
    trades = np.abs(np.diff(pos, prepend=0))
    ret = pnl / cap - COST * trades * 2             # 2 legs, charged on trade days
    return pd.Series(ret, index=y.index)


def sharpe(s, lo=None, hi=None):
    s = s.dropna()
    if lo:
        s = s[(s.index >= lo) & (s.index < hi)]
    return float(s.mean() / (s.std() + 1e-12) * np.sqrt(252)) if len(s) > 100 else np.nan


print(f"data ready t={time.time()-t0:.0f}s\n", flush=True)
print(f"{'pair':14s} {'IS Sharpe <2018':>15s} {'OOS Sharpe 2018+':>17s} {'OOS CAGR':>10s}",
      flush=True)
oos_all = []
for a, b in PAIRS:
    if a not in px.columns or b not in px.columns:
        continue
    d = px[[a, b]].dropna()
    if len(d) < 800:
        print(f"{a}/{b:8s} insufficient data", flush=True); continue
    ret = kalman_pairs(d[a].values.astype(float), pd.Series(d[b].values.astype(float), index=d.index)
                       ) if False else kalman_pairs(
        pd.Series(d[a].values.astype(float), index=d.index),
        d[b].values.astype(float))
    iss = sharpe(ret, "2008-01-01", "2018-01-01")
    oos = sharpe(ret, "2018-01-01", "2026-07-01")
    o = ret[(ret.index >= "2018-01-01")].dropna()
    cagr = (1 + o).prod() ** (252 / len(o)) - 1 if len(o) > 100 else np.nan
    oos_all.append(ret[ret.index >= "2018-01-01"])
    print(f"{a}/{b:11s} {iss:15.2f} {oos:17.2f} {cagr*100:9.1f}%", flush=True)

# equal-weight portfolio of all pairs (diversified stat-arb book)
port = pd.concat(oos_all, axis=1).mean(axis=1)
print(f"\nEqual-weight book of all pairs, OOS 2018+: Sharpe {sharpe(port):.2f}  "
      f"CAGR {((1+port.dropna()).prod()**(252/len(port.dropna()))-1)*100:.1f}%", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
