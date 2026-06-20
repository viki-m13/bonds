"""Exp 38 — reproduce the NDX Rotational Momentum (SystematicPeter) EXACTLY.
Trade only when QQQ>200dMA. Monthly: among NDX-100, eligible = price>own 200dMA
AND 250d return>0; rank by 250d ROC; buy top K equal-weight; hold 1 month; else
cash. Net of cost. CONTROLS: vs RANDOM-K from same eligible set (isolates whether
the momentum RANK adds value beyond gate+universe), vs QQQ, vs eligible-EW.
Current NDX list = survivorship-biased (inflates -> conservative)."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
ndx = [l.strip() for l in open("/tmp/wave/xuniverse_ndx.txt") if l.strip()]
raw = yf.download(ndx + ["QQQ"], start="2007-01-01", auto_adjust=True, progress=False)["Close"]
names = [t for t in ndx if t in raw.columns and raw[t].notna().sum() > 1200]
C = raw[names]; q = raw["QQQ"]
ma200 = C.rolling(200).mean(); qma = q.rolling(200).mean()
roc250 = C / C.shift(250) - 1
me = C.resample("ME").last().index
me = [d for d in me if d in C.index]
mret = raw.resample("ME").last(); mret.index = mret.index  # month-end levels


def strat(kind, K, cost=0.001, seed=0):
    rng = np.random.default_rng(seed); prev = set(); rets = []
    for i in range(len(me) - 1):
        d = me[i]; nxt = me[i + 1]
        if not (q.loc[:d].iloc[-1] > qma.loc[:d].iloc[-1]):       # NDX gate off -> cash
            rets.append((nxt, 0.0)); prev = set(); continue
        elig = [t for t in names if C.loc[:d, t].iloc[-1] > ma200.loc[:d, t].iloc[-1]
                and roc250.loc[:d, t].iloc[-1] > 0]
        if len(elig) < K:
            rets.append((nxt, 0.0)); prev = set(); continue
        if kind == "mom":
            sel = sorted(elig, key=lambda t: roc250.loc[:d, t].iloc[-1], reverse=True)[:K]
        elif kind == "rand":
            sel = list(rng.choice(elig, K, replace=False))
        else:
            sel = elig                                           # eligible equal-weight
        r = np.mean([(C.loc[:nxt, t].iloc[-1] / C.loc[:d, t].iloc[-1] - 1) for t in sel])
        turn = 1.0 - len(prev & set(sel)) / max(len(sel), 1)
        rets.append((nxt, r - turn * cost)); prev = set(sel)
    return pd.Series(dict(rets)).dropna()


def stats(s, lo=None, hi=None):
    if lo: s = s[(s.index >= lo) & (s.index < hi)]
    s = s.dropna(); eq = (1 + s).cumprod(); yrs = len(s) / 12
    return (eq.iloc[-1] ** (1/yrs) - 1, s.mean()/(s.std()+1e-12)*np.sqrt(12), float((eq/eq.cummax()-1).min()))


print(f"{len(names)} NDX names  t={time.time()-t0:.0f}s", flush=True)
for K in (5, 10):
    print(f"\n=== NDX rotational momentum, top-{K} (net 10bps) ===", flush=True)
    mom = strat("mom", K)
    rnd = pd.concat([strat("rand", K, seed=s) for s in range(6)], axis=1).mean(axis=1)
    idx = mom.index
    for nm, s in (("momentum top-K", mom), ("RANDOM-K (control)", rnd),
                  ("QQQ", q.resample("ME").last().pct_change().reindex(idx).dropna())):
        c, sh, dd = stats(s)
        c1, sh1, _ = stats(s, "2007-01-01", "2017-01-01"); c2, sh2, _ = stats(s, "2017-01-01", "2026-12-31")
        print(f"  {nm:20s} CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%  "
              f"[Sh: 07-16 {sh1:.2f} | 17-25 {sh2:.2f}]", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
