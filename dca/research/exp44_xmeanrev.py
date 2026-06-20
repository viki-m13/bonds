"""Exp 44 (loop1) — diversified short-term mean-reversion book (Connors RSI-2).
Idea: IBS on one ETF = in-mkt Sharpe 1.9 but 20% exposure. Run it across the
WHOLE S&P500 -> always hold a basket of oversold-in-uptrend names -> breadth
lifts Sharpe + exposure. Buy RSI2<10 & close>200dMA; exit RSI2>65 or close>10dMA.
Equal-weight held basket (cap 20), net of cost. OOS + corr to QQQ."""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
names = sorted(set(open("/tmp/wave/sp500_universe.txt").read().split()))
raw = yf.download(names+["QQQ"], start="2008-01-01", auto_adjust=True, progress=False)["Close"]
names = [t for t in names if t in raw.columns and raw[t].notna().sum()>2000]
C = raw[names]; ret = C.pct_change()
d1 = C.diff(); gain = d1.clip(lower=0).rolling(2).mean(); loss = (-d1.clip(upper=0)).rolling(2).mean()
rsi2 = 100 - 100/(1 + gain/(loss+1e-12))
ma200 = C.rolling(200).mean(); ma10 = C.rolling(10).mean()
elig = (rsi2 < 10) & (C > ma200)                       # oversold in uptrend
exit_ = (rsi2 > 65) | (C > ma10)
idx = C.index; CAP = 20; COST = 0.0005
held = {}; rets = []
for i in range(210, len(idx)-1):
    d = idx[i]; nxt = idx[i+1]
    # exits
    for t in list(held):
        if (not np.isfinite(C.at[d,t])) or bool(exit_.at[d,t]):
            del held[t]
    # entries: most-oversold eligible not held, fill to CAP
    cand = [t for t in names if bool(elig.at[d,t]) and t not in held and np.isfinite(ret.at[nxt,t])]
    cand = sorted(cand, key=lambda t: rsi2.at[d,t])[:max(0, CAP-len(held))]
    nnew = len(cand)
    for t in cand: held[t] = 1
    if held:
        r = np.mean([ret.at[nxt,t] for t in held if np.isfinite(ret.at[nxt,t])])
        turn = (nnew)/max(len(held),1)                 # fraction newly traded
        rets.append((nxt, r - turn*COST*2))
    else:
        rets.append((nxt, 0.0))
s = pd.Series(dict(rets)).dropna()
def st(x,lo=None,hi=None):
    if lo: x=x[(x.index>=lo)&(x.index<hi)]
    x=x.dropna(); eq=(1+x).cumprod(); yrs=len(x)/252
    return eq.iloc[-1]**(1/yrs)-1, x.mean()/(x.std()+1e-12)*np.sqrt(252), float((eq/eq.cummax()-1).min()), (x!=0).mean()
print(f"{len(names)} names  t={time.time()-t0:.0f}s", flush=True)
print("Diversified RSI-2 mean-reversion book (daily, net 5bps/side):", flush=True)
for tag,lo,hi in (("FULL","2009-01-01","2026-12-31"),("TRAIN<2017","2009-01-01","2017-01-01"),
                  ("TEST 2017+","2017-01-01","2026-12-31")):
    c,sh,dd,ex = st(s,lo,hi); print(f"  {tag:11s} CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%  exposure {ex*100:3.0f}%", flush=True)
qd = raw["QQQ"].pct_change().reindex(s.index)
print(f"  corr to QQQ (daily): {s.corr(qd):.2f}", flush=True)
sm = ((1+s).groupby(s.index.to_period("M")).prod()-1); sm.index=sm.index.to_timestamp(); sm.to_pickle("/tmp/wave/_xmr_m.pkl")
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
