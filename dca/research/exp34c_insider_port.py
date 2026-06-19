"""Exp 34c — deployable test: long-only portfolio of current net-insider-buyer
stocks (monthly rebalance, net of cost) vs SPY, QQQ, universe equal-weight, and
RANDOM same-size control. Full + sub-period. Is the stable +0.7pp/3m signal a
real, harvestable edge over the market (and how does it compare to QQQ)?"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
P = pd.read_pickle("/tmp/wave/_insider_panel.pkl")
uni = set()
for f in ("sp500_universe.txt", "xuniverse_sp400.txt", "xuniverse_ndx.txt"):
    txt = open(f"/tmp/wave/{f}").read()
    uni |= set(txt.split()) if " " in txt else set(l.strip() for l in txt.splitlines() if l.strip())
uni = sorted(t for t in uni if t and t.isalpha())
px = yf.download(uni + ["SPY", "QQQ"], start="2009-06-01", auto_adjust=True, progress=False)["Close"]
names = [t for t in uni if t in px.columns and px[t].notna().sum() > 800]
me = px[names + ["SPY", "QQQ"]].resample("ME").last(); me.index = me.index.to_period("M").to_timestamp()
mret = me.pct_change()
P = P[P.tk.isin(names)]
buy = P.pivot_table(index="ym", columns="tk", values="buy", aggfunc="sum").reindex(index=me.index, columns=names).fillna(0)
sell = P.pivot_table(index="ym", columns="tk", values="sell", aggfunc="sum").reindex(index=me.index, columns=names).fillna(0)
netbuyer = ((buy - sell).rolling(3, min_periods=1).sum() > 0)    # net buyer over trailing 3m
print(f"priced {len(names)} names  t={time.time()-t0:.0f}s", flush=True)

def port(kind, cost=0.0, seed=0):
    rng = np.random.default_rng(seed); prev = set(); rets = []
    for i in range(3, len(me.index) - 1):
        d = me.index[i]; nxt = me.index[i + 1]
        avail = [t for t in names if np.isfinite(mret.loc[nxt, t])]
        if kind == "insider":
            sel = [t for t in avail if netbuyer.loc[d, t]]
        elif kind == "ew":
            sel = avail
        else:                                        # random, matched count to insider set
            k = max(5, int(netbuyer.loc[d, avail].sum()))
            sel = list(rng.choice(avail, min(k, len(avail)), replace=False))
        if len(sel) < 5:
            continue
        turn = 1.0 - len(prev & set(sel)) / max(len(sel), 1)
        rets.append((nxt, mret.loc[nxt, sel].mean() - turn * cost)); prev = set(sel)
    return pd.Series(dict(rets)).dropna()

def stat(s, lo=None, hi=None):
    if lo: s = s[(s.index >= lo) & (s.index < hi)]
    s = s.dropna(); eq = (1 + s).cumprod(); yrs = len(s) / 12
    return (eq.iloc[-1] ** (1/yrs) - 1, s.mean()/(s.std()+1e-12)*np.sqrt(12), float((eq/eq.cummax()-1).min()))

ins = port("insider", 0.002); ins0 = port("insider", 0.0)
ew = port("ew", 0.0); rnd = pd.concat([port("rand", 0.002, s) for s in range(6)], axis=1).mean(axis=1)
idx = ins.index
for nm, s in (("Insider net-buyer NET 20bps", ins), ("Insider net-buyer GROSS", ins0),
              ("Universe equal-weight", ew), ("RANDOM same-size NET 20bps", rnd),
              ("SPY", mret["SPY"].reindex(idx).dropna()), ("QQQ", mret["QQQ"].reindex(idx).dropna())):
    c, sh, dd = stat(s)
    print(f"  {nm:30s} CAGR {c*100:5.1f}%  Sharpe {sh:.2f}  maxDD {dd*100:4.0f}%", flush=True)
print("\n  sub-period CAGR (insider-net20 / SPY / random):", flush=True)
for lo, hi, tg in (("2010-01-01","2018-01-01","2010-17"),("2018-01-01","2025-07-01","2018-25")):
    print(f"    {tg}: insider {stat(ins,lo,hi)[0]*100:5.1f}%  SPY {stat(mret['SPY'].reindex(idx),lo,hi)[0]*100:5.1f}%"
          f"  random {stat(rnd,lo,hi)[0]*100:5.1f}%  QQQ {stat(mret['QQQ'].reindex(idx),lo,hi)[0]*100:5.1f}%", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
