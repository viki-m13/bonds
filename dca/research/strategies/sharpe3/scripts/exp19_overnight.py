"""Invention track L: the overnight/intraday axis (OHLC, top-600 liquid names).

Honest execution grammar with opens available:
  - signal from close t  -> order placed overnight -> filled at OPEN t+1
    (opening auction; no look-ahead), or at CLOSE t+1 (old grammar).
  - gap signals (known at open) -> execute at close same day (conservative)
    or 'at open' (optimistic bound, labeled).

Tests:
  1. Decompose the crash-bounce: after a market-adjusted crash at close t,
     how much of the t+1 payoff is overnight (t close->t+1 open, untradeable)
     vs intraday (t+1 open->t+1 close, TRADEABLE via opening auction)?
  2. Crash-bounce book with open-entry: buy at open t+1, exit close t+1 /
     open t+2 / close t+2. Fees 5/10 bps per side.
  3. Gap-fade: fade residual overnight gaps open->close (optimistic bound).
  4. Overnight premium structure: per-name mean overnight vs intraday (doc).
Period: 2000-2019 (family-existence test; TEST 2020+ stays locked).
"""
import os, glob, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
D = "/tmp/sharpe3_work/ohlc"
tickers = [t.strip() for t in open(f"{D}/tickers.txt")]
ao, ac = {}, {}
for t in tickers:
    p = f"{D}/{t}.csv"
    if not os.path.exists(p) or os.path.getsize(p) < 1000: continue
    try:
        d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        if "adjOpen" not in d.columns: continue
        ao[t] = d["adjOpen"]; ac[t] = d["adjClose"]
    except Exception:
        continue
AO = pd.DataFrame(ao); AC = pd.DataFrame(ac)
AO, AC = AO.align(AC, join="outer")
print(f"OHLC panel: {AC.shape[1]} names, {AC.shape[0]} days  t={time.time()-t0:.0f}s", flush=True)

R_cc = AC.pct_change(fill_method=None)                       # close->close
R_on = (AO / AC.shift(1) - 1)                                # overnight (close->open)
R_id = (AC / AO - 1)                                         # intraday (open->close)
for X in (R_cc, R_on, R_id):
    X[np.abs(X) > 2] = np.nan
mkt_cc = R_cc.mean(axis=1); mkt_on = R_on.mean(axis=1); mkt_id = R_id.mean(axis=1)
xs_cc = R_cc.sub(mkt_cc, axis=0)
vol = xs_cc.rolling(63, min_periods=40).std().shift(1)
Zc = (xs_cc / vol)
idx = AC.index
per = (idx >= pd.Timestamp("2000-01-01")) & (idx <= pd.Timestamp("2019-12-31"))

# ---------- 1) bounce decomposition ----------
Zv = Zc.values
on_x = R_on.sub(mkt_on, axis=0).values
id_x = R_id.sub(mkt_id, axis=0).values
ev = np.argwhere((Zv < -2.5) & per[:, None])
ev = ev[ev[:, 0] < len(idx) - 4]
on1, id1, on2, id2 = [], [], [], []
for ti, ni in ev:
    a, b = on_x[ti+1, ni], id_x[ti+1, ni]
    c, d2 = on_x[ti+2, ni], id_x[ti+2, ni]
    if np.isfinite(a) and np.isfinite(b): on1.append(a); id1.append(b)
    if np.isfinite(c) and np.isfinite(d2): on2.append(c); id2.append(d2)
print(f"crash z<-2.5 (n={len(on1)}): t+1 overnight {np.mean(on1)*1e4:+.0f}bp  t+1 INTRADAY {np.mean(id1)*1e4:+.0f}bp  "
      f"t+2 overnight {np.mean(on2)*1e4:+.0f}bp  t+2 intraday {np.mean(id2)*1e4:+.0f}bp", flush=True)

# ---------- 2) crash-bounce with open entry ----------
def crash_book(exit_mode, th=-2.5, fee=10):
    pnl = []
    dates = []
    for ti in range(2, len(idx)-3):
        if not per[ti]: continue
        sel = np.where(Zv[ti] < th)[0]
        if len(sel) == 0:
            pnl.append(0.0); dates.append(idx[ti]); continue
        rets = []
        for ni in sel:
            if exit_mode == "c1":   r = id_x[ti+1, ni]
            elif exit_mode == "o2": r = id_x[ti+1, ni] + on_x[ti+2, ni]
            else:                    r = id_x[ti+1, ni] + on_x[ti+2, ni] + id_x[ti+2, ni]
            if np.isfinite(r): rets.append(r)
        gr = np.mean(rets) if rets else 0.0
        pnl.append(gr - 2*fee/1e4)
        dates.append(idx[ti])
    s = pd.Series(pnl, index=pd.DatetimeIndex(dates))
    act = s[s != 0]
    print(f"crash-book open-entry exit={exit_mode} fee={fee}: full Sharpe {G.sharpe(s):5.2f}  "
          f"active {len(act)/max(len(s),1):.0%}  mean/trade {act.mean()*1e4:+.0f}bp", flush=True)

for em in ("c1", "o2", "c2"):
    for fee in (5, 10):
        crash_book(em, fee=fee)

# ---------- 3) gap-fade (optimistic: execute at open) ----------
gap = R_on.sub(mkt_on, axis=0) / vol
gv = gap.values
pnl, dates = [], []
for ti in range(2, len(idx)-1):
    if not per[ti]: continue
    sel_l = np.where(gv[ti] < -2)[0]; sel_s = np.where(gv[ti] > 2)[0]
    rets = [id_x[ti, ni] for ni in sel_l if np.isfinite(id_x[ti, ni])]
    rets += [-id_x[ti, ni] for ni in sel_s if np.isfinite(id_x[ti, ni])]
    pnl.append((np.mean(rets) - 2*10/1e4) if rets else 0.0)
    dates.append(idx[ti])
s = pd.Series(pnl, index=pd.DatetimeIndex(dates))
act = s[s != 0]
print(f"gap-fade |z|>2 at-open fee=10: full Sharpe {G.sharpe(s):5.2f}  active {len(act)/max(len(s),1):.0%}  "
      f"mean/trade {act.mean()*1e4:+.0f}bp", flush=True)

# ---------- 4) overnight premium doc ----------
on_mean = R_on[per].mean().mean()*1e4; id_mean = R_id[per].mean().mean()*1e4
print(f"avg per-name daily: overnight {on_mean:+.1f}bp  intraday {id_mean:+.1f}bp  (the known overnight anomaly)", flush=True)
print(f"exp19 done t={time.time()-t0:.0f}s", flush=True)
