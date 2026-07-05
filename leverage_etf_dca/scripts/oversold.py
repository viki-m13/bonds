"""OVERSOLD / VALUE-TIMING cross-asset rotation — the honest test of the user's
hypothesis: "dynamically buy ETFs (gold, sectors, leveraged + normal) WHEN
UNDERSOLD so we outperform DCA-into-QQQ in every period, including dot-com."

This is mean-reversion (buy the dip), the OPPOSITE of the momentum rotation
already tested in rotation.py (which failed phase-robustness). We test several
oversold signals, with/without a long-term uptrend filter, base-only and
base+leveraged menus, across eras + full continuous + phase-robustness + dot-com.

No look-ahead: every monthly weight uses only data through the PRIOR month-end
(signals are .shift(1) on the monthly grid). DCA $1000/mo, 10 bps/side.
'Cash' = uninvested contribution earning 0 (conservative; no MMF yield credited).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl")
close = P["close"].sort_index(); kind = P["kind"]

retd = close.pct_change()
# per-asset "really trading" start = first date with a nonzero daily return
# (masks the flat reconstructed pre-underlying region of leveraged series)
valid_start = {}
for t in close.columns:
    nz = retd[t].ne(0) & retd[t].notna()
    valid_start[t] = nz.index[nz.argmax()] if nz.any() else close.index[-1]

BASE = [t for t in close.columns if kind[t] == "base"]
LEVL = [t for t in close.columns if kind[t] == "lev"]

def month_grid(nth=None):
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

def tradeable(dt, t, mg_px):
    return (dt >= valid_start[t]) and np.isfinite(mg_px.loc[dt, t])

def signals(grid):
    """all computed on daily close, sampled to grid, then these are 'as of dt'."""
    px = close.reindex(grid, method="ffill")
    ma200 = close.rolling(200, min_periods=120).mean().reindex(grid, method="ffill")
    ma50  = close.rolling(50,  min_periods=30).mean().reindex(grid, method="ffill")
    hi252 = close.rolling(252, min_periods=150).max().reindex(grid, method="ffill")
    # oversold measures (higher = more oversold)
    below_hi   = (hi252 / px - 1.0)                    # % below 1y high (drawdown depth)
    below_ma50 = (ma50 / px - 1.0)                     # % below 50d MA
    r1  = px / px.shift(1) - 1                          # 1-mo return (reversal: buy losers)
    r3  = px / px.shift(3) - 1
    r12 = px / px.shift(12) - 1                         # 12-mo trend
    up_trend = (px > ma200)                             # long-term uptrend filter
    return dict(px=px, below_hi=below_hi, below_ma50=below_ma50,
                r1=r1, r3=r3, r12=r12, up=up_trend, ma200=ma200)

_C = {}
def prep(nth):
    if nth not in _C:
        mg = month_grid(nth)
        _C[nth] = (mg, signals(mg))
    return _C[nth]

def score_fn(name, S, dt, universe):
    """return dict tk->score for oversold ranking (higher=buy). None-score = ineligible."""
    out = {}
    for t in universe:
        px = S["px"].loc[dt, t]
        if not (dt >= valid_start[t] and np.isfinite(px)): continue
        up = bool(S["up"].loc[dt, t]) if np.isfinite(S["ma200"].loc[dt, t]) else False
        if name == "dip_in_uptrend":
            # buy biggest 1-mo dip among assets still in a 200d uptrend
            if not up: continue
            v = -S["r1"].loc[dt, t]
        elif name == "deep_dd_uptrend":
            # biggest drawdown from 1y high, but still above 200d MA (dip not death)
            if not up: continue
            v = S["below_hi"].loc[dt, t]
        elif name == "pure_reversal":
            # most-down over 3 months, no trend filter (classic falling-knife MR)
            v = -S["r3"].loc[dt, t]
        elif name == "below_ma50_uptrend":
            if not up: continue
            v = S["below_ma50"].loc[dt, t]
        else:
            raise ValueError(name)
        if np.isfinite(v): out[t] = v
    return out

def rotate(start, end, sig="dip_in_uptrend", K=3, nth=None, universe=None,
           contrib=1000.0, cost=0.001, min_qual=1):
    mg, S = prep(nth)
    universe = universe if universe is not None else BASE
    grid = mg[(mg >= start) & (mg <= end)]
    px = S["px"]
    pos = {}; contributed = 0.0; rows = []; hold = []
    for dt in grid:
        # liquidate everything (full monthly re-selection), realize proceeds
        cash = contrib
        for t in list(pos.keys()):
            if t == "_CASH_":
                cash += pos.pop(t)  # carried cash, no trading cost
            elif np.isfinite(px.loc[dt, t]):
                cash += pos.pop(t) * px.loc[dt, t] * (1 - cost)
            else:
                pos.pop(t)
        contributed += contrib
        sc = score_fn(sig, S, dt, universe)
        picks = sorted(sc, key=lambda t: -sc[t])[:K]
        hold.append((dt, tuple(picks) if picks else ("CASH",)))
        if len(picks) >= min_qual and picks:
            per = cash / len(picks)
            for t in picks:
                pos[t] = per * (1 - cost) / px.loc[dt, t]
        else:
            pos["_CASH_"] = cash  # stays in cash this month (carried, 0 yield)
        V = 0.0
        for t, sh in pos.items():
            if t == "_CASH_": V += sh
            elif np.isfinite(px.loc[dt, t]): V += sh * px.loc[dt, t]
        rows.append((dt, V, contributed))
    eq = pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")
    return eq, pd.DataFrame(hold, columns=["date", "held"]).set_index("date")

def qqq_dca(start, end, nth=None, contrib=1000.0):
    g = month_grid(nth); g = g[(g >= start) & (g <= end)]
    r = close.reindex(month_grid(nth))["QQQ"].pct_change().reindex(g).fillna(0)
    V = 0; c = 0; rows = []
    for dt, x in r.items():
        V = V * (1 + x) + contrib; c += contrib; rows.append((dt, V, c))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

def maxdd(eq):
    V = eq["V"]; C = eq["contributed"].diff().fillna(eq["contributed"])
    r = ((V - C) / V.shift(1) - 1).dropna()
    cum = (1 + r).cumprod()
    return (cum / cum.cummax() - 1).min()

if __name__ == "__main__":
    ERAS = [("2006-01","2009-12"),("2010-01","2014-12"),("2015-01","2019-12"),
            ("2020-01","2026-06"),("2010-01","2026-06"),("2006-01","2026-06")]
    UNIS = {"base": BASE, "base+lev": BASE + LEVL}
    SIGS = ["dip_in_uptrend","deep_dd_uptrend","below_ma50_uptrend","pure_reversal"]
    print("RATIO vs QQQ-DCA (continuous DCA, month-end)\n")
    print(f"{'signal/universe/K':34}" + "".join(f"{a[:7]:>8}" for a,_ in ERAS))
    for uname, uni in UNIS.items():
        for sig in SIGS:
            for K in (1, 3):
                out = []
                for st, en in ERAS:
                    s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
                    eq,_ = rotate(s, e, sig=sig, K=K, universe=uni)
                    b = qqq_dca(s, e)
                    out.append(eq["V"].iloc[-1] / b["V"].iloc[-1])
                print(f"{sig[:20]:20} {uname:9} K{K} " + "".join(f"{v:>8.2f}" for v in out))
