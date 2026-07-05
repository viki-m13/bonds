"""REGIME SYNTHESIS — the one construction that could reconcile the two regimes.

Finding so far:
  * Oversold/diversified rotation BEATS QQQ-DCA in dot-com (1.1x, half the drawdown)
    but LOSES catastrophically 2010-26 (0.06-0.45x).
  * Leveraged-tech (VOLT) WINS 2010-26 but only protects (doesn't out-return) in dot-com.
They reward opposite behavior. Natural synthesis: a TREND GATE on the tech complex.
  RISK-ON  (QQQ > 200d MA): vol-targeted leveraged NASDAQ  (capture the tech decade)
  RISK-OFF (QQQ < 200d MA): diversified oversold sleeve / cash (survive & buy the bust)

If this survives PHASE-ROBUSTNESS (rebalance-day sensitivity) across every era AND
dot-com, it's a real edge. Every prior switch strategy died here — so this is the test.
No look-ahead: gate + weights use data through the PRIOR month-end (.shift(1)).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl")
close = P["close"].sort_index(); kind = P["kind"]
retd = close.pct_change()
valid_start = {}
for t in close.columns:
    nz = retd[t].ne(0) & retd[t].notna()
    valid_start[t] = nz.index[nz.argmax()] if nz.any() else close.index[-1]
BASE = [t for t in close.columns if kind[t] == "base"]

def month_grid(nth=None):
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

def tqqq_weight(mgrid, target=0.30, cap=1.0, volwin=63, fastwin=20, accel_k=2.0):
    def av(w):
        return (retd["TQQQ"].rolling(w, min_periods=int(w*0.7)).std()*np.sqrt(252)
                ).reindex(mgrid, method="ffill")
    vs = av(volwin)
    if accel_k:
        accel = (av(fastwin) / vs).clip(lower=1.0) ** accel_k
        vs = vs * accel
    return (target/vs).clip(0, cap).shift(1)

_S = {}
def prep(nth):
    if nth in _S: return _S[nth]
    mg = month_grid(nth)
    px = close.reindex(mg, method="ffill")
    ma200 = close.rolling(200, min_periods=120).mean().reindex(mg, method="ffill")
    ma50 = close.rolling(50, min_periods=30).mean().reindex(mg, method="ffill")
    up = (px > ma200)
    below50 = (ma50 / px - 1.0)
    # tech-complex trend gate, from PRIOR month-end (shift 1)
    gate = (close["QQQ"] > close["QQQ"].rolling(200, min_periods=120).mean()
            ).reindex(mg, method="ffill").shift(1).fillna(False)
    w = tqqq_weight(mg)
    _S[nth] = dict(mg=mg, px=px, up=up, below50=below50, gate=gate, w=w, ma200=ma200)
    return _S[nth]

def strat(start, end, nth=None, target=0.30, K=3, contrib=1000.0, cost=0.001,
          defense=("GLD", "TLT")):
    S = prep(nth); mg = S["mg"]; px = S["px"]
    grid = mg[(mg >= start) & (mg <= end)]
    pos = {}; contributed = 0.0; rows = []; log = []
    for dt in grid:
        cash = contrib
        for t in list(pos.keys()):
            if t == "_CASH_": cash += pos.pop(t)
            elif np.isfinite(px.loc[dt, t]): cash += pos.pop(t)*px.loc[dt, t]*(1-cost)
            else: pos.pop(t)
        contributed += contrib
        risk_on = bool(S["gate"].loc[dt])
        if risk_on:
            # vol-targeted leveraged NASDAQ; remainder to defensive blend (or cash pre-2005)
            w = float(S["w"].loc[dt]) if np.isfinite(S["w"].loc[dt]) else 0.0
            w = min(max(w, 0.0), 1.0)
            buy = cash * w
            if buy > 0 and np.isfinite(px.loc[dt, "TQQQ"]):
                pos["TQQQ"] = buy*(1-cost)/px.loc[dt, "TQQQ"]
            rem = cash - buy
            dtk = [t for t in defense if t in px.columns and dt >= valid_start[t]
                   and np.isfinite(px.loc[dt, t])]
            if rem > 0 and dtk:
                for t in dtk: pos[t] = pos.get(t, 0) + (rem/len(dtk))*(1-cost)/px.loc[dt, t]
            elif rem > 0:
                pos["_CASH_"] = pos.get("_CASH_", 0) + rem
            log.append((dt, "ON", tuple(["TQQQ"]) if buy > 0 else ("CASH",), round(w, 2)))
        else:
            # RISK-OFF: diversified oversold-in-uptrend sleeve; cash if nothing qualifies
            elig = {}
            for t in BASE:
                if not (dt >= valid_start[t] and np.isfinite(px.loc[dt, t])): continue
                if not (np.isfinite(S["ma200"].loc[dt, t]) and bool(S["up"].loc[dt, t])): continue
                v = S["below50"].loc[dt, t]
                if np.isfinite(v): elig[t] = v
            picks = sorted(elig, key=lambda t: -elig[t])[:K]
            if picks:
                per = cash/len(picks)
                for t in picks: pos[t] = per*(1-cost)/px.loc[dt, t]
            else:
                pos["_CASH_"] = pos.get("_CASH_", 0) + cash
            log.append((dt, "OFF", tuple(picks) if picks else ("CASH",), 0.0))
        V = sum((sh if t == "_CASH_" else sh*px.loc[dt, t])
                for t, sh in pos.items() if t == "_CASH_" or np.isfinite(px.loc[dt, t]))
        rows.append((dt, V, contributed))
    eq = pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")
    return eq, pd.DataFrame(log, columns=["date", "regime", "held", "w"]).set_index("date")

def qqq_dca(start, end, nth=None, contrib=1000.0):
    g = month_grid(nth); g = g[(g >= start) & (g <= end)]
    r = close.reindex(month_grid(nth))["QQQ"].pct_change().reindex(g).fillna(0)
    V = 0; c = 0; rows = []
    for dt, x in r.items(): V = V*(1+x)+contrib; c += contrib; rows.append((dt, V, c))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

def maxdd(eq):
    V = eq["V"]; C = eq["contributed"].diff().fillna(eq["contributed"])
    r = ((V-C)/V.shift(1)-1).dropna(); cum = (1+r).cumprod()
    return (cum/cum.cummax()-1).min()

if __name__ == "__main__":
    ERAS = [("1999-03","2003-12"),("2000-01","2010-12"),("2006-01","2009-12"),
            ("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),
            ("2006-01","2026-06")]
    print("REGIME-GATED (leveraged-tech ON / diversified-oversold OFF) vs QQQ-DCA\n")
    print(f"{'era':16}{'ratio':>8}{'strat maxDD':>13}{'qqq maxDD':>11}")
    for st, en in ERAS:
        s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
        eq, _ = strat(s, e); b = qqq_dca(s, e)
        print(f"{st+'..'+en[:7]:16}{eq['V'].iloc[-1]/b['V'].iloc[-1]:>8.2f}"
              f"{maxdd(eq):>13.0%}{maxdd(b):>11.0%}")
    print("\nPHASE ROBUSTNESS (full 1999-2026 continuous, by rebalance day):")
    for nth in [None, 4, 9, 14]:
        s, e = pd.Timestamp("1999-03-01"), pd.Timestamp("2026-07-01")
        eq, _ = strat(s, e, nth=nth); b = qqq_dca(s, e, nth=nth)
        print(f"  {'ME' if nth is None else 'day'+str(nth):5}: "
              f"{eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x  maxDD {maxdd(eq):.0%}")
    s, e = pd.Timestamp("1999-03-01"), pd.Timestamp("2026-07-01")
    eq, lg = strat(s, e)
    print("\nregime switches (count):", (lg['regime'] != lg['regime'].shift()).sum())
    print("recent:", lg.tail(4).to_dict("records"))
