"""TACTICAL QQQ-CORE (unleveraged) — the untested family: keep QQQ as the strategic
core (never structurally leave the compounder) and deviate only tactically.

Levers tested (all unleveraged, monthly, DCA $1000, 10bps/side, no look-ahead —
signals computed from data through the grid date, weight applied to the NEXT month):

  A. SIBLING SWITCH: hold SMH (semis) instead of QQQ when SMH's relative strength
     vs QQQ is positive (blended 3/6/12m relative momentum), else QQQ. Never leaves
     tech beta, so it dodges the structural wall that killed cross-asset rotation.
     Pre-2005 (no SMH data) it just holds QQQ.
  B. CREDIT GATE on the core: when HY OAS > its 252d MA (stress), shift the core
     to a GLD/TLT blend (cash pre-2005). The one phase-robust defensive signal found.
  C. TREND GATE (Faber 10m MA on QQQ) — the classic; tested for completeness.
  D. CONTRIBUTION ROUTER: never sell; route each month's NEW $1000 to the best of
     {QQQ, SMH, GLD, TLT} by 6m absolute momentum (QQQ if tie/none). Tax-realistic.
  Combos: A+B (sibling switch with credit-gated defense).

Gauntlet: eras incl. dot-com, continuous full-period, phase-robustness (ME/d4/d9/d14).
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl")
close = P["close"].sort_index()
retd = close.pct_change()
valid_start = {}
for t in close.columns:
    nz = retd[t].ne(0) & retd[t].notna()
    valid_start[t] = nz.index[nz.argmax()] if nz.any() else close.index[-1]

# HY OAS (credit stress), from 2000
_hy = pd.read_csv(f"{REPO}/data/fred/BAMLH0A0HYM2.csv")
_hy["Date"] = pd.to_datetime(_hy["Date"])
HY = _hy.set_index("Date").iloc[:, 0].replace(".", np.nan).astype(float).dropna()
HY_MA = HY.rolling(252, min_periods=150).mean()

def month_grid(nth=None):
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

def signals(mg):
    px = close.reindex(mg, method="ffill")
    # relative momentum SMH vs QQQ: blended 3/6/12m of the ratio
    rel = px["SMH"] / px["QQQ"]
    relmom = sum((rel/rel.shift(k)-1) for k in (3, 6, 12)) / 3.0
    # absolute 6m momentum for the router
    mom6 = {t: px[t]/px[t].shift(6)-1 for t in ("QQQ", "SMH", "GLD", "TLT")}
    # credit gate: stress if HY OAS above its 252d MA
    stress = (HY > HY_MA).reindex(mg, method="ffill")
    # Faber: QQQ above 10-month MA (monthly closes)
    faber = px["QQQ"] > px["QQQ"].rolling(10, min_periods=8).mean()
    return dict(px=px, relmom=relmom, mom6=mom6, stress=stress, faber=faber)

_C = {}
def prep(nth):
    if nth not in _C:
        mg = month_grid(nth)
        _C[nth] = (mg, signals(mg), close.reindex(mg).pct_change())
    return _C[nth]

def _def_ret(S, mret, dt):
    """defensive blend return for month ending dt: GLD/TLT if live, else 0 (cash)."""
    live = [t for t in ("GLD", "TLT") if dt >= valid_start[t]]
    if not live: return 0.0
    v = np.nanmean([mret.loc[dt, t] for t in live])
    return v if np.isfinite(v) else 0.0

def strat_returns(start, end, mode, nth=None, cost=0.001):
    """return monthly return series of the tactical strategy (fully-invested switch
    modes). Holding decided at month-end dt-1 from data through dt-1, earns dt's return."""
    mg, S, mret = prep(nth)
    g = mg[(mg >= start) & (mg <= end)]
    rr = []; prev = None
    for i, dt in enumerate(g):
        loc = mg.get_loc(dt)
        if loc == 0: rr.append((dt, 0.0)); continue
        d0 = mg[loc-1]                      # decision date = prior month-end
        # decide holding
        smh_ok = d0 >= valid_start["SMH"]
        rm = S["relmom"].get(d0, np.nan)
        stress = bool(S["stress"].get(d0, False))
        fab = bool(S["faber"].get(d0, True))
        if mode == "qqq":         hold = "QQQ"
        elif mode == "sibling":   hold = "SMH" if (smh_ok and np.isfinite(rm) and rm > 0) else "QQQ"
        elif mode == "credit":    hold = "DEF" if stress else "QQQ"
        elif mode == "faber":     hold = "QQQ" if fab else "DEF"
        elif mode == "sib+credit":
            hold = "DEF" if stress else ("SMH" if (smh_ok and np.isfinite(rm) and rm > 0) else "QQQ")
        else: raise ValueError(mode)
        if hold == "DEF": r = _def_ret(S, mret, dt)
        else:
            r = mret.loc[dt, hold]; r = r if np.isfinite(r) else 0.0
        if prev is not None and hold != prev: r -= 2*cost   # full switch cost
        rr.append((dt, r)); prev = hold
    return pd.Series(dict(rr))

def router_curve(start, end, nth=None, cost=0.001, contrib=1000.0):
    """no-sell DCA router: each month's contribution buys the best 6m-momentum asset
    (QQQ default). Existing positions are never sold."""
    mg, S, mret = prep(nth)
    g = mg[(mg >= start) & (mg <= end)]
    px = S["px"]
    pos = {}; contributed = 0.0; rows = []
    for dt in g:
        loc = mg.get_loc(dt); d0 = mg[loc-1] if loc > 0 else None
        pick = "QQQ"
        if d0 is not None:
            best = -np.inf
            for t, m in S["mom6"].items():
                if d0 < valid_start[t]: continue
                v = m.get(d0, np.nan)
                if np.isfinite(v) and v > best: best, pick = v, t
            if best <= 0: pick = "QQQ" if S["mom6"]["QQQ"].get(d0, 0) == best else "TLT" if best > -np.inf else "QQQ"
            if best <= 0: pick = "QQQ"   # nothing trending: default to core
        if np.isfinite(px.loc[dt, pick]):
            pos[pick] = pos.get(pick, 0) + contrib*(1-cost)/px.loc[dt, pick]
        contributed += contrib
        V = sum(sh*px.loc[dt, t] for t, sh in pos.items() if np.isfinite(px.loc[dt, t]))
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

def dca(r, contrib=1000.0):
    V = c = 0.0; rows = []
    for dt, x in r.items():
        V = V*(1+x)+contrib; c += contrib; rows.append((dt, V, c))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

def maxdd_r(r):
    cum = (1+r.fillna(0)).cumprod()
    return (cum/cum.cummax()-1).min()

ERAS = [("1999-03","2003-12"),("2000-01","2010-12"),("2006-01","2009-12"),
        ("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),
        ("2006-01","2026-06"),("1999-03","2026-06")]

if __name__ == "__main__":
    print("DCA final-wealth RATIO vs QQQ-DCA (unleveraged tactical, month-end)\n")
    print(f"{'mode':12}" + "".join(f"{a[:7]:>8}" for a, _ in ERAS) + f"{'mDD-full':>9}")
    for mode in ("sibling", "credit", "faber", "sib+credit"):
        out = []
        for st, en in ERAS:
            s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
            rs = strat_returns(s, e, mode); rq = strat_returns(s, e, "qqq")
            out.append(dca(rs)["V"].iloc[-1]/dca(rq)["V"].iloc[-1])
        s, e = pd.Timestamp("1999-03-01"), pd.Timestamp("2026-07-01")
        dd = maxdd_r(strat_returns(s, e, mode))
        print(f"{mode:12}" + "".join(f"{v:>8.2f}" for v in out) + f"{dd:>9.0%}")
    # router
    out = []
    for st, en in ERAS:
        s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
        eq = router_curve(s, e); rq = strat_returns(s, e, "qqq")
        out.append(eq["V"].iloc[-1]/dca(rq)["V"].iloc[-1])
    print(f"{'router':12}" + "".join(f"{v:>8.2f}" for v in out))

    print("\nPHASE ROBUSTNESS (ratio vs QQQ-DCA, full 1999-2026 / 2006-2026):")
    for mode in ("sibling", "sib+credit", "credit"):
        for span, (st, en) in [("99-26", ("1999-03", "2026-07")), ("06-26", ("2006-01", "2026-07"))]:
            cells = []
            for nth in [None, 4, 9, 14]:
                s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
                rs = strat_returns(s, e, mode, nth=nth); rq = strat_returns(s, e, "qqq", nth=nth)
                cells.append(f"{dca(rs)['V'].iloc[-1]/dca(rq)['V'].iloc[-1]:.2f}")
            print(f"  {mode:11} {span}:  ME={cells[0]}  d4={cells[1]}  d9={cells[2]}  d14={cells[3]}")
