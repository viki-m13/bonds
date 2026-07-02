"""LOOP iter 3 — adversarial mining of the LOSING decade: search ~40 causal
price-only signals for a top-5 basket that beat QQQ-DCA in BOTH 2003-09 and
2010-14 (mined in-sample ON those eras). Anything found is then judged on
2000-02 and 2015-26 as out-of-sample. If the mined set is empty or nothing
transfers, the all-era goal is infeasible by construction for this class.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats

t0 = time.time()
def p(*a): print(*a, flush=True)
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)
dvr = dv.rank(axis=1, ascending=False)
top1500 = dvr <= 1500
r = lambda df: df.where(liq & top1500).rank(axis=1, pct=True)

hi52 = me.rolling(12, min_periods=6).max()
lo52 = me.rolling(12, min_periods=6).min()
vol6 = me.pct_change().rolling(6, min_periods=4).std()
vol12 = me.pct_change().rolling(12, min_periods=8).std()
ma5 = me.rolling(5, min_periods=5).mean()
athigh = me.cummax()

SIGNALS = {
    "mom12":      r(me / me.shift(12) - 1),
    "mom6":       r(me / me.shift(6) - 1),
    "mom3":       r(me / me.shift(3) - 1),
    "mom12skip1": r(me.shift(1) / me.shift(13) - 1),
    "mom24":      r(me / me.shift(24) - 1),
    "mom36":      r(me / me.shift(36) - 1),
    "rev1":       r(-(me / me.shift(1) - 1)),
    "rev3":       r(-(me / me.shift(3) - 1)),
    "near52hi":   r(me / hi52),
    "off52hi":    r(-(me / hi52)),
    "range_pos":  r((me - lo52) / (hi52 - lo52)),
    "lowvol":     r(-vol6),
    "hivol":      r(vol6),
    "volcontr":   r(-(vol6 / vol12)),
    "dollarvol":  r(dv),
    "lowdollar":  r(-dv),
    "smoothtrend": r((me > ma5).rolling(12, min_periods=6).mean()),
    "dd_ath":     r(-(me / athigh)),          # deepest below all-time high
    "recov_ath":  r(me / athigh),
    "sharpe12":   r((me / me.shift(12) - 1) / (vol12 + 1e-9)),
    "gapup":      r((me / me.shift(1) - 1) - (me.shift(1) / me.shift(2) - 1)),
    "consist":    r((me.pct_change() > 0).rolling(12, min_periods=8).mean()),
    "accel":      r((me / me.shift(3) - 1) - (me / me.shift(12) - 1)),
    "decel":      r((me / me.shift(12) - 1) - (me / me.shift(3) - 1)),
    "pricehigh":  r(me),
    "pricelow":   r(-me),
    "mom6_lowvol": r(me / me.shift(6) - 1) + r(-vol6),
    "mom12_dv":    r(me / me.shift(12) - 1) + r(dv),
    "near52_lowvol": r(me / hi52) + r(-vol6),
    "consist_mom": r((me.pct_change() > 0).rolling(12, min_periods=8).mean()) + r(me / me.shift(12) - 1),
}

def run_static(pick, start, end, k=5, contrib=1000.0, cost=0.002, trail=-0.30, trend_gate=True):
    dates = M[(M >= start) & (M <= end)]
    elig_all = (liq & top1500 & (me >= 3.0) & ((me > ma10) if trend_gate else True))
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * (1 - cost); pos.pop(tk)
        srow = pick.loc[dt]; erow = elig_all.loc[dt]; marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            if (cp / e["peak_px"] - 1) <= trail or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk]):
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
        cand = cand[~cand.index.isin(pos)].sort_values(ascending=False)
        need = k - len(pos)
        if need > 0 and cash > 1e-9 and len(cand):
            if qqq_units > 0 and qqq_entry is not None and (dt - qqq_entry).days >= 30:
                cash += qqq_units * qp * (1 - 0.0005); qqq_units = 0.0; qqq_entry = None
            picks = list(cand.index[:need]); amt = cash / len(picks)
            for tk in picks:
                pos[tk] = {"val": amt * (1 - cost), "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
            cash = 0.0
        elif cash > 1e-9 and len(pos):
            hs = {tk: srow.get(tk, np.nan) for tk in pos}
            tp = sorted(hs, key=lambda t: -(hs[t] if np.isfinite(hs[t]) else -1))[:3]
            for tk in tp: pos[tk]["val"] += (cash / len(tp)) * (1 - cost)
            cash = 0.0
        if cash > 1e-9 and np.isfinite(qp):
            qqq_units += cash * (1 - 0.0005) / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0)
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"), dates

def ratio(pick, st, en, **kw):
    eq, dts = run_static(pick, pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"), **kw)
    b = dca_benchmark(bench["QQQ"], dts)
    return stats(eq)["final"] / b["V"].iloc[-1]

p(f"{'signal':16} {'03-09':>7} {'10-14':>7}   (mining eras — need BOTH > 1)")
mined = []
for nm, sc in SIGNALS.items():
    r1 = ratio(sc, "2003-01", "2009-12")
    r2 = ratio(sc, "2010-01", "2014-12")
    flag = " <== CANDIDATE" if (r1 > 1.0 and r2 > 1.0) else ""
    p(f"{nm:16} {r1:>7.2f} {r2:>7.2f}{flag}")
    if r1 > 1.0 and r2 > 1.0:
        mined.append(nm)
p(f"\nmined candidates (beat QQQ in BOTH bad eras): {mined or 'NONE'}")
for nm in mined:
    sc = SIGNALS[nm]
    p(f"  {nm}: OOS 2000-02 {ratio(sc, '2000-01', '2002-12'):.2f}  OOS 2015-19 "
      f"{ratio(sc, '2015-01', '2019-12'):.2f}  OOS 2020-26 {ratio(sc, '2020-01', '2026-06'):.2f}")
p(f"\nDONE t={time.time()-t0:.0f}s")
