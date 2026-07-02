"""ASCENT novel variants — ideas NOT tested in the prior program, adapted to the
sell-allowed / min-30d-hold mandate:
  1. QQQ-fallback: idle cash and low-conviction periods park in QQQ (not cash);
     stocks bought only when conviction is high. Floor = the benchmark itself.
  2. Conviction-threshold gating: buy stocks only when their ML prob clears an
     absolute bar (not just relative rank).
  3. Conviction-weighted sizing.
  4. Loser-recycling: rank-exit proceeds redeploy next period (already in engine).
Run AFTER run_candidates.py (uses same featmat/mlprob).
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats, twr

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
ma10 = me.rolling(10, min_periods=10).mean()
mom3 = me / me.shift(3) - 1
ELIG = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10) & (mom3 > 0))
accel = PROB - PROB.shift(2)
SCORE = PROB.where(accel > 0)

START, END = pd.Timestamp("2015-01-01"), pd.Timestamp("2026-06-01")
dates = M[(M >= START) & (M <= END)]
qqq = bench["QQQ"]
qqq_ma10 = qqq.rolling(10, min_periods=10).mean()
bq = dca_benchmark(qqq, dates)


def run_hybrid(score, elig, N=12, trail=-0.30, minhold_days=30, cost=0.0020,
               delist_ret=-0.25, conv_thresh=None, conv_weight=False,
               park="qqq", contrib=1000.0, ma=ma10):
    """DCA engine with QQQ as the parking asset for idle cash."""
    pos = {}; qqq_units = 0.0; qqq_entry = None
    cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.get(dt, np.nan)
        # mark/delist
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp
                e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= (1 + delist_ret); cash += e["val"] * (1 - cost); pos.pop(tk)
        srow = score.loc[dt]; erow = elig.loc[dt]; marow = ma.loc[dt] if ma is not None else None
        # exits
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < minhold_days: continue
            cp = e["last_px"]
            if (cp / e["peak_px"] - 1) <= trail or (marow is not None and np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk]):
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        # candidates
        cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
        if conv_thresh is not None and len(cand):
            # absolute conviction bar: prob percentile among all liquid names
            pct = srow.rank(pct=True)
            cand = cand[pct.reindex(cand.index) >= conv_thresh]
        cand = cand[~cand.index.isin(pos.keys())].sort_values(ascending=False)
        need = N - len(pos)
        if need > 0 and cash > 1e-9 and len(cand):
            picks = list(cand.index[:need])
            if conv_weight:
                w = np.array([max(cand[t], 1e-6) for t in picks]); w = w / w.sum()
            else:
                w = np.ones(len(picks)) / len(picks)
            for tk, wi in zip(picks, w):
                v = cash * wi * (1 - cost)
                pos[tk] = {"val": v, "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
            cash = 0.0
        # park leftover in QQQ (respecting min-hold on QQQ too)
        if cash > 1e-9 and park == "qqq" and np.isfinite(qp):
            qqq_units += cash * (1 - 0.0005) / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        # if slots open and we hold QQQ >= minhold, sell QQQ to fund stocks next period
        if park == "qqq" and qqq_units > 0 and len(pos) < N and qqq_entry is not None \
           and (dt - qqq_entry).days >= minhold_days and len(cand) > len(pos):
            cash += qqq_units * qp * (1 - 0.0005); qqq_units = 0.0; qqq_entry = None
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0.0)
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")


def show(nm, eq):
    s = stats(eq)
    ratio = s["final"] / bq["V"].iloc[-1]
    p(f"{nm:44} MOIC {s['moic']:5.2f}x IRR {s['irr']:6.1%} TWR {s['twr_cagr']:6.1%} "
      f"Sh {s['sharpe']:5.2f} DD {s['maxdd']:6.1%} vsQQQ {ratio:5.2f}x")
    r = twr(eq); qr = twr(bq)
    for lo, hi, tag in [("2015-01", "2021-12", "dev "), ("2022-01", "2026-06", "hold")]:
        rr = r[(r.index >= lo) & (r.index <= hi)]; qq = qr[(qr.index >= lo) & (qr.index <= hi)]
        c = (1 + rr).prod() ** (12 / len(rr)) - 1; qc = (1 + qq).prod() ** (12 / len(qq)) - 1
        p(f"    {tag}: {c:+6.1%} (QQQ {qc:+6.1%})")

sq = stats(bq)
p(f"QQQ-DCA   MOIC {sq['moic']:.2f}x IRR {sq['irr']:.1%} Sh {sq['sharpe']:.2f} DD {sq['maxdd']:.1%}\n")
show("HYBRID park-QQQ N12", run_hybrid(SCORE, ELIG))
show("HYBRID park-QQQ N12 conv>=0.95", run_hybrid(SCORE, ELIG, conv_thresh=0.95))
show("HYBRID park-QQQ N8 conv>=0.98", run_hybrid(SCORE, ELIG, N=8, conv_thresh=0.98))
show("HYBRID park-QQQ N12 conv-weighted", run_hybrid(SCORE, ELIG, conv_weight=True))
show("HYBRID park-QQQ N6 conv>=0.98 convw", run_hybrid(SCORE, ELIG, N=6, conv_thresh=0.98, conv_weight=True))
p(f"\nDONE t={time.time()-t0:.0f}s")
