"""LOOP iter 1 — pond rotation: pick the pond whose leaders beat QQQ, then buy
that pond's top stocks. Ponds are structurally distinct slices of the liquid
universe (price-only definitions so the full 2000-2026 history is testable):
  MEGA   $vol rank 1-100        (the QQQ engine)
  MID    $vol rank 300-800      (mid-tier liquidity)
  SMALL  $vol rank 800-2000
  LOWVOL lowest-vol quintile of top-1000 $vol
  BMOM   top-decile mom12 of top-1000 $vol
Meta-signal (causal): trailing 6m equal-weight return of each pond's CURRENT
top-5 basket (by $vol within pond, trend-gated) minus QQQ 6m. Hold the best
pond's top-5 if its spread > 0 else QQQ. Mandate mechanics: min-30d, cuts,
20bps. Era-sliced vs QQQ-DCA.
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
mom6 = me / me.shift(6) - 1
mom12 = me / me.shift(12) - 1
vol6 = FEAT["vol6"]
qmom6 = qqq / qqq.shift(6) - 1

dvr = dv.rank(axis=1, ascending=False)
top1000 = dvr <= 1000
PONDS = {
    "MEGA":   (dvr <= 100),
    "MID":    (dvr > 300) & (dvr <= 800),
    "SMALL":  (dvr > 800) & (dvr <= 2000),
    "LOWVOL": top1000 & (vol6.where(top1000).rank(axis=1, pct=True) <= 0.2),
    "BMOM":   top1000 & (mom12.where(top1000).rank(axis=1, pct=True) >= 0.9),
}
# in-pond pick score: $vol rank (leaders) — same convention as ASCENT
PICK = {nm: dv.rank(axis=1, pct=True).where(mask) for nm, mask in PONDS.items()}
ELIG = {nm: (liq & mask & (me >= 3.0) & (me > ma10)) for nm, mask in PONDS.items()}

# causal pond meta-signal: 6m EW return of the CURRENT top-5 (by pick score) per pond
SPREADS = {}
for nm in PONDS:
    r5 = PICK[nm].rank(axis=1, ascending=False) <= 5
    SPREADS[nm] = (mom6.where(r5).mean(axis=1) - qmom6).rolling(3, min_periods=1).mean()
SPR = pd.DataFrame(SPREADS)

def run_pond_rotation(start, end, k=5, contrib=1000.0, cost=0.002, trail=-0.30):
    dates = M[(M >= start) & (M <= end)]
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0
    rows = []; pond_hist = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * (1 - cost); pos.pop(tk)
        srow_all = SPR.loc[dt]
        best = srow_all.idxmax() if srow_all.notna().any() else None
        on = best is not None and srow_all[best] > 0
        pond_hist.append((dt, best if on else "QQQ"))
        score = PICK[best].loc[dt] if on else None
        elig = ELIG[best].loc[dt] if on else None
        marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            cut = (cp / e["peak_px"] - 1) <= trail or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk])
            # also rotate out if pond changed and name not in new pond's eligible set
            stale = on and (elig is not None) and (not bool(elig.get(tk, False)))
            if cut or (not on) or stale:
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        if on:
            if qqq_units > 0 and qqq_entry is not None and (dt - qqq_entry).days >= 30:
                cash += qqq_units * qp * (1 - 0.0005); qqq_units = 0.0; qqq_entry = None
            cand = score[elig.reindex(score.index).fillna(False).astype(bool)].dropna()
            cand = cand[~cand.index.isin(pos)].sort_values(ascending=False)
            need = k - len(pos)
            if need > 0 and cash > 1e-9 and len(cand):
                picks = list(cand.index[:need]); amt = cash / len(picks)
                for tk in picks:
                    pos[tk] = {"val": amt * (1 - cost), "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
                cash = 0.0
            elif cash > 1e-9 and len(pos):
                hs = {tk: (score.get(tk, np.nan) if score is not None else np.nan) for tk in pos}
                tp = sorted(hs, key=lambda t: -(hs[t] if np.isfinite(hs[t]) else -1))[:3]
                for tk in tp: pos[tk]["val"] += (cash / len(tp)) * (1 - cost)
                cash = 0.0
        if cash > 1e-9 and np.isfinite(qp):
            qqq_units += cash * (1 - 0.0005) / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0)
        rows.append((dt, V, contributed))
    return (pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"),
            dates, pd.DataFrame(pond_hist, columns=["date", "pond"]).set_index("date"))

ERAS = [("2000-01", "2002-12"), ("2003-01", "2009-12"), ("2010-01", "2014-12"),
        ("2015-01", "2019-12"), ("2020-01", "2026-06"), ("2000-01", "2026-06"),
        ("2015-01", "2026-06")]
p("=== LOOP1: pond rotation (top-5 of winning pond, QQQ fallback) ===")
for st, en in ERAS:
    stt, enn = pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01")
    eq, dts, ph = run_pond_rotation(stt, enn)
    b = dca_benchmark(bench["QQQ"], dts); s = stats(eq)
    mix = ph.pond.value_counts(normalize=True).round(2).to_dict()
    p(f"  {st}..{en}: vsQQQ {s['final']/b['V'].iloc[-1]:5.2f}x IRR {s['irr']:+6.1%} "
      f"Sh {s['sharpe']:4.2f} DD {s['maxdd']:6.1%}  mix {mix}")

# in-pond pick score variant: momentum instead of $vol
p("\n--- variant: in-pond pick = mom6 rank ---")
PICK = {nm: mom6.where(liq).rank(axis=1, pct=True).where(mask) for nm, mask in PONDS.items()}
SPREADS = {}
for nm in PONDS:
    r5 = PICK[nm].rank(axis=1, ascending=False) <= 5
    SPREADS[nm] = (mom6.where(r5).mean(axis=1) - qmom6).rolling(3, min_periods=1).mean()
SPR = pd.DataFrame(SPREADS)
for st, en in ERAS:
    stt, enn = pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01")
    eq, dts, ph = run_pond_rotation(stt, enn)
    b = dca_benchmark(bench["QQQ"], dts); s = stats(eq)
    p(f"  {st}..{en}: vsQQQ {s['final']/b['V'].iloc[-1]:5.2f}x IRR {s['irr']:+6.1%} DD {s['maxdd']:6.1%}")
p(f"\nDONE t={time.time()-t0:.0f}s")
