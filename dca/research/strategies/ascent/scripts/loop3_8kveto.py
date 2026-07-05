"""LOOP iter 3 — 8-K NEGATIVE-EVENT veto as a strategy overlay on a momentum book.

Loop7 found 8-K item COUNTS null as a cross-sectional score. Untested: a targeted
NEG-event VETO (drop names filing bad 8-Ks) as an overlay. 8-K item data is a
NON-PRICE signal testable at the 2003-09 wall (item codes reliable from ~2005),
unlike SEC fundamentals (2012+).

NEG (trailing 3m): 4.01 auditor change, 4.02 restatement/non-reliance, 1.03 bankruptcy,
    3.01 delisting notice, 2.05 exit/disposal, 2.06 impairment.
POS (trailing 3m): 1.01 material agreement, 2.01 completed acquisition.

Tests (monthly $1k DCA, 20bps, min-30d hold; era-sliced ratio vs QQQ-DCA; 5-seed null):
 1. NEG veto on a mom12 book vs same book WITHOUT the veto -> delta per era.
 2. NEG veto + POS tilt on the mom book.
 3. Equal-weight ex-NEG across the WHOLE eligible pond vs QQQ.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark, stats

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()

# ---- eligibility (standard) ----
elig_base = liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10)
mom12 = (me / me.shift(12) - 1)

# ---- 8-K NEG/POS indicators (trailing 3m, point-in-time on filing date) ----
E = pd.read_parquet(f"{HERE}/_8k_items.parquet")
E["date"] = pd.to_datetime(E.date)
E["ym"] = E.date.values.astype("datetime64[M]")
E = E[E.tk.isin(set(cols))]

def indicator(codes):
    mask = E["items"].str.contains("|".join(c.replace(".", r"\.") for c in codes), regex=True)
    sub = E[mask]
    pv = sub.assign(v=1.0).pivot_table(index="ym", columns="tk", values="v", aggfunc="sum")
    pv = pv.reindex(index=M, columns=cols).fillna(0.0)
    return (pv.rolling(3, min_periods=1).sum() > 0)

NEG = indicator(["4.01", "4.02", "1.03", "3.01", "2.05", "2.06"])
POS = indicator(["1.01", "2.01"])
p(f"NEG cells True (avg per month, 2005+): {NEG.loc['2005':].sum(axis=1).mean():.0f} names/mo")
p(f"POS cells True (avg per month, 2005+): {POS.loc['2005':].sum(axis=1).mean():.0f} names/mo")
# how many eligible names get vetoed on average
_el05 = elig_base.loc["2005":]
_vet = (_el05 & NEG.reindex_like(_el05)).sum(axis=1)
p(f"eligible names vetoed by NEG (avg/mo, 2005+): {_vet.mean():.1f} of {_el05.sum(axis=1).mean():.0f} eligible")

# ---- scoring panels ----
score_mom = mom12.where(elig_base).rank(axis=1, pct=True)

ERAS = [("2005-01", "2009-12"), ("2010-01", "2014-12"),
        ("2015-01", "2019-12"), ("2020-01", "2026-06")]

def run_ratio(score, elig, st, en, N=12, seed_shuffle=None):
    dates = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
    if seed_shuffle is not None:
        rng = np.random.default_rng(seed_shuffle)
        score = pd.DataFrame(rng.random(elig.shape), index=elig.index, columns=elig.columns).where(elig)
    r = dca_run(me, score, elig, dates, N=N)
    b = dca_benchmark(bench["QQQ"], dates)
    s = stats(r["equity"])
    return s["final"] / b["V"].iloc[-1], s

def null_band(elig, st, en, N=12, seeds=5):
    rs = []
    for sd in range(seeds):
        r, _ = run_ratio(None, elig, st, en, N=N, seed_shuffle=sd)
        rs.append(r)
    return np.mean(rs), np.max(rs)

# =====================================================================
p("\n=== TEST 1: NEG veto on mom12 book vs no-veto ===")
p(f"{'era':10} {'noveto':>7} {'veto':>7} {'delta':>7} | {'null_mn':>7} {'null_mx':>7} "
  f"| {'v_IRR':>6} {'v_Sh':>5} {'v_DD':>6}")
elig_veto = elig_base & ~NEG.reindex_like(elig_base).fillna(False)
T1 = {}
for st, en in ERAS:
    r_nov, s_nov = run_ratio(score_mom, elig_base, st, en)
    # re-rank momentum within the ex-NEG pond so vetoed names don't hold rank mass
    score_mom_v = mom12.where(elig_veto).rank(axis=1, pct=True)
    r_vet, s_vet = run_ratio(score_mom_v, elig_veto, st, en)
    nm, nx = null_band(elig_base, st, en)
    T1[(st, en)] = (r_nov, r_vet, nm, nx)
    p(f"{st[:7]:10} {r_nov:>7.2f} {r_vet:>7.2f} {r_vet-r_nov:>+7.2f} | {nm:>7.2f} {nx:>7.2f} "
      f"| {s_vet['irr']*100:>5.1f}% {s_vet['sharpe']:>5.2f} {s_vet['maxdd']*100:>5.0f}%")

# =====================================================================
p("\n=== TEST 2: NEG veto + POS tilt on mom book ===")
p(f"{'era':10} {'noveto':>7} {'veto+pos':>8} {'delta':>7} | {'null_mn':>7} {'null_mx':>7}")
# POS tilt: add a rank bonus to names with a POS event (still within ex-NEG pond)
posbonus = POS.reindex_like(elig_base).fillna(False).astype(float) * 0.15
score_vp = (mom12.where(elig_veto).rank(axis=1, pct=True) + posbonus).where(elig_veto)
for st, en in ERAS:
    r_nov = T1[(st, en)][0]
    r_vp, s_vp = run_ratio(score_vp, elig_veto, st, en)
    nm, nx = T1[(st, en)][2], T1[(st, en)][3]
    p(f"{st[:7]:10} {r_nov:>7.2f} {r_vp:>8.2f} {r_vp-r_nov:>+7.2f} | {nm:>7.2f} {nx:>7.2f}")

# =====================================================================
p("\n=== TEST 3: equal-weight ex-NEG across WHOLE pond vs QQQ (standalone) ===")
p(f"{'era':10} {'ewpond':>7} {'ew_exNEG':>8} {'delta':>7} | {'null_mn':>7} {'null_mx':>7}")
# equal-weight pond = flat score (all eligible equally); N large to hold a broad basket
Nbroad = 50
flat = pd.DataFrame(1.0, index=me.index, columns=me.columns)
for st, en in ERAS:
    r_all, _ = run_ratio(flat.where(elig_base), elig_base, st, en, N=Nbroad)
    r_exn, s_exn = run_ratio(flat.where(elig_veto), elig_veto, st, en, N=Nbroad)
    nm, nx = null_band(elig_base, st, en, N=Nbroad)
    p(f"{st[:7]:10} {r_all:>7.2f} {r_exn:>8.2f} {r_exn-r_all:>+7.2f} | {nm:>7.2f} {nx:>7.2f}")

# ---- save indicators ----
pd.to_pickle({"NEG": NEG, "POS": POS, "note": "trailing-3m 8-K NEG/POS event indicators, "
             "keyed on filing date, aligned to me panel"}, f"{HERE}/loop3_8kveto.pkl")
p(f"\nsaved loop3_8kveto.pkl  DONE t={time.time()-t0:.0f}s")
