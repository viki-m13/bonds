"""LOOP iter 1 — TWIN-MOMENTUM (Novy-Marx): price momentum AND fundamental
momentum must AGREE. Candidate only when a name is in the TOP tercile of BOTH
price momentum (rank of mom12/mom6) AND fundamental momentum (rank of rev_yoy
improving over trailing ~4q + rank of roa rising vs prior year). Rank within by
the average of the two legs. Contrast against each leg ALONE to prove agreement
adds value. Fundamentals begin ~2011 => pre-2011 months degrade to PRICE-ONLY
(fund leg unavailable); stated in output.
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

# ---- standard eligibility & ranking pond ----
elig = (liq & (me >= 3.0) & (dv >= 2e6) & (me > ma10))
pond = (liq & (me >= 3.0) & (dv >= 2e6))          # universe for cross-sectional ranks

# ---- PRICE momentum leg: rank of mom12 & mom6 (higher = stronger) ----
mom12 = FEAT["mom12"]; mom6 = FEAT["mom6"]
r_mom12 = mom12.where(pond).rank(axis=1, pct=True)
r_mom6  = mom6.where(pond).rank(axis=1, pct=True)
price_mom = (r_mom12 + r_mom6) / 2.0               # price-momentum composite rank

# ---- FUNDAMENTAL momentum leg: rev_yoy improving + roa rising (vs a year ago) ----
rev_yoy = FEAT["rev_yoy"]; roa = FEAT["roa"]
revImp = rev_yoy - rev_yoy.shift(12)               # rev_yoy improving over trailing ~4q
roaImp = roa - roa.shift(12)                        # roa rising vs prior year
r_revImp = revImp.where(pond).rank(axis=1, pct=True)
r_roaImp = roaImp.where(pond).rank(axis=1, pct=True)
fund_cov = (r_revImp.notna() & r_roaImp.notna())    # both fund components present
fund_mom = pd.concat([r_revImp, r_roaImp]).groupby(level=0).mean()
fund_mom = (r_revImp + r_roaImp) / 2.0             # fundamental-momentum composite rank

# months where fundamentals are essentially available (>=30 names with both legs)
fund_avail = fund_cov.sum(axis=1) >= 30
p(f"fundamentals available from {M[fund_avail].min().date()} "
  f"({int(fund_avail.sum())} of {len(M)} months)")

TOP = 2.0 / 3.0   # top tercile threshold on pct ranks

# ---- build the three scores (NaN = not a candidate) ----
# TWIN: top tercile of BOTH legs; rank within by average. On fund-missing months
# it degrades to price-only (honest: pre-2011 = price-only).
avg = (price_mom + fund_mom) / 2.0
twin = pd.DataFrame(np.nan, index=M, columns=cols)
priceonly = pd.DataFrame(np.nan, index=M, columns=cols)
fundonly = pd.DataFrame(np.nan, index=M, columns=cols)
for dt in M:
    pr = price_mom.loc[dt]
    fu = fund_mom.loc[dt]
    hasfund = fund_avail.loc[dt]
    # price-only leg (always available)
    pmask = pr >= TOP
    priceonly.loc[dt] = pr.where(pmask)
    # fund-only leg (only when fundamentals present)
    if hasfund:
        fmask = fu >= TOP
        fundonly.loc[dt] = fu.where(fmask)
        both = (pr >= TOP) & (fu >= TOP)
        twin.loc[dt] = avg.loc[dt].where(both)
    else:
        # degrade to price-only
        twin.loc[dt] = pr.where(pmask)

pd.to_pickle({"twin": twin.astype(np.float32), "price_mom": price_mom.astype(np.float32),
              "fund_mom": fund_mom.astype(np.float32), "priceonly": priceonly.astype(np.float32),
              "fundonly": fundonly.astype(np.float32),
              "note": "TWIN=top tercile of BOTH price_mom & fund_mom, ranked by avg; "
                      "pre-2011 (fund unavailable)=price-only"},
             f"{HERE}/loop1_twinmom.pkl")
p(f"saved loop1_twinmom.pkl  t={time.time()-t0:.0f}s")

# ---- era backtest harness ----
ERAS = [("2000-02", "2000-01", "2002-12"), ("2003-09", "2003-01", "2009-12"),
        ("2010-14", "2010-01", "2014-12"), ("2015-19", "2015-01", "2019-12"),
        ("2020-26", "2020-01", "2026-06"), ("2015-26", "2015-01", "2026-06"),
        ("2000-26", "2000-01", "2026-06")]

def run_score(score, st, en, N):
    dates = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
    res = dca_run(me, score, elig, dates, N=N, trail=-0.30, ma=ma10, minhold_days=30)
    b = dca_benchmark(bench["QQQ"], dates)
    s = stats(res["equity"])
    ratio = s["final"] / b["V"].iloc[-1]
    return ratio, s, dates

# random-in-pond null (score = uniform within eligible pond)
def null_ratios(st, en, N, seeds=5):
    dates = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
    b = dca_benchmark(bench["QQQ"], dates)
    rr = []
    for sd in range(seeds):
        rng = np.random.default_rng(1000 + sd)
        rnd = pd.DataFrame(rng.random(me.shape), index=M, columns=cols).where(pond)
        res = dca_run(me, rnd, elig, dates, N=N, trail=-0.30, ma=ma10, minhold_days=30)
        rr.append(stats(res["equity"])["final"] / b["V"].iloc[-1])
    return np.mean(rr), np.max(rr)

for N in [10, 20]:
    p(f"\n================  N={N}  ================")
    p(f"{'era':8} {'TWIN':>6} {'price':>6} {'fund':>6} | {'IRR':>6} {'Sh':>5} {'DD':>6} | {'null_mu':>7} {'null_mx':>7}")
    for lab, st, en in ERAS:
        rt, st_t, _ = run_score(twin, st, en, N)
        rp, _, _ = run_score(priceonly, st, en, N)
        # fund-only: only meaningful when fundamentals exist in the era
        if (fund_avail.reindex(M[(M >= pd.Timestamp(st+"-01")) & (M <= pd.Timestamp(en+"-01"))]).mean()) > 0.3:
            rf, _, _ = run_score(fundonly, st, en, N)
            rf_s = f"{rf:6.2f}"
        else:
            rf_s = "   n/a"
        nmu, nmx = null_ratios(st, en, N)
        p(f"{lab:8} {rt:6.2f} {rp:6.2f} {rf_s} | {st_t['irr']:6.1%} {st_t['sharpe']:5.2f} "
          f"{st_t['maxdd']:6.1%} | {nmu:7.2f} {nmx:7.2f}")

p(f"\nDONE t={time.time()-t0:.0f}s")
