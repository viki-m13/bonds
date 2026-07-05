"""LOOP iter 1 (expectancy) — maximize PAYOFF ASYMMETRY, not hit-rate.
Thesis: only 32-46% of names beat QQQ at 12m (structural wall). So instead of
chasing accuracy, select names with the fattest right tail (big-winner
potential) and cut losers fast via an ASYMMETRIC exit (trail -25% + trend, no
upper cap = winners ride). Three scores:
  (a) upside-convexity = rank(mom6)+rank(vol6)+rank(distHigh) gated roa>med & uptrend
  (b) coiled quality    = rank(roa)+rank(rev_yoy)+rank(-share_chg)+rank(distHigh)
  (c) ML _mlprob
Run each through dca_run with trail=-0.25, trend exit, minhold 30d, N in {8,12,20}.
Quantify win-rate / avg-win / avg-loss / top-5% share of return. Era-sliced
ratios vs QQQ-DCA with a 5-seed random-in-pond null.
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
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)

# standard eligibility
elig = (liq & (me >= 3) & (dv >= 2e6) & (me > ma10)).fillna(False)

def rk(df):  # cross-sectional pct rank within eligible pond
    return df.where(elig).rank(axis=1, pct=True)

# ---- score (a) upside-convexity, gated to quality & uptrend ----
roa = FEAT["roa"]
roa_med = roa.where(elig).median(axis=1)
quality = roa.ge(roa_med, axis=0)                 # roa > cross-sectional median
convex = rk(FEAT["mom6"]) + rk(FEAT["vol6"]) + rk(FEAT["distHigh"])
SC_convex = convex.where(elig & quality & (me > ma10))
# pure-technical convex (NO roa gate) -> testable in all eras incl pre-2012
# (SEC fundamentals roa/rev_yoy/share_chg are all-NaN before ~2012, so any
#  fundamental-gated score cannot trade in 2000-02 / 2003-09 by construction)
SC_convex_tech = convex.where(elig)

# ---- score (b) coiled quality ----
coiled = rk(roa) + rk(FEAT["rev_yoy"]) + rk(-FEAT["share_chg"]) + rk(FEAT["distHigh"])
SC_coiled = coiled.where(elig)

# ---- score (c) ML ----
SC_ml = PROB.where(elig)

SCORES = {"convex_tech": SC_convex_tech, "convex_qual": SC_convex, "coiled": SC_coiled, "ml": SC_ml}

ERAS = [("2000-02", "2000-01-01", "2002-12-01"),
        ("2003-09", "2003-01-01", "2009-12-01"),
        ("2010-14", "2010-01-01", "2014-12-01"),
        ("2015-19", "2015-01-01", "2019-12-01"),
        ("2020-26", "2020-01-01", "2026-06-01"),
        ("2000-26", "2000-01-01", "2026-06-01")]

def era_dates(lo, hi):
    return M[(M >= pd.Timestamp(lo)) & (M <= pd.Timestamp(hi))]

def trade_profile(tr):
    """win-rate, avg-win, avg-loss, top-5% share of total gross profit."""
    s = tr[tr.why != "buy"].dropna(subset=["ret"]).copy()
    if len(s) < 5:
        return dict(n=len(s), win=np.nan, aw=np.nan, al=np.nan, t5=np.nan)
    r = s["ret"].values
    win = (r > 0).mean()
    aw = r[r > 0].mean() if (r > 0).any() else np.nan
    al = r[r <= 0].mean() if (r <= 0).any() else np.nan
    # dollar profit per closed trade = val*ret/(1+ret)
    prof = (s["val"] * s["ret"] / (1 + s["ret"])).values
    pos = prof[prof > 0]
    if pos.sum() > 0:
        k = max(1, int(np.ceil(0.05 * len(prof))))
        top5 = np.sort(prof)[::-1][:k].sum()
        t5 = top5 / prof[prof > 0].sum()   # top-5% share of gross winning P&L
    else:
        t5 = np.nan
    return dict(n=len(s), win=win, aw=aw, al=al, t5=t5)

rng_master = np.random.default_rng(0)
NULLS = {}
def null_ratios(dates, N, qfinal, nseed=5):
    rats = []
    for sd in range(nseed):
        rgen = np.random.default_rng(1000 + sd)
        noise = pd.DataFrame(rgen.random(me.shape), index=M, columns=cols).where(elig)
        res = dca_run(me, noise, elig, dates, N=N, trail=-0.25, ma=ma10, minhold_days=30)
        rats.append(res["equity"]["V"].iloc[-1] / qfinal)
    return np.mean(rats), np.max(rats)

# ---------- MAIN: era-sliced table per score, N=12 primary ----------
for N in (12, 8, 20):
    p(f"\n{'='*78}\nN={N}  trail=-0.25  trend-exit  minhold=30d  (asymmetric: cut -25%, no upper cap)")
    for nm, SC in SCORES.items():
        p(f"\n  score={nm}")
        p(f"    era        ratio  IRR    Sh    DD    | null_mean null_max | win  aw    al    t5")
        for elab, lo, hi in ERAS:
            dts = era_dates(lo, hi)
            res = dca_run(me, SC, elig, dts, N=N, trail=-0.25, ma=ma10, minhold_days=30)
            eq = res["equity"]; st = stats(eq)
            qb = dca_benchmark(bench["QQQ"], dts)
            qf = qb["V"].iloc[-1]
            ratio = st["final"] / qf
            # null only for full + a couple eras to save time at N=12; else skip nulls for N=8,20
            if N == 12:
                nmean, nmax = null_ratios(dts, N, qf)
            else:
                nmean = nmax = np.nan
            tp = trade_profile(res["trades"])
            p(f"    {elab}   {ratio:5.2f}  {st['irr']*100:4.0f}%  {st['sharpe']:4.2f}  {st['maxdd']*100:4.0f}%  |"
              f"  {nmean:5.2f}    {nmax:5.2f}  | {tp['win']*100:3.0f}% {tp['aw']*100:+4.0f}% {tp['al']*100:+4.0f}%  {tp['t5']*100 if np.isfinite(tp['t5']) else float('nan'):3.0f}%")

# ---------- choose best score by min-era ratio at N=12 & save ----------
p(f"\n{'='*78}\nSelecting best score (highest MINIMUM era ratio, investable eras 2000-02..2020-26) at N=12:")
best = None
for nm, SC in SCORES.items():
    mins = []
    for elab, lo, hi in ERAS[:-1]:
        dts = era_dates(lo, hi)
        res = dca_run(me, SC, elig, dts, N=12, trail=-0.25, ma=ma10, minhold_days=30)
        qb = dca_benchmark(bench["QQQ"], dts)
        mins.append(res["equity"]["V"].iloc[-1] / qb["V"].iloc[-1])
    p(f"  {nm}: min-era ratio {min(mins):.2f}  eras {[round(m,2) for m in mins]}")
    if best is None or min(mins) > best[1]:
        best = (nm, min(mins))
p(f"  -> best = {best[0]} (min-era {best[1]:.2f})")
pd.to_pickle(SCORES[best[0]].astype(np.float32), f"{HERE}/loop1_expectancy.pkl")
p(f"saved loop1_expectancy.pkl ({best[0]})  DONE t={time.time()-t0:.0f}s")
