"""LOOP1 — Opportunistic (non-routine) insider buying, Cohen-Malloy-Pomorski (2012).
Literature: opportunistic insiders (no regular calendar cadence) earn ~82 bps/mo;
routine insiders ~zero. Repo validated insider buying broadly but never split them.

We lack per-insider IDs, so classify ROUTINE vs OPPORTUNISTIC at the FIRM level
using only the monthly open-market buy history (point-in-time, keyed on FILING date):
  buy_ind[t,firm]  = firm had an open-market P-buy filed in month t
  routine[t,firm]  = buy_ind AND habitual cadence, where habitual =
        (>=6 buy-months in the trailing 24m)  OR
        (bought this SAME calendar month in >=2 of the prior 3 years)
  opp[t,firm]      = buy_ind AND NOT routine   (non-routine / surprise buy)
  (strict surprise subset also computed: opp AND zero buys in trailing 12m)
All windows use STRICTLY PRIOR months -> no look-ahead. Entry only at the filing month.

Tests on the harness (me/dv/liq from _featmat.pkl):
 1. Cross-sectional fwd-12m rank-IC of opp vs routine vs all-buy, era-sliced.
 2. Monthly-DCA long-only tilt via dca_run: score=rank(opp signal, trailing 3m),
    N=20, standard eligibility, era-sliced ratio vs QQQ-DCA + random-in-buyers null.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark, stats, twr

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
liq, me, dv, bench, cols = D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
# insider era only
M = me.index[(me.index >= pd.Timestamp("2010-01-01")) & (me.index <= pd.Timestamp("2026-06-01"))]
me, dv, liq = me.loc[M], dv.loc[M], liq.loc[M]
ma10 = me.rolling(10, min_periods=10).mean()
STD_ELIG = liq & (me >= 3) & (dv >= 2e6) & (me > ma10)

# ---- build monthly insider-buy panels aligned to (M x cols) ----
d = pd.read_pickle(f"{HERE}/_insider_rich.pkl")
d["tk"] = d["tk"].str.replace('"', "", regex=False)
d = d[d.tk.isin(set(cols))]
d["ym"] = d["ym"].values.astype("datetime64[M]")
buy_ind = (d[d.buy > 0].assign(v=1.0)
           .pivot_table(index="ym", columns="tk", values="v", aggfunc="max")
           .reindex(index=M, columns=cols).fillna(0.0) > 0)
nbuy = (d[d.buy > 0].pivot_table(index="ym", columns="tk", values="nbuyers", aggfunc="sum")
        .reindex(index=M, columns=cols).fillna(0.0))
bi = buy_ind.astype(float)

# habitual: >=6 buy-months in trailing 24m (strictly prior)
trail24 = bi.shift(1).rolling(24, min_periods=1).sum()
# same calendar month in >=2 of prior 3 years (t-12, t-24, t-36)
same_cal = bi.shift(12).fillna(0) + bi.shift(24).fillna(0) + bi.shift(36).fillna(0)
habitual = (trail24 >= 6) | (same_cal >= 2)
routine = buy_ind & habitual
opp = buy_ind & (~habitual)
trail12 = bi.shift(1).rolling(12, min_periods=1).sum()
opp_strict = opp & (trail12 == 0)   # surprise: no buy in trailing 12m

p(f"panels built t={time.time()-t0:.0f}s  months={len(M)}  cols={len(cols)}")
p(f"buy-month firm-obs: all={int(bi.sum().sum())} routine={int(routine.sum().sum())} "
  f"opp={int(opp.sum().sum())} opp_strict={int(opp_strict.sum().sum())}")

# ================= PART 1: cross-sectional fwd-12m rank-IC =================
fwd12 = (me.shift(-12) / me - 1)
UNIV = STD_ELIG                        # rank/IC universe = standard eligible pond
FR = fwd12.where(UNIV)
ERAS = [("2011-01", "2014-12"), ("2015-01", "2019-12"), ("2020-01", "2025-12")]

def era_ic(sig):
    """mean cross-sectional Spearman(sig, fwd12) within eligible pond, per era.
    sig is a binary/intensity panel; corr computed over eligible names each month."""
    out = {}
    for st, en in ERAS:
        dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
        ics = []
        for dt in dts:
            f = FR.loc[dt]
            el = UNIV.loc[dt] & f.notna()
            s = sig.loc[dt].where(el)
            fr = f.where(el)
            m = s.notna() & fr.notna()
            if m.sum() < 30 or s[m].std() == 0:
                continue
            ics.append(s[m].rank().corr(fr[m].rank()))
        out[(st, en)] = (np.mean(ics) if ics else np.nan, len(ics))
    return out

# use nbuyers-weighted intensity within each cohort (0 outside cohort) for IC
sig_all = bi * (1 + np.log1p(nbuy))
sig_rout = routine.astype(float) * (1 + np.log1p(nbuy))
sig_opp = opp.astype(float) * (1 + np.log1p(nbuy))
sig_opps = opp_strict.astype(float) * (1 + np.log1p(nbuy))

p("\n=== PART 1: fwd-12m rank-IC (eligible pond), by era ===")
p(f"{'signal':14}" + "".join(f"{a}..{b[:4]:>10}" for a, b in [(s[:4], e) for s, e in ERAS]))
for nm, sg in [("all_buy", sig_all), ("routine", sig_rout),
               ("opportunistic", sig_opp), ("opp_strict", sig_opps)]:
    r = era_ic(sg)
    line = f"{nm:14}"
    for k in [( "2011-01","2014-12"),("2015-01","2019-12"),("2020-01","2025-12")]:
        ic, n = r[k]
        line += f"  {ic:+.4f}(n{n})"
    p(line)

# also: cohort mean fwd-12m rank spread (opp buyers vs non-buyers) for interpretability
p("\ncohort mean fwd-12m pct-rank (within eligible pond):")
for nm, ind in [("all_buy", buy_ind), ("routine", routine), ("opportunistic", opp)]:
    for st, en in ERAS:
        dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
        de, dn = [], []
        for dt in dts:
            el = UNIV.loc[dt] & FR.loc[dt].notna()
            r = FR.loc[dt].where(el).rank(pct=True)
            ev = ind.loc[dt].where(el).fillna(False)
            if ev.sum() < 5: continue
            de.append(r[ev].mean()); dn.append(r[el & ~ev].mean())
        if de:
            p(f"  {nm:14} {st[:4]}-{en[:4]}: buyers {np.mean(de):.3f} vs rest {np.mean(dn):.3f} "
              f"(spread {np.mean(de)-np.mean(dn):+.3f})")

# ================= PART 2: DCA tilt via dca_run =================
# tradeable signal: opportunistic buy in trailing 3 filing-months, intensity=nbuyers
def trail_score(ind, w=3):
    inten = (ind.astype(float) * (1 + np.log1p(nbuy))).rolling(w, min_periods=1).sum()
    return inten.where(inten > 0)   # NaN = not a candidate

score_opp = trail_score(opp)
score_rout = trail_score(routine)
score_all = trail_score(buy_ind)
dates = pd.DatetimeIndex(M)

def run_era(score, st, en):
    dts = dates[(dates >= pd.Timestamp(st + "-01")) & (dates <= pd.Timestamp(en + "-01"))]
    if len(dts) < 12: return None
    res = dca_run(me, score, STD_ELIG, dts, N=20)
    qqq = dca_benchmark(bench["QQQ"], dts)
    s = stats(res["equity"]); ratio = res["equity"]["V"].iloc[-1] / qqq["V"].iloc[-1]
    return ratio, s

# random-in-buyers null: random score in the pond of firms with ANY buy trailing 3m
buyer_pond = (buy_ind.astype(float).rolling(3, min_periods=1).sum() > 0)
def run_null_era(st, en, seeds=6):
    dts = dates[(dates >= pd.Timestamp(st + "-01")) & (dates <= pd.Timestamp(en + "-01"))]
    if len(dts) < 12: return None
    qqq = dca_benchmark(bench["QQQ"], dts)
    rr = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        rand = pd.DataFrame(rng.random(buyer_pond.shape), index=buyer_pond.index,
                            columns=buyer_pond.columns).where(buyer_pond)
        res = dca_run(me, rand, STD_ELIG, dts, N=20)
        rr.append(res["equity"]["V"].iloc[-1] / qqq["V"].iloc[-1])
    return np.mean(rr), np.max(rr)

p("\n=== PART 2: monthly-DCA long-only tilt (N=20) vs QQQ-DCA ===")
p(f"{'era':11} {'opp':>6} {'rout':>6} {'all':>6} | {'oppIRR':>7} {'oppSh':>6} {'oppDD':>6} | {'null_mu':>7} {'null_mx':>7}")
BT_ERAS = [("2011-01", "2014-12"), ("2015-01", "2019-12"), ("2020-01", "2025-12"),
           ("2011-01", "2025-12")]
for st, en in BT_ERAS:
    ro = run_era(score_opp, st, en)
    rr = run_era(score_rout, st, en)
    ra = run_era(score_all, st, en)
    nl = run_null_era(st, en)
    if ro is None: continue
    (rat_o, so), (rat_r, _), (rat_a, _) = ro, rr, ra
    nm_, nx_ = nl
    p(f"{st[:4]}-{en[:4]}  {rat_o:6.2f} {rat_r:6.2f} {rat_a:6.2f} | "
      f"{so['irr']*100:6.1f}% {so['sharpe']:6.2f} {so['maxdd']*100:6.1f}% | "
      f"{nm_:7.2f} {nx_:7.2f}")

# ---- save opportunistic score panel ----
score_opp.to_pickle(f"{HERE}/loop1_oppinsider.pkl")
p(f"\nsaved loop1_oppinsider.pkl  shape={score_opp.shape}  t={time.time()-t0:.0f}s")
