"""loop4_lazyprices2: Lazy-Prices 10-K text-similarity re-test at ~209-firm coverage.
Tasks:
 1. Rebuild monthly sim panel (most-recent trailing-12m 10-K, ffill<=12m, filing<=month-close). Coverage.
 2. Era-sliced rank-IC of sim_score (small change=bullish) vs fwd-12m ret + 10-shuffle null band.
 3. Tradeable overlay via dca_run vs QQQ-DCA + random-in-pond null (5 seeds), era-sliced:
    (a) score=rank(sim) standalone N=20; (b) sim as POSITIVE gate on mom12 book (above-median sim only).
 4. Top-decile precision: P(top-sim-decile name in true fwd-12m top decile) vs base rate (§6c tail test).
READ-ONLY on shard parquets.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark, stats
def p(*a): print(*a, flush=True)

# ================= load signal =================
files = sorted(glob.glob(f"{HERE}/loop_lp_shard_*.parquet")) + [f"{HERE}/loop_lazyprices_sims.parquet"]
parts = []
for f in files:
    try: parts.append(pd.read_parquet(f))
    except Exception as e: p("skip", f, e)
S = pd.concat(parts, ignore_index=True)
S["filing_date"] = pd.to_datetime(S.filing_date)
S = S.drop_duplicates(subset=["tk", "filing_date", "form"])
S = S[S.form.isin(["10-K", "10-K405", "10-KSB"])].copy()
p(f"raw signal rows: {len(S)}, unique firms: {S.tk.nunique()}, "
  f"filing range {S.filing_date.min().date()}..{S.filing_date.max().date()}")

# ================= featmat =================
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
liq, me, dv, cols, FEAT, bench = D["liq"], D["me"], D["dv"], D["cols"], D["FEAT"], D["bench"]
M = me.index
S = S[S.tk.isin(set(cols))]
p(f"signal firms in featmat cols: {S.tk.nunique()}")

# ================= standard eligibility =================
ma10 = me.rolling(10, min_periods=10).mean()
elig_std = liq & (me >= 3) & (dv >= 2e6) & (me > ma10)
fwd12 = (me.shift(-12) / me - 1)

# ================= 1. build monthly sim panel =================
# assign each 10-K to its filing month (known at filing-month close), keep most-recent, ffill<=12m
S["m"] = S.filing_date.values.astype("datetime64[M]").astype("datetime64[ns]")
piv = S.sort_values("filing_date").groupby(["m", "tk"]).sim.last().unstack()
piv = piv.reindex(index=M, columns=cols)
sim_score = piv.ffill(limit=12)          # most-recent trailing-12m 10-K sim; higher=less change=bullish

cov_any = sim_score.notna()
firms_ever = int(cov_any.any(axis=0).sum())
cov_elig = (cov_any & elig_std)
names_mo = cov_elig.sum(axis=1).loc["2006":"2025"]
names_mo = names_mo[names_mo > 0]
covall = cov_any.sum(axis=1).loc["2006":"2025"]; covall = covall[covall > 0]
p("\n===== 1. COVERAGE =====")
p(f"firms ever in panel                  : {firms_ever}")
p(f"avg names/month (any coverage)       : {covall.mean():.1f}  (max {covall.max():.0f})")
p(f"avg ELIGIBLE names/month (std elig)  : {names_mo.mean():.1f}  (max {names_mo.max():.0f})")
p(f"months with >=15 eligible covered    : {(cov_elig.sum(axis=1)>=15).sum()}")

# pond for IC / precision = liquid names (liq & dv>=2e6) matching loop3 baseline
pond = liq & (dv >= 2e6)

ERAS = [("2006-01", "2009-12"), ("2010-01", "2014-12"),
        ("2015-01", "2019-12"), ("2020-01", "2025-12")]

# ================= 2. era-sliced rank-IC + shuffle null =================
def month_ic(signal, dt, el):
    x = signal.loc[dt].where(el); y = fwd12.loc[dt].where(el)
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 15: return np.nan, 0
    return spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation, len(d)

def era_ic(signal, dts):
    ics, ns = [], []
    for dt in dts:
        ic, n = month_ic(signal, dt, pond.loc[dt])
        if np.isfinite(ic): ics.append(ic); ns.append(n)
    if not ics: return np.nan, np.nan, 0, np.nan
    return np.mean(ics), np.std(ics)/np.sqrt(len(ics)), len(ics), np.mean(ns)

def era_ic_shuffle(signal, dts, seed):
    r = np.random.default_rng(seed); ics = []
    for dt in dts:
        el = pond.loc[dt]
        d = pd.concat([signal.loc[dt].where(el), fwd12.loc[dt].where(el)], axis=1).dropna()
        if len(d) < 15: continue
        xv = d.iloc[:, 0].values.copy(); r.shuffle(xv)
        ic = spearmanr(xv, d.iloc[:, 1].values).correlation
        if np.isfinite(ic): ics.append(ic)
    return np.mean(ics) if ics else np.nan

p("\n===== 2. ERA-SLICED rank-IC of sim_score vs fwd-12m ret (small change=bullish, expect IC>0) =====")
p(f"{'era':10} {'n/mo':>5} {'nmo':>4} {'IC_sim':>9} {'se':>7} | {'shuf_mean':>9} {'shuf_lo':>8} {'shuf_hi':>8} {'sig?':>5}")
ic_by_era = {}
for st, en in ERAS:
    dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
    ics, ses, nmo, avgn = era_ic(sim_score, dts)
    sh = [era_ic_shuffle(sim_score, dts, sd) for sd in range(10)]
    sh = [x for x in sh if np.isfinite(x)]
    smean = np.mean(sh) if sh else np.nan
    slo, shi = (np.min(sh), np.max(sh)) if sh else (np.nan, np.nan)
    sig = "Y" if (np.isfinite(ics) and (ics > shi or ics < slo)) else "n"
    ic_by_era[st[:7]] = ics
    p(f"{st[:7]:10} {avgn if np.isfinite(avgn) else 0:>5.0f} {nmo:>4} "
      f"{ics:>9.4f} {ses:>7.4f} | {smean:>9.4f} {slo:>8.4f} {shi:>8.4f} {sig:>5}")
# full-sample
dts_all = M[(M >= pd.Timestamp("2006-01-01")) & (M <= pd.Timestamp("2025-12-01"))]
ica, sea, nmoa, avgna = era_ic(sim_score, dts_all)
p(f"{'FULL 06-25':10} {avgna:>5.0f} {nmoa:>4} {ica:>9.4f} {sea:>7.4f} |")
p("\n114-firm baseline (prior read): full-sample IC ~0.015, INSIDE null band, sign-flip in 2010-14.")

# ================= 3. tradeable overlay =================
p("\n===== 3. TRADEABLE OVERLAY vs QQQ-DCA (std eligibility, era-sliced terminal ratio) =====")

def run_ratio(score, elig, dts, N, seed_random=None):
    """Return strat_final/qqq_final over dts. If seed_random given, replace score with random in same pond."""
    if len(dts) < 6: return np.nan, np.nan, np.nan
    if seed_random is not None:
        r = np.random.default_rng(seed_random)
        rnd = pd.DataFrame(r.random(elig.shape), index=elig.index, columns=elig.columns)
        sc = rnd.where(elig.reindex_like(rnd).fillna(False).astype(bool))
    else:
        sc = score
    res = dca_run(me, sc, elig, dts, N=N)
    qqq = dca_benchmark(bench["QQQ"], dts)
    sf = res["equity"]["V"].iloc[-1]; qf = qqq["V"].iloc[-1]
    st = stats(res["equity"])
    return sf/qf, st.get("irr", np.nan), st.get("sharpe", np.nan)

# --- (a) score = sim rank, standalone N=20 ---
# candidate = covered & standard-eligible; score = sim_score (rank handled by dca_run sort)
elig_a = elig_std & cov_any
score_a = sim_score.where(elig_a)
# --- (b) sim as POSITIVE gate on mom12 book: only names with above-median sim (among covered) each month ---
mom12 = FEAT["mom12"]
med = sim_score.where(cov_any).median(axis=1)          # cross-sectional median of covered sims each month
above_med = sim_score.ge(med, axis=0) & cov_any
elig_b = elig_std & above_med
score_b = mom12.where(elig_b)

for label, score, elig, N in [("(a) rank(sim) N=20", score_a, elig_a, 20),
                              ("(b) mom12|sim>med N=20", score_b, elig_b, 20)]:
    p(f"\n--- overlay {label} ---")
    p(f"{'era':10} {'ratio':>7} {'IRR':>7} {'Sh':>6} | {'null_mean':>9} {'null_max':>9} {'beats_max?':>10}")
    for st, en, lab in [(a, b, a[:7]) for a, b in ERAS] + [("2006-01", "2025-12", "FULL06-25")]:
        dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
        # restrict to months that have >=N candidates so the book can fill
        havecand = (elig & score.notna()).sum(axis=1)
        dts = dts[havecand.reindex(dts).fillna(0).values >= 3]
        if len(dts) < 6:
            p(f"{st[:7]:10} {'--':>7}  (insufficient candidate-months)"); continue
        ratio, irr, sh = run_ratio(score, elig, dts, N)
        nulls = [run_ratio(None, elig, dts, N, seed_random=sd)[0] for sd in range(5)]
        nulls = [x for x in nulls if np.isfinite(x)]
        nm, nx = (np.mean(nulls), np.max(nulls)) if nulls else (np.nan, np.nan)
        beat = "Y" if (np.isfinite(ratio) and ratio > nx) else "n"
        p(f"{lab:10} {ratio:>7.3f} {irr*100:>6.1f}% {sh:>6.2f} | {nm:>9.3f} {nx:>9.3f} {beat:>10}")

# ================= 4. top-decile precision (§6c tail test) =================
p("\n===== 4. TOP-SIM-DECILE PRECISION vs fwd-12m TOP-DECILE (covered eligible names) =====")
hits = tot = 0
lifts = []
for dt in M:
    el = pond.loc[dt] & cov_any.loc[dt]
    x = sim_score.loc[dt].where(el); y = fwd12.loc[dt].where(el)
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 20: continue
    n = len(d); k = max(1, int(round(n*0.10)))
    top_sim = d.iloc[:, 0].nlargest(k).index
    top_ret = set(d.iloc[:, 1].nlargest(k).index)
    h = sum(t in top_ret for t in top_sim)
    hits += h; tot += len(top_sim)
    base = k / n
    lifts.append((h/len(top_sim)) / base if base > 0 else np.nan)
prec = hits/tot if tot else np.nan
p(f"pooled top-sim-decile precision      : {prec:.4f}  (n picks={tot})")
p(f"random base rate (decile)            : ~0.10")
p(f"mean per-month lift over base        : {np.nanmean(lifts):.3f}x  (1.0=no skill)")
p(f"months evaluated                     : {len(lifts)}")
p("\nDONE")
