"""loop3: PRELIMINARY (underpowered, directional) Lazy-Prices IC test.
Concat all shard parquets + loop_lazyprices_sims.parquet, dedup on (tk,filing_date,form).
Build monthly sim panel (most-recent 10-K with filing_date<=month-close, ffill<=12m, no look-ahead).
change_score=(1-sim) [big change=bearish], sim_score=sim [small change=bullish].
Tests: era-sliced rank-IC of both vs fwd-12m ret; shuffle-null (10); factor orthogonality; sign stability.
Small-n: DIRECTIONAL ONLY.
"""
import os, glob, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__))
def p(*a): print(*a, flush=True)

# ---- load signal: concat shards + sims, dedup ----
files = sorted(glob.glob(f"{HERE}/loop_lp_shard_*.parquet")) + [f"{HERE}/loop_lazyprices_sims.parquet"]
parts = []
for f in files:
    try: parts.append(pd.read_parquet(f))
    except Exception as e: p("skip", f, e)
S = pd.concat(parts, ignore_index=True)
S["filing_date"] = pd.to_datetime(S.filing_date)
S = S.drop_duplicates(subset=["tk", "filing_date", "form"])
S = S[S.form.isin(["10-K", "10-K405", "10-KSB"])].copy()
p(f"raw signal rows after concat+dedup+10K-filter: {len(S)}, unique firms: {S.tk.nunique()}")
p(f"filing_date range: {S.filing_date.min().date()} .. {S.filing_date.max().date()}")

# ---- featmat ----
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
liq, me, dv, cols, FEAT = D["liq"], D["me"], D["dv"], D["cols"], D["FEAT"]
M = me.index
S = S[S.tk.isin(set(cols))]
p(f"signal firms in featmat cols: {S.tk.nunique()}")

# ---- eligibility: liquid names (liq & dv>=2e6) ----
pond = liq & (dv >= 2e6)
fwd12 = (me.shift(-12) / me - 1)

# ---- build monthly sim panel: most-recent 10-K with filing_date<=month close, ffill<=12m ----
# month-start index M; filing known at its filing month. assign to filing month, ffill.
S["m"] = S.filing_date.values.astype("datetime64[M]").astype("datetime64[ns]")
piv = S.sort_values("filing_date").groupby(["m", "tk"]).sim.last().unstack()
piv = piv.reindex(index=M, columns=cols)
sim_score = piv.ffill(limit=12)
change_score = 1.0 - sim_score

cov = (sim_score.notna() & pond)
firms_ever = int((sim_score.notna()).any(axis=0).sum())
avg_names = cov.sum(axis=1).loc["2006":"2025"]
avg_names = avg_names[avg_names > 0]
p(f"\nCOVERAGE: {firms_ever} firms ever in panel; "
  f"avg {avg_names.mean():.1f} eligible names/month (over months with any coverage, 2006-2025)")

ERAS = [("2006-01", "2009-12"), ("2010-01", "2014-12"),
        ("2015-01", "2019-12"), ("2020-01", "2025-12")]

def month_ic(signal, dt):
    el = pond.loc[dt]
    x = signal.loc[dt].where(el); y = fwd12.loc[dt].where(el)
    d = pd.concat([x, y], axis=1).dropna()
    if len(d) < 15: return np.nan, 0
    return spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation, len(d)

def era_ic(signal, dts):
    ics, ns = [], []
    for dt in dts:
        ic, n = month_ic(signal, dt)
        if np.isfinite(ic): ics.append(ic); ns.append(n)
    if not ics: return np.nan, np.nan, 0, np.nan
    return np.mean(ics), np.std(ics)/np.sqrt(len(ics)), len(ics), np.mean(ns)

def era_ic_shuffle(signal, dts, seed):
    r = np.random.default_rng(seed)
    ics = []
    for dt in dts:
        el = pond.loc[dt]
        row = signal.loc[dt].where(el)
        y = fwd12.loc[dt].where(el)
        d = pd.concat([row, y], axis=1).dropna()
        if len(d) < 15: continue
        xv = d.iloc[:, 0].values.copy(); r.shuffle(xv)
        ic = spearmanr(xv, d.iloc[:, 1].values).correlation
        if np.isfinite(ic): ics.append(ic)
    return np.mean(ics) if ics else np.nan

p("\n=== 1&2. ERA-SLICED rank-IC vs fwd-12m return (liquid: liq & dv>=2e6) ===")
p("change_score=(1-sim), Lazy-Prices predicts BEARISH -> expect IC < 0")
p("sim_score=sim,          small change BULLISH      -> expect IC > 0")
p(f"\n{'era':10} {'n/mo':>5} {'IC_change':>10} {'se':>7} | {'IC_sim':>9} {'se':>7} | "
  f"{'shuf_mean':>9} {'shuf_lo':>8} {'shuf_hi':>8}")
sign_change, sign_sim = [], []
for st, en in ERAS:
    dts = M[(M >= pd.Timestamp(st + "-01")) & (M <= pd.Timestamp(en + "-01"))]
    icc, sec, nc, avgn = era_ic(change_score, dts)
    ics, ses, ns, _ = era_ic(sim_score, dts)
    # shuffle null on sim (change is just 1-sim -> same |IC|, opposite sign)
    sh = [era_ic_shuffle(sim_score, dts, sd) for sd in range(10)]
    sh = [x for x in sh if np.isfinite(x)]
    smean = np.mean(sh) if sh else np.nan
    slo, shi = (np.min(sh), np.max(sh)) if sh else (np.nan, np.nan)
    p(f"{st[:7]:10} {avgn if np.isfinite(avgn) else 0:>5.0f} "
      f"{icc:>10.4f} {sec:>7.4f} | {ics:>9.4f} {ses:>7.4f} | "
      f"{smean:>9.4f} {slo:>8.4f} {shi:>8.4f}")
    sign_change.append(icc); sign_sim.append(ics)

# ---- 3. factor orthogonality ----
p("\n=== 3. FACTOR ORTHOGONALITY (cross-sectional corr of sim_score ranks vs factor ranks) ===")
p("spec §6c requires |corr| < 0.15. pooled across all months w/ coverage, eligible names.")
factors = {"mom12": FEAT["mom12"], "log_dv": np.log(dv.where(dv > 0)),
           "vol6": FEAT["vol6"], "roa": FEAT["roa"]}
p(f"{'factor':10} {'mean_xs_corr':>13} {'|mean|<0.15?':>13}")
for fn, F in factors.items():
    corrs = []
    for dt in M:
        el = pond.loc[dt]
        x = sim_score.loc[dt].where(el)
        f = F.loc[dt].where(el) if dt in F.index else None
        if f is None: continue
        d = pd.concat([x, f], axis=1).dropna()
        if len(d) < 15: continue
        c = spearmanr(d.iloc[:, 0], d.iloc[:, 1]).correlation
        if np.isfinite(c): corrs.append(c)
    mc = np.mean(corrs) if corrs else np.nan
    p(f"{fn:10} {mc:>13.4f} {'YES' if abs(mc) < 0.15 else 'NO':>13}  (nmo={len(corrs)})")

# ---- 4. sign stability ----
p("\n=== 4. SIGN STABILITY across eras ===")
p(f"IC_change by era: {[round(x,4) for x in sign_change]}")
p(f"IC_sim    by era: {[round(x,4) for x in sign_sim]}")
cc = [x for x in sign_change if np.isfinite(x)]
ss = [x for x in sign_sim if np.isfinite(x)]
change_stable = all(x < 0 for x in cc)   # Lazy-Prices direction: change bearish
sim_stable_pos = all(x > 0 for x in ss)
p(f"change_score IC<0 in ALL eras (Lazy-Prices direction)? {change_stable}")
p(f"sim_score    IC>0 in ALL eras?                          {sim_stable_pos}")
p(f"same sign across all eras (change)? {all(np.sign(x)==np.sign(cc[0]) for x in cc)}")
p(f"same sign across all eras (sim)?    {all(np.sign(x)==np.sign(ss[0]) for x in ss)}")
p("\nDONE")
