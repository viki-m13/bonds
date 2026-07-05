"""LOOP2 FEASIBILITY PROBE — can ANY price-only monthly-DCA selection rule beat
QQQ-DCA in the two eras where every strategy dies: 2000-02 and 2003-09?

Constraint: only price/volume signals exist pre-2012 (SEC fundamentals all-NaN
pre-2012), so features are built ONLY from me / dv / liq.

Battery: momentum(3/6/12 + skip-month), low-vol, near-52w-high, dollar-volume
leaders, mean-reversion(1m/3m), trend-quality/smoothness, deep-value drawdown-
recovery, low-price, dual-momentum(only names beating QQQ). Each is run PLAIN and
with a Faber-style DEFENSIVE/REGIME overlay (rotate held book -> QQQ when
QQQ<10mo-MA, back to picks when QQQ>10mo-MA). N in {5,12,20}. Each via dca_run
(monthly, $1k, 20bps, min-30d hold), ratio = strat_final / QQQ-DCA_final,
reported SEPARATELY for 2000-02 and 2003-09.

Deliverable: the CEILING ratio any price-only rule reaches in each bad era, and
whether anything clears 1.1.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import dca_run, dca_benchmark

t0 = time.time()
def p(*a): print(*a, flush=True)

D = pd.read_pickle(f"{HERE}/_featmat.pkl")
liq, me0, dv0, bench = D["liq"], D["me"], D["dv"], D["bench"]
M = me0.index

# ---- subset universe to names ever-eligible 1999-2009 (speed + memory) --------
base_elig = (liq & (me0 >= 3) & (dv0 >= 2e6))
win = (M >= pd.Timestamp("1999-01-01")) & (M <= pd.Timestamp("2009-12-31"))
uni = base_elig.loc[win].any(axis=0)
cols = uni[uni].index
me = me0[cols].copy()
dv = dv0[cols].copy()
lq = liq[cols].copy()
elig_base = (lq & (me >= 3) & (dv >= 2e6))       # standard pond, NO per-stock trend gate

qqq = bench["QQQ"].reindex(M)
qma10 = qqq.rolling(10, min_periods=10).mean()
regime_off = (qqq < qma10)                        # True = risk-off month

# ---- append QQQ as a tradable column for the defensive overlay ----------------
me_q = me.copy(); me_q["QQQ"] = qqq
stock_cols = list(cols)

# ---- price-only signal panels (higher = better), rank within eligible pond -----
rets = me.pct_change()
hi52 = me.rolling(12, min_periods=6).max()
lo52 = me.rolling(12, min_periods=6).min()
vol6 = rets.rolling(6, min_periods=4).std()
vol12 = rets.rolling(12, min_periods=8).std()
ma5 = me.rolling(5, min_periods=5).mean()
athigh = me.cummax()
mask = elig_base.replace(False, np.nan)           # NaN out ineligible before ranking
def R(df):                                        # pct-rank within eligible names
    return (df * mask.reindex_like(df).notna()).where(mask.notna()).rank(axis=1, pct=True)

qmom12 = (qqq / qqq.shift(12) - 1)
SIG = {
    "mom3":        R(me / me.shift(3) - 1),
    "mom6":        R(me / me.shift(6) - 1),
    "mom12":       R(me / me.shift(12) - 1),
    "mom12skip1":  R(me.shift(1) / me.shift(13) - 1),
    "mom6skip1":   R(me.shift(1) / me.shift(7) - 1),
    "lowvol":      R(-vol6),
    "near52hi":    R(me / hi52),
    "dollarvol":   R(dv),
    "rev1":        R(-(me / me.shift(1) - 1)),
    "rev3":        R(-(me / me.shift(3) - 1)),
    "smoothtrend": R((me > ma5).rolling(12, min_periods=6).mean()),
    "trendqual":   R((me / me.shift(12) - 1) / (vol12 + 1e-9)),   # 12m sharpe-of-trend
    "ddrecover":   R((me / athigh) * (me / me.shift(1) - 1 > 0)),  # near ATH & turning up
    "deepvalrec":  R((1 - me / hi52) * (me / me.shift(3) - 1)),    # deep below hi but 3m up
    "lowprice":    R(-me),                                          # lowest nominal price
    # dual-momentum: score = mom12 but only names beating QQQ's 12m return
    "dualmom":     R((me / me.shift(12) - 1).sub(qmom12, axis=0).where(
                       (me / me.shift(12) - 1).sub(qmom12, axis=0) > 0)),
}

ERAS = {"2000-02": ("2000-01-01", "2002-12-31"),
        "2003-09": ("2003-01-01", "2009-12-31")}

BIG = 1e18
def run_ratio(score, era, N, overlay):
    st, en = ERAS[era]
    dates = M[(M >= pd.Timestamp(st)) & (M <= pd.Timestamp(en))]
    if not overlay:
        eq = dca_run(me, score, elig_base, dates, N=N)["equity"]
    else:
        # Faber regime overlay: risk-off -> force-exit all stocks, rotate into QQQ
        sc = score.reindex(columns=me_q.columns).copy()
        sc["QQQ"] = np.nan
        el = elig_base.reindex(columns=me_q.columns, fill_value=False).copy()
        el["QQQ"] = False
        off_dates = dates[regime_off.reindex(dates).fillna(False).values]
        # risk-off: kill stock scores, make QQQ the only eligible top pick
        sc.loc[off_dates, stock_cols] = np.nan
        sc.loc[off_dates, "QQQ"] = 10.0
        el.loc[off_dates, "QQQ"] = True
        # ma-exit panel: force-exit held stocks on risk-off months (>minhold)
        ma = pd.DataFrame(np.nan, index=me_q.index, columns=me_q.columns)
        ma.loc[off_dates, stock_cols] = BIG
        eq = dca_run(me_q, sc, el, dates, N=N, ma=ma)["equity"]
    b = dca_benchmark(bench["QQQ"], dates)
    return eq["V"].iloc[-1] / b["V"].iloc[-1]

# ---- run the battery ----------------------------------------------------------
rows = []
for nm, sc in SIG.items():
    for N in (5, 12, 20):
        for ov in (False, True):
            r00 = run_ratio(sc, "2000-02", N, ov)
            r03 = run_ratio(sc, "2003-09", N, ov)
            rows.append({"sig": nm, "N": N, "ovl": "def" if ov else "plain",
                         "r_2000_02": r00, "r_2003_09": r03})
    p(f"  done {nm}  t={time.time()-t0:.0f}s")

res = pd.DataFrame(rows)

# ---- random null (context): random score in same pond, N=12, 5 seeds ----------
null00, null03 = [], []
for seed in range(5):
    rng = np.random.default_rng(seed)
    rnd = pd.DataFrame(rng.random(me.shape), index=me.index, columns=me.columns)
    rnd = rnd.where(mask.notna())
    null00.append(run_ratio(rnd, "2000-02", 12, False))
    null03.append(run_ratio(rnd, "2003-09", 12, False))
p(f"\nRANDOM NULL N12 plain: 2000-02 mean {np.mean(null00):.2f} max {np.max(null00):.2f} | "
  f"2003-09 mean {np.mean(null03):.2f} max {np.max(null03):.2f}")

# ---- report -------------------------------------------------------------------
p("\n=== TOP 10 RULES BY 2003-09 RATIO ===")
top = res.sort_values("r_2003_09", ascending=False).head(10)
p(f"{'sig':12} {'N':>3} {'ovl':>6} {'2003-09':>8} {'2000-02':>8}")
for _, x in top.iterrows():
    p(f"{x.sig:12} {x.N:>3} {x.ovl:>6} {x.r_2003_09:>8.2f} {x.r_2000_02:>8.2f}")

p("\n=== TOP 10 RULES BY 2000-02 RATIO ===")
top2 = res.sort_values("r_2000_02", ascending=False).head(10)
p(f"{'sig':12} {'N':>3} {'ovl':>6} {'2000-02':>8} {'2003-09':>8}")
for _, x in top2.iterrows():
    p(f"{x.sig:12} {x.N:>3} {x.ovl:>6} {x.r_2000_02:>8.2f} {x.r_2003_09:>8.2f}")

c03 = res.loc[res.r_2003_09.idxmax()]
c00 = res.loc[res.r_2000_02.idxmax()]
p("\n=== CEILING ===")
p(f"best 2003-09: {c03.sig} N{c03.N} {c03.ovl} -> {c03.r_2003_09:.2f} (its 2000-02 {c03.r_2000_02:.2f})")
p(f"best 2000-02: {c00.sig} N{c00.N} {c00.ovl} -> {c00.r_2000_02:.2f} (its 2003-09 {c00.r_2003_09:.2f})")
# rules clearing 1.1 in BOTH bad eras
both = res[(res.r_2003_09 > 1.1) & (res.r_2000_02 > 1.1)]
p(f"\nrules clearing 1.1 in BOTH bad eras: {len(both)}")
for _, x in both.sort_values('r_2003_09', ascending=False).iterrows():
    p(f"   {x.sig:12} N{x.N} {x.ovl:>6}  03-09 {x.r_2003_09:.2f}  00-02 {x.r_2000_02:.2f}")
res.to_pickle(f"{HERE}/loop2_feasibility_results.pkl")
p(f"\nDONE t={time.time()-t0:.0f}s")
