"""Forward projection: DCA $1,500 every two weeks into QQQ.

Method (honest by construction):
  - Contributions: $1,500 every 10 trading days (26/yr), start today, no
    stopping, no timing, dividends reinvested.
  - Return engine: STATIONARY BLOCK BOOTSTRAP of QQQ's own daily total returns
    (1999-03 to today), 21-day blocks. Blocks preserve volatility clustering
    and crash sequences, so drawdown paths and sequence-of-returns risk are
    represented rather than assumed away. 20,000 paths per horizon.
  - Three drift scenarios, because the future is not the past:
      HIST      history as-is (QQQ's realized drift over its whole life)
      HAIRCUT   history minus 2 pp/yr  (base case: valuations start higher,
                the mega-cap concentration that drove the last decade is
                unlikely to repeat at the same rate)
      SOBER     history minus 4 pp/yr  (a persistently weaker regime)
  - Reported in NOMINAL dollars and in REAL (today's) dollars at 2.5%/yr
    inflation, because a 30-year nominal number is misleading on its own.

Run:  python3 scripts/dca_qqq_projection.py
"""
import csv, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRIB = 1500.0
PER_YEAR = 26
DAYS_PER_CONTRIB = 10
HORIZONS = [5, 10, 15, 20, 25, 30]
NSIM = 20000
BLOCK = 21
INFL = 0.025
SEED = 20260809

# ---------- load QQQ daily total-return series ----------
rows = []
with open(f"{ROOT}/data/etfs/QQQ.csv") as f:
    for r in csv.DictReader(f):
        v = r.get("Adj Close") or r.get("Close")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        rows.append((r["Date"], v))
rows.sort()
px = np.array([v for _, v in rows])
ret = px[1:]/px[:-1] - 1
ret = ret[np.isfinite(ret)]
yrs_hist = len(ret)/252.0
cagr_hist = (px[-1]/px[0])**(1/((len(px)-1)/252.0)) - 1
vol_hist = ret.std()*math.sqrt(252)
print(f"QQQ history: {rows[0][0]} -> {rows[-1][0]}  ({yrs_hist:.1f} years)")
print(f"  realized CAGR {cagr_hist:6.2%}   annualized vol {vol_hist:5.1%}")
print(f"  worst 1-day {ret.min():.1%}   best 1-day {ret.max():.1%}\n")

SCEN = {"HIST": 0.00, "HAIRCUT (base case)": -0.02, "SOBER": -0.04}

def simulate(years, drift_adj, rng):
    """Block-bootstrap paths; return terminal wealth array."""
    n_contrib = years*PER_YEAR
    n_days = n_contrib*DAYS_PER_CONTRIB
    daily_adj = (1+drift_adj)**(1/252.0) - 1 if drift_adj else 0.0
    n_blocks = int(np.ceil(n_days/BLOCK))
    starts = rng.integers(0, len(ret)-BLOCK, size=(NSIM, n_blocks))
    # build (NSIM, n_days) return matrix from sampled blocks
    offs = np.arange(BLOCK)
    idx = (starts[:, :, None] + offs[None, None, :]).reshape(NSIM, -1)[:, :n_days]
    R = ret[idx] + daily_adj
    # wealth: contribute at day 0 of each 10-day step, then compound
    growth = np.cumprod(1.0 + R, axis=1)                 # value of $1 from t=0
    g_at = growth[:, ::DAYS_PER_CONTRIB]                 # growth factor at each contribution date
    # each contribution i grows by growth_end / g_at[i]
    end = growth[:, -1][:, None]
    wealth = (CONTRIB * (end / g_at)).sum(axis=1)
    return wealth

rng = np.random.default_rng(SEED)
print(f"DCA ${CONTRIB:,.0f} every 2 weeks ({PER_YEAR}/yr = ${CONTRIB*PER_YEAR:,.0f}/yr), "
      f"{NSIM:,} bootstrap paths per cell\n")

for label, adj in SCEN.items():
    print(f"===== {label}"
          + (f"  (drift {adj*100:+.0f} pp/yr vs history)" if adj else "  (history as-is)") + " =====")
    print(f"{'yrs':>4} {'you put in':>12} | {'10th %ile':>12} {'25th':>12} {'MEDIAN':>13} {'75th':>12} {'90th':>12} | "
          f"{'median real':>12} {'multiple':>8} {'P(loss)':>8}")
    for y in HORIZONS:
        w = simulate(y, adj, rng)
        put_in = CONTRIB*PER_YEAR*y
        p10, p25, p50, p75, p90 = np.percentile(w, [10, 25, 50, 75, 90])
        real50 = p50/((1+INFL)**y)
        ploss = float((w < put_in).mean())
        print(f"{y:>4} {put_in:>12,.0f} | {p10:>12,.0f} {p25:>12,.0f} {p50:>13,.0f} {p75:>12,.0f} {p90:>12,.0f} | "
              f"{real50:>12,.0f} {p50/put_in:>7.2f}x {ploss:>7.1%}")
    print()

# implied annualized return on contributions (money-weighted), base case
print("=== money-weighted (IRR-like) return implied by the base-case median ===")
rng2 = np.random.default_rng(SEED+1)
for y in HORIZONS:
    w = simulate(y, -0.02, rng2)
    p50 = np.median(w)
    # solve for r such that FV of an annuity of CONTRIB per period == p50
    lo, hi = -0.5, 0.5
    for _ in range(200):
        mid = (lo+hi)/2
        rp = (1+mid)**(1/PER_YEAR) - 1
        fv = CONTRIB*(((1+rp)**(y*PER_YEAR) - 1)/rp) if rp != 0 else CONTRIB*y*PER_YEAR
        if fv < p50: lo = mid
        else: hi = mid
    print(f"  {y:>2}y: median ${p50:>12,.0f}  ->  {(lo+hi)/2:5.2%}/yr money-weighted")
