"""IMPROVE_VOLEST — can a faster/smarter causal vol estimator cut VOLT's true
worst-case drawdown (the -65% in the dot-com first-leg-down, 2000) WITHOUT hurting
returns in the other eras?

Infrastructure = dotcom.py (QQQ back to 1999-03, reconstructed 3x TQQQ, cash defense
from FRED T-bill). Base rule unchanged:  w = clip(0.30/vol_est, 0, 1), weight read at
PRIOR month-end (shift 1), monthly DCA $1000.  ALL estimators are causal (no look-ahead).

We swap ONLY the volatility estimator and measure:
  - dot-com lump-sum max drawdown (2000-01..2002-12, the -65% leg)
  - full 1999-2026 lump-sum: CAGR, Sharpe, maxDD
  - DCA final-wealth ratio vs QQQ-DCA, era-sliced (2003-09, 2010-19, 2020-26, full)
  - phase-robustness (rebalance-day sensitivity, the killer test)

Run:  python3 improve_volest.py
"""
import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
REPO = "/home/user/bonds"

# ---------- data (same as dotcom.py, plus High/Low/Open for range vol) ----------
d = pd.read_csv(f"{REPO}/data/etfs/QQQ.csv"); d["Date"] = pd.to_datetime(d["Date"])
d = d.drop_duplicates("Date").set_index("Date").sort_index()
qqq, qh, ql, qo = d["Close"], d["High"], d["Low"], d["Open"]
idx = qqq.index

# cash: 1-mo T-bill, fill pre-2001 with 1-yr, daily = annual_yield/252 (conservative)
t1mo = pd.read_csv(f"{REPO}/data/fred/DGS1MO.csv"); t1mo["Date"] = pd.to_datetime(t1mo["Date"]); t1mo = t1mo.set_index("Date")["DGS1MO"]
t1 = pd.read_csv(f"{REPO}/data/fred/DGS1.csv"); t1["Date"] = pd.to_datetime(t1["Date"]); t1 = t1.set_index("Date")["DGS1"]
cashy = t1mo.reindex(idx).ffill(); cashy = cashy.fillna(t1.reindex(idx).ffill()).fillna(4.0)
cash_dret = (cashy/100.0)/252.0
cash_lvl = (1+cash_dret).cumprod()

# reconstruct TQQQ (3x daily QQQ, fees) — identical to dotcom.py
qret = qqq.pct_change().fillna(0)
exp, borrow, L = 0.0095, 0.03, 3
tqqq_dret = L*qret - (exp+(L-1)*borrow)/252.0
tqqq_lvl = (1+tqqq_dret).cumprod()*100

# monthly grid = last trading day of each calendar month (main table matches dotcom.py)
mg = pd.DatetimeIndex(sorted(qqq.groupby(qqq.index.to_period("M")).apply(lambda x: x.index[-1]).values))

TARGET = 0.30  # applied to TQQQ-equivalent vol
SQ = np.sqrt(252)

# =====================================================================
# VOL ESTIMATORS — every one returns a DAILY series of TQQQ-equivalent
# annualized vol (so 0.30/vol reproduces the base weight).  QQQ-based
# range estimators are x3 to convert to TQQQ vol (leveraged daily range
# scales ~linearly with leverage; std(3x-c) = 3*std, verified below).
# =====================================================================
def v_c2c(win):                                   # close-to-close over `win`
    return tqqq_dret.rolling(win, min_periods=max(8, win*2//3)).std()*SQ

def v_ewma(lam):                                  # RiskMetrics EWMA vol
    var = tqqq_dret.pow(2).ewm(alpha=1-lam, adjust=False).mean()
    v = np.sqrt(var)*SQ
    v[:40] = np.nan                               # warm-up
    return v

def v_parkinson(win):                             # Parkinson high-low range vol (x3)
    lr = np.log(qh/ql)
    var = (lr.pow(2)).rolling(win, min_periods=max(15, win*2//3)).mean()/(4*np.log(2))
    return np.sqrt(var)*SQ*3.0

def v_gk(win):                                    # Garman-Klass (uses O,H,L,C) (x3)
    hl = np.log(qh/ql).pow(2); co = np.log(qqq/qo).pow(2)
    daily = 0.5*hl - (2*np.log(2)-1)*co
    var = daily.rolling(win, min_periods=max(15, win*2//3)).mean()
    var = var.clip(lower=1e-8)
    return np.sqrt(var)*SQ*3.0

def v_max(fast, slow):                            # max(slow, fast) close-to-close vol
    return pd.concat([v_c2c(fast), v_c2c(slow)], axis=1).max(axis=1)

def v_accel(fast=20, slow=63, k=1.5, thr=1.0):    # acceleration overlay on slow
    vf, vs = v_c2c(fast), v_c2c(slow)
    ratio = (vf/vs).clip(lower=1.0)               # only when fast>slow (vol rising)
    return vs*np.power(ratio, k)                  # overshoot slow when accelerating

# --- sanity: base c2c on QQQ*3 == TQQQ c2c (drift constant cancels in std) ---
_a = (qret.rolling(63).std()*SQ*3).dropna(); _b = v_c2c(63).dropna()
_j = _a.index.intersection(_b.index)
assert (abs(_a[_j]-_b[_j]) < 1e-9).all(), "x3 scaling check failed"

# =====================================================================
# WEIGHT BUILDERS
# =====================================================================
def w_from_vol(vol_daily, grid):
    """monthly weight = clip(0.30/vol,0,1) read at PRIOR month-end."""
    vm = vol_daily.reindex(grid, method="ffill")
    return (TARGET/vm).clip(0, 1).shift(1)

def w_asymmetric(grid, fast=21, slow=63, beta=0.34):
    """De-lever FAST (use short-window weight immediately when it drops),
    re-lever SLOW (creep toward the long-window weight with speed beta).
    Hysteresis on the weight itself. Read at prior month-end (shift 1)."""
    wf = (TARGET/v_c2c(fast).reindex(grid, method="ffill")).clip(0, 1)
    ws = (TARGET/v_c2c(slow).reindex(grid, method="ffill")).clip(0, 1)
    out = pd.Series(index=grid, dtype=float); state = np.nan
    for dt in grid:
        tf, ts = wf.get(dt, np.nan), ws.get(dt, np.nan)
        if not np.isfinite(tf) or not np.isfinite(ts):
            out[dt] = np.nan; continue
        if not np.isfinite(state):
            state = ts
        elif tf < state:                # risk rising -> drop immediately (fast)
            state = tf
        else:                            # risk falling -> creep up slowly (slow)
            state = state + beta*(ts - state)
        out[dt] = state
    return out.shift(1)                 # no look-ahead

# =====================================================================
# BACKTEST ENGINE (given grid + monthly weight series)
# =====================================================================
def gret(level, grid):                 # point-to-point monthly gross returns
    return level.reindex(grid, method="ffill").pct_change()

def strat_monthly(wser, grid, cost=0.001):
    tqm = gret(tqqq_lvl, grid); csm = gret(cash_lvl, grid)
    rr = {}; prevw = 0.0
    for dt in grid:
        wt = wser.get(dt, 0.0); wt = float(wt) if np.isfinite(wt) else 0.0
        rt = tqm.get(dt, 0.0); rt = rt if np.isfinite(rt) else 0.0
        rc = csm.get(dt, 0.0); rc = rc if np.isfinite(rc) else 0.0
        rr[dt] = wt*rt + (1-wt)*rc - abs(wt-prevw)*2*cost; prevw = wt
    return pd.Series(rr)

def dca_final(r):
    V = 0.0
    for x in r: V = V*(1+x) + 1000
    return V

def dcaq_final(grid):
    qm = gret(qqq, grid); V = 0.0
    for dt in grid:
        x = qm.get(dt, 0.0); x = x if np.isfinite(x) else 0.0; V = V*(1+x)+1000
    return V

def lstat(r):
    r = r.dropna()
    if len(r) < 3: return np.nan, np.nan, np.nan
    cum = (1+r).cumprod()
    cagr = cum.iloc[-1]**(12/len(r))-1; sh = r.mean()/r.std()*SQ if r.std() else np.nan
    dd = (cum/cum.cummax()-1).min()
    return cagr, sh, dd

def turnover(wser, grid):
    w = wser.reindex(grid).fillna(0.0); return (w.diff().abs()).mean()

# =====================================================================
# ESTIMATOR REGISTRY
# =====================================================================
# each entry -> function(grid) -> monthly weight series (shifted, no look-ahead)
ESTS = {
    "base 63d c2c":        lambda g: w_from_vol(v_c2c(63), g),
    "42d c2c":             lambda g: w_from_vol(v_c2c(42), g),
    "21d c2c":             lambda g: w_from_vol(v_c2c(21), g),
    "EWMA lam0.94":        lambda g: w_from_vol(v_ewma(0.94), g),
    "EWMA lam0.97":        lambda g: w_from_vol(v_ewma(0.97), g),
    "Parkinson 21d":       lambda g: w_from_vol(v_parkinson(21), g),
    "GarmanKlass 21d":     lambda g: w_from_vol(v_gk(21), g),
    "max(63d,21d)":        lambda g: w_from_vol(v_max(21, 63), g),
    "accel 20/63 k1.5":    lambda g: w_from_vol(v_accel(20, 63, 1.5), g),
    "accel 20/63 k2.0":    lambda g: w_from_vol(v_accel(20, 63, 2.0), g),
    "accel 20/63 k2.5":    lambda g: w_from_vol(v_accel(20, 63, 2.5), g),
    "accel 10/63 k2.0":    lambda g: w_from_vol(v_accel(10, 63, 2.0), g),
    "max+accel 20/63 k2":  lambda g: w_from_vol(pd.concat([v_c2c(21), v_accel(20,63,2.0)], axis=1).max(axis=1), g),
    "asym fast21/slow63":  lambda g: w_asymmetric(g),
}

ERAS = [("2003-01","2009-12","2003-09"), ("2010-01","2019-12","2010-19"),
        ("2020-01","2026-06","2020-26")]
FULL = ("1999-04","2026-06")
DOTCOM = ("2000-01","2002-12")

def era_ratio(wfun, a, b):
    s, e = pd.Timestamp(a+"-01"), pd.Timestamp(b+"-01"); g = mg[(mg>=s)&(mg<=e)]
    return dca_final(strat_monthly(wfun(g), g)) / dcaq_final(g)

def dd_over(wfun, a, b):
    s, e = pd.Timestamp(a+"-01"), pd.Timestamp(b+"-01"); g = mg[(mg>=s)&(mg<=e)]
    return lstat(strat_monthly(wfun(g), g))[2]

# ---------- phase robustness: 21-trading-day grids at 5 offsets ----------
def phase_grids(n_off=5):
    grids = []
    for off in np.linspace(0, 20, n_off).astype(int):
        sel = idx[off::21]
        grids.append(pd.DatetimeIndex(sel))
    return grids

def phase_full_dd(wfun):
    dds = []
    for g in phase_grids():
        g = g[(g >= pd.Timestamp("1999-04-01"))]
        dds.append(lstat(strat_monthly(wfun(g), g))[2])
    return min(dds), max(dds)      # worst and best full maxDD across phases

# =====================================================================
# RUN
# =====================================================================
print("QQQ history:", idx[0].date(), "->", idx[-1].date(), " | monthly points:", len(mg))
print("\n"+"="*118)
print(f"{'estimator':20} {'dotcomDD':>9} {'fullDD':>8} {'fullCAGR':>9} {'fullSh':>7} "
      f"{'fullRatio':>10} {'2003-09':>8} {'2010-19':>8} {'2020-26':>8} {'wEraRat':>8} {'turnovr':>8}")
print("-"*118)

rows = {}
for name, wfun in ESTS.items():
    gF = mg[(mg>=pd.Timestamp(FULL[0]+"-01")) & (mg<=pd.Timestamp(FULL[1]+"-01"))]
    rF = strat_monthly(wfun(gF), gF)
    cagr, sh, dd = lstat(rF)
    fullratio = dca_final(rF)/dcaq_final(gF)
    ddom = dd_over(wfun, *DOTCOM)
    eras = {lab: era_ratio(wfun, a, b) for a, b, lab in ERAS}
    werara = min(eras.values())
    tov = turnover(wfun(gF), gF)
    rows[name] = dict(dotcomDD=ddom, fullDD=dd, cagr=cagr, sh=sh, fullratio=fullratio,
                      eras=eras, werara=werara, tov=tov)
    print(f"{name:20} {ddom*100:>8.1f}% {dd*100:>7.1f}% {cagr*100:>8.1f}% {sh:>7.2f} "
          f"{fullratio:>9.2f}x {eras['2003-09']:>7.2f}x {eras['2010-19']:>7.2f}x "
          f"{eras['2020-26']:>7.2f}x {werara:>7.2f}x {tov*100:>7.1f}%")

print("-"*118)
print("dotcomDD = lump-sum max drawdown over 2000-01..2002-12 | fullDD/CAGR/Sh = lump-sum 1999-2026")
print("fullRatio & era cols = DCA final-wealth vs QQQ-DCA | wEraRat = worst of the 3 post-2003 era ratios")
print("turnovr = mean |Δweight| per month")

# ---------- phase robustness for the standouts ----------
print("\n"+"="*70)
print("PHASE ROBUSTNESS (5 rebalance offsets, 21-trading-day grids, 1999-2026)")
print(f"{'estimator':20} {'full maxDD range':>26} {'span':>7}")
print("-"*70)
for name in ["base 63d c2c","21d c2c","42d c2c","EWMA lam0.94","max(63d,21d)",
             "accel 20/63 k1.5","asym fast21/slow63","Parkinson 21d"]:
    lo, hi = phase_full_dd(ESTS[name])
    print(f"{name:20} {lo*100:>11.1f}% .. {hi*100:>7.1f}% {(hi-lo)*100:>6.1f}pp")

# ---------- base reference DCA weight during the crash (does it de-risk?) ----------
print("\n--- base 63d weight vs a fast estimator during the 2000 crash ---")
gcrash = mg[(mg>=pd.Timestamp("2000-01-01")) & (mg<=pd.Timestamp("2001-06-01"))]
wb = ESTS["base 63d c2c"](gcrash); wf = ESTS["21d c2c"](gcrash); wm = ESTS["max(63d,21d)"](gcrash)
print(f"{'month':>10} {'base63 w':>9} {'21d w':>7} {'max w':>7}")
for dt in gcrash:
    print(f"{str(dt.date()):>10} {wb.get(dt,np.nan)*100:>8.0f}% {wf.get(dt,np.nan)*100:>6.0f}% {wm.get(dt,np.nan)*100:>6.0f}%")
