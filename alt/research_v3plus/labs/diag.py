"""Diagnostics on IS window only: raw conditional mean returns for holding day d
(hold_ret[d] = open[d]->open[d+1] o2o of TQQQ), pre-cost."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
import numpy as np, pandas as pd
from cal_lib import *

opens, closes, cal = load_panel()
o2o = opens.pct_change()
hold = o2o["TQQQ"].shift(-1)                    # return of holding day d
IS = (cal >= IS_START) & (cal <= IS_END)
h = hold[IS].dropna()
idx = h.index

def show(mask, label):
    a = h[mask.reindex(idx).fillna(False)]
    b1 = a.loc[:SPLIT]; b2 = a.loc[SPLIT:]
    t = a.mean() / (a.std() / np.sqrt(len(a))) if len(a) > 5 else np.nan
    print(f"{label:28s} n={len(a):4d}  mean={a.mean()*1e4:7.1f}bp  t={t:5.2f}  "
          f"| 10-14 {b1.mean()*1e4:7.1f}bp  15-18 {b2.mean()*1e4:7.1f}bp")

print("=== sanity: holidays detected (IS) ===")
pre1 = holiday_flags(cal)
print("count pre-holiday sessions IS:", int(pre1[IS].sum()), "(expect ~9/yr * 8.8yr ~ 79)")
print(pre1[IS][pre1[IS]].index[:12].strftime("%Y-%m-%d").tolist())

print("\n=== day of week (holding day) ===")
for dw, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
    show(pd.Series(idx.dayofweek == dw, index=idx), nm)

print("\n=== month of year (holding day) ===")
for m in range(1, 13):
    show(pd.Series(idx.month == m, index=idx), f"month {m:2d}")

print("\n=== pre-holiday ===")
show(pre1, "pre-holiday d (session before)")
pre2 = pre1.shift(-1).fillna(False)             # two sessions before holiday
show(pre2, "pre-holiday d-1")
post1 = pre1.shift(1).fillna(False)             # first session after holiday
show(post1, "post-holiday d+1")
show(~(pre1 | pre2 | post1), "all other days")

print("\n=== OpEx week (week of 3rd Friday) ===")
months = pd.PeriodIndex(idx, freq="M").unique()
opex_days = set()
post_opex_days = set()
for p in months:
    f3 = third_friday(p.year, p.month)
    opex_days |= set(week_of(cal, f3))
    post_opex_days |= set(week_of(cal, f3 + pd.Timedelta(days=7)))
show(pd.Series(idx.isin(list(opex_days)), index=idx), "OpEx week")
show(pd.Series(idx.isin(list(post_opex_days)), index=idx), "post-OpEx week")
show(pd.Series(~idx.isin(list(opex_days | post_opex_days)), index=idx), "other weeks")

print("\n=== month-turn offsets (fwd 1..5, bwd -5..-1), quarter vs non-quarter ===")
fwd, bwd = month_session_offsets(cal)
# quarter-turn month boundary: month m ends and m+1 starts a new quarter
mth = pd.Series(idx.month, index=idx)
q_end_month = mth.isin([3, 6, 9, 12])           # for bwd offsets (end of quarter)
q_start_month = mth.isin([1, 4, 7, 10])         # for fwd offsets (start of quarter)
for off in range(-8, 0):
    m = pd.Series(bwd.reindex(idx) == off, index=idx)
    show(m & q_end_month, f"bwd {off} Qend")
    show(m & ~q_end_month, f"bwd {off} nonQ")
for off in range(1, 7):
    m = pd.Series(fwd.reindex(idx) == off, index=idx)
    show(m & q_start_month, f"fwd +{off} Qstart")
    show(m & ~q_start_month, f"fwd +{off} nonQ")

print("\n=== DOW x trend (QQQ>200sma, known at close t-1) ===")
sma = closes["QQQ"].rolling(200).mean()
up = (closes["QQQ"] > sma).shift(1).fillna(False)   # usable for holding day t
for dw, nm in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri"]):
    show(pd.Series(idx.dayofweek == dw, index=idx) & up.reindex(idx), nm + " up")
    show(pd.Series(idx.dayofweek == dw, index=idx) & ~up.reindex(idx), nm + " dn")

print("\n=== QQQ gap at open[t] -> hold day t+1 ===")
gap = (opens["QQQ"] / closes["QQQ"].shift(1) - 1)
for thr in [0.003, 0.005, 0.01]:
    gu = (gap > thr).shift(1).fillna(False)     # gap day t, hold day t+1
    gd = (gap < -thr).shift(1).fillna(False)
    show(gu.reindex(idx), f"gap up>{thr*100:.1f}% next d")
    show(gd.reindex(idx), f"gap dn<-{thr*100:.1f}% next d")

print("\n=== TMF hold day by month offset (bond EOM) ===")
htmf = o2o["TMF"].shift(-1)[IS].dropna()
i2 = htmf.index
for off in [-3, -2, -1, 1, 2]:
    m = (bwd.reindex(i2) == off) if off < 0 else (fwd.reindex(i2) == off)
    a = htmf[m.fillna(False)]
    t = a.mean() / (a.std() / np.sqrt(len(a)))
    print(f"TMF off {off:+d}: n={len(a):3d} mean={a.mean()*1e4:6.1f}bp t={t:5.2f} "
          f"| 10-14 {a.loc[:SPLIT].mean()*1e4:6.1f} 15-18 {a.loc[SPLIT:].mean()*1e4:6.1f}")
a = htmf[(bwd.reindex(i2) >= -3) | (fwd.reindex(i2) <= 0)]
print("TMF all days mean bp:", round(htmf.mean() * 1e4, 1))
