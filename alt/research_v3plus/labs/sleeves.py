"""Calendar sleeve backtests. IS metrics only ever printed."""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
import numpy as np, pandas as pd
from cal_lib import *

opens, closes, cal = load_panel()
phx = load_phoenix()
fwd, bwd = month_session_offsets(cal)
pre1 = holiday_flags(cal)
pre2 = pre1.shift(-1).fillna(False)
post1 = pre1.shift(1).fillna(False)
dow = pd.Series(cal.dayofweek, index=cal)
mth = pd.Series(cal.month, index=cal)
tom = (fwd <= 3) | (bwd >= -4)
sma = closes["QQQ"].rolling(200).mean()
up = (closes["QQQ"] > sma).shift(1).astype("boolean").fillna(False).astype(bool)

# OpEx / post-OpEx weeks
opex_wk = pd.Series(False, index=cal); post_opex_wk = pd.Series(False, index=cal)
for p in pd.PeriodIndex(cal, freq="M").unique():
    f3 = third_friday(p.year, p.month)
    opex_wk[opex_wk.index.isin(week_of(cal, f3))] = True
    post_opex_wk[post_opex_wk.index.isin(week_of(cal, f3 + pd.Timedelta(days=7)))] = True

# QQQ gap known at open[t] -> drives W[t+1]
gap = (opens["QQQ"] / closes["QQQ"].shift(1) - 1)
gap_up1 = (gap > 0.01).shift(1).astype("boolean").fillna(False).astype(bool)
gap_dn1 = (gap < -0.01).shift(1).astype("boolean").fillna(False).astype(bool)

def sleeve(mask, asset="TQQQ", w=1.0):
    W = blank_W(cal, [asset, "BIL"])
    W.loc[mask.reindex(cal).fillna(False), asset] = w
    return W

results = {}
def ev(name, W):
    res = run(W, opens)
    results[name] = (is_stats(res, name, phx), res)
    return res

print("--- Day-of-week ---")
ev("tue_tqqq            (Tue hold)", sleeve(dow == 1))
ev("tue_qld             (Tue hold)", sleeve(dow == 1, "QLD"))
ev("tuewed_tqqq         (Tue+Wed)", sleeve(dow.isin([1, 2])))
ev("tuethu_up_tqqq      (Tue-Thu in uptrend)", sleeve(dow.isin([1, 2, 3]) & up))
ev("tue_dn_tqqq         (Tue only downtrend)", sleeve((dow == 1) & ~up))
ev("tue_extom_tqqq      (Tue ex-TOM)", sleeve((dow == 1) & ~tom))

print("--- Holiday ---")
ev("prehol1_tqqq        (session before hol)", sleeve(pre1))
ev("prehol2_tqqq        (2 sessions before)", sleeve(pre1 | pre2))
ev("posthol_tqqq        (session after hol)", sleeve(post1))
ev("hol_straddle_tqqq   (pre1+post1)", sleeve(pre1 | post1))
ev("posthol_qld", sleeve(post1, "QLD"))

print("--- OpEx ---")
ev("opexwk_tqqq         (OpEx week)", sleeve(opex_wk))
ev("postopex_tqqq       (week after OpEx)", sleeve(post_opex_wk))
ev("postopex_extom_tqqq", sleeve(post_opex_wk & ~tom))

print("--- Month seasonality ---")
ev("halloween_tqqq      (Nov-Apr)", sleeve(mth.isin([11, 12, 1, 2, 3, 4])))
ev("skipsep_tqqq        (all but Sep)", sleeve(mth != 9))
ev("skipaugsep_tqqq     (all but Aug-Sep)", sleeve(~mth.isin([8, 9])))
ev("jul_tqqq            (Jul only)", sleeve(mth == 7))

print("--- Quarter turn ---")
qs = mth.isin([1, 4, 7, 10])
qe = mth.isin([3, 6, 9, 12])
ev("qtrext_tqqq         (fwd+4,+5 Qstart)", sleeve(qs & fwd.isin([4, 5])))
ev("qtrpre_tqqq         (bwd-6,-5 Qend)", sleeve(qe & bwd.isin([-6, -5])))

print("--- Gap conditioning (t+1) ---")
ev("gapfollow_tqqq      (QQQ gap>+1% prev d)", sleeve(gap_up1))
ev("gapfade_tqqq        (QQQ gap<-1% prev d)", sleeve(gap_dn1))

print("--- TMF month-end ---")
ev("tmf_eom32           (bwd -3,-2)", sleeve(bwd.isin([-3, -2]), "TMF"))
ev("tmf_eom321          (bwd -3..-1)", sleeve(bwd.isin([-3, -2, -1]), "TMF"))
ev("tmf_eom2            (bwd -2)", sleeve(bwd == -2, "TMF"))

print("--- Blend: Tue|post-holiday TQQQ, TMF on EOM-3/-2 else ---")
Wb = blank_W(cal, ["TQQQ", "TMF", "BIL"])
eq_days = ((dow == 1) | post1).reindex(cal).fillna(False)
tmf_days = bwd.isin([-3, -2]).reindex(cal).fillna(False) & ~eq_days
Wb.loc[eq_days, "TQQQ"] = 1.0
Wb.loc[tmf_days, "TMF"] = 1.0
ev("cal_blend           (Tue+posthol / TMF eom)", Wb)

import json
summ = {k: {kk: vv for kk, vv in v[0].items() if kk != "name"} for k, v in results.items()}
with open("sleeve_summary.json", "w") as f:
    json.dump(summ, f, indent=1, default=float)
