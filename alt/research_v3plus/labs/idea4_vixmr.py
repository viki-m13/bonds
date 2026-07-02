"""Idea 4: VIX mean-reversion post-panic state.
VIX 60d z-score > Z then falling (VIX < VIX max over last k days by margin)
-> long equity LETF for N days. Rare-event sleeve, BIL otherwise.
FRED VIXCLS is published same-day (close); use with shift(1) like everything else.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

TICKS = ["TQQQ", "QLD", "SSO", "SVXY", "BIL"]
opens = panel(TICKS, "Open")
vix = load_fred("VIXCLS").reindex(CAL).ffill()

z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()

def vix_mr(zthr=2.0, fall_k=3, fall_frac=0.90, hold=15, asset="QLD", wt=1.0,
           stopz=None):
    """Panic armed when z>zthr; fire when vix < fall_frac * max(vix, last fall_k d)
    while still elevated (z>0.5). Hold `hold` days; re-arm allowed."""
    armed = False
    sig = pd.Series(0.0, index=CAL)
    cnt = 0
    vmax = vix.rolling(fall_k).max()
    for i, t in enumerate(CAL):
        zt = z.get(t, np.nan)
        if cnt > 0:
            sig.iloc[i] = 1.0
            cnt -= 1
            if stopz is not None and zt > stopz:  # re-panic stop
                cnt = 0; sig.iloc[i] = 0.0; armed = True
            continue
        if not armed and zt > zthr:
            armed = True
        if armed and not np.isnan(zt):
            if vix[t] < fall_frac * vmax.get(t, np.inf) and zt > 0.5:
                cnt = hold
                sig.iloc[i] = 1.0
                armed = False
        if armed and zt < 0:
            armed = False  # panic faded without trigger
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = sig * wt
    W["BIL"] = 1.0 - sig * wt
    return W.shift(1).fillna(0.0)

grid = [
 ("V4a_z2_f90_h15_QLD",   dict()),
 ("V4b_z2_f90_h10_QLD",   dict(hold=10)),
 ("V4c_z2_f90_h21_QLD",   dict(hold=21)),
 ("V4d_z2_f85_h15_QLD",   dict(fall_frac=0.85)),
 ("V4e_z25_f90_h15_QLD",  dict(zthr=2.5)),
 ("V4f_z15_f90_h15_QLD",  dict(zthr=1.5)),
 ("V4g_z2_f90_h15_TQQQ",  dict(asset="TQQQ", wt=0.7)),
 ("V4h_z2_f90_h15_SSO",   dict(asset="SSO")),
 ("V4i_z2_f90_h15_stop",  dict(stopz=2.5)),
 ("V4j_z2_f90_h21_TQQQ",  dict(asset="TQQQ", wt=0.7, hold=21)),
]
for name, kw in grid:
    W = vix_mr(**kw)
    evaluate(W, opens, name)

grid2 = [
 ("V4k_z25_h21",      dict(zthr=2.5, hold=21)),
 ("V4l_z25_h10",      dict(zthr=2.5, hold=10)),
 ("V4m_z25_f85",      dict(zthr=2.5, fall_frac=0.85)),
 ("V4n_z25_TQQQ07",   dict(zthr=2.5, asset="TQQQ", wt=0.7)),
 ("V4o_z225_h15",     dict(zthr=2.25)),
 ("V4p_z25_f95",      dict(zthr=2.5, fall_frac=0.95)),
]
for name, kw in grid2:
    evaluate(vix_mr(**kw), opens, name)

evaluate(vix_mr(zthr=2.5, hold=21), opens, "SAVE_vix_postpanic", save="vix_postpanic")
