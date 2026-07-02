"""Idea 2: Crash-recovery capture.
Deep drawdown state: SPY 15-40% below 252d high. Stabilization trigger ->
long QLD/TQQQ for a multi-week window. Flat (BIL) otherwise.
Triggers tested: (a) 20d low holding N days, (b) 10d vol falling,
(c) close > 20dma while in DD state.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

TICKS = ["TQQQ", "QLD", "SSO", "BIL"]
opens = panel(TICKS + ["SPY"], "Open")
closes = panel(TICKS + ["SPY"], "Close")
spy = closes["SPY"]

hi252 = spy.rolling(252).max()
dd = spy / hi252 - 1.0
lo20 = spy.rolling(20).min()
vol10 = spy.pct_change().rolling(10).std()
sma20 = spy.rolling(20).mean()

def crash_recovery(dd_lo=-0.40, dd_hi=-0.12, trigger="lowhold", hold=40,
                   asset="QLD", wt=1.0, nhold=5):
    """State: dd in [dd_lo, dd_hi]. Trigger fires -> long `asset` for `hold` days
    (exit early if new 20d low made => stop, back to waiting)."""
    in_dd = (dd <= dd_hi) & (dd >= dd_lo)
    if trigger == "lowhold":
        # 20d low unchanged for nhold days (price stopped making new lows)
        trig = (lo20 == lo20.shift(nhold)) & in_dd
    elif trigger == "volfall":
        trig = (vol10 < vol10.shift(nhold)) & (vol10 < vol10.rolling(nhold*2).max()*0.8) & in_dd
    elif trigger == "sma20":
        trig = (spy > sma20) & in_dd
    sig = pd.Series(0.0, index=CAL)
    cnt = 0
    for i, t in enumerate(CAL):
        if cnt > 0:
            # stop: new 20d low made -> exit
            if spy.get(t, np.nan) <= lo20.shift(1).get(t, np.inf) * 0.999:
                cnt = 0
            else:
                sig.iloc[i] = 1.0
                cnt -= 1
                continue
        if trig.get(t, False):
            cnt = hold
            sig.iloc[i] = 1.0
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = sig * wt
    W["BIL"] = 1.0 - sig * wt
    W = W.shift(1).fillna(0.0)
    return W

grid = [
 ("V2a_lowhold5_QLD_h40",  dict(trigger="lowhold", nhold=5,  hold=40, asset="QLD")),
 ("V2b_lowhold10_QLD_h40", dict(trigger="lowhold", nhold=10, hold=40, asset="QLD")),
 ("V2c_lowhold5_TQQQ_h40", dict(trigger="lowhold", nhold=5,  hold=40, asset="TQQQ", wt=0.7)),
 ("V2d_volfall_QLD_h40",   dict(trigger="volfall", nhold=5,  hold=40, asset="QLD")),
 ("V2e_sma20_QLD_h40",     dict(trigger="sma20",   hold=40, asset="QLD")),
 ("V2f_sma20_QLD_h60",     dict(trigger="sma20",   hold=60, asset="QLD")),
 ("V2g_sma20_TQQQ_h40",    dict(trigger="sma20",   hold=40, asset="TQQQ", wt=0.7)),
 ("V2h_sma20_dd8",         dict(trigger="sma20",   hold=40, asset="QLD", dd_hi=-0.08)),
 ("V2i_lowhold5_dd8",      dict(trigger="lowhold", nhold=5, hold=40, asset="QLD", dd_hi=-0.08)),
]
for name, kw in grid:
    W = crash_recovery(**kw)
    evaluate(W, opens, name)

grid2 = [
 ("V2j_volfall_n3",        dict(trigger="volfall", nhold=3,  hold=40, asset="QLD")),
 ("V2k_volfall_n7",        dict(trigger="volfall", nhold=7,  hold=40, asset="QLD")),
 ("V2l_volfall_h25",       dict(trigger="volfall", nhold=5,  hold=25, asset="QLD")),
 ("V2m_volfall_h60",       dict(trigger="volfall", nhold=5,  hold=60, asset="QLD")),
 ("V2n_volfall_TQQQ07",    dict(trigger="volfall", nhold=5,  hold=40, asset="TQQQ", wt=0.7)),
 ("V2o_volfall_SSO",       dict(trigger="volfall", nhold=5,  hold=40, asset="SSO")),
 ("V2p_volfall_dd10",      dict(trigger="volfall", nhold=5,  hold=40, asset="QLD", dd_hi=-0.10)),
 ("V2q_volfall_dd15",      dict(trigger="volfall", nhold=5,  hold=40, asset="QLD", dd_hi=-0.15)),
]
for name, kw in grid2:
    W = crash_recovery(**kw)
    evaluate(W, opens, name)

W = crash_recovery(trigger="volfall", nhold=5, hold=60, asset="QLD")
evaluate(W, opens, "SAVE_crash_recovery", save="crash_recovery")
