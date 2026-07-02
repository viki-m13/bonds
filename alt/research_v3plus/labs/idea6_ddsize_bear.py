"""Idea 6: drawdown-depth-conditional sizing.
Above 20dma (short-term stabilized), size QLD anticyclically with SPY drawdown
depth: w = clip(base + k*|dd|, 0, cap). Deeper dip + stabilization -> bigger size.
Idea 7: bear-state inverse book: SPY<200dma & vol rising -> SQQQ/TMF book.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

TICKS = ["QLD","TQQQ","SQQQ","SDS","TMF","BIL"]
opens = panel(TICKS + ["SPY"], "Open")
closes = panel(TICKS + ["SPY"], "Close")
spy = closes["SPY"]
dd = spy / spy.rolling(252).max() - 1.0
sma20 = spy.rolling(20).mean()
sma200 = spy.rolling(200).mean()
vol21 = spy.pct_change().rolling(21).std() * np.sqrt(252)

# ---- Idea 6: anticyclical dd sizing ----
def dd_size(base=0.3, k=3.0, cap=1.0, asset="QLD", floor_dd=0.0):
    w = (base + k * (-dd).clip(lower=floor_dd)).clip(0, cap)
    w[spy <= sma20] = 0.0          # stabilization qualifier
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = w
    W["BIL"] = 1.0 - w
    return W.shift(1).fillna(0.0)

for nm, kw in [
    ("V6a_b30_k3_QLD",  dict()),
    ("V6b_b20_k4_QLD",  dict(base=0.2, k=4.0)),
    ("V6c_b40_k2_QLD",  dict(base=0.4, k=2.0)),
    ("V6d_b30_k3_TQQQ", dict(asset="TQQQ", cap=0.7)),
    ("V6e_b0_k5_QLD",   dict(base=0.0, k=5.0)),   # pure dip-buyer
    ("V6f_b30_k0_QLD",  dict(base=0.3, k=0.0)),   # control: no dd conditioning
]:
    evaluate(dd_size(**kw), opens, nm)

# smooth the 20dma gate with 3d persistence to cut whipsaw
def dd_size2(base=0.3, k=3.0, cap=1.0, asset="QLD", persist=3):
    above = (spy > sma20).rolling(persist).min().astype(bool)
    below = (spy <= sma20).rolling(persist).min().astype(bool)
    gate = pd.Series(np.nan, index=CAL)
    gate[above] = 1.0; gate[below] = 0.0
    gate = gate.ffill().fillna(0.0)
    w = ((base + k * (-dd)).clip(0, cap)) * gate
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = w
    W["BIL"] = 1.0 - w
    return W.shift(1).fillna(0.0)

for nm, kw in [
    ("V6g_persist3",  dict()),
    ("V6h_persist5",  dict(persist=5)),
    ("V6i_p3_b20k4",  dict(base=0.2, k=4.0, persist=3)),
]:
    evaluate(dd_size2(**kw), opens, nm)

# ---- Idea 7: bear inverse book ----
def bear_book(book, vol_rise=True):
    m = (spy < sma200)
    if vol_rise:
        m = m & (vol21 > vol21.shift(10))
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    for tk, w in book.items():
        W.loc[m, tk] = w
    W["BIL"] = 1.0 - W.drop(columns="BIL").sum(axis=1)
    return W.shift(1).fillna(0.0)

evaluate(bear_book({"SQQQ":0.25, "TMF":0.25}), opens, "V7a_sqqq_tmf")
evaluate(bear_book({"SQQQ":0.3}), opens, "V7b_sqqq_only")
evaluate(bear_book({"SDS":0.3, "TMF":0.3}), opens, "V7c_sds_tmf")
evaluate(bear_book({"SQQQ":0.25, "TMF":0.25}, vol_rise=False), opens, "V7d_novolcond")
