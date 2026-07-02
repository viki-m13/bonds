"""Idea 8: post-panic carry-capture sleeves.
8A: VIX z>2 then falling -> long SVXY (short-vol premium richest post-panic).
    SVXY data starts 2011-10; sleeve holds BIL before that.
8B: HY OAS 120d z>1.5 then tightening -> long HYG for N days (credit snapback).
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

TICKS = ["SVXY","HYG","QLD","BIL"]
opens = panel(TICKS, "Open")
vix = load_fred("VIXCLS").reindex(CAL).ffill()
oas = load_fred("BAMLH0A0HYM2").reindex(CAL).ffill()

def panic_then_fall(series, zwin, zthr, fall_k, fall_frac, hold, lag):
    z = (series - series.rolling(zwin).mean()) / series.rolling(zwin).std()
    vmax = series.rolling(fall_k).max()
    armed, cnt = False, 0
    sig = pd.Series(0.0, index=CAL)
    for i, t in enumerate(CAL):
        zt = z.get(t, np.nan)
        if cnt > 0:
            sig.iloc[i] = 1.0; cnt -= 1; continue
        if not armed and zt > zthr: armed = True
        if armed and not np.isnan(zt):
            if series[t] < fall_frac * vmax.get(t, np.inf) and zt > 0.5:
                cnt = hold; sig.iloc[i] = 1.0; armed = False
            elif zt < 0:
                armed = False
    return sig.shift(lag).fillna(0.0)

def wrap(sig, asset, wt):
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    W[asset] = sig * wt
    W["BIL"] = 1.0 - sig * wt
    return W

# 8A SVXY (VIX observable at close -> lag 1)
for nm, kw, wt in [
    ("V8Aa_svxy_z2_h15_w03",  dict(zwin=60, zthr=2.0, fall_k=3, fall_frac=0.90, hold=15, lag=1), 0.3),
    ("V8Ab_svxy_z25_h21_w03", dict(zwin=60, zthr=2.5, fall_k=3, fall_frac=0.90, hold=21, lag=1), 0.3),
    ("V8Ac_svxy_z2_h15_w05",  dict(zwin=60, zthr=2.0, fall_k=3, fall_frac=0.90, hold=15, lag=1), 0.5),
    ("V8Ad_svxy_z25_h30_w03", dict(zwin=60, zthr=2.5, fall_k=3, fall_frac=0.90, hold=30, lag=1), 0.3),
]:
    evaluate(wrap(panic_then_fall(vix, **kw), "SVXY", wt), opens, nm)

# 8B HYG credit snapback (OAS published with a lag -> lag 2)
for nm, kw, wt in [
    ("V8Ba_hyg_z15_h40",  dict(zwin=120, zthr=1.5, fall_k=5, fall_frac=0.97, hold=40, lag=2), 1.0),
    ("V8Bb_hyg_z20_h40",  dict(zwin=120, zthr=2.0, fall_k=5, fall_frac=0.97, hold=40, lag=2), 1.0),
    ("V8Bc_hyg_z15_h60",  dict(zwin=120, zthr=1.5, fall_k=5, fall_frac=0.97, hold=60, lag=2), 1.0),
    ("V8Bd_hyg_z15_h40_f95", dict(zwin=120, zthr=1.5, fall_k=10, fall_frac=0.95, hold=40, lag=2), 1.0),
    ("V8Be_qld_z15_h40",  dict(zwin=120, zthr=1.5, fall_k=5, fall_frac=0.97, hold=40, lag=2), 0.6),
]:
    a = "QLD" if nm.startswith("V8Be") else "HYG"
    evaluate(wrap(panic_then_fall(oas, **kw), a, wt), opens, nm)

# robustness probe around V8Ad
for nm, kw, wt in [
    ("V8Ae_z225_h30", dict(zwin=60, zthr=2.25, fall_k=3, fall_frac=0.90, hold=30, lag=1), 0.3),
    ("V8Af_z275_h30", dict(zwin=60, zthr=2.75, fall_k=3, fall_frac=0.90, hold=30, lag=1), 0.3),
    ("V8Ag_z25_h25",  dict(zwin=60, zthr=2.5,  fall_k=3, fall_frac=0.90, hold=25, lag=1), 0.3),
    ("V8Ah_z25_h35",  dict(zwin=60, zthr=2.5,  fall_k=3, fall_frac=0.90, hold=35, lag=1), 0.3),
    ("V8Ai_z25_h30_w025", dict(zwin=60, zthr=2.5, fall_k=3, fall_frac=0.90, hold=30, lag=1), 0.25),
    ("V8Aj_z25_h30_f85",  dict(zwin=60, zthr=2.5, fall_k=3, fall_frac=0.85, hold=30, lag=1), 0.3),
    ("V8Ak_z25_h30_zw90", dict(zwin=90, zthr=2.5, fall_k=3, fall_frac=0.90, hold=30, lag=1), 0.3),
]:
    evaluate(wrap(panic_then_fall(vix, **kw), "SVXY", wt), opens, nm)

evaluate(wrap(panic_then_fall(vix, zwin=60, zthr=2.5, fall_k=3, fall_frac=0.90,
        hold=30, lag=1), "SVXY", 0.3), opens, "SAVE_svxy_postpanic", save="svxy_postpanic")
