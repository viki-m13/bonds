"""Idea 3: Dispersion regime gating an LETF momentum book.
Dispersion = cross-sectional std of 21d returns across 9 XL* sectors,
percentile over trailing 756d. Base book: top-2 LETFs by blended momentum
(63d+126d), require positive momentum, 0.35 each, rest BIL.
Gate tests: trade only in high-dispersion tercile / low / mid, and scaling.
"""
import sys
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from common import *

SECT = ["XLK","XLE","XLF","XLV","XLI","XLP","XLU","XLY","XLB"]
LETFS = ["TQQQ","SOXL","TECL","FAS","ERX","EDC","YINN","DRN","UGL","TMF"]
TICKS = LETFS + ["BIL"]
opens = panel(TICKS, "Open")
closes = panel(TICKS, "Close")
sect_c = panel(SECT, "Close")

disp = sect_c.pct_change(21).std(axis=1)
disp_pct = disp.rolling(756).apply(lambda x: (x[-1] >= x).mean(), raw=True)

mom = 0.5 * closes[LETFS].pct_change(63) + 0.5 * closes[LETFS].pct_change(126)

def mom_book(top=2, wt=0.35, rebal=5):
    """Weekly-rebalanced top-N positive-momentum LETF book (signal at close t-1)."""
    W = pd.DataFrame(0.0, index=CAL, columns=TICKS)
    last = None
    for i, t in enumerate(CAL):
        if i % rebal == 0:
            m = mom.loc[:t].iloc[-1] if t in mom.index else None
            row = pd.Series(0.0, index=TICKS)
            if m is not None:
                mm = m.dropna()
                mm = mm[mm > 0].sort_values(ascending=False)
                for tk in mm.index[:top]:
                    row[tk] = wt
            row["BIL"] = 1.0 - row.sum()
            last = row
        W.iloc[i] = last if last is not None else 0.0
    return W.shift(1).fillna(0.0)   # decision lag

base = mom_book()

def gated(Wb, mode, lo=1/3, hi=2/3):
    g = pd.Series(0.0, index=CAL)
    if mode == "hi":   g[disp_pct >= hi] = 1.0
    elif mode == "lo": g[disp_pct <= lo] = 1.0
    elif mode == "mid": g[(disp_pct > lo) & (disp_pct < hi)] = 1.0
    elif mode == "hilo_scale":  # scale linearly with dispersion percentile
        g = disp_pct.clip(0, 1).fillna(0.0)
    g = g.shift(1).fillna(0.0)  # gate also decided at close t-1
    W = Wb.mul(g, axis=0)
    W["BIL"] = W["BIL"] * 0 + (1.0 - W.drop(columns="BIL").sum(axis=1))
    return W

evaluate(base, opens, "V3a_mom_uncond")
for mode in ["hi", "mid", "lo", "hilo_scale"]:
    evaluate(gated(base, mode), opens, f"V3_{mode}")

# variant: top-1, and 21d dispersion of raw levels vs percentile threshold 0.5
base1 = mom_book(top=1, wt=0.5)
evaluate(base1, opens, "V3b_top1_uncond")
evaluate(gated(base1, "hi"), opens, "V3b_top1_hi")
evaluate(gated(base1, "hi", hi=0.5), opens, "V3b_top1_hi50")
evaluate(gated(base, "hi", hi=0.5), opens, "V3c_top2_hi50")

# smooth the gate: 10d mean of dispersion percentile, weekly gate updates
def gated_smooth(Wb, thresh=1/3, smooth=10, rebal=5, invert=False):
    dp = disp_pct.rolling(smooth).mean()
    raw = (dp >= thresh) if invert else (dp <= thresh)
    g = pd.Series(np.nan, index=CAL)
    for i in range(0, len(CAL), rebal):
        g.iloc[i] = 1.0 if raw.iloc[i] else 0.0
    g = g.ffill().fillna(0.0).shift(1).fillna(0.0)
    W = Wb.mul(g, axis=0)
    W["BIL"] = 1.0 - W.drop(columns="BIL").sum(axis=1)
    return W

for th in [1/3, 0.5]:
    for sm in [5, 10, 21]:
        evaluate(gated_smooth(base, thresh=th, smooth=sm),
                 opens, f"V3d_lo_sm{sm}_th{th:.2f}")

evaluate(gated_smooth(base, thresh=1/3, smooth=5), opens,
         "SAVE_disp_lo_mom", save="disp_lo_mom")
