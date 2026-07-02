"""Exp1: 52-week-high proximity ranking (George-Hwang) vs plain 252d momentum.
Universe: 12 underlyings ranked, expressed via matched 3x LETFs."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

UMAP = {"QQQ":"TQQQ","SPY":"UPRO","SMH":"SOXL","XLK":"TECL","XLF":"FAS","XLE":"ERX",
        "EEM":"EDC","FXI":"YINN","VNQ":"DRN","USO":"UCO","GLD":"UGL","TLT":"TMF"}
UND = list(UMAP); LETF = list(UMAP.values())

closes = panel(UND, "Close")
opens = panel(LETF + ["BIL"], "Open")

def build(sig, K, step, trend=None, wscheme="eq", bil=True):
    """sig: DataFrame of ranking signal on UND (already info<=close t-1 will be applied
    via shift(1) here). trend: optional bool DataFrame (underlying uptrend filter)."""
    s = sig.shift(1)                       # decision from close[t-1]
    tr = trend.shift(1) if trend is not None else None
    idx = s.index
    reb = monthly_mask(idx, step)
    ranks = s.rank(axis=1, ascending=False)
    sel = ranks <= K
    if tr is not None:
        sel = sel & tr
    if wscheme == "eq":
        Wu = sel.div(K).fillna(0.0)
    else:  # signal-proportional among selected
        sw = s.where(sel)
        sw = sw.sub(sw.min(axis=1), axis=0) + 0.05
        Wu = sw.div(sw.sum(axis=1), axis=0).fillna(0.0)
    Wu = hold_between(Wu, reb)
    W = Wu.rename(columns=UMAP)
    W = W.reindex(columns=LETF + ["BIL"]).fillna(0.0)
    if bil:
        W["BIL"] = (1.0 - W[LETF].sum(axis=1)).clip(lower=0.0)
    return W

mom12_1 = closes.shift(21) / closes.shift(252) - 1.0
mom252 = closes / closes.shift(252) - 1.0
prox = closes / closes.rolling(252).max()
sma200 = closes > closes.rolling(200).mean()
vol63 = closes.pct_change().rolling(63).std()
momvol = (closes / closes.shift(126) - 1.0) / (vol63 * np.sqrt(252))

results = []
for name, sig in [("gh52", prox), ("mom252", mom252), ("mom12_1", mom12_1),
                  ("momvol126", momvol)]:
    for K in (2, 3, 4):
        for step in (5, 21):
            st, _ = evaluate(build(sig, K, step), opens, f"e1_{name}_K{K}_s{step}")
            results.append(st)

# trend-filtered variants (slot to BIL when underlying below 200dma)
for name, sig in [("gh52", prox), ("mom252", mom252)]:
    for K in (2, 3):
        st, _ = evaluate(build(sig, K, 21, trend=sma200), opens, f"e1_{name}tf_K{K}_s21")
        results.append(st)

# GH with proximity threshold: only hold names within 2%/5% of 52w high
for thr in (0.95, 0.98):
    near = prox >= thr
    st, _ = evaluate(build(prox, 3, 21, trend=near), opens, f"e1_gh52thr{int(thr*100)}_K3_s21")
    results.append(st)

pd.DataFrame(results).to_csv(f"{SCRATCH}/e1_results.csv", index=False)
