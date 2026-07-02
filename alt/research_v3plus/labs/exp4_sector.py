"""Exp4: sector rotation with breadth-sized inverse hedge.
Top-3 of 9 SPDRs by momentum; XLK->TECL, XLF->FAS, XLE->ERX (3x), others 1x.
Hedge sleeve: PSQ or SH sized h = hmax * breadth (frac of sectors < 200dma).
Long book gets (1-h). All long positions, gross <= 1."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

SECT = ["XLK","XLF","XLE","XLV","XLY","XLP","XLI","XLU","XLB"]
LMAP = {"XLK":"TECL","XLF":"FAS","XLE":"ERX"}
closes = panel(SECT, "Close")
TRADE = ["TECL","FAS","ERX","XLV","XLY","XLP","XLI","XLU","XLB","PSQ","SH","BIL"]
opens = panel(TRADE, "Open")

breadth = (closes < closes.rolling(200).mean()).mean(axis=1)   # frac below 200dma

def build(mom, K, step, hmax, hedge, lev=True):
    s = mom.shift(1)
    b = breadth.shift(1).fillna(0.0)
    reb = monthly_mask(s.index, step)
    sel = (s.rank(axis=1, ascending=False) <= K).astype(float)
    h = hold_between((hmax * b).to_frame("h"), reb)["h"]
    Wu = hold_between(sel.div(K), reb).mul(1.0 - h, axis=0)
    cols = {c: (LMAP.get(c, c) if lev else c) for c in SECT}
    W = Wu.rename(columns=cols).reindex(columns=TRADE).fillna(0.0)
    W[hedge] = h
    W["BIL"] = (1.0 - W.drop(columns="BIL").sum(axis=1)).clip(lower=0.0)
    return W

mom126 = closes.pct_change(126)
mom12_1 = closes.shift(21) / closes.shift(252) - 1.0
prox = closes / closes.rolling(252).max()

results = []
for name, sig in [("mom126", mom126), ("mom12_1", mom12_1), ("gh52", prox)]:
    for hmax in (0.0, 0.3, 0.5):
        for hedge in (["PSQ"] if hmax == 0 else ["PSQ", "SH"]):
            st, _ = evaluate(build(sig, 3, 21, hmax, hedge), opens,
                             f"e4_{name}_h{hedge if hmax else 'no'}{int(hmax*10)}")
            results.append(st)

# unlevered version (plain SPDRs) with hedge — lower beta book
for hmax in (0.0, 0.5):
    st, _ = evaluate(build(mom126, 3, 21, hmax, "SH", lev=False), opens,
                     f"e4_mom126_1x_h{int(hmax*10)}")
    results.append(st)

# weekly rebalance on best config
st, _ = evaluate(build(prox, 3, 5, 0.5, "SH"), opens, "e4_gh52_hSH5_s5")
results.append(st)
st, _ = evaluate(build(mom126, 3, 5, 0.5, "SH"), opens, "e4_mom126_hSH5_s5")
results.append(st)

pd.DataFrame(results).to_csv(f"{SCRATCH}/e4_results.csv", index=False)
