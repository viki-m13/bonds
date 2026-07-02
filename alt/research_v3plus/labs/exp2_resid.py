"""Exp2: residual (beta-adjusted) momentum on the 12-underlying universe.
Rolling beta of each underlying's daily returns on SPY (252d window),
residual e = r_i - beta*r_spy; rank on sum(resid, L)/std(resid, L)."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

UMAP = {"QQQ":"TQQQ","SPY":"UPRO","SMH":"SOXL","XLK":"TECL","XLF":"FAS","XLE":"ERX",
        "EEM":"EDC","FXI":"YINN","VNQ":"DRN","USO":"UCO","GLD":"UGL","TLT":"TMF"}
UND = list(UMAP); LETF = list(UMAP.values())
closes = panel(UND, "Close")
opens = panel(LETF + ["BIL"], "Open")

r = closes.pct_change()
rm = r["SPY"]
beta = r.rolling(252).cov(rm).div(rm.rolling(252).var(), axis=0)
resid = r - beta.mul(rm, axis=0)

def build(sig, K, step):
    s = sig.shift(1)
    reb = monthly_mask(s.index, step)
    sel = (s.rank(axis=1, ascending=False) <= K)
    Wu = hold_between(sel.div(K).fillna(0.0), reb)
    W = Wu.rename(columns=UMAP).reindex(columns=LETF + ["BIL"]).fillna(0.0)
    W["BIL"] = (1.0 - W[LETF].sum(axis=1)).clip(lower=0.0)
    return W

results = []
for L in (63, 126, 252):
    rs = resid.rolling(L).sum() / resid.rolling(L).std()   # t-stat style resid mom
    for K in (3, 4):
        for step in (5, 21):
            st, _ = evaluate(build(rs, K, step), opens, f"e2_resid{L}_K{K}_s{step}")
            results.append(st)

# raw residual sum (no vol scaling)
for L in (126,):
    rs = resid.rolling(L).sum()
    for K in (3, 4):
        st, _ = evaluate(build(rs, K, 5), opens, f"e2_residraw{L}_K{K}_s5")
        results.append(st)

# blend: average rank of resid-mom(126) and gh52 proximity
prox = closes / closes.rolling(252).max()
rs126 = resid.rolling(126).sum() / resid.rolling(126).std()
blend = prox.rank(axis=1) + rs126.rank(axis=1)
for K in (3, 4):
    for step in (5, 21):
        st, _ = evaluate(build(blend, K, step), opens, f"e2_ghresid_K{K}_s{step}")
        results.append(st)

pd.DataFrame(results).to_csv(f"{SCRATCH}/e2_results.csv", index=False)
