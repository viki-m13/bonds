"""Exp3: lead-lag. (a) SMH momentum leading QQQ (trade TQQQ); (b) HYG leading SPY
(trade UPRO). Baselines: follower's own momentum. Also leader+follower AND gates."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

closes = panel(["SMH", "QQQ", "HYG", "SPY"], "Close")
opens = panel(["TQQQ", "UPRO", "QLD", "SSO", "BIL"], "Open")

def onoff(sig_on, letf, name, w=1.0):
    """sig_on: bool Series (info at close t). Hold letf next day if True, else BIL."""
    s = sig_on.shift(1).fillna(False)
    W = pd.DataFrame(0.0, index=s.index, columns=[letf, "BIL"])
    W[letf] = w * s.astype(float)
    W["BIL"] = 1.0 - W[letf]
    return evaluate(W, opens, name)

results = []
c = closes
for L in (10, 21):
    smh = c["SMH"].pct_change(L) > 0
    qqq = c["QQQ"].pct_change(L) > 0
    hyg = c["HYG"].pct_change(L) > 0
    spy = c["SPY"].pct_change(L) > 0
    results.append(onoff(smh, "TQQQ", f"e3_smh{L}_tqqq")[0])       # leader signal
    results.append(onoff(qqq, "TQQQ", f"e3_qqq{L}_tqqq_base")[0])  # baseline own-mom
    results.append(onoff(smh & qqq, "TQQQ", f"e3_smh&qqq{L}_tqqq")[0])
    results.append(onoff(hyg, "UPRO", f"e3_hyg{L}_upro")[0])
    results.append(onoff(spy, "UPRO", f"e3_spy{L}_upro_base")[0])
    results.append(onoff(hyg & spy, "UPRO", f"e3_hyg&spy{L}_upro")[0])

# HYG with 63d lookback (credit trends slower) and HYG vs its 100dma
hyg63 = c["HYG"].pct_change(63) > 0
hygma = c["HYG"] > c["HYG"].rolling(100).mean()
results.append(onoff(hyg63, "UPRO", "e3_hyg63_upro")[0])
results.append(onoff(hygma, "UPRO", "e3_hygma100_upro")[0])
results.append(onoff(hygma, "SSO", "e3_hygma100_sso")[0])
# SMH lead with 2x instead of 3x
smh21 = c["SMH"].pct_change(21) > 0
results.append(onoff(smh21, "QLD", "e3_smh21_qld")[0])
# relative lead: SMH/QQQ ratio rising (21d) AND QQQ uptrend
rel = (c["SMH"] / c["QQQ"]).pct_change(21) > 0
qqq21 = c["QQQ"].pct_change(21) > 0
results.append(onoff(rel & qqq21, "TQQQ", "e3_relsmh&qqq21_tqqq")[0])

pd.DataFrame(results).to_csv(f"{SCRATCH}/e3_results.csv", index=False)
