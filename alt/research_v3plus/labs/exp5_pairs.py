"""Exp5: relative-strength switch pairs, long leader's LETF only if leader in
absolute uptrend (>200dma), else BIL. Pairs: SMH/XLK (SOXL/TECL), EEM/SPY
(EDC/UPRO), QQQ/IWM (TQQQ/IWM 1x), FXI/EEM (YINN/EDC).
Plus robustness sweep of the exp3 HYG gate."""
import sys; sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from lib import *

UND = ["SMH","XLK","EEM","SPY","QQQ","IWM","FXI","HYG"]
closes = panel(UND, "Close")
TR = ["SOXL","TECL","EDC","UPRO","TQQQ","IWM","YINN","SSO","BIL"]
opens = panel(TR, "Open")

def pair(a, b, la, lb, L, name, step=5, w=1.0):
    ca, cb = closes[a], closes[b]
    rel = (ca / cb).pct_change(L)
    upa = ca > ca.rolling(200).mean()
    upb = cb > cb.rolling(200).mean()
    sa = ((rel > 0) & upa).shift(1).fillna(False)
    sb = ((rel <= 0) & upb).shift(1).fillna(False)
    reb = monthly_mask(closes.index, step)
    W = pd.DataFrame(0.0, index=closes.index, columns=TR)
    W[la] = w * sa.astype(float); W[lb] = W[lb] + w * sb.astype(float)
    W = hold_between(W, reb)
    W["BIL"] = (1.0 - W.drop(columns="BIL").sum(axis=1)).clip(lower=0.0)
    return evaluate(W, opens, name)

results = []
for L in (63, 126):
    results.append(pair("SMH","XLK","SOXL","TECL",L,f"e5_smhxlk{L}")[0])
    results.append(pair("EEM","SPY","EDC","UPRO",L,f"e5_eemspy{L}")[0])
    results.append(pair("QQQ","IWM","TQQQ","IWM",L,f"e5_qqqiwm{L}")[0])
    results.append(pair("FXI","EEM","YINN","EDC",L,f"e5_fxieem{L}")[0])

# combined 3-pair book, 1/3 each
def pairW(a,b,la,lb,L,step=5):
    ca, cb = closes[a], closes[b]
    rel = (ca/cb).pct_change(L)
    upa = ca > ca.rolling(200).mean(); upb = cb > cb.rolling(200).mean()
    sa = ((rel>0)&upa).shift(1).fillna(False); sb = ((rel<=0)&upb).shift(1).fillna(False)
    W = pd.DataFrame(0.0, index=closes.index, columns=TR)
    W[la]=sa.astype(float); W[lb]=W[lb]+sb.astype(float)
    return hold_between(W, monthly_mask(closes.index, step))
Wc = (pairW("SMH","XLK","SOXL","TECL",126)
      + pairW("EEM","SPY","EDC","UPRO",126)
      + pairW("QQQ","IWM","TQQQ","IWM",126)) / 3.0
Wc["BIL"] = (1.0 - Wc.drop(columns="BIL").sum(axis=1)).clip(lower=0.0)
results.append(evaluate(Wc, opens, "e5_3pair126")[0])

# HYG gate robustness sweep (exp3 follow-up)
for ma in (50, 80, 100, 120, 150, 200):
    g = (closes["HYG"] > closes["HYG"].rolling(ma).mean()).shift(1).fillna(False)
    for letf in ("UPRO","SSO"):
        W = pd.DataFrame(0.0, index=closes.index, columns=[letf,"BIL"])
        W[letf] = g.astype(float); W["BIL"] = 1.0 - W[letf]
        results.append(evaluate(W, opens, f"e5_hygma{ma}_{letf.lower()}")[0])

pd.DataFrame(results).to_csv(f"{SCRATCH}/e5_results.csv", index=False)
