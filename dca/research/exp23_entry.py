"""Experiment 23 — ENTRY-TIMING within a fixed DCA cadence (accepting QQQ).
Cash arrives each cycle (biweekly ~10 td / monthly ~21 td) and MUST be deployed
by cycle end (cash never sits >1 cycle -> bounded drag). Question: does any
achievable trigger beat buying IMMEDIATELY on arrival, on (a) average entry
price and (b) terminal wealth? perfect_low = hindsight ceiling. OOS train/test.
"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
q = yf.download("QQQ", start="1999-03-01", auto_adjust=True, progress=False)["Close"]
q = q.dropna()
if isinstance(q, pd.DataFrame):
    q = q.iloc[:, 0]
c = q.values
n = len(c)
# indicators
sma10 = pd.Series(c).rolling(10).mean().values
d = np.diff(c, prepend=c[0])
up = pd.Series(np.clip(d, 0, None)).rolling(14).mean().values
dn = pd.Series(np.clip(-d, 0, None)).rolling(14).mean().values
rsi = 100 - 100 / (1 + up / (dn + 1e-12))
dates = q.index


def buy_index(blk, rule):
    """blk = array of close indices for the cycle; return chosen index."""
    p = c[blk]; start = p[0]
    if rule == "immediate":
        return blk[0]
    if rule == "end":
        return blk[-1]
    if rule == "perfect_low":
        return blk[int(np.argmin(p))]
    for i in range(1, len(blk)):
        gi = blk[i]
        if rule == "down_day" and c[gi] < c[gi - 1]:
            return gi
        if rule == "below_start" and c[gi] < start:
            return gi
        if rule == "dip2" and c[gi] < np.max(p[:i + 1]) * 0.98:
            return gi
        if rule == "sma_dip" and c[gi] < sma10[gi]:
            return gi
        if rule == "rsi45" and rsi[gi] < 45:
            return gi
    return blk[-1]                                   # forced deploy at cycle end


RULES = ["immediate", "down_day", "below_start", "dip2", "sma_dip", "rsi45",
         "end", "perfect_low"]


def evaluate(N, lo, hi):
    # build consecutive blocks of N trading days within [lo,hi)
    sel = np.where((dates >= pd.Timestamp(lo)) & (dates < pd.Timestamp(hi)))[0]
    sel = sel[sel >= 14]                              # need indicator warmup
    blocks = [sel[k:k + N] for k in range(0, len(sel) - N, N)]
    res = {}
    base_ratios = None
    for rule in RULES:
        ratios = []; shares = 0.0
        for blk in blocks:
            gi = buy_index(blk, rule)
            ratios.append(c[gi] / c[blk[0]])          # entry vs cycle-start price
            shares += 1.0 / c[gi]
        term = shares * c[blocks[-1][-1]]
        res[rule] = (np.mean(ratios), term, np.array(ratios))
    imm_term = res["immediate"][1]
    print(f"\n  cadence={N}td  {lo[:4]}-{hi[:4]}  ({len(blocks)} cycles):", flush=True)
    for rule in RULES:
        mr, term, ratios = res[rule]
        beat = 100 * (ratios < res["immediate"][2]).mean()
        print(f"     {rule:12s} avg entry {mr:.4f}  termWealth/imm {term/imm_term:.4f}"
              f"  (beat-immediate entry {beat:3.0f}% of cycles)", flush=True)


for N in (10, 21):
    evaluate(N, "2002-01-01", "2018-01-01")          # TRAIN
    evaluate(N, "2018-01-01", "2026-07-01")          # TEST
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
