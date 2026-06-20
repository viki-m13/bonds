# STRATA OOS-robustness lab (HL crypto)

Techniques to lift STRATA's OOS Sharpe, kept only if they help. HL era, net, IS=first60/OOS=last40.

| technique | Sharpe | IS | OOS | maxDD |
|---|---|---|---|---|
| BASELINE (fixed shrunk-MV) | **+1.58** | +1.37 | +1.85 | -11% |
| WF-ADAPT | **+0.86** | +0.44 | +1.43 | -17% |
| WF + VOLOFVOL | **+0.62** | +0.26 | +1.15 | -20% |
| WF + HURST regime | **+0.98** | +0.50 | +1.63 | -17% |
| ALL combined | **+0.74** | +0.31 | +1.38 | -20% |

## Verdict

- Baseline STRATA OOS +1.85. Best OOS improvement: **BASELINE (fixed shrunk-MV)** -> OOS +1.85 (+0.00 vs baseline). The overlays don't reliably lift OOS — STRATA's shrunk-MV is already near its robust OOS; walk-forward adapts but the recent regime caps it.
- These are honest, causal, anti-overfit techniques (walk-forward, vol-of-vol de-risking, regime tilt). If your screenshot shows a specific indicator instead, name it and I'll slot it in here.
