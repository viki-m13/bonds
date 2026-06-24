# Improving TIDE round-2 — different MECHANISMS (honest)

Walk-forward OOS + deflated (14 cumulative trials). base OOS +1.98.

| variant | Sharpe(HL) | IS | OOS | dOOS | CAGR | maxDD | deflated P |
|---|---|---|---|---|---|---|---|
| base TIDE | +2.01 | +2.04 | +1.98 | +0.00 | +32% | -8% | 0.68 |
| multiH | +2.03 | +2.00 | +2.06 | +0.08 | +32% | -9% | 0.71 |
| rank weights | +1.70 | +2.09 | +1.08 | -0.89 | +26% | -11% | 0.29 |
| rank + multiH | +1.87 | +2.26 | +1.23 | -0.75 | +29% | -9% | 0.35 |
| state machine (held) | +1.06 | +1.58 | +0.26 | -1.72 | +19% | -19% | 0.07 |
| state + multiH | +1.42 | +1.17 | +1.81 | -0.17 | +23% | -12% | 0.69 |
| asym long/short | +0.86 | +1.29 | +0.11 | -1.87 | +12% | -16% | 0.05 |

## Best: multiH

- OOS +2.06 (base +1.98, delta +0.08); full HL +2.03; pre-HL +1.02.
- 4-fold WF: +2.31, +0.84, +2.08, +2.90; bootstrap 95% CI [+0.93,+3.00].

## Verdict

- **No mechanism robustly beats base TIDE** (best multiH +2.06 vs +1.98). The base construction is at the honest ceiling for a single x-sectional breakout book; different mechanisms trade turnover/shape but not edge.
- Honest standalone level ~2.1.
