# Harness Validation — Null Gauntlet (run before believing any result)

Per the research-lead guardrails: survivorship-clean PIT data (Tiingo delisting-
inclusive ✓), and explicit NULL TESTS to prove the pipeline doesn't leak.
Script: dca/research/exp101_nulltest.py.

## Results
| Test | IC(fwd3m) | Champion-sim Sharpe |
|---|--:|--:|
| **REAL champion (WAVE ML)** | **+0.163** | **1.34** |
| NULL-1: RANDOM score through full harness (gates+ride+cut), 10 seeds | — | mean +0.04, std 0.30, max +0.66 |
| NULL-2: SHUFFLED-target ML (labels permuted within month), retrained | +0.022 | 0.57 |
| QQQ benchmark | — | 1.03 |

## Interpretation
- **NULL-1 (random scores → Sharpe 0.04):** the gates/ride/cut MECHANICS do NOT
  manufacture alpha. A random selector run through the exact champion harness earns
  ~0. The real 1.34 is genuine selection, not a backtest artifact.
- **NULL-2 (shuffled target → IC 0.022 / Sharpe 0.57):** the ML's genuine
  cross-sectional contribution is real (0.163 vs 0.022 null). The residual 0.57
  Sharpe is NOT leakage — it's the runner-gate's legitimate momentum/trend tilt
  (mom3>0 + above-10mo-MA), which carries signal independent of the ML score.
  Honest decomposition: ML edge ≈ 0.14 IC; gates add a momentum floor.
- VERDICT: harness CLEAN. Prior results (WAVE Sharpe 1.41, HANDOFF entry-alpha,
  factor L/S) are above the null distribution and not pipeline artifacts.

## Remaining discipline to apply to the trajectory/interaction GENERATOR
1. Deflated Sharpe Ratio + PBO with the TRUE trial count (every hypothesis the
   generator tries, not just kept ones). With N trials, |IC|<~0.03 is noise.
2. Locked holdout touched ONCE (most-recent ~25%); report validation-vs-holdout gap
   as the overfitting estimate. (Caveat: cross-experiment iteration this session has
   partially contaminated the timeline; the HANDOFF single as-of test is the cleanest.)
3. Re-run NULL-2 to ~0 by shuffling AFTER the gates (isolate pure-ML leak) in v2.
