# Rotating across ALL leveraged ETFs — tested, and it's worse than the single sleeve

Per the request to "use all leveraged ETFs, buying monthly and selling when necessary,"
I built and rigorously tested a full-menu leveraged-ETF momentum rotation. **It does not
beat QQQ-DCA over the full period and fails robustness — the concentrated vol-targeted
NASDAQ sleeve (see README) is strictly better.** Script: `scripts/rotation.py`.

## The strategy tested
Universe = all 16 leveraged ETFs (leveraged tech TQQQ/TECL/SOXL/QLD, broad UPRO/SSO/TNA,
sectors FAS/ERX/DRN/LABU, international EDC/YINN, commodity UGL/UCO, bonds TMF). Each month:
hold the top-K by blended 3/6/12-month momentum among names above their 200-day MA *and*
whose underlying is in an uptrend; equal-weight (± vol-targeting); **sell on a 200-day-MA
break**; park unfilled slots in a GLD-TLT defensive blend. Monthly DCA, 10 bps/side, no look-ahead.

## Results (ratio vs QQQ-DCA)
| config | 2006–09 | 2010–14 | 2015–19 | 2020–26 | **2006–26 (continuous)** |
|---|--:|--:|--:|--:|--:|
| all-lev rotate K1 | 0.79 | 1.18 | 0.80 | 1.15 | **0.36** |
| all-lev rotate K3 | 1.21 | 1.17 | 1.10 | 1.12 | **0.90** |
| all-lev rotate K5 | 1.18 | 1.18 | 1.09 | 1.03 | **0.75** |
| all-lev K3 vol-tgt 30% | 1.19 | 1.17 | 1.07 | 1.11 | **0.81** |

The per-era numbers look positive, but those are *isolated windows that each restart the
DCA from zero* (no drawdown carryover). The honest **continuous full-period run loses to
QQQ-DCA (0.75–0.90×)** for every K — the leveraged decay from rotation whipsaw and the
diversification away from leveraged-tech beta more than offset the sub-window gains.

## The kill: phase-robustness
Full-period 2006–26 ratio for the *same* strategy, by rebalance day:
| rebalance on | ratio vs QQQ-DCA | max DD |
|---|--:|--:|
| month-end | 0.66× | −80% |
| day 4 | 1.34× | −65% |
| day 9 | 0.81× | −73% |
| day 14 | 2.42× | −65% |

A 0.66×→2.42× swing on *which day you trade* is the textbook signature of timing-luck, not
signal — the same failure that killed the leveraged-tech **trend-switch** strategy. Combined
with −65% to −80% drawdowns, this is not deployable.

## Why (the mechanism)
**Leverage only pays on an asset with strong, *persistent* positive drift.** 3×-ing NASDAQ
works because tech trends durably upward; 3×-ing gold / oil / emerging markets / financials /
bonds just amplifies choppy, mean-reverting assets, where daily-rebalance decay is a large
negative carry. A momentum rotation keeps buying whichever leveraged sector just ran (UGL,
EDC, SOXL…) right before it reverts — paying the decay tax repeatedly. So the full leveraged
menu is **worse** than concentrating the leverage on the single best-drift asset (NASDAQ) and
managing *its* risk with vol-targeting.

## Verdict
The deployable answer to "which ETF to DCA into, with leverage, to beat QQQ-DCA" is **not**
a rotation across many leveraged ETFs. It is the **vol-targeted leveraged-NASDAQ sleeve**
(`README.md`): beats QQQ-DCA in every era (2.41× full), phase-robust, −47% drawdown. Rotating
the full leveraged menu adds turnover, decay, and phase-luck while *reducing* return.
