# Momentum / Trend / 52-Week-High family — results

Builders: `research/signals_momentum.py` (all pass `audit.audit_builder`
truncation tests: mom_12_1, mom_ret189, mom_clenow252, mom_ma_align,
mom_dist200, mom_trend_tstat126, mom_high52_fresh, mom_x_smooth audited,
max|Δ|=0, no NaN mismatches). Eval: `protocol.evaluate_signal`, biweekly
(every=10) unless noted, 5 bps, full 244-window grid. Date run: 2026-06-12.

## Full sweep, k=3 biweekly

| signal | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | weak regimes (vs QQQ) |
|---|---|---|---|---|---|
| mom_ret63 | 62% | 82% | +7.6% | -37.2% | GFC, recovery09-12, 2022 |
| mom_ret126 (baseline) | 73% | 88% | +14.0% | -39.7% | GFC -21%, rec09-12 -25%, 2022 -16% |
| mom_ret189 | 76% | 88% | +17.3% | -41.3% | GFC -20%, rec09-12 -28%, 2022 -18% |
| mom_ret252 | 73% | 87% | +16.9% | -36.4% | same pattern |
| **mom_12_1** (252d skip 21) | **80%** | **91%** | **+24.6%** | **-30.9%** | GFC -14%, rec09-12 -27%, 2022 -18% |
| mom_int_12_7 (252 skip 126) | 62% | 87% | +7.1% | -27.8% | mild everywhere, low upside |
| mom_accel (Δ63d ret) | 47% | 77% | -1.3% | -34.1% | dead signal |
| mom_accel2 (2nd deriv) | 48% | 76% | -0.7% | -32.5% | dead signal |
| mom_accel_pos (gated) | 51% | 79% | +0.2% | -29.7% | dead signal |
| mom_high52_prox | 3% | 34% | -17.7% | -50.8% | loses to QQQ everywhere |
| mom_high52_fresh (days since high) | 3% | 39% | -18.2% | -50.4% | loses to QQQ everywhere |
| mom_frac50dma_126 | 15% | 74% | -9.9% | -40.4% | low-vol tilt, QQQ drag |
| mom_frac50dma_252 | 23% | 81% | -6.7% | -34.0% | low-vol tilt |
| mom_trend_tstat126 (corr) | 17% | 70% | -11.8% | -40.1% | low-vol tilt |
| mom_clenow126 (slope·R²) | 65% | 86% | +7.9% | -34.0% | — |
| mom_clenow252 | 75% | 89% | +16.2% | -33.3% | GFC -20%, rec09-12 -20%, 2022 -15% |
| mom_fip126 (mom × up-share) | 66% | 89% | +10.6% | -34.0% | — |
| mom_ma_align (p>50>100>200) | 70% | 86% | +14.0% | -30.9% | GFC -20%, rec09-12 -17% |
| mom_dist200 (close/200dma) | 75% | 86% | +16.6% | -35.0% | GFC -20%, rec09-12 -22%, 2022 -17% |
| mom_x_smooth (rank mom + rank corr) | 42% | 85% | -2.8% | -27.4% | smoothness drags vs QQQ |
| mom_high52_gated (prox if 6m>0) | 5% | 45% | -16.8% | -50.6% | gating doesn't save it |
| mom126_x_high52 (rank blend) | 35% | 79% | -4.8% | -28.8% | high-52 dilutes momentum |

## Lookback / skip sensitivity (k=3 biweekly)

| variant | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq |
|---|---|---|---|---|
| mom_6_1 (126 skip 21) | 70% | 90% | +10.6% | -34.0% |
| mom_9_1 (189 skip 21) | 81% | 90% | +18.6% | -38.2% |
| mom_11_1 (231 skip 21) | 77% | 91% | +19.2% | -38.5% |
| mom_12_0.5 (252 skip 10) | 77% | 90% | +22.2% | -34.7% |
| mom_12_1 (252 skip 21) | 80% | 91% | +24.6% | -30.9% |
| mom_12_2 (252 skip 42) | 80% | 93% | +23.7% | -31.2% |

Graceful degradation: every 9–12-month lookback with a 0.5–2-month skip
lands at 77–81% win_qqq. Not a single-point optimum — a plateau. Skipping
the most recent month is what adds value over raw trailing returns
(short-term reversal); the exact skip length barely matters.

## k sweep (biweekly)

| signal | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| mom_12_1 win_qqq / med | 78% / +45.1% | 80% / +32.3% | 80% / +24.6% | 72% / +12.1% |
| mom_12_1 worst_vs_qqq | -43.8% | -37.8% | -30.9% | -25.1% |
| mom_clenow252 win_qqq / med | 68% / +15.7% | 80% / +22.1% | 75% / +16.2% | 63% / +5.4% |
| mom_ret189 win_qqq / med | 79% / +28.6% | 79% / +23.7% | 76% / +17.3% | 74% / +10.3% |
| mom_dist200 win_qqq / med | 73% / +26.1% | 77% / +21.4% | 75% / +16.6% | 64% / +8.7% |
| mom_12_2 win_qqq / med | — | **82% / +32.2%** | 80% / +23.7% | — |

Concentration raises the median sharply but fattens the left tail
(k=1 worst -44%, p10 -13.8%). k=2–3 is the sweet spot.

## Cadence (every=21, monthly)

| config | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq |
|---|---|---|---|---|
| mom_12_1 k=2 monthly | 81% | 89% | +36.6% | -40.0% |
| mom_12_1 k=3 monthly | 78% | 89% | +31.2% | -30.7% |
| mom_12_2 k=3 monthly | 80% | 94% | +31.8% | -30.7% |

Monthly cadence ≈ biweekly (slightly higher medians, similar win rates):
the signal is robust to rebalance frequency, not an every=10 artifact.

## Survivorship control

Random top-k picks over the same eligible universe (15 draws, files
`research/random_control_k{2,3}.csv`): k=3 median draw beats QQQ in only
11.5% of windows (median excess -14.6%); k=2: 10.2% (-14.5%). The
recommended configs (80-82% win_qqq, +25…+32% median) are far above the
control, so the edge is not the universe's survivorship bias.

## Honest bottom line

**No config in this family reaches the win_qqq ≥ 85% bar.** The family
plateaus at 80–82% win_qqq. The binding constraint is regime-shaped, not
parameter-shaped: every momentum variant loses to QQQ in
(a) recovery_2009_2012 — the classic post-crash momentum crash (-24% to
-43% vs QQQ), (b) GFC 2007-09 (-11% to -20%), (c) vol_2018 (~-10% to -15%),
(d) bear_2022 (-15% to -19%, momentum was long expensive tech into the
rate shock). Wins come from trending bulls (2013-17 +22…+29%, COVID
+10…+14%, sideways 2015-16, AI bull). Fixing the remaining ~20% of windows
likely needs a regime/defensive overlay or a sell rule, not another
momentum parameterization.

## Fragile / dead ends (do not pursue as-is)

* **52w-high proximity (all variants)**: 3–5% win_qqq. Selects low-beta
  steady names (staples/utilities pinned at highs); massive QQQ drag.
  Gating on positive momentum doesn't rescue it. Win_spy < 50% too.
* **Trend-quality-only scores** (frac>50dma, corr/t-stat): same low-vol
  failure mode; they beat SPY sometimes but never QQQ.
* **Momentum acceleration** (1st and 2nd derivative): indistinguishable
  from noise (47–51% win_qqq, ~0 median).
* **Smoothness blends** (rank-mom + rank-corr): smoothness leg dilutes the
  momentum leg; strictly worse than momentum alone.

## Top-3 recommended configs

1. **mom_12_2, k=2, biweekly** — 82% / 93% / +32.2% / worst -34.6%,
   p10 -5.7%, full-sample 11.0x. Best overall balance in the family; sits
   in the middle of the robust 9-12m plateau.
2. **mom_12_1, k=3, biweekly** — 80% / 91% / +24.6% / worst -30.9%.
   The canonical, simplest formulation; best left-tail control at
   biweekly cadence; preferred if simplicity is the tiebreaker.
3. **mom_12_1, k=2, monthly (every=21)** — 81% / 89% / +36.6% /
   worst -40.0%, full-sample 12.7x. Highest median; half the trading;
   demonstrates cadence robustness. Take this if costs/taxes matter more
   than the worst window.

All three are the *same economic signal* (12-month winners ex the latest
month) — recommend treating "12-1 momentum, k=2-3" as the family output
and spending the next iteration on a defensive overlay (e.g. crash/regime
sell rule) to attack the 2009/2018/2022 windows.
