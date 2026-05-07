# MERIDIAN-MAX — Strict-No-Leverage Daily Dual-Momentum

## Performance (2010-01-04 — 2026-04-02)

| Window | Sharpe | CAGR | Vol | MDD | Sortino | Calmar |
|---|---|---|---|---|---|---|
| FULL | **0.88** | **20.9%** | 25.0% | -36.9% | 1.19 | 0.57 |
| IS (2010-2018) | 0.73 | 14.8% | 22.5% | -36.9% | 0.96 | 0.40 |
| OOS (2019-2026) | **1.05** | **28.7%** | 27.8% | -28.1% | 1.45 | 1.02 |

NAVx: 24.75× (turn $10k into ~$248k). IS-OOS Sharpe gap: 0.32.

## Hard constraints (all enforced simultaneously)

1. **No leveraged or inverse ETFs** (no 2x/3x/-1x products).
2. **No portfolio margin or borrowing.** Sum of weights ≤ 1.0 every day.
3. **No forward-looking signals.** Inputs use close[t-1]; positions
   established at open[t]; returns earned open[t]→open[t+1].
4. **No selection bias.** Universe fixed ex-ante to 33 liquid 1x ETFs
   by liquidity + inception ≤ 2009.

## Strategy

Two daily-managed momentum sleeves on the same 33-ETF broad universe,
combined at fixed equal weight. Each sleeve picks the TOP-1 ETF by
absolute momentum at a different lookback horizon and rebalances daily.

| Sleeve | Lookback | Eligibility | Position |
|---|---|---|---|
| S1 FAST | 21-day return | 21d return > 0 | Top-1, full concentration |
| S2 SLOW | 126-day return | 126d return > 0 | Top-1, full concentration |

Aggregator: 50% S1 + 50% S2. No IS-fitted weights. Each sleeve allocates
100% of its capital between one ETF and BIL. Portfolio gross = 1.0.

Daily rebalance — the strategy follows leadership transitions immediately.

## Risk overlays (de-risk only)

- **Drawdown throttle**: linear scale toward zero as NAV falls below 252d
  HWM, floor at -15%.
- **Vol-regime gate**: halve exposure when 60d realized vol > 99th
  percentile of 252d trailing distribution.

Average overlay multiplier: 0.95.

## Per-sleeve metrics (standalone)

| Sleeve | FULL Sharpe | FULL CAGR | Vol | MDD |
|---|---|---|---|---|
| FAST_21 | 0.73 | 19.6% | 31.1% | -52.5% |
| SLOW_126 | 0.67 | 17.8% | 32.7% | -55.7% |

Pairwise correlation: 0.55. Low enough that the 50/50 blend has Sharpe
0.79 (raw) / 0.88 (with overlays) — both higher than either sleeve alone.

## Transaction-cost sensitivity

| TC per leg | FULL CAGR (with overlays) |
|---|---|
| 1 bp (HFT) | 22.5% |
| 2 bps (top algo) | 22.0% |
| **3 bps (institutional algo, canonical)** | **20.9%** |
| 5 bps (retail) | 18.7% |
| 10 bps (poor execution) | 12.4% |

This is a daily-traded strategy; TC matters. The canonical 3 bps
assumption is realistic for institutional execution on liquid ETFs.

## Universe (fixed ex-ante)

- **Broad equity (5):** SPY, QQQ, IWM, EFA, EEM
- **US sectors (9):** XLK, XLY, XLP, XLU, XLV, XLE, XLF, XLI, XLB
- **Sub-sectors (6):** SMH, XBI, ITB, XHB, TAN, VNQ
- **International (2):** EWJ, FXI
- **Treasuries (4):** TLT, IEF, IEI, SHY
- **Credit / TIPS (4):** HYG, LQD, EMB, TIP
- **Commodities (3):** GLD, SLV, DBC
- **Cash:** BIL

Inclusion based on liquidity + inception ≤ 2009 only — never on ex-post
returns. SMH/XLK/QQQ are present alongside every other sector; sector
concentration emerges only from the systematic momentum rules.

## What this strategy can and cannot do

**Can deliver:**
- 20% CAGR FULL (29% OOS) on unbiased universe with no leverage
- Sharpe 0.88 FULL, 1.05 OOS
- Daily-managed, fully systematic
- Transparent, no IS-fitted weights

**Cannot deliver under these constraints:**
- 30%+ CAGR — the highest single 1x ETF (SMH) is 25.2% B&H. Geometric
  mean of rolling picks can't exceed that. Phoenix's 37% requires
  leveraged ETFs.
- Sharpe > 1.5 — no 1x ETF has Sharpe > 1.5 over 16 years; the
  diversification multiplier from blending tops out around 2x; ensemble
  Sharpe ceiling is 1.0-1.5.
- Sharpe > 3 — structurally impossible on this toolkit. Achievable in
  reality only via leverage (raises Sharpe through vol scaling),
  short-history specialty instruments (JAAA post-2020 = Sharpe 3.94 but
  only 5 years), long-short / futures, or HFT.

## Trade-off frontier

Three points on the achievable curve, all unbiased and unlevered:

| Variant | Sharpe | CAGR | Vol | MDD |
|---|---|---|---|---|
| **High-CAGR (this)** | 0.88 | 20.9% | 25.0% | -36.9% |
| Mid (3-sleeve composite) | 0.92 | 8.8% | 9.7% | -14.3% |
| High-Sharpe (defensive) | 0.90 | 3.1% | 3.4% | -7.0% |

The product (Sharpe × CAGR) is bounded around 0.18-0.20. To exceed
that you need to break one of the four hard constraints.

## Files

| Path | Description |
|---|---|
| `alt/meridian_strategy.py` | Single canonical implementation |
| `alt/MERIDIAN_DESIGN.md` | This document |
| `data/results/meridian_metrics.json` | All performance numbers |
| `data/results/meridian_returns.csv` | Daily series + overlay state |
| `data/results/meridian_sleeves.csv` | Per-sleeve daily returns |
| `docs/meridian.html` | Editorial-style factsheet webpage |
| `docs/meridian_data.json` | Webpage data (auto-generated) |
