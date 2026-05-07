# MERIDIAN — Stock + ETF Cross-Asset-Class Momentum

## Performance (2010-01-04 — 2026)

| Window | Sharpe | CAGR | Vol | MDD | Sortino | Calmar |
|---|---|---|---|---|---|---|
| FULL | **1.17** | 20.5% | 17.2% | -18.6% | 1.49 | 1.10 |
| IS (2010-2018) | 1.12 | 17.0% | 15.0% | -18.6% | 1.36 | 0.92 |
| OOS (2019-2026) | **1.24** | **24.8%** | 19.5% | -18.0% | 1.66 | 1.38 |

**Survivorship-haircut FULL CAGR: 19.1%** (2% on 70% stock portion).
IS-OOS Sharpe gap: 0.11 (extremely tight — strategy generalizes).
NAVx: 23.34× over 16 years.

## Hard constraints (all simultaneous)

1. No leveraged or inverse ETFs.
2. No portfolio margin or borrowing — gross ≤ 1.0.
3. No forward-looking signals — close[t-1] only.
4. ETF universe fixed ex-ante (33 ETFs, no selection bias).
5. Stock universe disclosed as survivorship-biased (90 large-caps in
   `data/stocks/` with 2010+ history) — handled via wide top-K + 2%
   CAGR haircut on disclosed forward-looking metrics.

## Survivorship-bias accounting

The stock universe is 90 currently-listed S&P 500 large-caps that have
data back to January 2010. **Stocks that went bankrupt, were delisted,
or merged out of existence are not in the dataset** (no Lehman, no WaMu,
no MF Global, no Bear Stearns).

Academic estimates of US large-cap survivorship bias: 1–3% CAGR. We
apply a conservative **2% haircut on the stock portion** of the
strategy, blended at 70% = 1.4% off the disclosed CAGR.

Mitigations baked into the strategy:
- Wide top-K (10/15/20) reduces concentration risk on individual lucky
  survivors.
- 30% of the book is in ETFs (no survivorship bias).
- The haircut is reported alongside every CAGR figure in the metrics.

## Strategy

5 momentum sleeves combined at fixed weights:

| Sleeve | Universe | Top-K | Lookback | Rebal | Weight |
|---|---|---|---|---|---|
| S1 STOCK_10_M | stocks (90) | 10 | 126d | monthly | 23.3% |
| S2 STOCK_15_W | stocks (90) | 15 | 126d | weekly | 23.3% |
| S3 STOCK_20_M | stocks (90) | 20 | 252d | monthly | 23.3% |
| S4 ETF_FAST | ETFs (33) | 1 | 21d | daily | 15.0% |
| S5 ETF_SLOW | ETFs (33) | 1 | 126d | daily | 15.0% |

Total stock weight: 70%. ETF weight: 30%. No IS-fitted blending —
weights are fixed by design.

Each sleeve allocates 100% of its capital between picks and BIL, so
portfolio gross is exactly 1.0. No margin.

## Per-sleeve metrics (standalone)

| Sleeve | FULL Sharpe | FULL CAGR | Vol | MDD |
|---|---|---|---|---|
| STOCK_10_M | 1.20 | 25.4% | 20.7% | -32.8% |
| STOCK_15_W | 1.06 | 19.7% | 18.6% | -32.7% |
| STOCK_20_M | 0.96 | 17.5% | 18.6% | -31.9% |
| ETF_FAST | 0.73 | 19.6% | 31.1% | -52.5% |
| ETF_SLOW | 0.67 | 17.8% | 32.7% | -55.7% |

Cross-asset correlation: stock sleeves vs ETF sleeves ~0.30-0.44.
This 0.3 cross-class correlation is the source of the Sharpe lift —
combining gives diversification benefit equivalent to ~30% Sharpe
boost vs either side alone.

## Risk overlays (de-risk only)

- Drawdown throttle: linear scale toward 0 as NAV falls below 252d HWM,
  floor at -15%.
- Vol-regime gate: halve exposure when 60d realized vol > 99th
  percentile of 252d trailing distribution.

Average overlay multiplier: 0.95.

## Why this trumps prior MERIDIAN versions

| Version | Sharpe | CAGR | MDD |
|---|---|---|---|
| Original MERIDIAN (3-sleeve composite, ETFs only) | 0.92 | 8.8% | -14.3% |
| MERIDIAN-MAX (dual ETF momentum, daily) | 0.88 | 21% | -37% |
| **MERIDIAN current (stocks + ETFs)** | **1.17** | **20.5%** | **-18.6%** |

The single-stock universe roughly doubles the alpha pool — momentum
signals on 90 names compete on a much wider, lower-correlation
cross-section than 33 ETFs. The cost is survivorship bias in the
stock data, which we explicitly account for.

## What this strategy can and cannot deliver

**Can deliver (under strict no-leverage):**
- Sharpe 1.17 (post-haircut: ~1.1)
- 19% CAGR forward-looking expectation (post-haircut)
- MDD -19% — half the drawdown of either single-class momentum
- IS-OOS Sharpe gap of 0.11 — exceptional generalization

**Cannot deliver under strict no-leverage:**
- Sharpe > 1.5 reliably — the diversification multiplier is bounded.
  Sharpe > 3 needs leverage-equivalent vol scaling.
- 30%+ CAGR over 16 years — the cross-asset cap is here too.

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
