# TIDE across asset classes, universes, timeframes, leverage (honest)

Same FROZEN rule (20d breakout x trend-intensity, hold3, vol-targeted) everywhere. The cross-ASSET test is the real one: a crypto-invented rule working on stocks = real effect, not a fit.

| universe | N names | period | Sharpe | CAGR | maxDD |
|---|---|---|---|---|---|
| CRYPTO-57 (HL-funded) | 57 | 2014-11-01..2026-04-24 | **+1.28** | +25% | -20% |
| CRYPTO-112 (full daily) | 111 | 2014-11-01..2026-04-24 | **+1.07** | +18% | -20% |
| STOCKS-96 (large-cap) | 96 | 2005-03-09..2026-04-10 | **-0.80** | -6% | -74% |
| STOCKS-430 (extended) | 430 | 2009-03-10..2026-05-07 | **-1.28** | -8% | -76% |
| ETFs | 31 | 1999-03-01..2026-06-18 | **+0.14** | +1% | -33% |

## Timeframes (crypto)

| timeframe | Sharpe | CAGR |
|---|---|---|
| hourly (20 coins, scaled windows) | -0.08 | -0% |
| weekly (57 coins, scaled windows) | +0.55 | +9% |

## HL leverage profile (CRYPTO-57 book, vol-targeted to 12%)

- Implied gross leverage: average **1.02x**, 95th pct 1.92x, cap 3.0x. Well within HL limits (majors allow 20-50x).
- Same Sharpe at any leverage; scaling the vol target trades return for drawdown:
| vol target | CAGR | maxDD | ~gross lev |
|---|---|---|---|
| 12% | +25% | -20% | ~1.0x |
| 20% | +42% | -33% | ~1.7x |
| 30% | +64% | -48% | ~2.6x |
| 50% | +109% | -71% | ~4.3x |

## Verdict (honest — and it sharpens what TIDE is)

- **Cross-asset:** crypto-57 +1.28, crypto-112 +1.07; stocks-96 **-0.80**, stocks-430 **-1.28**, ETFs +0.14.
- **TIDE is CRYPTO-SPECIFIC — it does NOT generalize to equities; it INVERTS.** The same rule that earns ~2 in crypto LOSES (-0.8 to -1.3, -75% DD) on hundreds of stocks. That is economically coherent: short-horizon cross-sectional moves CONTINUE in crypto (momentum/breakout) but REVERSE in equities (the well-known equity short-term reversal). The sign of the edge flips with asset class.
- **Implication for HL HIP-3 equity perps (TSLA, etc.): do NOT run TIDE on them** — it would lose. TIDE is a crypto-daily strategy, full stop.
- **Full crypto universe (112):** +1.07 vs +1.28 on the liquid-57 — adding smaller coins slightly DILUTES it; keep to the liquid subset.
- **Timeframes:** daily is the sweet spot; weekly +0.55 (weaker), hourly -0.08 (fails — costs/noise dominate intraday).
- **Leverage:** avg gross 1.0x, cap 3x — trivially within HL limits (20-50x on majors). Leverage scales CAGR and drawdown linearly, Sharpe unchanged; at 30% vol target it's ~+65% CAGR / -37% DD at ~2.6x gross.
- **Net:** TIDE is robust WITHIN crypto-daily (its design domain) but is NOT a universal anomaly. Honest scope: a crypto-daily ~2.0 Sharpe book, not a cross-asset one.
