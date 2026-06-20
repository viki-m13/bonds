# Does more breadth (tickers) or more leverage raise Sharpe?

Price-based sleeves (TREND+BAB+SQUEEZE+ACCEL), equal-risk, vol-targeted, 4.5bps taker, liquidity filter $3M/30d. IS=first60/OOS=last40 of HL era. (Funding sleeves excluded here so the 57 vs 111 comparison is apples-to-apples; extra coins lack HL funding.)

## 1. BREADTH — 57 funded coins vs full 111-coin universe

| universe | med eligible names | Sharpe | IS | OOS | maxDD |
|---|---|---|---|---|---|
| 57 (funded/HL) | 44 | **+1.33** | +1.32 | +1.36 | -9.9% |
| 111 (all data/crypto) | 62 | **+1.01** | +1.03 | +1.00 | -10.1% |

**Breadth effect: -0.32 Sharpe** from ~doubling the universe. A modest lift — crypto is single-factor so extra names are mostly more BTC-beta, not independent bets; the liquidity filter keeps dead microcaps out (which would otherwise INFLATE the backtest with untradeable names). Real but diminishing — not a path to 3.

## 2. LEVERAGE — scaling the SAME book to higher vol targets

Sharpe = mean/vol is INVARIANT to leverage (lever k -> k*return AND k*vol). Leverage scales RETURN, drawdown, and liquidation risk — not the ratio. We scale the 111-coin book to each vol target:

| vol target | ~gross leverage | Sharpe | ann return | maxDD | 1-day worst |
|---|---|---|---|---|---|
| 12% | ~1.4x | +1.05 | +15% | -11% | -6.4% |
| 24% | ~2.7x | +1.00 | +28% | -22% | -12.9% |
| 36% | ~3.6x | +0.98 | +36% | -26% | -13.3% |
| 60% | ~4.0x | +1.02 | +42% | -28% | -13.3% |
| 100% | ~4.1x | +1.01 | +42% | -28% | -13.3% |

**Verdict on leverage:** the Sharpe column is flat — leverage does NOT improve risk-adjusted return. It multiplies returns AND drawdowns together. Past the growth-optimal (Kelly) point, geometric return actually FALLS (vol drag) and a bad day can liquidate the account. Leverage is the lever for RETURNS (and risk), never for Sharpe. The vol target — not leverage — is the real control, and the binding constraint is drawdown tolerance, not the ratio.
