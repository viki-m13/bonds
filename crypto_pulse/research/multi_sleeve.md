# Multi-sleeve crypto book on Hyperliquid — honest Sharpe-2 attempt

HL-tradeable era 2023-05-12 -> 2026-04-24, 57 coins, real HL funding + 4.5bps taker, 10% vol/sleeve. IS = first 60%, OOS = last 40%.

| sleeve | Sharpe | IS | OOS | ann | maxDD |
|---|---|---|---|---|---|
| TREND | +0.75 | +1.13 | +0.11 | +8.2% | -11.0% |
| CARRY | +0.92 | +0.48 | +1.60 | +12.8% | -14.9% |
| REVERSAL | -0.35 | -0.50 | -0.15 | -4.0% | -23.0% |

Sleeve correlations: TREND-CARRY=+0.08, TREND-REVERSAL=-0.17, CARRY-REVERSAL=-0.20

**blend (risk-weighted): Sharpe +1.13 (IS +1.07, OOS +1.24), ann +10.3%, maxDD -9.9%**
**blend + 12% vol target: Sharpe +0.93 (IS +0.95, OOS +0.89), ann +13.2%, maxDD -13.4%**

IS risk weights: TREND 55%, CARRY 45%

## Verdict

- **TREND + trend-filtered CARRY blends to a stable, validated Sharpe ~1.1** on the HL-tradeable era — IS 1.07, OOS 1.24, −9.9% maxDD, net of real HL funding + 4.5 bps taker. That is a genuine lift over PULSE-trend alone (0.75) and the two sleeves are uncorrelated (+0.08).
- **CARRY must be trend-filtered** (don't short coins still trending up) or it is unstable (IS −0.28) with −24% negative-skew drawdowns. REVERSAL is **dead net of taker** (the intraday edge is maker-only, per hft.md).
- **A robust net Sharpe of 2 is NOT achievable here.** Daily crypto trend+carry tops out ~1.1–1.2 net; the only signals with Sharpe-2+ breadth (intraday reversal) require maker execution. Leverage raises return, not Sharpe. The honest, validated, deployable number is **~1.1 (HL era) / ~1.2 (full sample)** — real, but short of 2.
