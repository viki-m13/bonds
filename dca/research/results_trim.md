# Concentration TRIM vs never-sell (and vs full liquidation)

Instead of selling everything, on each rebalance boundary sell only the
**excess** of any holding above a weight cap and redeploy it into the current
top-2. Most of the book — and its deferred gains — is left untouched.
244-window grid, biweekly, 5 bps. `python research/trim_study.py`.

## Results

| config | beat QQQ | beat SPY | median vs QQQ | p10 | worst vs QQQ | full mult | biggest holding |
|---|---|---|---|---|---|---|---|
| **never sell (SUMMIT)** | 93% | 98% | +28.8% | +3.0% | −10.6% | 20.0× | 36% (NVDA) |
| trim 33% / year | 94% | 97% | +30.4% | +2.3% | −10.8% | 20.4× | 32% |
| trim 25% / year | 92% | 96% | +30.0% | +2.1% | −11.6% | 20.4× | 25% |
| trim 20% / year | 92% | 97% | +26.0% | +2.4% | −12.1% | 23.6× | 19% |
| trim 33% / quarter | 94% | 97% | +29.4% | +2.2% | −10.7% | 23.0× | 29% |
| trim 25% / quarter | 92% | 96% | +27.7% | +1.7% | −11.9% | 24.5× | 25% |
| trim 20% / quarter | 92% | 97% | +25.4% | +1.1% | −10.9% | 28.7× | 26% |

## Findings — trim works; it is the opposite of full liquidation

**1. It caps concentration at almost no cost to the robust metrics.** Every
trim variant keeps win-rate at 92-94%, median lead +25-30%, and worst window
−10.6 to −12.1% — statistically the same as never-sell — while pulling the
biggest position down from 36% to anywhere between 19% and 32% depending on the
cap. You buy the diversification almost for free.

**2. It stays phase-stable — unlike full liquidation.** Full multiple across
the 10 schedule offsets:

| | median | range |
|---|---|---|
| never sell | 19.6× | 18.9–21.3× |
| trim 25% / quarter | 22.6× | 18.7–26.0× |
| trim 25% / year | 19.0× | 18.1–20.4× |
| (full liquidation, quarterly) | 44.9× | **15.2–130.3×** |

Trim's outcome barely depends on timing (1.1–1.4× spread, like never-sell),
whereas full liquidation swung 8.6× on offset alone. Trim does not reintroduce
the fragility.

**3. It even nudges the multiple up.** Quarterly trims raise the full multiple
(23–29× vs 20×) by rotating a slice of the appreciated winners into current
leaders, which compounds in a momentum-persistent market — but the gain is
modest and sits inside the robust phase range, not a lucky artifact.

**4. Annual ≈ quarterly on the robust metrics**, with far less turnover. Trim
only sells the excess of the 1–2 oversized names a handful of times a year (vs
full liquidation's 417 sells of the entire book), so the realized-gains tax is
a small fraction. Annual trimming is the tax-efficient sweet spot.

## Recommendation

If you want to cap single-name risk, an **annual trim at ~25-33%** is the clean
lever: it holds SUMMIT's 93%/+29%/−11% profile, stays phase-robust, caps the
biggest position to the chosen ceiling, and triggers only a small, infrequent
taxable event on the excess of the largest names. This is exactly what full
liquidation failed to be — a real, low-cost concentration control rather than a
high-turnover, high-tax, timing-dependent mirage.

(Default SUMMIT stays never-sell for maximum tax deferral and simplicity; trim
is an optional overlay for investors who specifically want to limit how large
any one name can get.)
