# Ernie Chan "Algorithmic Trading" techniques applied to crypto (what helps)

Mined the repo recreating Chan's *Algorithmic Trading: Winning Strategies and
their Rationale* (github.com/zazhang/ep-chan-book-algo-trading). It covers
stationarity tests (ADF, Hurst exponent, variance ratio), cointegration, Kalman-
filter dynamic hedging, Bollinger/cross-sectional mean reversion, and cross-
sectional momentum.

## What we tested
**Kalman-filter dynamic-beta residual mean reversion** (Chan's signature: estimate
a time-varying hedge ratio via a Kalman filter, trade the mean-reverting residual
spread). Implemented a per-coin random-walk Kalman beta vs the crypto market
factor, residual spread z-score, fade it. Net of 4.5bps taker, HL era:
  hold 3d: -1.31 | 5d: -1.23 | 10d: -1.71  (negative in BOTH halves)
It is *negatively* correlated with our momentum sleeves (-0.13 to -0.23) but
*negative-Sharpe*, so adding it as a 4th sleeve HURTS the blend (1.12 -> 0.26).

## The honest lesson (and it's valuable)
Chan's machinery is built for genuinely mean-reverting markets (FX, equity/ETF
pairs, calendar spreads — where ADF/Hurst<0.5 hold). **Crypto residuals do NOT
mean-revert at the daily/taker horizon — they trend (momentum).** The Kalman
sophistication doesn't change the signal's sign; it just confirms, via Chan's own
framework, that crypto-at-taker is a MOMENTUM market. This validates our
trend+carry+order-flow book (all momentum/carry) and explains why every reversion
idea (VELOCITY-at-taker, residual reversal, Kalman pairs) fails at taker — crypto
mean reversion is a MAKER-only, intraday phenomenon.

So the repo's real value is diagnostic (Hurst/variance-ratio confirm the regime),
not a new alpha sleeve. The transferable tools worth keeping: Hurst/variance-ratio
to classify trending-vs-reverting per asset, and the Kalman dynamic-beta for
cleaner factor hedging in the carry/residual sleeves (marginal). No 4th sleeve
from Chan.
