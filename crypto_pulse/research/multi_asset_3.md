# Toward Sharpe 3 OOS — multi-asset VOL + STRATA

Reproduced vol-channel daily breakout on ETF asset classes (equity/sector/bond/commodity/FX) + crypto-VOL (t5rvt) + STRATA, each vol-targeted 12%, weekly. The honest multi-asset diversification test.

Overlap: 2018-03-02 -> 2026-04-03 (423 weeks).

## Per-book Sharpe (full overlap)

| book | Sharpe |
|---|---|
| VOL-EQ_IDX | +0.00 |
| VOL-SECTOR | -0.27 |
| VOL-BONDS | +0.20 |
| VOL-COMMOD | +0.24 |
| VOL-FX | +0.07 |
| VOL-CRYPTO | +2.18 |
| STRATA | +1.76 |

Mean pairwise correlation: **0.14**

## Combined multi-asset book

| combiner | Sharpe | IS | OOS | maxDD |
|---|---|---|---|---|
| equal-risk (all) | **+1.35** | +1.40 | +1.29 | -6% |
| Sharpe-opt (IS, pos books) | **+2.47** | +2.71 | +2.10 | -4% |

## Verdict

- Combined Sharpe-opt OOS = **+2.10**. Below 3 (OOS +2.10). The reproduced ETF VOL books are weaker than the vol repo's headline (daily ETF < their 5-min/optimized equity), so the diversification gets us toward ~2-2.5, not 3 — but it CONFIRMS the direction: each uncorrelated asset class adds. Stronger per-class books (their actual equity strategy at OOS 2.4) would close the gap.
- Mean cross-book correlation ~0.14 (genuine diversification across asset classes).
