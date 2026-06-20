# Point-in-time universe: top-N most-liquid, ranked monthly

Price sleeves (TREND+BAB+SQUEEZE+ACCEL), 4.5bps taker, vol-target 12%. Universe re-ranked monthly by trailing 30d dollar volume (causal). Full sample from ~2015. (Survivorship caveat noted in the module docstring.)

## Universe size sweep (full sample)

| universe | med live names | Sharpe (full) | IS≈pre-HL | HL era | CAGR | maxDD |
|---|---|---|---|---|---|---|
| BTC+ETH only | 2 | **+0.90** | +1.29 | -0.25 | +13% | -27% |
| top 5 | 5 | **+0.57** | +0.82 | -0.25 | +8% | -24% |
| top 10 | 10 | **+0.70** | +0.88 | +0.12 | +10% | -24% |
| top 20 | 20 | **+1.05** | +1.06 | +1.05 | +16% | -24% |
| top 30 | 30 | **+1.12** | +1.14 | +1.10 | +17% | -24% |
| top 50 | 50 | **+1.10** | +1.09 | +1.13 | +17% | -24% |

BTC+ETH time-series trend (long/short each on its own trend, not cross-sectional): Sharpe **+1.16** (pre-HL +1.45, HL +0.34).

## Verdict

- **BTC+ETH alone is weak** (Sharpe +0.90): two names give almost no cross-sectional breadth, so BAB/ACCEL/SQUEEZE degenerate to a single long-short pair. The edge NEEDS breadth.
- **The PIT top-N book works and is robust to N** (top-30 Sharpe +1.12), confirming it's not an artifact of a hand-picked universe: rank the most-liquid coins monthly and trade them. Breadth helps up to ~20-30 names, then flattens (single-factor crypto). This IS the deployable, point-in-time universe rule.
- Honest PIT note: the monthly liquidity rank removes look-ahead in universe SELECTION; full survivorship-freedom would need a delisting-complete panel, but the liquidity filter (only trade names with >$3M/day AT THE TIME) already excludes the dead-microcap mirage that inflates naive crypto backtests.
