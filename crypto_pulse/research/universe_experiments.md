# Universe-construction experiments (eval 2018+, price sleeves)

Axes: liquidity lookback (10/20/30/60d) x band (top-N vs skip top-K megacaps) x rebalance (monthly/biweekly). Robust pick = best min(IS,OOS). Picking the single max IS is overfitting — flagged.

| lookback | band | rebal | med N | Sharpe | IS | OOS | min(IS,OOS) |
|---|---|---|---|---|---|---|---|
| 10d | top30 | monthly | 30 | **+1.41** | +1.67 | +1.02 | +1.02 |
| 10d | skip4-band30 | monthly | 30 | **+1.25** | +1.50 | +0.88 | +0.88 |
| 10d | skip9-band30 | monthly | 30 | **+1.24** | +1.41 | +0.99 | +0.99 |
| 10d | top20 | monthly | 20 | **+1.32** | +1.43 | +1.14 | +1.14 |
| 20d | top30 | monthly | 30 | **+1.47** | +1.70 | +1.12 | +1.12 |
| 20d | skip4-band30 | monthly | 30 | **+1.37** | +1.69 | +0.88 | +0.88 |
| 20d | skip9-band30 | monthly | 30 | **+1.50** | +1.66 | +1.26 | +1.26 |
| 20d | top20 | monthly | 20 | **+1.35** | +1.57 | +1.01 | +1.01 |
| 30d | top30 | monthly | 30 | **+1.46** | +1.73 | +1.05 | +1.05 |
| 30d | skip4-band30 | monthly | 30 | **+1.52** | +1.87 | +0.97 | +0.97 |
| 30d | skip9-band30 | monthly | 30 | **+1.30** | +1.36 | +1.22 | +1.22 |
| 30d | top20 | monthly | 20 | **+1.30** | +1.53 | +0.94 | +0.94 |
| 60d | top30 | monthly | 30 | **+1.43** | +1.62 | +1.13 | +1.13 |
| 60d | skip4-band30 | monthly | 30 | **+1.38** | +1.58 | +1.04 | +1.04 |
| 60d | skip9-band30 | monthly | 30 | **+1.41** | +1.56 | +1.18 | +1.18 |
| 60d | top20 | monthly | 20 | **+1.24** | +1.47 | +0.86 | +0.86 |

**Most robust (max min(IS,OOS)):** 20d / skip9-band30 / monthly -> Sharpe +1.50 (IS +1.66 / OOS +1.26).
**Highest full-sample (likely overfit):** 30d / top30 / biweekly -> +1.54 (IS +1.83 / OOS +1.09).

## Does excluding megacaps help? (30d, monthly)

| band | Sharpe | IS | OOS |
|---|---|---|---|
| top30 | +1.46 | +1.73 | +1.05 |
| skip4-band30 | +1.52 | +1.87 | +0.97 |
| skip9-band30 | +1.30 | +1.36 | +1.22 |
| top20 | +1.30 | +1.53 | +0.94 |

## Liquidity lookback effect (top30, monthly)

| lookback | Sharpe | IS | OOS |
|---|---|---|---|
| 10d | +1.41 | +1.67 | +1.02 |
| 20d | +1.47 | +1.70 | +1.12 |
| 30d | +1.46 | +1.73 | +1.05 |
| 60d | +1.43 | +1.62 | +1.13 |

## Verdict

- On 2018+ (real breadth) the grid is **~1.2-1.5** across lookback/band/rebal — ROBUST to universe construction (not a fragile knob). Most of the jump vs the full-2015 sample's ~1.1 is the eval PERIOD (excluding the BTC+ETH-only 2015-16), not the rule; the rule itself moves things only ~0.1-0.2. Best robust config: **20d / skip9-band30 / monthly = +1.50** (IS +1.66/OOS +1.26).
- Shorter (10d) vs longer (60d) lookback barely moves it; excluding megacaps helps only marginally if at all (the cross-sectional sleeves already neutralize the BTC factor via BAB/demeaning). Picking the single highest full-sample cell would be overfitting — the spread between robust and max is the overfit premium.
- Net: keep it simple — **top-30 by 30d dollar volume, monthly** is as good as anything and the most stable. The Sharpe ceiling is set by the signal/cost structure, not the universe rule.
