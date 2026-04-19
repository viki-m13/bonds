# HYDRA — 17-sleeve diversified ensemble

Alternative to NOVA METEOR. Targets high risk-adjusted return via
uncorrelated-sleeve diversification rather than concentrated leverage.

## Architecture

17 independently-constructed sleeves, each:
- produces a daily return stream (long, short, or long-short)
- is vol-targeted to 10% annualised per-sleeve (rolling 63d, floor at 5%,
  scaling capped at 1.5x)
- uses 1-bar signal lag and 15 bps transaction cost on turnover
- is monthly-rebalanced (no daily churn)

Sleeves span 8 orthogonal alpha categories:
- **Equity trend**: vol-contingent SPY, sector top-3 momentum
- **Fixed income**: bond duration regime, credit trend, yield-curve carry
- **Commodity**: DBC trend, gold-silver regime
- **FX**: DBV carry, dollar regime
- **Volatility**: VIX contango carry
- **Crypto**: BTC trend
- **Cross-asset**: absolute momentum on 6 assets
- **Alternative**: defensive rotation, semis trend, SPY mean reversion,
  EEM trend, TIP vs IEF inflation hedge

Ensemble: inverse-vol risk parity weighting, then final portfolio vol-target
at 20%, gross cap 5x.

## Honest results (2005-04-05 .. 2026-04-10, 21.0y)

|               | HYDRA    | SPY      |
|---------------|----------|----------|
| CAGR          | 16.0%    | 12.0%    |
| Vol           | 10.2%    | 19.1%    |
| Sharpe        | **1.57** | 0.63     |
| Max DD        | −25.0%   | −55.2%   |
| NAVx ($10k→)  | $257k    | $85k     |

### IS / OOS split at 2018-01-01
- IS  (2005-2017): SR 1.27, CAGR 13.7%
- OOS (2018-2026): SR **2.14**, CAGR 19.5%, MDD −15.7%

### Recent annual performance
| Year | Ret    | Vol   | SR    | MDD    |
|------|--------|-------|-------|--------|
| 2019 | 28.0%  | 6.5%  | 3.86  | −3.2%  |
| 2020 | 1.2%   | 16.6% | 0.16  | −15.7% |
| 2021 | 2.7%   | 5.7%  | 0.50  | −2.9%  |
| 2022 | 3.6%   | 5.4%  | 0.68  | −4.1%  |
| 2023 | 33.0%  | 9.6%  | 3.04  | −3.4%  |
| 2024 | 55.7%  | 7.3%  | 6.13  | −2.0%  |
| 2025 | 41.7%  | 8.5%  | 4.20  | −3.8%  |

### Diagnostics
- Mean |pairwise correlation| across sleeves: **0.18**
- Median |corr|: 0.13, Max: 0.77 (eq-regime ↔ curve-carry in duration regimes)
- 17 sleeves, all positive full-window SR except the 2 borderline (s6, s8)

## Relative to targets

The brief asked for **20%+ CAGR and SR 3+**. The realistic honest ceiling
we could hit in this framework (15 bps TC, 1-bar lag, no lookahead, 21y
window including 2008 and 2020) is:

- Full-period SR ≈ 1.6
- OOS SR ≈ 2.1
- CAGR 16% full, 19.5% OOS

**SR 3 is not achievable** over a 21-year backtest without either (a)
hindsight-biased sleeve selection, (b) concentrated leverage like
NOVA METEOR (which produced −78% MDD in the proxy), (c) much shorter
windows that miss the two major crises, or (d) sleeves that exploit
one specific regime that won't repeat.

HYDRA beats SPY by 400bp/yr with half the volatility and half the drawdown,
and achieves OOS SR > 2. That is an institutional-grade diversified book
and a strong defensible alternative to METEOR for real client capital.

## Running

```bash
cd alt/hydra
python hydra_run.py
```

Writes:
- `data/results/hydra_returns.csv` — daily HYDRA + SPY returns
- `data/results/hydra_sleeves.csv` — daily per-sleeve returns
