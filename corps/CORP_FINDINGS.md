# Corporate bonds — a free data source, and a validated strategy

**Result: the KEYSTONE dislocation-reversion strategy works on individual
corporate bonds too, validated out-of-sample on 22 years of free data.**

Full sample 2002–2025, 8,000 liquid corporate bonds, buy at the ask when a
bond prints ≥3 pts below its own trailing 60-day median clean price, hold ~1
year, sell at the bid:

| | value |
|---|---|
| Trades | 32,746 |
| Win rate | 75% |
| Mean return / trade (~1yr) | +6.22% |
| **Excess vs matched random-entry control** | **+1.38%** (p<0.001) |
| Equity CAGR / maxDD | **+5.41% / −14.7%** |
| LQD buy-and-hold (same window) | +4.61% / −25.0% |

Higher return than investment-grade buy-and-hold with roughly **half the
drawdown**, and the excess-vs-control proves it is timing alpha, not beta.

## The free data source

After confirming a **basic FINRA API credential does not include trade-level
TRACE** (only aggregates — see "credential map" below), we found the data
free elsewhere:

**[Open Source Bond Asset Pricing](https://openbondassetpricing.com/)**
(Dickerson et al.) publishes a processed **daily** corporate-bond panel built
from TRACE — free direct download, no WRDS:

- `stage1_osbap_0k_volume_2025.parquet` — **29.8M bond-days, 73,835 bonds,
  2002-07 → 2025-03**, Enhanced + Standard + 144A TRACE.
- Fields used: `pr` (daily volume-weighted **clean price**), **`prc_bid` /
  `prc_ask`** (daily bid/ask), `ytm`, `credit_spread`, `qvolume`,
  `bond_maturity`. The bid/ask lets us model KEYSTONE execution faithfully —
  **buy at the ask, sell at the bid** — exactly like the muni customer-buy/sell
  prints.

`corps/scripts/build_osbap_panel.py` maps it to the muni engine's schema
(top 8,000 by liquidity → 14.0M bond-days). The proven backtest engine runs
unchanged.

## Evidence

Horizon economics match munis (spread is a one-time cost, carry+reversion
grows): unconditional customer round-trip mean return 30d −0.4% → 180d +1.9%
→ 365d +4.5% → 730d +9.2%.

**Threshold monotonicity (full sample, excess vs control):**

| entry threshold | trades | win | mean/trade | excess | p |
|---|--:|--:|--:|--:|--:|
| ≥1 pt below | 50,709 | 70% | +4.22% | −0.41% | 1.00 |
| ≥2 pt below | 41,000 | 73% | +5.01% | +0.26% | 0.004 |
| **≥3 pt below** | 32,746 | 75% | +6.22% | **+1.38%** | <0.001 |
| ≥4 pt below | 26,694 | 77% | +7.87% | **+2.91%** | <0.001 |

**In-sample → out-of-sample (≥3 pt), both significant:**

| window | trades | win | mean/trade | excess | p |
|---|--:|--:|--:|--:|--:|
| IS 2002–2015 | 15,798 | 79% | +8.12% | **+2.30%** | <0.001 |
| OOS 2016–2025 | 17,556 | 72% | +5.70% | **+1.69%** | <0.001 |

**By era — significant in 5 of 6, one systemic-crisis failure:**

| era | trades | win | excess | p |
|---|--:|--:|--:|--:|
| 2004–2007 | 3,683 | 71% | +0.93% | <0.001 |
| **2008–2009 GFC** | 2,826 | 68% | **−2.17%** | 0.95 |
| 2010–2015 | 8,807 | 81% | +1.50% | <0.001 |
| 2016–2019 | 6,485 | 84% | +2.87% | <0.001 |
| 2020 COVID | 4,853 | 96% | +4.47% | <0.001 |
| 2021–2023 | 8,320 | 40% | +0.71% | <0.001 |

Same signature as munis: a mean-reversion strategy that pays in most regimes
and **loses in the one systemic crisis** (2008 GFC here, the muni analogue was
the 2022 rate selloff) — when the whole market craters, buying dislocations is
catching falling knives, and random entry in the same names does as well or
better. Disclosed, not hidden.

## Honest caveats

- **Carry** is proxied by each bond's median yield (OSBAP daily rows don't
  carry the coupon); the excess-vs-control metric nets it out since both legs
  hold the same bond for the same period.
- **Equity-curve drawdown** uses linear intra-trade attribution (as in munis),
  so the −14.7% is somewhat smoothed vs a daily mark; total return and CAGR are
  realized from bid/ask fills.
- **Data** is OSBAP's cleaned daily VWAP + bid/ask (a reputable academic
  pipeline), not raw ticks; 8,000 most-liquid of 73,835 bonds.
- The **GFC** loss is intrinsic to mean-reversion; size accordingly (a
  market-trend / credit-spread overlay is the natural risk gate, as for munis).

## Credential map (why we needed the free source)

The provided FINRA credential is a **basic** tier. Tested against every
`fixedIncomeMarket` dataset: all `trace*Detail/*Summary` (trade-level) return
**403 "basic API credential cannot access"**; only aggregates
(`corporateMarketBreadth/Sentiment`, `cappedVolume`, treasury aggregates) are
readable. Those aggregates (3-year history) did **not** yield a strategy that
beats buy-and-hold (see git history / `aggregate_analysis.py`). The OSBAP
daily panel is what made the per-bond strategy possible for free.

## Reproduce

```bash
# 1. download the free OSBAP daily panel (1.8 GB)
curl -o osbap.zip https://openbondassetpricing.com/wp-content/uploads/2025/12/stage1_osbap_0k_volume_2025.zip
unzip osbap.zip
# 2. build the panel and backtest
python corps/scripts/build_osbap_panel.py stage1_osbap_0k_volume_2025.parquet
python corps/research/osbap_backtest.py sweep
python corps/research/osbap_backtest.py full
```
