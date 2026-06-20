# SUMMIT — Prop Market-Neutral Alpha Engine (SHORTING + LEVERAGE)

> **Status: ARCHIVED for future implementation. NOT the current deployment.**
> Requires shorting + leverage/margin — out of scope for the current long-only,
> no-margin mandate. Recorded here so we can pick it up and improve it later.
> Build/validate scripts: `dca/research/exp73_factorzoo.py`, `exp74_alpha.py`,
> `exp75_reality.py`, `exp76_mlls.py`. Data: PIT survivorship-clean Tiingo +
> SEC fundamentals/revenue + Form-4 insider. Window 2012–2025 (ML 2015–2025).

## What it is
A dollar-neutral long/short book that strips market beta to harvest pure alpha.
Two engines, both ~zero correlation to QQQ:

1. **Linear multi-factor L/S** — equal-weight combine of 11 single-factor
   decile L/S sleeves: value (B/M, E/P), Novy-Marx gross profitability (GP/A),
   sales/assets, Piotroski, ROA, ROE, momentum 12-1, 52w-high, buyback
   (−share issuance), revenue acceleration, insider cluster.
2. **ML L/S** — walk-forward HistGBM (trained only on past data) predicting
   cross-sectional return terciles from all 36 features, long top decile /
   short bottom decile.

## Validated results (PIT survivorship-clean)

| Engine | ann return | Sharpe (gross) | Sharpe (net 6%/yr borrow) | corr QQQ | maxDD |
|---|--:|--:|--:|--:|--:|
| Single best factor — Value (B/M) | 21.0% | 1.38 | — | −0.15 | — |
| Linear 11-factor L/S | 12–14% | 1.55–1.75 | ~1.0 | −0.09 | −16% |
| **ML L/S (decile)** | **46.4%** | **2.00** | **1.74** | **−0.01** | −40% |

**Portable alpha (overlay on QQQ beta):**
- QQQ + 1× linear alpha → 30% / Sharpe 1.66
- QQQ + 1× ML alpha (net) → **59% / Sharpe 2.01**
- QQQ + 2× alpha → 41–79% / Sharpe ~1.83–2.03

Regime robustness (ML L/S net Sharpe): 2015-18 **1.89**, 2019-21 0.66, 2022-25 **2.57** — never negative (linear book WAS negative 2017-20).

## Honest caveats (do not strip)
- Sharpe 2.0 is **gross of commissions / slippage / market impact**; decile L/S
  turns over ~100%+/month across ~200 small/mid-cap names. Realistic net is
  materially lower.
- Requires **shorting hard-to-borrow small/mid-caps** (borrow cost modeled at
  6%/yr; squeeze + availability risk on top) and **leverage/margin** for the
  overlay.
- Pure L/S max drawdown ~−40%.
- Feature-set choice used some full-sample judgment (training itself is OOS).

## Turnover & cost-aware analysis — DEPLOYMENT-READY (exp78)
Realistic net-of-cost results for the ML L/S, incl 6%/yr borrow on the short leg.
Rank-buffering (hold a name until it leaves a 2× band) is the key efficiency lever
— it cuts turnover AND raises Sharpe (less noise trading).

| Config | turnover/mo | gross Sharpe | net @10bps | net @20bps | net ann |
|---|--:|--:|--:|--:|--:|
| monthly decile | 160% | 2.02 | 1.68 | 1.60 | 37% |
| monthly + buffer2× | 99% | 2.22 | 1.91 | 1.85 | 43% |
| quarterly decile | 80% | 1.98 | 1.67 | 1.63 | 38% |
| **quarterly + buffer2× (DEPLOY)** | **58%** | 2.12 | 1.83 | **1.80** | **43%** |
| monthly quintile | 137% | 1.74 | 1.35 | 1.26 | 25% |

**Cost robustness (quarterly+buffer2×, incl borrow):** net Sharpe 1.87 / 1.85 /
1.83 / 1.80 / 1.77 at 0 / 5 / 10 / 20 / 30 bps-per-side. maxDD ~−37%.
=> **Deployable config: quarterly rebalance, decile L/S, 2× rank buffer — net
Sharpe ~1.8 after realistic costs.** Capacity: equal-weight decile of the liquid
(>$3, >$5 short) PIT universe = a few hundred names/side; size limited by ADV of
the smaller names (cap each position at a few % of 20-day ADV; large AUM should
tilt to the larger-cap half). Borrow: restrict shorts to easy-to-borrow to realize
the modeled 6%/yr.

## Roadmap to improve (when we revisit)
1. **Turnover/cost-aware construction** — rank-buffering, monthly→quarterly
   rebalance, name-count vs cost trade-off → get a *tradeable* net Sharpe.
2. **Sector neutralization** — remove sector bets (likely fixes 2019-21 dip).
3. **Borrow-aware shorting** — restrict shorts to easy-to-borrow / large names;
   measure realistic borrow.
4. **Vol-targeting** the book to a fixed risk; dynamic factor weighting.
5. **New alpha data** — 13F institutional flows, short-interest/FTD, options skew.
6. **Ensemble** ML + linear (corr 0.76, modest diversification) + add tree/NN.
