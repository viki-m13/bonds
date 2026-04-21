# LETF strategy audit — honest critique

**Question asked:** "is this overfitting or cherry picking? are we searching
through the whole universe of LETFs when rebalancing?"

**Answer:** yes, substantially. The headline numbers survive some robustness
checks and collapse under others. Deflated Sharpe **cannot reject the null
hypothesis of zero skill** for any of the headline strategies once you
account for the 323 configs we tested.

## TL;DR

| Test | Verdict |
| --- | --- |
| Full-universe (all 17 LETFs, not core6) | **~12 pp CAGR gap** — headline is basket-selection effect |
| IS/OOS holdout (2011-18 → 2019-26) | Mixed — `invvol` survives, `static` + `invvol-scaled` crush OOS |
| Block-bootstrap 95% CIs | **Wide** — every Sharpe CI overlaps SPY's [0.37, 1.34] |
| Deflated Sharpe (N=323) | **Max DSR = 57%** — fails the 95% skill threshold by a mile |
| Family distributions | `invvol` stable (Spearman +0.61), `static` unstable (Spearman −0.11) |
| Regime stress | Every LETF strategy had a 1-yr period of −50% to −81% |

## 1. Basket selection (`letf_robust_universe.py`)

The headline "29% CAGR" inv-vol used `core6` = UPRO/TQQQ/SOXL/TECL/TMF/UGL —
a hand-picked winner basket. Running identical strategy on the full universe:

| Basket | inv-vol CAGR |
|---|---|
| core6 (hand-picked) | 28.4% |
| tech4 (UPRO/TQQQ/SOXL/TECL) | 34.4% |
| all17 (full long universe) | 15.8% |
| long_defensive (TMF/UGL/TYD/UBT) | 3.9% |

**The basket is worth more than the strategy.** The "29% CAGR" reduces to
~16% once we stop cherry-picking tickers.

## 2. IS / OOS holdout (`letf_is_oos.py`)

Fit on 2011-2018, freeze, evaluate 2019-04-2026 blind.

| Family | IS-best | IS SR | OOS SR | Δ |
|---|---|---|---|---|
| `invvol` | invvol clean4 lb=21 | 0.99 | **0.86** | −0.13 |
| `mom` | mom core6 lb=126 top5 | 0.95 | 0.80 | −0.15 |
| `invvol-scaled` | s-core6 lb=126 tv=60% | 0.94 | 0.70 | −0.24 |
| `static` | HFEA-Tech 50/50 | 1.22 | **0.58** | **−0.64** |
| EW-all17 baseline | — | 0.50 | **0.75** | +0.25 |
| SPY buy-hold | — | 0.81 | 0.86 | +0.05 |

**EW-all17 naive equal-weight (no selection) beat 3 of the 4 IS-optimised
winners on OOS.** The `static` family winner (HFEA-Tech 50/50) lost more
than half its Sharpe OOS.

Spearman rank correlation IS↔OOS by family:

| Family | Spearman(SR) | Top-quartile persistence |
|---|---|---|
| invvol | +0.61 | 80% |
| invvol-scaled | +0.28 | 48% |
| mom | +0.20 | 40% |
| static | **−0.11** | **12%** |

Static rank is essentially random across windows — picking winners there
is pure curve-fitting.

## 3. Block-bootstrap CI (`letf_bootstrap.py`)

500 block-bootstrap resamples (block=21 d) on the 2011-2026 daily returns:

| Strategy | CAGR pt [95% CI] | Sharpe pt [95% CI] | MDD pt [95% CI] |
|---|---|---|---|
| 100% TQQQ | 36.4 [1.2, 79.6] | 0.82 [0.35, 1.31] | −81.7 [−96.4, −57.9] |
| HFEA-Tech 50/50 | 24.6 [6.6, 46.4] | 0.82 [0.36, 1.34] | −75.0 [−79.2, −38.6] |
| EW5 UPRO/TQQQ/SOXL/TMF/UGL | 31.3 [9.6, 53.6] | 0.90 [0.43, 1.34] | −68.7 [−77.9, −40.6] |
| invvol clean4 lb=21 | 23.3 [10.5, 39.8] | **0.95 [0.52, 1.49]** | −52.5 [−62.6, −28.8] |
| invvol core6 lb=63 | 28.4 [11.5, 48.5] | 0.92 [0.49, 1.40] | −60.7 [−74.3, −35.4] |
| mom core6 lb=126 top4 | 32.9 [5.5, 64.6] | 0.84 [0.36, 1.33] | −72.6 [−90.8, −46.4] |
| **SPY BH** | **13.5 [5.3, 22.3]** | **0.82 [0.37, 1.34]** | **−33.7 [−51.8, −17.1]** |

Every LETF strategy's Sharpe CI **overlaps SPY's**. Claiming "LETF Sharpe
= 0.95 > SPY 0.82" is not statistically defensible — on resampled histories
the ordering routinely flips. Point estimates are ~15 years of one draw.

## 4. Deflated Sharpe (`letf_deflated_sharpe.py`)

Across 323 tested strategies on the IS period:

- SR distribution: min −0.52, median 0.73, **max 1.22**
- Expected max SR under null (N=323, non-skill) ≈ **0.90**
- **Observed max beats expected max by only ~0.05** — entirely consistent with
  luck given we searched 323 configs.

| Strategy | SR | Skew | Kurt | E[max SR] | **DSR** |
|---|---|---|---|---|---|
| HFEA-Tech 50/50 | 0.82 | −0.13 | 8.06 | 0.90 | **37.2%** |
| invvol clean4 lb=21 | 0.95 | −0.09 | 9.09 | 0.90 | **56.8%** |
| invvol-s core6 tv=60% | 0.90 | −0.19 | 10.29 | 0.90 | **49.5%** |
| mom core6 top5 | 0.87 | −0.20 | 8.76 | 0.90 | **44.6%** |
| EW-all17 | 0.63 | −0.47 | 13.12 | 0.90 | **14.6%** |

**Nothing passes DSR > 95%.** The best (57%) is a coin-flip. Given
multiple testing, we cannot claim statistical skill for any of the
headline strategies.

This is the single most important number in the audit.

## 5. Regime stress (`letf_regime_stress.py`)

Worst rolling periods and drawdown recovery:

| Strategy | Worst 1y | Worst 3y | Deepest DD | Recovery |
|---|---|---|---|---|
| 100% TQQQ | −81.0% | −11.3% | −81.7% (2021-11→2022-12) | 1108 d |
| HFEA-Tech 50/50 | −74.7% | −24.6% | −75.0% | **STILL UNDERWATER (4+ yr)** |
| EW5 LETF | −64.5% | −8.1% | −68.7% | 925 d |
| EW-all17 | −51.1% | −7.4% | −62.2% | 968 d |
| invvol clean4 lb=21 | −50.1% | −6.0% | −52.5% | 869 d |
| invvol core6 lb=63 | −56.0% | −0.1% | −60.7% | 800 d |
| mom core6 top4 | −69.2% | −3.6% | −72.6% | 882 d |
| **SPY BH** | **−19.7%** | +0.4% | **−33.7%** | **172 d** |

**Every LETF strategy subjects the investor to a 1-year period of −50% to
−81%.** HFEA-Tech (the IS winner) is still underwater 4+ years after its
2021 peak. SPY buy-and-hold had a −20% worst year and recovered in 172
days.

For a retail client, the experience of any LETF strategy in this set is
categorically different from SPY — the bootstrap says the Sharpes might be
statistically similar, but the lived volatility and drawdown-recovery
paths are not similar at all.

## 6. Family distributions (`letf_family_dist.py`)

| Family | n | IS SR median [max] | OOS SR median [max] | OOS CAGR median | OOS MDD median |
|---|---|---|---|---|---|
| invvol | 20 | 0.79 [0.99] | **0.71 [0.92]** | 19.0% | **−54.3%** |
| invvol-scaled | 80 | 0.76 [0.94] | 0.67 [0.97] | 15.0% | **−91.7%** |
| mom | 60 | 0.51 [0.95] | 0.59 [0.91] | 18.6% | −72.4% |
| static | 163 | 0.74 [1.22] | 0.81 [1.11] | 25.9% | −62.2% |

Median is the honest number for "a random pick in this family". On that
basis:

- `invvol` is the most robust family: narrow Sharpe dispersion, highest
  rank-stability IS→OOS, lowest median OOS MDD.
- `invvol-scaled` has a median OOS MDD of **−91.7%** — a coin-flip of
  bankruptcy in ~half the configs.
- `static` has high best-case but negative IS→OOS rank correlation —
  choosing by IS Sharpe is anti-informative.

## 7. What we CAN defensibly claim

After all checks, the one class of strategies that survives:

- **Inverse-vol across a small, fixed, unlevered-underlying-diverse basket,
  21-63 day lookback, monthly rebal, next-day-open execution.**
- Specifically `invvol clean4 lb=21` (UPRO/TQQQ/TMF/UGL equal-risk):
  OOS Sharpe 0.86, CAGR 23.6%, MDD −52.5%. Best IS→OOS rank stability
  (80% top-Q persistence within family).
- But: **DSR only 57%**, so we cannot claim the point estimate is
  statistically better than chance after selection bias.
- 1-year drawdown at worst is −50%, 3-year flat, recovery took 2.4 years.

## 8. What we CANNOT defensibly claim

- "LETF portfolios deliver 30%+ CAGR." The cross-universe median is
  closer to 15-18%.
- "Inv-vol beats SPY on a risk-adjusted basis." Sharpe CIs overlap
  entirely.
- "Our top strategy has Sharpe 1+." Deflated Sharpe says no.
- "This is safer than HFEA." Only `invvol` family shows rank stability;
  the rest are curve-fit to the 2011-18 regime.
- Any crypto inclusion result: the 2015+ / 2018+ windows are too short
  to survive even a weak multiple-testing correction and BTC/ETH have
  been in a historically anomalous bull regime.

## 9. Recommendation for the webapp strategy page

Ship one strategy, not a sweep:

- **Name:** "Levered risk-parity (clean4)" — UPRO / TQQQ / TMF / UGL,
  inverse-vol weights with 21-day lookback, monthly rebalance at next
  open, 15 bps costs.
- **Disclosure:** CAGR 23.6% OOS, MDD −52.5%, worst 1y −50%, 3-year
  flat stretches are expected. Statistically indistinguishable from
  SPY buy-hold at 95% confidence on historical data.
- **Position sizing:** recommend no more than 20–30% of a client's
  risk-asset allocation, complementary to SPY/AGG core — not a
  replacement.

All the "65% CAGR with BTC" / "HFEA-Tech 1.22 Sharpe" variants stay in
the backtest notebook, not in a client product.
