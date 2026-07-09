# Trading individual munis like stocks — what actually works

**Bottom line:** short-term price-action *timing* on individual munis does
not work — the dealer bid-ask spread eats every round trip. But a
**deep-dislocation reversion** strategy does: buy a bond when it prints
**≥3 points below its own trailing 60-day median** (a forced-seller
dislocation), hold **~1 year**, sell into a customer bid. Full-sample
2013–2025, across 1,350 trades in 1,416 bonds:

| | value |
|---|---|
| Win rate | **73%** |
| Mean return / trade (~1yr, incl. coupon) | **+4.42%** |
| **Excess vs matched random-entry control** | **+3.10%** |
| Bootstrap p-value | **<0.001** |

The excess is measured against random entries *in the same bonds over the
same window*, so it is not carry, not credit premium, and not
bond-selection — it is genuine within-bond timing alpha from buying
idiosyncratic dislocations that mean-revert. It survived a real
out-of-sample test (+4.45% excess, p<0.001, on 2023-2025 entries the
strategy was locked before seeing).

**And the honest caveat, up front:** this is a mean-reversion / liquidity-
provision strategy, so it has one clear failure mode — a *sustained*
rate selloff. In 2022 it lost (−1.99% excess, 32% win): when the whole
market reprices down, "below the 60-day median" fires on everything and
there is no reversion, only more selling. You catch falling knives. It
works in 4 of the 5 eras we split out; it is net-positive across all 13
years *including* 2022; but it is not all-weather.

---

## How we got here (the honest path, not a highlight reel)

### 1. The cost that kills naive trading

Munis are a dealer market: you buy at the dealer's ask (a customer-buy
"S" print) and sell at the dealer's bid (a customer-sell "P" print). The
median same-day round-trip spread in our data is **~0.44 points** (mean
~0.9). Validation (`data/VALIDATION.md`) confirms the data is real: 100%
agreement with an independent EMMA summary endpoint, Δprice/Δyield
correlation −0.98, and 96.9% of two-sided days have customer-buy ≥
customer-sell.

### 2. Short-term timing fails — as it must

Reversion ("firesale"), momentum, and post-issuance ("new-issue")
strategies that round-trip in days–weeks all lose ~1.2–1.8% per trade and
**none beats its matched random control**, in-sample or out:

| family (short hold) | OOS n | win | mean/trade | excess vs ctrl | p |
|---|--:|--:|--:|--:|--:|
| firesale | 2186 | 32% | −1.24% | +0.23% | — |
| momentum | 5167 | 32% | −1.17% | −0.04% | 0.87 |
| new_issue (≤10d) | 463 | 45% | −0.38% | −0.03% | 0.72 |

The signal identifies *when a bond is trading*, not *when it is
mispriced*. Two weeks of carry (~0.2%) can't cover a ~0.9-point spread.

### 3. The economics flip with holding period

The spread is a **one-time** cost; carry + pull-to-par is a **growing**
return. Forward return of a customer round-trip (buy ask → sell bid,
coupon credited), unconditional across all bond-days:

| hold | mean return | win rate |
|--:|--:|--:|
| 30d | −1.00% | 37% |
| 90d | −0.41% | 49% |
| 180d | +0.46% | 62% |
| 365d | +1.78% | 69% |
| 730d | +4.64% | 77% |

Past ~6 months you're positive. The earlier strategies simply used the
wrong holding period for the instrument.

### 4. Selection: cheapness is monotone — but mostly cross-sectional

Sorting all 1-year holds by entry cheapness is cleanly monotone (cheapest
price decile +4.8%, cheapest yield decile +4.4%, vs +1.8% average). **But
most of that is which bonds you pick, not when.** Once the matched control
nets out bond identity, a *yield*-cheap signal adds almost nothing
(+0.07% excess) — it's confounded by rate moves. The clean signal is a
*price* dislocation relative to the bond's own recent level.

### 5. The edge that survives the control

`price_discount` — buy when the customer-buy price is ≥ N points below the
trailing 60-day median mid — is monotone and significant **in-sample**
(entries 2012–2022):

| threshold | IS n | win | mean/trade | excess vs ctrl | p |
|---|--:|--:|--:|--:|--:|
| ≥1pt below | 1146 | 59% | +0.65% | −0.13% | 0.65 |
| ≥2pt below | 960 | 62% | +1.79% | +0.95% | 0.002 |
| **≥3pt below** | 791 | 63% | +2.55% | **+2.07%** | **<0.001** |

Locked mechanically (`lock_configs.py`) and run **once** out-of-sample:

| window | n | win | mean/trade | excess vs ctrl | p |
|---|--:|--:|--:|--:|--:|
| OOS 2023-01 → 2025-04 (1yr holds) | 608 | 87% | +6.32% | **+4.45%** | **<0.001** |

### 6. The regime breakdown (the most important table here)

`price_discount ≥3pt`, excess vs control, by era:

| era | trades | win | excess | p |
|---|--:|--:|--:|--:|
| 2013–2016 | 108 | 94% | +2.79% | <0.001 |
| 2017–2019 | 63 | 94% | +2.17% | 0.007 |
| 2020–2021 (COVID crash+recovery) | 238 | 86% | +10.65% | <0.001 |
| **2022 (rate selloff)** | 401 | **32%** | **−1.99%** | 1.00 |
| 2023–2025 (recovery) | 564 | 85% | +4.22% | <0.001 |

The OOS window (2023–2025) overlaps a muni **bull market** — yields peaked
in late 2022 and fell through 2024 — which is why OOS (+4.45%) prints
hotter than IS (+2.07%) and win rates hit the high 80s. **The in-sample
+2.07% (multi-regime, includes the 2022 loss) is the durable estimate;
the +4.45% is what a good regime looks like.** The full-sample number
(+3.10% excess over 2013–2025) sits between them and is the honest headline.

### 7. A regime overlay helps but costs trades

Gating entries on the broad market (only buy when MUB ≥ its 100-day
average) nearly removes the 2022 losses and lifts full-sample excess to
+4.26%, but cuts the trade count ~6× (791→128). It is offered as a risk
overlay, not baked into the core — filtering on the event that most
motivated it (2022) risks curve-fitting, so we disclose both.

## Is it live right now?

No — and that's the strategy working as designed. As of the last data
date (2026-07-08), **zero** bonds trade ≥3pt below their own trend; the
deepest of 696 recently-active names is −1.6pt and the median bond is
slightly *above* trend. A liquidity-provision strategy is supposed to be
dormant when nobody is force-selling. `current_picks.py` runs the screen
daily; it lights up in a selloff.

## What this is, honestly

- **A patient, value / liquidity-provision strategy in individual CUSIPs**
  — systematically choosing *which* munis to own when they dislocate,
  which is "trading them like stocks" on the timescale the instrument
  actually rewards (~1 year, not intraday).
- **Real and significant** (+3.1% excess over a matched control across 13
  years, p<0.001) — not carry, not credit premium, not composition.
- **Not all-weather**: it is short volatility / short a rate-shock. Size
  it as the mean-reversion strategy it is, optionally with the market
  overlay, and expect it to sit in cash when the market is calm.
- **Capacity-limited**: deep idiosyncratic dislocations are rare
  (~100–250/yr across ~1,400 liquid bonds), and each fill is one customer
  buying into someone else's forced sale.

## Reproduce

```bash
python scripts/build_universe.py && python scripts/download_trades.py
python scripts/validate_data.py
python research/panel.py
python research/run_backtest.py is      # in-sample grid
python research/lock_configs.py         # mechanical config lock
python research/run_backtest.py oos     # one-shot out-of-sample
python research/current_picks.py        # today's actionable screen
```
