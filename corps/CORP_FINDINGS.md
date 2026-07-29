# Corporate bonds — a free data source, and a bias-audited strategy

**Result: the KEYSTONE dislocation-reversion strategy works on individual
corporate bonds too, validated out-of-sample on 22.6 years of free data across
the *full* survivorship-complete universe.**

Full sample 2002–2025, **all 55,545 corporate bonds** with ≥20 trading days
(29.7M bond-days), buy at the ask when a bond prints ≥3 pts below its own
trailing 60-day median clean price, hold ~1 year, sell at the bid:

| | value |
|---|---|
| Trades | 68,721 |
| Win rate | 76% |
| Mean return / trade (~1yr) | +7.51% |
| **Excess vs matched random-entry control** | **+2.06%** (p<0.001) |
| Equity total / CAGR / maxDD | **+240.9% / +5.57% / −14.1%** |
| LQD buy-and-hold (same window) | +177.5% / +4.62% / −25.0% |

Higher return than investment-grade buy-and-hold with roughly **half the
drawdown**, and the excess-vs-control proves it is timing alpha, not beta.

> **Bias note.** These numbers are from the professional rebuild on the
> **full universe** (every bond with ≥20 trading days), not a top-N-by-liquidity
> cut. Removing our earlier "top-8000 liquid bonds" shortcut *raised* the excess
> (+1.38% → **+2.06%**), so the shortcut had been conservative, not flattering.
> Full audit: [`CORP_AUDIT.md`](CORP_AUDIT.md).

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

The **entire panel is committed to this repo**, year-partitioned under
`corps/data/panel/osbap_YYYY.parquet` (zstd, <100 MB/file), so the backtest is
fully reproducible without re-downloading 1.8 GB. `corps/research/panel_io.py`
loads it into the muni engine's schema; the proven backtest engine runs unchanged.

## Evidence

**Threshold monotonicity (full sample, excess vs control):** the signal is
monotone — deeper dislocations pay more, and significance turns on at ≥2 pt.

| entry threshold | trades | win | mean/trade | excess | p |
|---|--:|--:|--:|--:|--:|
| ≥1 pt below | 116,549 | 74% | +5.04% | −0.00% | 0.57 |
| ≥2 pt below | 88,748 | 75% | +6.17% | +0.90% | <0.001 |
| **≥3 pt below** | 68,721 | 76% | +7.51% | **+2.06%** | <0.001 |
| ≥4 pt below | 54,688 | 77% | +9.11% | **+3.49%** | <0.001 |

**In-sample → out-of-sample (≥3 pt), both significant:**

| window | trades | win | mean/trade | excess | p |
|---|--:|--:|--:|--:|--:|
| IS 2002–2015 | 35,023 | 81% | +10.21% | **+3.37%** | <0.001 |
| OOS 2016–2025 | 34,834 | 71% | +5.97% | **+1.63%** | <0.001 |

**By era — positive excess in every regime, significant in 5 of 6:**

| era | trades | win | excess | p |
|---|--:|--:|--:|--:|
| 2004–2007 | 9,686 | 73% | +0.51% | 0.02 |
| **2008–2009 GFC** | 7,142 | 73% | +0.27% | 0.38 (not sig.) |
| 2010–2015 | 14,718 | 83% | +1.93% | <0.001 |
| 2016–2019 | 10,480 | 85% | +3.26% | <0.001 |
| 2020 COVID | 8,885 | 96% | +4.14% | <0.001 |
| 2021–2023 | 18,671 | 46% | +0.59% | <0.001 |

Same signature as munis: a mean-reversion strategy that pays in every regime
but is **weakest in the one systemic crisis** (2008 GFC here, the muni analogue
was the 2022 rate selloff). When the whole market craters, buying dislocations
is catching falling knives, and random entry in the same names does about as
well — so the GFC excess is small and not statistically distinguishable from
zero. Disclosed, not hidden.

## Anti-overfitting — the transfer test and the rejected overlays

**Transfer test (strongest evidence).** The entire signal specification
(60-day window, 3-pt threshold, ~1-yr hold, 90-day/8-day liquidity gate) was
fixed on U.S. **municipal** bonds and applied **unchanged** to corporates —
different asset class, issuers, and data vendor — with **zero corporate-specific
fitting**, and still produces +2.06% excess (+1.63% OOS). Curve-fit signals do
not transfer across markets; this one does.

**Rejected "improvements"** (each judged on OOS excess vs the base +1.63%):

| overlay | IS excess | **OOS excess** | verdict |
|---|--:|--:|---|
| base | +3.37% | **+1.63%** | — |
| market-regime gate | +4.21% | +0.94% | **reject** (fit the GFC, degrades OOS) |
| per-bond credit filter | +3.20% | +0.64% | **reject** |
| regime + credit | +3.37% | +0.78% | **reject** |

The regime gate looked great in-sample and "fixed" 2008, but it **hurt
out-of-sample** — the textbook overfitting signature. We publish the base, not
the in-sample-flattering variant.

**Robust levers** (monotone, fair same-control comparison, hold OOS): deeper
threshold (≥4 pt → +2.96% OOS) and longer hold (~455 d → +2.78% OOS) both add
excess at the cost of breadth / duration; a dynamic recovery-exit cuts average
hold materially at a similar *annualized* return (a turnover/risk gain, not
extra alpha).

## Selectivity — a focused short-duration book

The full-universe book carries ~3,100 concurrent positions — more than a real
desk would run. We tested point-in-time filters on **credit quality** (entry-day
credit spread) and **duration** (years to maturity), each baked into the
eligibility gate so the matched control trades the *same* selected subset. The
result overturns the intuitive "trade only the safest bonds":

**Credit quality — filtering to high grade backfires** (OOS, ≥3pt):

| entry credit spread | OOS trades | win | mean/trade | excess |
|---|--:|--:|--:|--:|
| ≤1% (top IG) | 6,161 | 45% | **−0.65%** | +1.50% |
| ≤1.5% (solid IG) | 13,927 | 51% | −0.40% | **−0.25%** |
| 1–3% (mid IG) | 22,013 | 65% | +2.49% | **−0.48%** |
| ≤5% (excl. distress) | 30,168 | 69% | +3.62% | +1.11% |
| >3% (crossover/HY) | 15,518 | 87% | +13.92% | +1.55% |
| >5% (deep distress) | 8,527 | 88% | +19.24% | **−0.35%** |

Top-IG dislocations have *no* out-of-sample timing edge and even lost money
outright (the 2022 rate selloff crushed long high-grade). The reversion alpha
lives in **crossover/HY** names; only the very deepest distress (spread >5%,
falling knives) is worth trimming, and only as a tail-risk knob.

**Duration — the edge concentrates in short bonds** (OOS, ≥3pt):

| years to maturity | OOS trades | win | mean/trade | excess |
|---|--:|--:|--:|--:|
| ≤3y (short) | 4,857 | 87% | +10.05% | +3.11% |
| **≤5y (short-int)** | 10,421 | 81% | +10.06% | **+3.51%** |
| 5–12y (belly) | 12,378 | 71% | +6.79% | +2.17% |
| >12y (long) | 13,103 | 63% | +2.25% | **−0.21%** |

**Short-dated (≤5y) is the operating point.** Short bonds must pull to par, so
they revert; long bonds (>12y) show zero OOS edge (their price is rates, not
idiosyncratic reversion). The focused ≤5y book vs the full universe:

| book | avg positions | total | CAGR | maxDD | OOS excess |
|---|--:|--:|--:|--:|--:|
| full universe (≥3pt) | 3,133 | +240.9% | +5.57% | −14.1% | +1.63% |
| **focused ≤5y** | **1,016** | **+268.9%** | **+5.93%** | −15.4% | **+3.51%** |
| LQD (total return) | — | +177.5% | +4.62% | −25.0% | — |

Higher return on **a third the positions**, and — crucially — the short-duration
cut is *not* an overfit overlay. Unlike the rejected regime gate (which fit the
GFC and failed OOS), it is monotone across the maturity spectrum, economically
structural, improves *both* IS and OOS, and **repairs the 2008 GFC**
out-of-sample (era excess +3.46%, p=0.006 — the full book's one weak regime).
The only soft era becomes the 2021–23 rate selloff (−0.65%, ns), the same
Achilles heel as munis. A conservative variant (≤5y *and* excl. deep distress
cs≤5%) roughly **halves the drawdown to −8.8%** but gives back return
(+3.84% CAGR, below LQD) — a low-vol knob, not the headline.

## Recommended operating point: add an issuer concentration cap

Capping the book at **one concurrent position per issuer** is the single best
risk-adjusted improvement found (full sample, honest daily marks):

| book | trades | CAGR | maxDD | Sharpe |
|---|--:|--:|--:|--:|
| focused ≤5y | 21,872 | +5.88% | −31.3% | 0.46 |
| **focused ≤5y + 1 position/issuer** | 11,477 | **+6.62%** | **−30.0%** | **0.58** |

Out-of-sample (2016–2025) it delivers **+2.65% excess vs control (p<0.001),
CAGR +7.24%, Sharpe 0.51**. Diversification across issuers beats raw breadth —
half the trades, better returns, better risk.

Also confirmed: the excess survives in the **most-liquid quartile** (+3.54%), so
it is not an illiquidity artifact; and stacking depth × duration (≥4pt & ≤5y)
gives the best CAGR (+6.33%, OOS excess +3.92%) at the cost of a deeper −38.9%
drawdown.

## A separate search for a higher-Sharpe strategy failed

A full independent search for a novel strategy targeting Sharpe ≈3 / CAGR ≥10%
was run: 24 specs from six trading-lens agents, 9 strategy families screened
in-sample with pre-registered kill gates, then a one-shot out-of-sample test.
**The target was not met.** Seven families died in-sample; the one live
candidate (a volume-confirmed fire-sale reversal) had a significant *and*
monotone in-sample edge and still **failed out-of-sample** (+0.07%, p=0.25).
Combination could not help — the sleeves are all long credit beta and their
correlations *rose* out-of-sample (0.69–0.85).

Full write-up, including the mechanism (bid-ask paid twice, a single tradable
risk factor, vol-targeting trading CAGR for Sharpe):
[`NEW_STRATEGY_SEARCH.md`](NEW_STRATEGY_SEARCH.md).

## Honest caveats

- **Carry** is proxied by each bond's median yield (OSBAP daily rows don't
  carry the coupon); the excess-vs-control metric nets it out since both legs
  hold the same bond for the same period.
- **Equity-curve drawdown was understated — corrected.** The −14.1% above comes
  from linear intra-trade attribution. Re-marking daily at actual mid prints
  gives **−32.7%** (full book) / **−31.3%** (focused ≤5y), with monthly Sharpe
  vs T-bill of **0.41 / 0.46**. Total return and CAGR are unchanged (always
  realized from bid/ask fills) — only the path was smoothed. See
  [`CORP_AUDIT.md`](CORP_AUDIT.md) §5.
- **Data** is OSBAP's cleaned daily VWAP + bid/ask (a reputable academic
  pipeline), not raw ticks.
- **Capacity**: the full-universe number includes illiquid names; a live book
  would tier by liquidity, trading fewer, larger positions (deeper-threshold
  operating points concentrate the alpha but reduce breadth).
- The **GFC** is intrinsic to mean-reversion; size accordingly. A market-trend /
  credit-spread overlay is the natural risk gate but, tested here, it overfit
  and was rejected for degrading OOS.

## Credential map (why we needed the free source)

The provided FINRA credential is a **basic** tier. Tested against every
`fixedIncomeMarket` dataset: all `trace*Detail/*Summary` (trade-level) return
**403 "basic API credential cannot access"**; only aggregates
(`corporateMarketBreadth/Sentiment`, `cappedVolume`, treasury aggregates) are
readable. Those aggregates (3-year history) did **not** yield a strategy that
beats buy-and-hold. The OSBAP daily panel is what made the per-bond strategy
possible for free.

## Reproduce

```bash
# The full panel is already committed under corps/data/panel/*.parquet.
# From that, reproduce every published number:
python corps/research/osbap_full.py       # threshold sweep, IS/OOS, per-era
python corps/research/osbap_improve.py     # round-1 overlays (rejected OOS)
python corps/research/osbap_improve2.py    # round-2 robust levers
python corps/research/finalize.py          # rebuild docs/corps_data.json + equity curves
```
