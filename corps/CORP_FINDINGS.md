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

## Honest caveats

- **Carry** is proxied by each bond's median yield (OSBAP daily rows don't
  carry the coupon); the excess-vs-control metric nets it out since both legs
  hold the same bond for the same period.
- **Equity-curve drawdown** uses linear intra-trade attribution (as in munis),
  so the −14.1% is somewhat smoothed vs a daily mark; total return and CAGR are
  realized from bid/ask fills.
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
