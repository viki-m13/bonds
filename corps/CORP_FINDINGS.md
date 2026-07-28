# Corporate bonds — what the FINRA credential unlocks, and the honest result

**Two findings, stated plainly:**

1. **A basic/public FINRA Data credential does _not_ include trade-level
   TRACE.** It unlocks only aggregate datasets. The per-CUSIP trade tape that
   the KEYSTONE dislocation strategy needs sits behind an upgraded (paid)
   credential.
2. **On the aggregate data that _is_ accessible, no corporate strategy beat
   buy-and-hold.** We tested customer-flow and breadth signals honestly; one
   is directionally sensible but none survives as a tradable edge on the
   available (short, one-regime) history.

The per-bond framework (`corps/scripts/`, `corps/research/pipeline.py`) is
built and verified end-to-end; it runs the moment trade-level access exists.

## Credential access map (basic credential `d506…`)

Tested every FINRA `fixedIncomeMarket` dataset:

| dataset | access | content |
|---|---|---|
| `traceCorporateBondDetail` | **403** — "basic API credential cannot access" | per-trade prints (what KEYSTONE needs) |
| `traceCorporateBondSummary` | **403** | per-firm trade summaries |
| `traceAgency* / traceSecuritized* / traceTreasuries*` | **403** | per-trade detail/summary |
| `corporateMarketBreadth` | ✅ 200 | daily advances/declines/52wk-hi-lo, by grade |
| `corporateMarketSentiment` | ✅ 200 | daily trades/volume by grade × flow (cust buy/sell) |
| `corporatesAndAgenciesCappedVolume` | ✅ 200 | daily volume by grade/144A |
| `treasuryDailyAggregates`, `treasuryMonthlyAggregates` | ✅ 200 | treasury aggregates |

So the credential sees **market-level aggregates**, not individual bonds.

## What we tested on the accessible aggregates (2023-07 → 2026-07, ~3y)

Data downloaded by `corps/scripts/download_aggregates.py`; ETF benchmarks
(LQD/HYG/AGG/VCIT/USHY) from Yahoo.

**a) Customer-flow imbalance → forward ETF return.** Net customer flow
`(custBuyVol − custSellVol)/(…)` by grade vs forward LQD/HYG returns:

- Investment grade: correlation ≈ 0 at every horizon (noise).
- High yield: weak *contrarian* tilt (net customer selling → mildly higher
  forward returns), |corr| ≤ 0.10. Directionally consistent with the
  dislocation thesis, but thin.

**b) Breadth → forward ETF return.** The 52-week-high/low breadth is more
strongly *contrarian*, especially in HY (corr **−0.47** at 42 days: many
bonds at new highs → lower forward returns; washed-out breadth → recovery).

**c) Does the breadth signal trade?** No. As a timing overlay (long HYG when
breadth not extended, else cash / LQD / AGG), every configuration
**underperformed buy-and-hold HYG** on both the in-sample (2023-24) and
out-of-sample (2025-26) splits:

| strategy | ann return | Sharpe | maxDD |
|---|--:|--:|--:|
| breadth-timed HYG (best config) | +8.4% | 1.49 | −4.6% |
| **buy-and-hold HYG** | **+8.0%** | **1.54** | −4.6% |
| buy-and-hold LQD | +5.0% | 0.72 | −6.3% |

The correlation is real but not monetizable: 2023-2025 was a persistent bond
rally, so any time out of the market cost more upside than it saved in
drawdown. And with ~3 years of one regime and overlapping forward windows,
the effective sample is far too small to claim an edge regardless.

## Honest conclusion

- **The basic FINRA credential cannot power a per-bond corporate KEYSTONE.**
  Trade-level TRACE requires an upgraded credential; the code is ready for it.
- **The aggregate data alone does not yield a corporate strategy that beats
  buy-and-hold** on the available history.

## Path forward

1. **Upgrade the FINRA Data credential** to a tier that grants
   `traceCorporateBondDetail`. Then:
   `python corps/scripts/download_trades.py build-universe` →
   `download` → `python corps/research/pipeline.py is|oos`. The per-bond
   dislocation strategy runs unchanged (same engine as the 3,085-muni result).
2. Or source trade-level corporate TRACE from a licensed feed
   (Bloomberg/ICE) into the same `date,price,ytw,par,side` schema.

Reproduce the analysis above: `corps/research/aggregate_analysis.py`.
