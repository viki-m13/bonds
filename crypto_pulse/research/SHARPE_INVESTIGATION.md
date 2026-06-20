# The hunt for a price-action strategy with Sharpe > 3 — an honest record

Goal set by the brief: **invent a price-action / technical trading strategy with
an annualized Sharpe ratio over 3**, "however necessary, honestly." This document
is the honest record of that hunt across every dataset available in the repo.

## TL;DR

A genuine, causal, cost-aware, out-of-sample **Sharpe > 3 was not attainable**
on any daily-or-hourly OHLCV data available here. Every construction that *prints*
a Sharpe above ~2 turned out to be a **bid-ask-bounce / stale-price artifact**
that evaporates — and usually flips sign — the moment you stop trading at the
exact bar used to form the signal. The best **honest** price-action result found
is **PULSE**, a vol-targeted daily-crypto trend+breakout book at **Sharpe ≈ 1.2
net of 10 bps, −16% max drawdown** (`strategy_daily.py`). That is a real,
tradeable edge; 3 is not, on this data.

This is not a defeatist conclusion — it is the single most important fact about
high-Sharpe claims, demonstrated rather than asserted.

## What was tested (all causal: signal at close of bar t, traded later)

### 1. US equities, daily (PIT S&P 500 panel, 720 names, 2004–2026)
Market-neutral cross-sectional books, executed open[t+1]→open[t+2] (no
look-ahead), sector-neutralized, vol-targeted, realistic 3 bps/side:

| signal | IS Sharpe | OOS Sharpe |
|---|---|---|
| short-term reversal (3–10d) | ~0.9 gross / ~0 net | ~0.3 gross / negative net |
| residual (beta-neutral) reversal | 0.26 | 0.38 (negative net) |
| multi-horizon reversal composite | 1.08 gross | 0.68 gross (negative net) |
| reversal + momentum + low-vol ensemble | ≤1 | ≤0 |

Best honest equity result: **net Sharpe ~0.9 in-sample, ~0 out-of-sample** — the
short-term reversal alpha has decayed since ~2016 (well documented). Nowhere near 3.

**The equity mirage, quantified (5-day reversal, IS Sharpe):**
trade *at* the formation close (impossible) → **2.13**; skip one day (tradeable
market-on-close) → 1.36 gross, **0.89 net**; open-to-open → 0.76. The drop from
2.13 to ~0.9 is pure bid-ask bounce.

### 2. Crypto, daily (data/crypto, 111 coins, 2014–2026, real OHLC)
Net of 10 bps/side:

| family | full Sharpe | note |
|---|---|---|
| time-series trend (TSMOM, inv-vol) | ~1.0 | weak out-of-sample |
| cross-sectional momentum | **negative** | crypto XS-momentum has not worked recently |
| cross-sectional short-term reversal | negative net | killed by turnover |
| **trend + 20d Donchian, vol-targeted (PULSE)** | **1.20** | **−16% maxDD; the keeper** |

PULSE sub-periods: 2020–22 **+1.31**, 2023–24 +0.93, 2025–26 −0.06 (trend
struggled in the recent chop). A real, attractive market-neutral strategy — but
its Sharpe is ~1.2, and it touches ~2 only in the strongest trending regime.

### 3. Crypto, hourly (binance.us, 20 liquid coins, 5 years, fetched live)
This was the most promising route — 24/7 hourly bars give thousands of bets/year,
the regime where high Sharpe legitimately lives. It produced the cleanest
illustration of why "Sharpe 3" claims are usually fake.

`mirage_demo.py` (cross-sectional 3-hour reversal, net 5 bps):

| execution | Sharpe | ann. return |
|---|---|---|
| trade **at** the formation close (impossible) | **+16.98** | +504% |
| **skip 1 bar** (causal, tradeable) | **−6.83** | −187% |
| skip 2 bars | −11.31 | −300% |

A +17 Sharpe collapses to −7 by skipping a single bar. The "edge" is entirely
bid-ask bounce and stale prints on a thin venue; once you trade where you
actually could, it is not just gone but negative. Time-series and cross-sectional
*momentum* on the same data are symmetrically negative artifacts. **The hourly
data is not clean enough to support an honest hourly strategy**, and the apparent
high Sharpe is the artifact, not an edge.

## Why Sharpe 3 is hard here — the arithmetic

Annualized Sharpe ≈ (information ratio per bet) × √(independent bets per year).
- The honest per-bet edge in liquid daily price action is small (daily rank-IC
  ~0.01–0.03) and has **decayed** over time as it got arbitraged.
- Daily large-cap/major-coin strategies place only ~50–250 weakly-independent
  bets/year. To reach Sharpe 3 you would need ~36 *truly uncorrelated* Sharpe-0.5
  sleeves — there are not 36 independent price-action signals on this universe.
- The frequency that *does* supply enough bets (intraday) requires **clean
  tick/L1 data and realistic microstructure costs**; the free hourly data here is
  too noisy, and once costs/bounce are handled honestly the edge disappears.

## What a *real* Sharpe-3 price-action strategy would require (not available here)

1. **Clean high-frequency data** (consolidated tick / L1 quotes, not thin-venue
   hourly OHLC) for intraday mean-reversion or opening-range strategies, with a
   proper spread/impact model.
2. **A much broader, illiquid universe** (small/micro-cap equities or long-tail
   alt-coins) where reversal/illiquidity premia are large — and where this repo's
   own prior research notes the alpha actually lives ("small/illiquid corners").
3. **Non-price signals that price action only proxies** — order-flow imbalance,
   funding/basis carry in perps, options flow — i.e. not pure OHLCV.

None of these are present in the repo, so the honest deliverable is **PULSE
(Sharpe ~1.2)** plus this transparent account of why the bigger number is a
mirage. Claiming a 3 here would require trading at the formation bar, hiding
costs, or cherry-picking a regime — exactly the practices this record exists to
expose.

## Reproduce
```bash
python strategy_daily.py     # PULSE: honest daily-crypto strategy + equity curve
python mirage_demo.py        # the +17 -> -7 hourly bounce artifact (needs data/crypto_hourly)
# hourly data fetch (binance.us klines, ~5 yrs x 20 coins) — re-run if data/crypto_hourly absent:
#   see the fetch snippet committed below / in git history (api.binance.us /api/v3/klines, interval=1h)
```
The hourly CSVs (~45 MB) are git-ignored; the fetch is reproducible from the
public binance.us klines endpoint.
