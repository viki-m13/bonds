# SHARPE3 — the honest hunt for a Sharpe-3 stock-picking strategy

**Mandate**: run until a stock-picking strategy with net Sharpe ≥ 3 is found,
using only the point-in-time data already in this repo (daily + intraday).

**Status**: research complete — see verdict at the bottom. All numbers below
are net of costs, PIT-universe, next-day execution unless labeled otherwise.
Everything is reproducible: `sharpe3/experiments/exp01..exp20`, results in
`sharpe3/results/*.json`.

## Data used

| dataset | span | contents |
|---|---|---|
| S&P 500 PIT panel (`data/pit/summit_panel.parquet`) | 2004–2026 | open/close/volume + daily PIT membership, ~410 members/day |
| NASDAQ-100 PIT panel | 2004–2026 | OHLCV + membership |
| Tiingo full US universe | 1990–2026 | 24k tickers incl. 8.9k delisted (survivorship-clean), adjClose+volume |
| SEC 8-K filings | 1994–2026 | 1.02M filings with item codes (2.02 = earnings) |
| SEC XBRL revenue | 2011–2026 | quarterly revenue panel |
| Intraday 5-min | 2016–2026 | 7 ETFs (SPY, QQQ, IWM, DIA, GLD, TLT, XLF) |
| Intraday-derived daily + VWAP | 2016–2026 | 118 large-cap stocks |

## Methodology

- **Causality contract**: signal row d uses info through close of day d;
  execution at next open (default), next close, or same-close MOC (flagged).
- **Costs**: 5 bps/side large-cap, 10 mid, 20 small; sweeps 1–40 bps.
- **OOS discipline**: primary split 2019-01-01 (some tables 2020/2022);
  yearly Sharpe tables for every candidate; ML strictly walk-forward with
  embargo, refit annually.
- **Universe hygiene**: PIT membership masks; delisting-inclusive broad panel;
  data-error guards (|r|>100%/day → NaN); no shorting of names outside the
  liquid tiers.

## The map: every family tested

| # | family (best variant) | gross SR | net SR | IS | OOS (2019+) | verdict |
|---|---|---|---|---|---|---|
| 01 | 1–21d close reversal (S&P500 LS deciles) | 0.6–0.9 | ≤0.6 | ≤0.8 | ≈ 0 / negative | dead OOS |
| 01 | 12-1 / 6-1 momentum LS | −0.3 | −0.4 | −0.5 | ≈ 0 | dead |
| 01/11 | overnight-return momentum (21–252d) | 1.2–1.6 | 1.2–1.4 | 1.4–1.9 | −0.8…+0.2 | pre-2015 artifact-heavy; dead OOS |
| 01/02 | intraday-cum reversal 5d (best price signal) | 1.9 | 1.5 | 1.9 | −0.1 | decayed to zero |
| 02 | residual/beta-neutral, sector-neutral reversal | 0.3–0.6 | ≤0.1 | ≤0.4 | negative | worse than raw |
| 03 | PEAD on 8-K 2.02 reactions (large-cap) | 0.2–0.4 | ≤ −0.1 | −0.3 | −0.7…−0.2 | arbed away |
| 03 | no-news reversal conditioning | 1.9 | 1.5 | 1.9 | −0.3 | conditioning real, edge still dead OOS |
| 04 | same-close MOC execution of reversal | — | worse | — | — | overnight leg is *adverse*; next-open is optimal |
| 05 | broad-universe (Tiingo, delisting-incl.) reversal by ADV tier | 0.5–1.2 | −8…−0.1 | — | — | all killed by costs; small-cap "alpha" = bid-ask bounce |
| 06 | earnings-announcement premium (PIT-predicted windows) | 0.3 | 0.0 | −0.2 | +0.3 | noise |
| 07 | sector lead-lag, big-brother→little-brother | ≤0.7 | ≤ −0.2 | — | negative | dead |
| 08 | ML v1: 34-feature walk-forward Ridge/LightGBM, 5d | 0.2–0.5 | ≈ 0 | −0.5 | +0.5 | IC 0.010 (t=2.6), too small net |
| 09 | 8-K item-type event drifts (17 item codes) | — | — | t≤6 IS | t≤2.7 OOS | small broad drift; sleeve material only |
| 10 | calendar (turn-of-month, weekday) | ≤0.8 | ≤0.7 | — | — | TOM premium no longer differentiated |
| 11 | overnight-only session strategies | up to 5 (full) | — | 6.3 | ≤0.7, ≤0 at 5bps | **textbook artifact**: SR 15–36 in 2004–06, ~0 post-2019 |
| 12 | S&P 500 add/delete reconstitution | — | 0.2 | 0.5 | −0.6 | dead since 2012 (pre-2012 CARs were huge) |
| 13 | close-vs-VWAP / range-position (118 stocks) | ≤0.7 | ≤ −0.3 | — | negative | dead |
| 14 | ML v2: 2010+ training, h=1+5, buffered portfolio | see below | see below | — | — | best ML result; still ≪ 3 |
| 15 | intraday 5-min ETF anomalies (overnight, intraday-mom, gap-fade, last-30-min) | — | ≤0.8 | — | ≤1.1 | famous anomalies ≤1 net today (not stock-picking anyway) |
| 16 | conditional liquidity provision (reversal after stress) | — | ≤1.1 | — | ≤0.4 | conditioning can't rescue a dead edge |
| 17 | NDX OHLC range/Parkinson/gap signals | ≤0.3 | ≤ −0.1 | — | negative | dead |
| 18 | revenue growth/acceleration/surprise (XBRL, 70d lag) | ≤0.1 | ≈ 0 | — | ≈ 0 | dead |

(Values are the best variant per family; full grids in `results/`.)

## The uniform decay pattern

Nearly every family shows the same signature: strong 2004–2012, weakening
2013–2018, ≈ 0 or negative 2019+. This is not a bug in the harness — it is
the well-documented institutionalization of short-horizon equity stat-arb.
The pre-2012 numbers replicate the classic literature (short-term reversal,
overnight anomalies, index reconstitution, PEAD); the post-2018 numbers
replicate its documented decay.

<!-- ENSEMBLE_SECTION -->

<!-- VALIDATION_SECTION -->

<!-- VERDICT_SECTION -->
