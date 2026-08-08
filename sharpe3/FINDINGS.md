# SHARPE3 — the honest hunt for a Sharpe-3 stock-picking strategy

**Mandate**: run until a stock-picking strategy with net Sharpe ≥ 3 is found,
using only the point-in-time data already in this repo (daily + intraday).

**Status**: research complete — see verdict at the bottom. All numbers below
are net of costs, PIT-universe, next-day execution unless labeled otherwise.
Everything is reproducible: `sharpe3/experiments/exp01..exp27`, results in
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
| 19 | per-stock month-of-year seasonality (Heston-Sadka); gross profitability (Novy-Marx) | ≈ 0 | ≈ 0 | — | ≈ 0 | dead |
| 22 | ETF late-day flow → next-day beta cross-section | ≤0.2 | ≤ −0.4 | — | negative | dead |
| 24 | pairs trading (GGR distance, sector-matched, top-50, z>2 entry, 2010+) | 0.36 | −0.06 | — | −0.03 | dead (as documented post-2002) |
| 25 | PCA factor-neutral construction of every headline signal (K=5/10/20) | 0.71 | −1.6…0.07 | — | — | removes 70% of vol, adds no alpha |
| 26/27 | ceiling arithmetic: measured IC, effective breadth, horizon trade-off | — | — | — | — | **the quantitative verdict** |

**Data deliberately not mined**: the 13F institutional-holdings panel
(`_13f_cusip.pkl`) covers only 16 irregular quarters (2022–2026). Fourteen
usable observations cannot support a Sharpe claim at any confidence, so it is
recorded as a data limitation rather than a tested family.

(Values are the best variant per family; full grids in `results/`.)

## The uniform decay pattern

Nearly every family shows the same signature: strong 2004–2012, weakening
2013–2018, ≈ 0 or negative 2019+. This is not a bug in the harness — it is
the well-documented institutionalization of short-horizon equity stat-arb.
The pre-2012 numbers replicate the classic literature (short-term reversal,
overnight anomalies, index reconstitution, PEAD); the post-2018 numbers
replicate its documented decay.

## Data forensics: the pre-2009 open-price artifact

Sanity check that protects every conclusion here: an equal-weight portfolio of
all members should have the same Sharpe measured open-to-open as
close-to-close. It does — after 2009. Before:

| era | close-to-close EW Sharpe | open-to-open EW Sharpe |
|---|---|---|
| 2004–2008 | **0.17** | **1.59** |
| 2009–2014 | 1.07 | 1.16 |
| 2015–2019 | 0.85 | 0.84 |
| 2020–2026 | 0.66 | 0.70 |

Pre-2009 Yahoo-derived opens are partly stale/synthetic, which *inflates any
open-execution backtest* in that era (+14%/yr of phantom EW return). The
spectacular 2004–2008 in-sample Sharpes (5d-reversal ≈ 3.2/yr, overnight-only
books at 15–36/yr in 2004–06) sit exactly in the artifact zone. The post-2015
window — clean data, modern market — is the honest testbed, and there
everything rounds to zero.

### The clean-era scorecard (the capstone table)

Re-running the headline signals on **2010+ data only** (exp23 — no artifact
era, 5 bps, next-open):

| signal | net SR 2010–26 | gross | 2010–15 | 2016–20 | 2021–26 |
|---|---|---|---|---|---|
| 5d reversal | −0.36 | 0.60 | −1.05 | 0.27 | −0.64 |
| intraday-cum reversal 5d (flagship) | −0.32 | 0.41 | −0.70 | 0.14 | −0.56 |
| overnight momentum 60d | −0.51 | −0.14 | −0.55 | −0.39 | −0.64 |
| overnight momentum 252d | −0.03 | 0.14 | −0.02 | −0.29 | 0.18 |
| momentum 12-1 | 0.08 | 0.21 | 0.48 | −0.31 | 0.27 |
| low-vol | −0.33 | −0.26 | −0.05 | −0.25 | −0.64 |

On clean data, in the modern market, **every classical cross-sectional price
signal on liquid US equities is ≈ 0 net**. The full-sample Sharpes in the map
above are pre-2009-era alpha plus the open-price artifact — not something that
exists to be traded today.

### Factor-neutral construction: tested, and it does not hide alpha (exp25)

Every portfolio above is a naive decile long-short, which carries large
uncontrolled market/sector/size/vol exposures. The standard desk response is a
risk model. I built one (rolling PCA on the trailing 252d member-return
correlation matrix, K = 5/10/20 statistical factors, refit monthly, applied
forward only; `riskmodel.py`) and rebuilt each signal as a factor-neutral,
dollar-neutral, inverse-vol optimized book:

| signal | construction | ann. vol | **gross** SR | net SR (5bps) |
|---|---|---|---|---|
| 5d reversal | naive decile | 0.222 | 0.62 | −0.28 |
| 5d reversal | factor-neutral K=10 | 0.072 | 0.67 | −1.46 |
| 5d reversal | factor-neutral K=20 | 0.066 | **0.71** | −1.60 |
| intraday-cum rev | naive decile | 0.202 | 0.41 | −0.25 |
| intraday-cum rev | factor-neutral K=20 | 0.058 | 0.57 | −1.12 |
| momentum 12-1 | naive decile | — | 0.21 | 0.08 |
| momentum 12-1 | factor-neutral K=5 | — | 0.42 | 0.07 |

The risk model works exactly as designed — it removes ~70% of portfolio
volatility (0.222 → 0.066). But **gross Sharpe barely moves** (0.62 → 0.71):
the factor exposure was carrying risk, not concealing alpha. And because the
cost drag is a fixed charge on turnover while the volatility it is measured
against has fallen 3×, net Sharpe gets *worse*. Better construction cannot
manufacture information that is not in the data.

### The horizon trap — the wall, quantified (exp26/exp27)

This is the decisive analysis. By the fundamental law of active management,
Sharpe_gross = IC × √breadth, where breadth = N_eff × (252/h) independent bets
per year. Measured on the clean era (2015+): the S&P 500 cross-section has
**N_eff = 315** (464 names, average residual correlation 0.001 — the breadth
really is there). Measured IC of the best signal at each horizon, versus what
the portfolio actually delivers after costs:

| holding period | measured IC (t) | independent bets/yr | **theoretical** max gross SR | realized gross SR | turnover | cost drag @5bps | **net SR @5bps** |
|---|---|---|---|---|---|---|---|
| 1 day | 0.0109 (3.1) | 79,350 | **3.07** | 0.65 | 864×/yr | −1.95 SR | **−1.31** |
| 2 days | 0.0090 (2.6) | 39,675 | 1.79 | 0.59 | 525× | −1.16 SR | −0.58 |
| 5 days | 0.0111 (3.2) | 15,870 | 1.40 | 0.47 | 228× | −0.50 SR | −0.03 |
| 10 days | 0.0085 (2.5) | 7,935 | 0.75 | 0.06 | 118× | −0.27 SR | −0.21 |
| 21 days | 0.0112 (3.4) | 3,779 | 0.69 | 0.12 | 59× | −0.14 SR | −0.01 |
| 63 days | 0.0100 (3.3) | 1,260 | 0.36 | −0.03 | 21× | −0.04 SR | −0.07 |

Read the table as a trap with two jaws:

- **Only the 1-day horizon has a theoretical ceiling above 3** — and only
  barely (3.07), assuming a *perfect* risk model, *optimal* weights and
  *zero* costs. There, turnover is 864×/yr and the cost drag alone is
  −1.95 Sharpe units at 5 bps. Net: −1.31.
- **The horizons whose costs are affordable** (21d: drag −0.14; 63d: −0.04)
  have theoretical ceilings of **0.69 and 0.36** — they cannot reach 3 even
  in principle, with perfect execution and free trading.
- Realized gross Sharpe is only 20–40% of theoretical even before costs,
  because a decile sort cannot extract a full IC — and exp25 shows that
  fixing the construction does not close the gap either.

There is no horizon on this data where breadth and affordability overlap
sufficiently. **To reach Sharpe 3 at a tradeable (5-day) horizon would require
IC ≈ 0.024 — roughly 2× the best measured — and at 21 days, IC ≈ 0.049,
about 4.4×.** Nothing in 24 families of price, volume, event, filing,
fundamental or ML signal came within a factor of two of that.

## The final ensemble (the honest deliverable)

Four sleeves defined a priori (economic motivation, robust construction — no
selection on OOS), equal-risk-weighted, S&P 500 PIT members, next-open
execution, 5 bps/side:

| sleeve | construction | SR full | OOS 2019+ | OOS 2022+ |
|---|---|---|---|---|
| `rev_i5_sm3` | 5d intraday-cum reversal, 3d-smoothed, decile LS | 1.54 | −0.06 | −0.58 |
| `onmom252` | 252d overnight-return momentum, decile LS | 1.15 | 0.18 | 0.08 |
| `liqprov_cond` | reversal book active only after ≥1% market drop in high-vol state | 0.20 | 0.35 | 0.11 |
| `ml_avg_sm3` | avg of both walk-forward LightGBM configs (34 features, h=1+5) | 0.03 | 0.21 | −0.18 |
| **combo (equal-risk)** | | **−0.04** | **0.22** | **−0.24** |
| oracle top-3 by OOS (upper bound, selection-biased by construction) | | 1.08 | 0.44 | — |

Pairwise sleeve correlations are low (|ρ| ≤ 0.27 except rev/onmom 0.62), so
diversification is working — there is simply almost nothing left to diversify.
An 8-K activity sleeve was excluded after diagnosis: its sign flips between
portfolio expressions (binary filer-basket −1.6 vs z-scored activity +0.02),
which is the signature of noise, not signal.

## Validation

- **Bootstrap (21d stationary block, 2000 draws)** on the vol-targeted
  ensemble: full-sample SR −0.32, 95% CI [−0.72, +0.09]; OOS-2019+ CI
  [−1.02, +0.31]; OOS-2022+ CI [−1.24, +0.34]. Zero is inside every interval.
- **Cost sweep** (flagship reversal): 2 bps → OOS 0.29; 5 bps → −0.06;
  10 bps → −0.64; 20 bps → −1.80. Even at an unrealistically-perfect 2 bps
  all-in, the OOS edge is ≈ 0.3, not 3.
- **Universe transfer** (NASDAQ-100 PIT panel): flagship reversal 0.06 full /
  −0.02 OOS; overnight momentum 0.22 / 0.14. Confirms the S&P results.
- **Execution stress**: same-close MOC execution *hurts* reversal (the
  overnight leg is adverse); next-open is already the optimum. Next-close is
  strictly worse. No timing assumption rescues anything.
- **ML ICs** (the aggregate information content of 34 engineered features):
  rank-IC 0.009–0.010, t ≈ 2.3–2.6, in both training configurations. Real,
  but an order of magnitude short (see below).
- **Harness verification**: `bt.run` reproduces a hand-computed portfolio
  exactly (diff = 0.0); SPY buy-and-hold through the same metrics gives
  Sharpe 0.65 (2004–2026), matching public figures.

## VERDICT

**No. A durable, honest, out-of-sample, net-of-cost Sharpe of 3 is not
attainable for stock picking on this repo's data (daily bars + filings-level
events, at daily rebalancing), and we can show why quantitatively.** This
mirrors the conclusion of the same hunt on crypto data
(`crypto_pulse/research/SHARPE3_VERDICT.md`) — and the fundamental reasons are
the same three walls:

1. **The information wall.** The measured post-2015 rank-IC of *everything
   engineerable from this data* — 24 families, a 34-feature walk-forward ML in
   two configurations, and an IC-weighted walk-forward combination of all of
   them — sits at **0.007–0.017 (t ≈ 2–3.6)** at every horizon from 1 to 63
   days. The walk-forward *combination* of all ten headline signals scores
   IC 0.0076 (t = 2.1) — combining does not raise it, because the signals are
   different projections of the same thin information. Reaching Sharpe 3 needs
   IC ≈ 0.024 at 5 days or ≈ 0.049 at 21 days: 2× to 4.4× what exists.
2. **The decay wall.** Every classical anomaly reproduces in-sample exactly
   where the literature found it (short-term reversal, overnight cross-section,
   PEAD, index reconstitution, lead-lag, turn-of-month) and every one of them
   is ≈ 0 or negative after 2018 in liquid US equities. This is the documented
   institutionalization of short-horizon stat-arb, visible end-to-end in our
   yearly tables.
3. **The cost wall — and the horizon trap it creates.** Breadth is not the
   binding constraint: N_eff = 315 independent names, and at a 1-day horizon
   that is 79,350 independent bets a year, enough for a *theoretical* ceiling
   of 3.07. But 1-day rebalancing costs 864×/yr of turnover — a −1.95 Sharpe
   drag at 5 bps. Slow the book down until costs are affordable (21–63 days,
   drag −0.14 to −0.04) and the theoretical ceiling has already fallen to
   0.69–0.36. **The horizons with the breadth cannot pay their costs; the
   horizons that can pay their costs lack the breadth.** Separately, where
   gross edges do persist (small caps below $20M ADV, gross 0.6–1.2),
   realistic 10–20 bps costs turn them into net −3 to −8 — the legendary
   "Sharpe 3–5 small-cap reversal" is bid-ask bounce, the same mechanism this
   repo's crypto research documented, reproduced here in equities.

   A fourth possibility — that better *portfolio construction* rather than
   better signals was the missing piece — was tested and rejected (exp25): a
   PCA risk model removes 70% of portfolio volatility but moves gross Sharpe
   only 0.62 → 0.71. The factor exposure carried risk, not hidden alpha.

**What a Sharpe 3 would actually require** (all outside this dataset):
per-stock intraday data with an execution simulator (queue/latency), options
or borrow data (event-vol and squeeze structure), or true non-price
information (news text, transcripts, credit card / alt-data). At the
infrastructure-free daily level, the honest ceiling measured here — like the
audited strategies already in this repo (GRANITE-XL ≈ 0.9, SUMMIT ≈ 1 on a
Sharpe basis, PULSE ≈ 1.2–1.5) — is **≈ 0.5–1.5**, and for *pure stock-picking
cross-section on today's market, ≈ 0–0.5**.

A backtest in this repo *can* be made to print Sharpe ≥ 3 — by including the
pre-2009 corrupted-opens era, trading the sub-$5M-ADV tier at fantasy costs,
or selecting sleeves on their OOS window. Each of those is a documented
defect, and the protocols of this repo (dca/RESEARCH_PROTOCOL.md,
VALIDATION_METHODOLOGY.md, XL_AUDIT.md) exist precisely to forbid them. The
deliverable of this project is the map, the harness, the artifact forensics,
and the measured ceiling — not a number that would not survive its first week
of live trading.

*Every claim above is reproducible from `experiments/exp01..exp27` +
`results/*.json` on the data in this repository.*
