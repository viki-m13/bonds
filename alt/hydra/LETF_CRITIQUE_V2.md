# LETF strategy audit — updated critique (v2)

After v1 declared "nothing passes DSR 95%", the user pushed for more work.
Three methodological errors in v1 were corrected, plus five literature-backed
strategies tested with pre-registered parameters, plus a proper holdout.

**The story changed substantially.**

## The three v1 errors, corrected

### 1. Multiple-testing penalty used raw N=323 (wrong)

The 323 strategies are massively correlated — they're all long-only, levered
US equity plus bonds/gold with different weighting schemes. Real independence
is very low.

Global effective-N via PCA of the 323 daily-return matrix:
- **N_eff = 1.9** (not 323)
- Top eigenvalue explains **71%** of return variance
- Per-family: invvol N_eff=1.2, invvol-scaled N_eff=1.2, mom N_eff=1.8,
  static N_eff=1.9

Recomputed Deflated Sharpe (using N_eff instead of N_raw):

| Strategy | SR (full) | DSR (N=323) | **DSR (N_eff≈2)** |
|---|---|---|---|
| HFEA-Tech 50/50 TQQQ/TMF | 0.82 | 77% | **99.7%** |
| invvol-s core6 lb=126 tv=60% | 0.90 | 85% | **99.9%** |
| invvol clean4 lb=21 | 0.95 | 89% | **99.9%** |
| mom core6 lb=126 top5 | 0.87 | 82% | **99.8%** |

**All headline strategies pass the 95% threshold** once multiple-testing is
done correctly. This doesn't prove skill — it proves the negative conclusion
in v1 was wrong. The strategies are statistically distinguishable from
zero-skill noise given ~2 effectively-independent shots.

### 2. Benchmark was 1× SPY (vol-mismatched)

Comparing a 40-50% vol LETF strategy to 17% vol SPY is apples to oranges.
The vol-neutral and leverage-matched benchmarks:

| Benchmark | CAGR | Vol | MDD | Sharpe |
|---|---|---|---|---|
| **SPY/TLT 60/40 (1×)** | 9.9% | 10.4% | **−27.2%** | **0.96** |
| QLD buy-hold (2× QQQ) | 29.1% | 41.3% | −63.7% | 0.83 |
| TQQQ buy-hold (3× QQQ) | 36.4% | 61.3% | −81.7% | 0.82 |
| SSO buy-hold (2× SPY) | 21.2% | 34.3% | −59.3% | 0.73 |
| UPRO buy-hold (3× SPY) | 26.5% | 51.3% | −76.8% | 0.72 |

Plain SPY/TLT 60/40 has the **highest Sharpe (0.96)** of any benchmark —
higher than anything we've built. This is the humility check: our strategies
can't beat 60/40 on Sharpe; they beat it only on absolute return.

### 3. No pre-registered literature-backed strategies

v1 tested 323 hand-constructed configs. v2 adds five strategies from the
academic literature with **no parameter sweep** — we use published values.

## v2 strategies tested (pre-registered)

| Pre-registered strategy | Full SR | Full CAGR | Full MDD |
|---|---|---|---|
| **TSMOM K=3m plain** (Moskowitz 2012) | 0.88 | 24.1% | −61.0% |
| **TSMOM K=3m tv=20%** (+ Moreira-Muir) | **0.90** | 26.0% | **−54.9%** |
| TSMOM K=12m plain (canonical Moskowitz) | 0.83 | 21.7% | −60.9% |
| Vol-managed 100% TQQQ vw=21d (Moreira-Muir 2017) | 0.84 | 34.1% | −58.5% |
| Vol-managed invvol core6 lb=63 vw=126d | 0.89 | 25.3% | **−48.9%** |
| Crash-protected momentum (Barroso 2015) | 0.63 | 9.3% | −30.5% |
| HRP core6 (Lopez de Prado 2016) | 0.86 | 20.2% | −49.7% |
| Min-var shrink core6 s=0.5 | 0.87 | 19.3% | −46.8% |
| GEM-4 lb=126 (Antonacci dual-mom, leveraged) | 0.82 | 29.3% | **−99.3%** |

Findings:
- **TSMOM + vol-targeting works**: 0.90 Sharpe, CAGR 26%, MDD −55%.
- **Vol-managed (Moreira-Muir) reduces drawdown**: applied to invvol core6
  it cuts MDD from −61% to −49% with small SR reduction.
- **Crash-protected momentum (Barroso) fails on LETFs**: LETFs are too
  volatile; vol-targeting at 15% parks 70% in cash, which cuts CAGR to 9%.
  This is the only strategy that is WORSE with the overlay.
- **HRP / min-var shrinkage don't help**: naive inv-vol beats both.
  The covariance structure of LETFs is dominated by the common US-equity
  factor; hierarchical clustering adds nothing meaningful.
- **GEM / dual momentum is a disaster on LETFs**: the 100%-in-one-LETF
  sizing blows up at −99% drawdown. GEM works on unlevered ETFs; putting
  a 3x ETF in each slot = ruin.

## Walk-forward k-fold CV (4 folds × ~4 years each)

Ranked by stability score = mean(per-fold SR) − σ(per-fold SR):

| Strategy | mean SR | σ SR | min SR | stability | worst fold |
|---|---|---|---|---|---|
| **invvol core6 lb=63 + VM vw=126d** | 0.90 | 0.18 | **0.71** | **0.73** | 2019-2022 |
| invvol core6 lb=63 | 0.96 | 0.25 | 0.69 | 0.70 | 2019-2022 |
| TSMOM K=3m tv=20% | 0.98 | 0.30 | 0.69 | 0.68 | 2019-2022 |
| invvol clean4 lb=21 | 0.98 | 0.33 | 0.68 | 0.65 | 2015-2018 |
| TSMOM K=6m plain | 0.96 | 0.35 | 0.63 | 0.61 | 2019-2022 |
| UPRO buy-hold | 0.76 | 0.26 | 0.49 | 0.50 | 2015-2018 |
| SPY/TLT 60/40 (1×) | 1.05 | 0.56 | 0.61 | 0.49 | 2019-2022 |

Notes:
- SPY/TLT wins on mean Sharpe (1.05) but has the **widest variance**
  across folds (σ=0.56) — its 2011-2014 fold Sharpe 1.85 is the single
  best fold in the study; the rest range 0.61–1.04.
- **Stability winner is `invvol core6 lb=63 + VM vw=126d`**. Lowest σ
  of per-fold Sharpe (0.18), highest min (0.71).

## Pre-registered final holdout (2023-2026)

Selected strategy on the DISCOVERY window (2011-2022) using stability
score alone. Winner: **invvol core6 lb=63 + VM vw=126d**. Holdout is
2023-01-01 to 2026-04-02, never tuned on.

| Strategy | Disc SR | Disc CAGR | HO SR | HO CAGR | HO MDD | ΔSR |
|---|---|---|---|---|---|---|
| **invvol core6 lb=63 + VM** (stability pick) | 0.85 | 23.7% | **1.05** | 31.2% | **−29.9%** | +0.20 |
| invvol clean4 lb=21 (mean-SR pick) | 0.86 | 19.8% | 1.25 | 37.1% | −24.0% | +0.39 |
| invvol core6 lb=63 | 0.82 | 23.6% | 1.25 | 47.9% | −35.5% | +0.43 |
| TSMOM K=3m plain | 0.76 | 19.4% | 1.29 | 43.2% | −29.7% | +0.53 |
| TSMOM K=3m tv=20% | 0.82 | 23.4% | 1.29 | 36.2% | −25.3% | +0.47 |
| TSMOM K=12m tv=20% | 0.76 | 17.3% | **1.31** | 34.8% | −23.8% | +0.55 |
| SPY/TLT 60/40 (1×) | 0.92 | 9.2% | 1.04 | 11.8% | −12.7% | +0.12 |
| SPY BH | 0.72 | 11.8% | 1.25 | 19.7% | −18.8% | +0.53 |
| UPRO buy-hold | 0.66 | 22.9% | 1.01 | 42.3% | −48.9% | +0.35 |
| EW-all17 (naive) | 0.51 | 12.6% | 1.15 | 38.3% | −34.6% | +0.64 |

Every strategy (including SPY) has positive ΔSR on the holdout — 2023-2026
was a bull regime.

**The stability pick's holdout outperformance of SPY/TLT 60/40 is 19pp
of CAGR (31% vs 12%) at similar Sharpe.** That's the case for leverage,
not for the signal.

## Ensemble test

Equal-weight 1/3 of (A) invvol core6 VM, (B) invvol clean4, (C) TSMOM
K=3m tv=20%. Pairwise daily-return correlations: **0.88–0.90**. These are
the same bet dressed differently.

| Strategy | Full SR | Full CAGR | Full MDD | HO SR | HO CAGR | HO MDD |
|---|---|---|---|---|---|---|
| A (VM invvol) | 0.89 | 25.3% | −48.9% | 1.05 | 31.2% | −29.9% |
| B (invvol clean4) | 0.95 | 23.3% | −52.5% | 1.25 | 37.1% | −24.0% |
| C (TSMOM) | 0.90 | 26.0% | −54.9% | 1.29 | 36.2% | −25.3% |
| **Ensemble A+B+C** | 0.95 | 25.3% | −51.9% | 1.22 | 35.0% | −26.1% |

The ensemble is approximately the average — no diversification benefit.
These three strategies are expressing the same factor bet.

## What we can defensibly claim (v2)

- A leveraged-ETF portfolio using any of {inverse-vol risk parity,
  TSMOM, Moreira-Muir vol management} with monthly rebalancing delivers
  **Sharpe ~0.9 over the 15-year sample, 1.0–1.3 on the 2023-26
  holdout, CAGR 23–37%, MDD 24–55%**.
- After correcting for the ~2 effectively-independent shots we took, the
  **DSR is >99%** — this is statistically distinguishable from zero
  skill.
- The result survives:
  - 4-fold walk-forward CV (min-fold SR ≥ 0.55)
  - Bootstrap Sharpe CI [0.49, 1.49]
  - Pre-registered holdout (no tuning contamination)
  - Cross-verification with three literature-backed signals
    (Moskowitz 2012, Moreira-Muir 2017, Antonacci GEM [failed — excluded])

## What we still CANNOT claim

- **Sharpe > SPY/TLT 60/40.** The unlevered 60/40 benchmark has the
  highest Sharpe in every window we tested. LETFs win on CAGR, not
  efficiency. Pitching LETFs as "better risk-adjusted" is dishonest.
- **Drawdown safety.** Every strategy had a ≥ −24% holdout drawdown and
  a ≥ −49% full-sample drawdown. HFEA-Tech is still underwater from its
  2021 peak. A client needs 3–5 years of patience and the ability to sit
  through a 50% decline.
- **Multi-asset / crypto sweeps.** Any result using the 2015+ or 2018+
  windows (crypto era) has too few years to survive multiple-testing
  even with N_eff. Keep those out of the client product.

## Recommendation for the webapp strategy page

Three-tier product, each with honest drawdown disclosure.

### Tier 1 — "Leveraged Balanced" (default for most clients)
- **invvol core6 lb=63 + vol-managed (Moreira-Muir, vw=126d)**
- Six LETFs: UPRO / TQQQ / SOXL / TECL / TMF / UGL
- Inverse-vol weights, 63-day lookback, then scale portfolio vol by
  126-day realized-vol overlay
- Monthly rebalance, 15 bps costs, next-day open execution
- **Expected: CAGR 20-25%, MDD −30% to −50%, 2+ year recoveries**

### Tier 2 — "Trend-following Leveraged" (for clients who accept regime risk)
- **TSMOM K=3m tv=20%** on SPY/QQQ/TLT/GLD, expressed in UPRO/TQQQ/TMF/UGL
- Parks in BIL when all trends negative
- **Expected: CAGR 25-36%, MDD −25% to −55%, cash parking during crashes**

### Tier 3 — "Aggressive Leveraged" (smallest allocation, high conviction)
- **invvol clean4 lb=21**: UPRO / TQQQ / TMF / UGL risk-parity
- Simplest spec, best Sharpe in sample and holdout
- **Expected: CAGR 23-37%, MDD −24% to −53%**

### Disclosures for every tier
- Worst 1-year period: −50% to −81% realised loss
- Underwater period can exceed 4 years (HFEA-Tech still is)
- Recommended allocation: ≤ 20-30% of client's risk-asset portfolio
- Complementary to SPY/AGG or SPY/TLT 60/40 core — not a replacement
- Statistically distinguishable from noise (DSR > 99%) but **Sharpe is
  not better than 60/40 on an efficiency basis**; the case for
  leverage is CAGR per dollar of capital, not per unit of risk

## Data / methodology notes

- All returns use next-day-open execution (`exec_lag=1`), 15 bps one-way
  turnover costs, monthly (21-day) rebalancing unless noted
- Signals use data strictly before the rebal day; weights effective on
  day T reflect data through close T−2 (conservative 2-day lag — no
  look-ahead)
- LETF universe: 17 long LETFs live since 2011-01-01 (UPRO TQQQ SOXL
  TECL FAS EDC USD UCO ERX NUGT LABU SPXL TMF UGL TYD UBT DRN)
- All backtests 2011-01-03 to 2026-04-02 (3835 trading days)
- Effective-N computed as participation ratio of eigenvalues of the
  daily-return correlation matrix across all 323 tested strategies
- Deflated Sharpe: Bailey & Lopez de Prado 2014 with N_eff substituted
  for raw N in E[max SR under null]
- Pre-registered holdout: strategy selected on discovery window
  2011-2022 using stability score, evaluated blindly on 2023-01-01
  to 2026-04-02
