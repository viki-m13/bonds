# CORPS — trading individual corporate bonds (dislocation-reversion)

Ports the validated **KEYSTONE** muni strategy to individual **corporate**
bonds. The thesis is identical: corporate bonds trade OTC through dealers,
most CUSIPs trade intermittently, and customers pay the bid–ask spread — so
short-term timing loses but a patient buyer of forced-seller dislocations
(buy ≥3 pts below a bond's own trailing 60-day median, hold ~1 year) can earn
a reversion premium. Corporate spreads are wider and credit dispersion richer
than munis, which *should* strengthen the signal — a hypothesis to test on the
data.

## Status

**Working corporate strategy, validated out-of-sample on 22 years of free
data.** Full sample 2002–2025 (8,000 liquid bonds): buy ≥3 pts below a bond's
own trailing 60-day median, hold ~1 year → **75% win, +6.22%/trade, +1.38%
excess vs a matched control (p<0.001)**; equity **+5.41% CAGR / −14.7% maxDD
vs LQD +4.61% / −25.0%**. OOS (2016–2025) excess +1.69% (p<0.001). Loses in
the 2008 GFC (the systemic-crisis failure mode, mirroring munis' 2022). Full
tables in **[`CORP_FINDINGS.md`](CORP_FINDINGS.md)**.

## Data — the free source (primary)

**[Open Source Bond Asset Pricing](https://openbondassetpricing.com/)** (Dickerson
et al.) publishes a processed **daily** corporate-bond panel from TRACE — free
direct download, no WRDS: 29.8M bond-days, 73,835 bonds, 2002-07 → 2025-03,
with daily clean price, **bid, ask**, yield, and credit spread. The bid/ask
lets us model KEYSTONE execution faithfully (buy at ask, sell at bid).

```bash
curl -o osbap.zip https://openbondassetpricing.com/wp-content/uploads/2025/12/stage1_osbap_0k_volume_2025.zip
unzip osbap.zip
python corps/scripts/build_osbap_panel.py stage1_osbap_0k_volume_2025.parquet
python corps/research/osbap_backtest.py sweep    # threshold + horizon scan
python corps/research/osbap_backtest.py full     # IS/OOS + per-era
```

## Data — FINRA TRACE API (secondary path)

FINRA's own Data API (`api.finra.org`) serves trade-level TRACE, but a **basic**
(free) credential grants only aggregates — `trace*Detail/*Summary` return 403.
The per-CUSIP pipeline (`finra_client.py`, `download_trades.py`,
`research/pipeline.py`) is built and verified, and runs unchanged the moment an
upgraded credential or licensed feed is available. The accessible aggregates
alone did **not** beat buy-and-hold (`aggregate_analysis.py`).

```bash
export FINRA_API_CLIENT_ID=... FINRA_API_CLIENT_SECRET=...   # basic tier: aggregates only
python corps/scripts/download_aggregates.py
python corps/research/aggregate_analysis.py
```

## What is reused vs corporate-specific

- **Reused unchanged** (the proven core): the honest backtest engine, the
  matched random-entry control, the price-dislocation signal, and the daily
  per-bond aggregation — imported directly from `munis/research/`. Corporate
  TRACE is normalized to the *same* schema (`date, price, ytw, par, side`
  with `side ∈ {S customer-buy, P customer-sell, D inter-dealer}`), so the
  engine operates identically.
- **Corporate-specific**:
  - `finra_client.py` — FINRA OAuth2 + dataset paging.
  - `download_trades.py::_side()` / `_normalize()` — maps TRACE's
    reporting-party / contra-party fields to the S/P/D convention. Field
    names follow FINRA's documented TRACE schema and are isolated in `COLS`
    for one-line adjustment against the live schema on first credentialed run.
  - **Coupon accrual**: TRACE trades carry no coupon, so absolute returns use
    a default coupon until a coupon-by-CUSIP reference join is supplied. The
    headline metric — *excess vs a matched control in the same bond* — nets
    coupon out almost entirely, so the core result is robust to this.

## Files

```
corps/
├── scripts/
│   ├── finra_client.py      # FINRA Data API OAuth2 client (+ public selftest)
│   └── download_trades.py   # universe scan + per-CUSIP TRACE download
├── research/
│   └── pipeline.py          # panel + backtest (reuses munis engine) + selftest
└── data/                    # universe/ and trades/ (populated when credentialed)
```
