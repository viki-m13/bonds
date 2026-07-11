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

**Code complete and wired; needs a (free) FINRA data credential to pull the
trade tape.** The pipeline is verified end-to-end on synthetic data
(`pipeline.py selftest`) and the HTTP/auth plumbing is verified against
FINRA's public API (`finra_client.py`).

## Data — FINRA TRACE (the corporate analogue of MSRB EMMA)

Every corporate-bond trade is reported to FINRA and disseminated via **TRACE**
(price, yield, size, side). Trade-level TRACE is served by the **FINRA Data
API** (`api.finra.org`, dataset `fixedIncomeMarket/traceCorporateBond`) and
requires OAuth2 client credentials — **free** registration at the FINRA API
Developer Center. (We verified the API is live: public datasets return data
unauthenticated; the TRACE datasets return `401` until a token is supplied.
The Morningstar-hosted legacy Bond Center is deprecated/gated and not used.)

```bash
export FINRA_API_CLIENT_ID=...        # from FINRA API Developer Center (free)
export FINRA_API_CLIENT_SECRET=...
```

## Pipeline

```bash
# 0. verify wiring (no creds needed)
python corps/research/pipeline.py selftest
python corps/scripts/finra_client.py           # public-API self-test

# 1. discover the liquid corporate universe from a recent TRACE window
python corps/scripts/download_trades.py build-universe 2024-01-01 2025-12-31

# 2. download full per-CUSIP trade histories for the top-N liquid bonds
python corps/scripts/download_trades.py download 1500

# 3. build the daily panel and run the honest backtest
python corps/research/pipeline.py panel
python corps/research/pipeline.py is           # in-sample
python corps/research/pipeline.py oos          # locked out-of-sample
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
