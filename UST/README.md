# UST — trading individual Treasury bonds/bills like stocks

Self-contained research folder: download **CUSIP-level** daily prices for
every outstanding US Treasury, build a validated total-return panel, and
backtest cross-sectional strategies that go long/short **individual bonds**
(not futures, not ETFs) with a pre-registered out-of-sample protocol.

## TL;DR results

*(filled in after the single frozen OOS run — see `results/final_results.json`)*

## Data

| What | Source | Coverage here |
|---|---|---|
| Daily per-CUSIP prices (buy / sell / end-of-day, per $100 par) | [FedInvest / TreasuryDirect](https://www.treasurydirect.gov/GA-FI/FedInvest/securityPriceDetail) (Bureau of the Fiscal Service) | 2010-01-04 → 2026-07-07, 4,133 trading days, 2,466 CUSIPs, 1.4M rows |
| Security metadata (issue/dated dates, original term, coupon) | [TreasuryDirect securities API](https://www.treasurydirect.gov/TA_WS/securities/search) | all 11,969 auction records |
| Cross-check yields | FRED DGS2 / DGS10 / DGS30 | validation only |

The FedInvest file lists **every outstanding marketable CUSIP each day**, so
the universe is point-in-time by construction (no survivorship bias). TIPS
and FRNs are excluded up front (they need CPI index ratios / FRN reference
rates for correct returns); the tradeable universe is nominal fixed-coupon
notes and bonds, with bills used for the cash-rate proxy and curve fitting.

### Derived panel (`src/build_dataset.py`)

For each (date, CUSIP): clean price, accrued interest (street convention,
actual/actual semiannual anchored at maturity), dirty price, **coupon-adjusted
daily total return**, solved YTM, modified duration, remaining maturity, and
the FedInvest buy/sell spread (used as the transaction-cost model).

### Validation (`src/validate.py`, output in `results/validation_report.txt`)

- Solved YTMs of ~2y/~10y/~30y securities vs FRED constant-maturity series:
  **median gap ≈ 1bp** (p95 < 8bp) over 4,126 overlapping days — independent
  confirmation of both the price data and the bond math.
- Coupon capture per bond-year = coupon rate exactly (median ratio 1.000).
- Zero daily returns outside duration-scaled bounds; the largest moves in the
  panel are the real March-2020 30-year dislocations (±7%).
- No calendar gaps > 5 days; ~350 securities per day.

## Honest out-of-sample protocol

See `VALIDATION.md`, which was **committed before any experiment ran**
(check the git history). Summary:

- **IS 2010–2019** for all design and tuning; the harness refuses to load
  data past 2019-12-31. Every configuration tried is recorded in
  `results/is_experiments.csv` (so you can discount the IS Sharpe for
  selection across N configs).
- Config frozen in `config/final_strategy.json`, then **one** full run
  (`src/run_final.py`) reports IS and OOS separately — including the
  March-2020 dislocation and the 2022 hiking cycle.
- Execution: signals at close t, trade at close t, earn returns from t+1.
- Costs: per-CUSIP FedInvest half-spread per side (floor 1bp), 2× stress
  reported.

## How to reproduce

```bash
pip install -r requirements.txt
python3 src/download_fedinvest.py          # ~4,150 requests, ~15 min
python3 src/download_metadata.py
python3 src/build_dataset.py               # ~4 min, writes data/processed/panel.parquet
python3 src/validate.py
python3 src/run_experiments.py             # IS grid only (refuses OOS data)
python3 src/run_final.py                   # frozen config, IS+OOS report
```

`data/processed/panel.parquet` is committed, so steps 2–4 are optional.

## Layout

```
UST/
├── VALIDATION.md            # pre-registered protocol (committed before results)
├── config/final_strategy.json
├── src/
│   ├── download_fedinvest.py   # daily CUSIP price files
│   ├── download_metadata.py    # TreasuryDirect auction metadata
│   ├── build_dataset.py        # panel: accrued, returns, YTM, duration
│   ├── bondmath.py             # vectorized street-convention bond math
│   ├── curve.py                # daily NSS curve fits (grid + weighted LS)
│   ├── strategies.py           # cross-sectional signals & L/S construction
│   ├── backtest.py             # cost-aware engine, no look-ahead
│   ├── validate.py             # data-quality + FRED cross-checks
│   ├── run_experiments.py      # IS-only tuning harness
│   └── run_final.py            # the single frozen IS+OOS run
├── data/processed/panel.parquet
└── results/
```

## Limitations (read before believing any backtest)

- FedInvest prices are the Fiscal Service's official daily marks, not
  executable dealer quotes; fills at mid ± half the FedInvest spread are an
  approximation of retail-ish execution. Institutional on-the-run spreads
  are tighter; deep off-the-runs can be wider in stress.
- Short legs are assumed to finance at general collateral; bonds trading
  *special* in repo would add cost to shorts (notably on-the-runs).
- No position-size vs. issue-size constraint (fine at small AUM; the
  strategy trades ~$100M+ issues).
- Bills are not in the long/short universe; the strategy is a coupon
  notes/bonds relative-value trade.
