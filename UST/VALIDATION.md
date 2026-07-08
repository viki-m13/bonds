# Validation protocol (written before any backtest was run)

This file was committed **before** running any strategy experiments, and the
protocol below was not modified afterwards. It exists so the out-of-sample
claim is auditable from git history.

## Data split — fixed in advance

| Window | Range | Use |
|---|---|---|
| In-sample (IS) | 2010-01-04 → 2019-12-31 | all exploration, tuning, selection |
| Out-of-sample (OOS) | 2020-01-01 → last available date | run **once** with the frozen config |

Rules:

1. `src/run_experiments.py` refuses to load any data past 2019-12-31.
   Every design decision (signal family, lookbacks, fraction traded,
   rebalance frequency, weighting scheme, cost assumptions) is made from
   IS results only.
2. When IS exploration is finished, the chosen configuration is frozen in
   `config/final_strategy.json` and committed.
3. `src/run_final.py` is then run **one time** on the full panel. Whatever
   comes out — good or bad — is reported in `results/` and the README.
   No re-tuning after seeing OOS numbers. If OOS fails, the honest result
   *is* that it fails.
4. The OOS period includes March 2020 (Treasury market dislocation) and the
   2022 hiking cycle — a genuinely hard test, not a quiet regime.

## No-look-ahead rules in the engine

- Signals at rebalance date t use only data ≤ t (rolling windows, curve fit
  of day t's cross-section).
- Trades execute at the close of t and earn returns from t+1.
- The NSS curve is re-fit each day from that day's cross-section only.
- Universe membership at t uses only information known at t (remaining
  maturity, presence of a price).

## Costs

- Base case: per-trade cost = |Δweight| × half the FedInvest buy/sell spread
  for that CUSIP on trade date (floor 1bp per side). FedInvest spreads are
  wider than institutional inter-dealer spreads, so the base case is
  conservative for an institutional reader; a 2× cost stress is reported too.
- Shorting individual Treasuries is assumed to net to the general-collateral
  repo rate (no specialness). This is optimistic for bonds trading special;
  flagged as a limitation in the README.

## Survivorship / selection bias

- The FedInvest files list **every outstanding CUSIP on that day** —
  the universe is point-in-time by construction; no survivorship filter is
  possible even by accident (Treasuries also do not default in-sample).
- TIPS and FRNs are excluded up front (different cashflow mechanics, would
  require CPI/FRN reference data); this exclusion is made before any
  backtesting and applies to IS and OOS alike.

## Multiple-testing honesty

The IS grid is recorded in full in `results/is_experiments.csv` (every
configuration evaluated, not just the winner). The number of configurations
tried bounds the selection bias a reader should assume when discounting the
IS Sharpe. OOS is a single run of a single config.
