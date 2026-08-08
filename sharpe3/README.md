# sharpe3 — the honest hunt for a Sharpe-3 stock-picking strategy

Mandate (2026-08): *"invent a stock-picking strategy with a Sharpe of 3+,
using PIT data already in the repo (daily and intraday)."*

**Read [`FINDINGS.md`](FINDINGS.md) for the full record and the verdict.**

## Layout

```
sharpe3/
├── datalib.py         # PIT data loaders (S&P500 panel, NDX-100, Tiingo broad universe)
├── bt.py              # vectorized daily cross-sectional backtester + metrics
├── sleeves.py         # final candidate sleeve generator (daily net returns)
├── build_broad.py     # builds the filtered broad-universe panel to cache/
├── experiments/       # exp01..exp27 — the systematic map (each writes results/*.json)
├── results/           # scorecards + logs for every experiment
└── FINDINGS.md        # the record: methodology, every family tested, verdict
```

## Reproduction

```bash
cd sharpe3
python3 build_broad.py                    # once: broad panel cache (~2 min)
python3 experiments/exp01_baselines.py    # any experiment, standalone
python3 experiments/exp14_ml2.py          # ML sleeve (writes cache/exp14_pred_*.parquet)
python3 experiments/exp20_ensemble.py     # final ensemble
python3 experiments/exp21_validation.py   # bootstrap CIs, cost/exec stress
python3 experiments/exp27_horizon_tradeoff.py  # the ceiling arithmetic
```

Causality contract and cost model are documented in `bt.py` and `FINDINGS.md`.
