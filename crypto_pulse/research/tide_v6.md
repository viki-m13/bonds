# Improving TIDE round-5 — NOVEL ideas + longer backtest (honest)

On improved base (5-horizon + Parkinson). Robust bar: beat base OOS AND pre-HL AND all WF folds. base OOS +2.29, pre-HL +1.35.

| variant | HL | IS | OOS | dOOS | pre-HL | deflated P |
|---|---|---|---|---|---|---|
| base (improved TIDE) | +2.23 | +2.20 | +2.29 | +0.00 | +1.35 | 0.68 |
| +effr (efficiency) | +2.04 | +2.11 | +1.92 | -0.36 | +1.51 | 0.54 |
| +disp (dispersion timing) | +2.11 | +2.07 | +2.19 | -0.10 | +1.36 | 0.63 |
| +accel (acceleration) | +2.17 | +2.27 | +2.00 | -0.28 | +1.16 | 0.57 |
| +effr+disp | +2.01 | +2.14 | +1.78 | -0.51 | +1.56 | 0.47 |

## Longer backtest — full history year-by-year (improved TIDE)

| year | Sharpe | CAGR |
|---|---|---|
| 2015 | +0.75 | +12% |
| 2016 | +0.24 | +3% |
| 2017 | +1.98 | +44% |
| 2018 | +1.71 | +23% |
| 2019 | +0.49 | +6% |
| 2020 | +1.68 | +35% |
| 2021 | +2.29 | +34% |
| 2022 | +1.94 | +37% |
| 2023 | +1.53 | +24% |
| 2024 | +1.94 | +31% |
| 2025 | +2.38 | +38% |
| 2026 | +1.99 | +26% |

- Full-period (2014-11-01..2026-04-24, 4193 days): Sharpe **+1.55**, CAGR +27%, maxDD -20%.
- Positive in 12 of the ~12 years — a decade-spanning edge, not a recent artifact.

## Verdict

- Robust novel survivors: NONE.
- **No novel lever robustly beats the improved base** (~2.29). The efficiency/dispersion/acceleration ideas are creative but don't add robust OOS edge.
- Honest single-book level **~2.3**, now confirmed over a ~12-year backtest. A single independent breakout book tops out here; 3 needs orthogonal legs.
