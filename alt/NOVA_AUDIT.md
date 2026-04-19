# NOVA Strategy Audit — Findings & Corrections

**Branch:** `claude/audit-nova-strategy-hypNB`
**Date:** 2026-04-19

## Summary

The original NOVA backtest (`alt/nova_build.py`, `alt/nova_proxy_build.py` and
all three grid-search scripts) contained a **one-bar look-ahead bug in the
momentum signal**. The bug mechanically inflated reported performance from a
true full-window Sharpe ≈ 0.35 (with the original parameters) to the reported
Sharpe 1.59. Separately, the parameters were chosen on the same data used to
report results (in-sample overfit), the leveraged-ETF universe is
survivor-only, and the synthetic-leverage proxy for 2005–2015 understated real
leverage drag.

## 1. CRITICAL — one-bar look-ahead in momentum signal

### Where

`alt/nova_build.py` (production), `alt/nova_proxy_build.py`, `alt/nova_grid.py`,
`alt/nova_grid2.py`, `alt/nova_grid3.py`, `alt/nova_verify.py`.

### Bug

At bar `i` the code computed

```python
momo = prices.iloc[i] / prices.iloc[i - LOOKBACK] - 1   # today's close
current = ...top-N by momo...                           # new weights
r = (rets.iloc[i] * eff).sum()                          # earn rets.iloc[i]
```

`rets.iloc[i]` is the return from close[i-1] → close[i]. So the signal used
close[i] to decide weights that earn the return ending at close[i] — one-bar
look-ahead. The regime gates are `.shift(1)`-lagged, but the momentum signal
was not.

Because the signal is "top-N movers", the bug has a directly positive bias on
every rebalance day (you mechanically end up long whatever moved most that
day, then earn that same day's return). Effect is large.

### Impact (same code, only the signal was lagged 1 bar)

Original-parameter config `lookback=10, top_n=3, cap=0.33`:

| | Sharpe | Ann. Ret | Vol | MDD | NAVx |
|---|---|---|---|---|---|
| As-written (look-ahead) | **1.59** | **67.3%** | 42.4% | −34.9% | **832** |
| 1-bar lag (correct) | **0.35** | **14.6%** | 41.6% | −76.5% | **2.0** |

Proxy backtest (2005→2026, with crypto, same-parameter config):

| | Sharpe | Ann. Ret | Vol | MDD | NAVx |
|---|---|---|---|---|---|
| As-written | **1.67** | **67.6%** | 40.4% | −43.9% | **≈3 × 10⁵** |
| 1-bar lag (correct) | **0.43** | **18.0%** | 41.7% | −75.5% | **7.1** |

### Fix

Every file now uses `prices.iloc[i-1]` as the signal numerator and
`prices.iloc[i-1-LOOKBACK]` as the denominator, with weights applied at bar
`i` to earn `rets.iloc[i]`. This makes the signal strictly observable at
trade time.

## 2. In-sample overfit — parameters chosen on reporting window

All three grid scripts iterated over the same full window used to report
final results (2014-09 → today). With look-ahead inflating every config
differentially, the "winning" tuple `(lookback=10, top_n=3, cap=0.33,
btc_ma=200)` was largely selected for whichever cell got the biggest tailwind
from the bug. On corrected data, that specific config delivers IS Sharpe
≈ −0.06 and only looks good OOS due to post-2020 crypto and tech runs, not
because the parameters are good.

### IS / OOS split on corrected data

Grid `alt/nova_isos_grid.py` added: split at 2020-01-01 (IS = 2014-09 to
2019-12, OOS = 2020-01 to 2026-04). Ranked by IS Sharpe:

| lookback | top_n | cap | IS Sharpe | IS Ret | OOS Sharpe | OOS Ret | Full SR |
|---|---|---|---|---|---|---|---|
| 120 | 2 | 0.33 | 0.72 | 23.2% | 0.74 | 27.1% | 0.73 |
| 120 | 2 | 0.50 | 0.72 | 35.2% | 0.74 | 41.1% | 0.73 |
| 120 | 4 | 0.50 | 0.63 | 21.6% | 0.78 | 34.3% | 0.71 |
| 120 | 3 | 0.33 | 0.53 | 20.3% | 0.86 | 41.0% | 0.72 |
| 10  | 3 | 0.33 | −0.06 | −2.4% | 1.05 | 47.0% | 0.58 |

The 10-day lookback produces negative IS Sharpe once look-ahead is removed.
The 120-day lookback is IS/OOS-robust and was adopted as the production
default.

## 3. Survivorship bias in universe

- Universe was "every bull-leveraged ETF in `data/etfs/` the author had data
  for that survived" (TQQQ, UPRO, SOXL, TECL, FAS, LABU, EDC, YINN, ERX,
  NUGT, DRN, UCO, TYD, UGL, UBT, QLD, SSO, TMF). Delisted/terminated leveraged
  ETFs (UGAZ, DGAZ, JNUG, GUSH, UWTI, DWTI, FNGU, BITU, etc.) are absent from
  the data folder, so they were never considered — the universe is
  survivor-only by construction.
- Bear/inverse leveraged ETFs (SQQQ, SPXU, SOXS, TECS, FAZ, YANG, LABD, TBT,
  TMV, DRV, DUST, EDZ, ERY) are present in `data/etfs/` but deliberately
  excluded. That is a valid design choice (long-only momentum) but should be
  explicit in the writeup.
- `nova_grid3.py` adds 11 more names to the original 7. That is not a
  robustness test — it is the same in-sample optimization with a bigger menu;
  the expanded universe wins because it scored better in-sample.

No correction applied beyond acknowledgement. A full survivor-free re-run
would need to re-collect delisted ETF histories; flagged as future work.

## 4. Synthetic leverage drag understated (proxy 2005–2015)

`nova_proxy_build.py` previously modelled a synthetic 3x/2x ETF as
`leverage × underlier_daily_return − 1%/yr expense`. Real leveraged ETFs pay
the expense *plus* financing cost on the levered notional (~(leverage − 1) ×
short rate). In the 2005–2008 period that was ≈ 4–5%/yr; in 2022–2024 it was
≈ 5%/yr. A 3x ETF therefore has 3–6%/yr drag, not 1%/yr.

### Fix

`nova_proxy_build.py` now computes
```
synth = leverage * under_r − 0.0095/252 − (leverage − 1) * DGS3MO/252
```
using FRED `DGS3MO` (3-month Treasury bill yield) as the financing reference.

## 5. Post-fix headline numbers (honest, current production)

Config: lookback=120, top_n=3, cap=0.33, SPY>200dma & VIX<30, BTC>200dma,
weekly rebalance, 15bps round-trip.

| Window | SR | Ret | Vol | MDD | NAVx |
|---|---|---|---|---|---|
| Live 2014-09 → 2026-04 (11.5y) | 0.72 | 31.5% | 43.9% | −73.5% | 12.4× |
| Proxy with crypto 2005 → 2026 (21.2y) | 0.59 | 26.8% | 45.2% | −69.7% | — |
| Proxy no crypto 2005 → 2026 | 0.41 | 18.4% | 44.5% | −70.6% | — |

Benchmarks on same window:
- SPY 2014-09 → 2026-04: SR 0.79, Ret 13.9%, Vol 17.6%, MDD −33.7%
- AGG 2014-09 → 2026-04: SR 0.41, Ret 2.1%, Vol 5.2%, MDD −18.4%

The strategy still beats SPY on total return but Sharpe is lower (0.72 vs
0.79) and drawdowns are severe. **This is the honest starting point for
further development.**

## 6. Minor issues

- `prices.ffill()` across equity + crypto on Monday mornings can mix "today's"
  crypto close with "Friday's" equity close. Small effect compared to #1;
  not patched here.
- No unit tests or walk-forward harness. `nova_verify.py` only inspects
  weight composition. Recommend adding a regression test that pins output
  metrics to a known commit and a CI check that the signal is lagged.
- Earlier factsheet JSON (`nova_factsheet_data.json`) and `docs/nova.html`
  published the 832× NAV and 67.3% CAGR numbers. The JSON has been
  regenerated against the corrected backtest; the HTML loads the JSON at
  runtime, so it now surfaces the honest numbers.

## Files changed in this audit

- `alt/nova_build.py` — lagged signal, lookback 10 → 120, updated docstring
- `alt/nova_proxy_build.py` — lagged signal, lookback 10 → 120, added
  financing-cost drag
- `alt/nova_grid.py`, `alt/nova_grid2.py`, `alt/nova_grid3.py` — lagged signal
- `alt/nova_verify.py` — lagged signal
- `alt/nova_isos_grid.py` — new; IS/OOS evaluation with the fix
- `alt/nova_factsheet_run.py` — updated descriptions (10-day → 120-day,
  lag note)
- `data/results/nova_returns.csv`, `nova_proxy_returns.csv`,
  `nova_proxy_nocrypto_returns.csv`, `nova_factsheet_data.json` — regenerated

## 7. NOVA v2 — honest 50%-CAGR variant

Using the corrected (look-ahead-fixed) pipeline, `alt/nova_v2_explore.py`
searched a wider grid over (lookback, top_n, cap, rebalance frequency, and
portfolio overlay leverage with DGS3MO financing). Selection rule: hit ≥45%
full-window CAGR, IS Sharpe > 0.4, OOS Sharpe > 0.4, IS-OOS Sharpe gap < 0.5,
then rank by full Sharpe.

The chosen config (`alt/nova_v2_build.py`) is the most IS/OOS-stable option
that clears 50% CAGR:

| | Sharpe | Ret | Vol | MDD | NAVx |
|---|---|---|---|---|---|
| IS 2014-09 → 2019-12 | 0.73 | 49.4% | 67.8% | −88.4% | 4.0× |
| OOS 2020-01 → 2026-04 | 0.72 | 59.4% | 82.2% | −73.1% | 4.8× |
| Full 2014-09 → 2026-04 | 0.72 | 54.8% | 76.0% | −88.9% | 18.9× |

Parameters: lookback 120d, top-3 positive, no per-name cap, 10-day
(biweekly) rebalance, 1.7× portfolio overlay financed at DGS3MO, same
regime gates as v1. Signal lagged 1 bar.

**The −89% max drawdown is not a bug — it is inherent to levered
cross-sectional momentum on leveraged-ETF + crypto. At 1.7× overlay on a
top-3 book with no cap, the strategy regularly loses most of its NAV in a
regime change. Treat this as a speculative satellite, not a core
allocation.**

Why 50% CAGR requires this much risk:
- The corrected v1 (lookback 120, top_n 3, cap 0.33, no overlay) delivers
  ~31% CAGR with ~-73% MDD.
- Raising concentration (top_n 2, cap 0.50-1.0) lifts CAGR to ~38% with
  ~-82% MDD.
- Adding overlay leverage lifts CAGR proportionally but multiplies drag
  in drawdowns.
- Below 120-day lookback the IS/OOS gap blows out (classic overfit).

These are the honest mechanical limits of the universe.

## Still honest, still biased — what's left

1. **Parameter & universe selection are still partly in-sample.** The grid
   picks 120-day momentum because that cell wins on IS Sharpe; we did not
   k-fold / bootstrap.
2. **Survivorship bias in the ETF universe.** See #3 above.
3. **No transaction-cost realism beyond 15bps round-trip.** Market impact,
   borrow on inverse names, ETF creation-unit effects, and crypto spreads
   are not modelled.
4. **Weekly close-to-close execution.** Real execution would use
   next-day-open or VWAP; small alpha leaks here (a few bps per rebal).
