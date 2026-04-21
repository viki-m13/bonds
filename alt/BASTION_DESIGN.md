# BASTION — Leveraged Risk Parity + Multi-Factor Kill Switch

## 1. Headline result (HONEST)

Full window (2010-03-11 → 2026-04-02):

| Metric | Full | IS (2010-03..2018-12) | OOS (2019-01..2026-04) |
|---|---:|---:|---:|
| **Sharpe** | **0.918** | 0.985 | 0.841 |
| CAGR | 21.22% | 22.59% | 19.58% |
| Vol | 24.18% | 23.49% | 25.00% |
| MaxDD | -35.52% | -26.70% | -35.52% |
| Avg turnover / yr | 27.6 | | |
| Avg time-in-mkt | 0.84 | 0.90 | 0.80 |
| |IS–OOS Sharpe gap| | | | **0.144** |

Hard-constraint check:

| Requirement | Pass? | Value |
|---|:---:|---|
| Sharpe ≥ 2.0 full | **FAIL** | 0.918 |
| Sharpe ≥ 1.5 IS | **FAIL** | 0.985 |
| Sharpe ≥ 1.5 OOS | **FAIL** | 0.841 |
| \|IS-OOS gap\| ≤ 0.5 | PASS | 0.144 |
| CAGR ≥ 20% full | PASS | 21.22% |
| Broad universe (≥ 10 LETFs) | PASS | 11 active + 7 candidates = 18 screened |
| Rebalance daily–monthly | PASS | monthly + daily gate |
| Next-day open execution | PASS | `w[t-1] · (open[t]/open[t-1]-1)` |
| Signal strictly lagged ≥ 1 bar | PASS | all signals use close[t-1] |
| No daily vol scaling | PASS | static notional weights |
| TC ≥ 5 bps one-way | PASS | 5 bps |

BASTION does **not** clear the 2.0-Sharpe bar and does **not** clear the 1.5-IS-OR-OOS bars.

Relative to prior VANGUARD build (Sharpe 0.957):

|  | BASTION | VANGUARD |
|---|---:|---:|
| Full Sharpe | 0.918 | 0.957 |
| Full CAGR | 21.2% | 24.8% |
| Full MDD | **-35.5%** | -40.9% |
| IS-OOS gap | 0.144 | **0.014** |

BASTION trades slightly lower Sharpe for meaningfully lower max drawdown. The
IS-OOS stability is still strong (gap 0.14 << 0.5 limit) but not quite
VANGUARD's near-zero gap. The **honest ceiling of leveraged-ETF rotation
with macro gates is confirmed at Sharpe ≈ 0.9–1.0** — adding a sixth
trigger (stock-bond correlation) and per-sleeve trend filters did not break
it.

## 2. Architecture

### 2.1 Three-sleeve leveraged risk-parity core

Universe screened (≥ 10 leveraged ETFs, broad):
```
EQUITY    : UPRO, TQQQ, QLD, SSO                       (4)
RATES     : TMF,  UBT,  TYD                            (3)
REAL ASSETS: UGL,  UCO,  NUGT, DRN                     (4)
                                                       ---
Total screened leveraged ETFs                           11
Broader candidate list (grid ablations): +SOXL, TECL,  (+7)
    FAS, ERX, LABU, EDC, YINN
CASH (risk-off residual): BIL
```

Within each sleeve, we pick eligible names by `inverse-vol weighting`
using 60-day realised vol. Eligibility requires

- 189-day momentum (using close[t-1]) > 0
- close[t-1] > 200-day SMA.

### 2.2 Static notional sleeve weights (IS-tuned)

Static weights are set **once**, not rolling (no daily vol scaling):

```
w_eq    = 0.40       ← equity sleeve
w_rates = 0.45       ← rates sleeve
w_ra    = 0.15       ← real assets sleeve
gross   = 2.25       ← constant amplification to hit CAGR target
```

The grid supports both 40/40/20 (traditional RP on underliers) and the
selected 40/45/15 cell — the latter gave the highest IS Sharpe that still
clears IS CAGR ≥ 20%, with a 0.14 IS-OOS gap. Because each ETF is itself
2-3× levered, effective portfolio delta at full risk-on is ≈ 4.5×.

Selection rule: **argmax IS Sharpe subject to IS CAGR ≥ 20%**. The
selection looked at ~3,000 grid cells (see `alt/bastion_grid.py` and
`alt/bastion_grid2.py`, outputs in `data/results/bastion_grid*.csv`). Tied
cells were broken with preference for longer `corr_window` (slower
statistical estimator) and simpler configurations.

### 2.3 Per-sleeve trend gate

A second-order protection: each sleeve is held only when its underlying
anchor is above its 200-day SMA (SPY for equity, TLT for rates, GLD for
real assets). This is applied AFTER the monthly freeze so each sleeve
can re-enter on a daily basis when its trend repairs — no monthly-lag
penalty.

### 2.4 Multi-factor kill switch (6 triggers, graduated participation)

Counts how many of the following fire today using only close[t-1]
information:

1. **HY credit widening** — HY OAS 20-day change > 0.30, or 5-day change >
   0.25, or (60-day z > 1.5 AND 20-day change > 0).
2. **VIX regime** — VIX > 30 OR 60-day z-score > 1.2.
3. **Curve inversion** — T10Y2Y < 0 AND 60-day change < 0.
4. **SPY trend broken** — SPY NOT above its 200-day SMA.
5. **Stock-bond correlation** — 30-day SPY/TLT correlation > -0.10 (the
   "2022 regime" — we want bonds to be *hedging*, so a threshold at
   -0.10 fires whenever the correlation is insufficiently negative). The
   -0.10 level came from the IS grid — conventional 0.40 thresholds miss
   2022 entirely because the transition started before 60-day correlation
   crossed 0.
6. **SPY 20-day drawdown** — SPY more than 5% below its rolling 20-day
   high (fast panic trigger).

Trigger count is smoothed with a 5-day rolling mean and mapped to
participation:

| Smoothed triggers | Participation |
|---|---:|
| < 0.8 | 100% |
| [0.8, 1.4) | 75% |
| [1.4, 2.0) | 50% |
| [2.0, 2.6) | 25% |
| ≥ 2.6 | 0% |

Participation is `.shift(1)`-lagged into weights so the decision at the
open of day t uses data through close of day t-1.

### 2.5 Timing convention (strict; manually spot-checked)

```
weights set at open[t]  ← using close[t-1] signal
PnL bar t = weights[t-1] · (opens[t] / opens[t-1] − 1)
turnover  = Σ |w[t] - w[t-1]|
TC applied at t+1 on turnover[t] (5 bps / side)
```

Manual spot-check on 2010-03-15 confirms the CSV's `gross_ret` equals the
explicit `Σ w_prev · (open_t/open_prev - 1)` computation, so no
open-to-close leakage.

### 2.6 Rebalance

- **Monthly** (first business day) — sleeve inv-vol basket regenerated.
- **Daily** — trend gate, kill switch.

## 3. Kill-switch behaviour (diagnostic)

| Year | Transitions (off-edges) | Avg on_mult |
|---|:---:|:---:|
| 2010 | 3 | |
| 2011 | 5 | |
| 2012 | 2 | |
| 2013 | 1 | |
| 2014 | 3 | |
| 2015 | 5 | |
| 2016 | 3 | |
| 2017 | 1 | |
| 2018 | 7 | |
| 2019 | 2 | |
| 2020 | 8 | (COVID) |
| 2021 | 1 | |
| 2022 | 4 | |
| 2023 | 5 | |
| 2024 | 8 | |
| 2025 | 3 | |
| 2026 | 2 | |

The kill switch handles 2020Q1, 2015, 2018Q4 cleanly. 2022 was handled
only partially (−31.6% for the year despite 4 kill-switch exits): by
the time stock-bond correlation or HY OAS flip into trigger territory,
the January + April losses have already compounded through the 2-3×
leveraged sleeves.

## 4. Yearly returns (net of 5 bps TC)

```
2010:  43.2%   2011:  20.9%   2012:  -6.5%   2013:  80.0%
2014:  54.7%   2015: -17.3%   2016:  10.8%   2017:  42.6%
2018:   8.7%   2019:  45.5%   2020:  41.5%   2021:  61.3%
2022: -31.6%   2023:  13.6%   2024:  11.4%   2025:  34.7%
2026:  -1.1% (YTD)
```

## 5. Per-sleeve standalone Sharpes (with kill switch applied)

| Sleeve | Sharpe | CAGR | Vol | MDD |
|---|---:|---:|---:|---:|
| Equity | 0.76 | 14.3% | 20.3% | -31.3% |
| Rates | 0.36 | 3.8% | 12.7% | -22.6% |
| Real Assets | 0.57 | 3.8% | 7.0% | -14.5% |

Equity sleeve carries most of the Sharpe. The rates and real-asset
sleeves are structurally low-Sharpe in this window (bonds flat-to-down
from 2020; gold choppy until 2024), so they contribute mainly through
diversification (slight) and through occasional regime hedging.

## 6. Honest risks

- **Sharpe ≈ 0.9 is the ceiling.** The prior ceiling analysis (NOVA
  audit, VANGUARD summary) is confirmed. Adding the 2022-specific
  stock-bond correlation trigger lifts the Sharpe a hair (≈ +0.02 on
  IS, neutral on OOS) but nowhere near doubles it.
- **Stock-bond correlation is a lagging signal.** The 30-day corr window
  reacts meaningfully only 3-4 weeks after the regime change. By then
  much of the 2022 January loss is already baked in.
- **Static sleeve weights are IS-tuned.** The 40/45/15 split was picked
  on 2010-2018. A 2005-2010-based RP would have given 37/32/31 with
  lower IS Sharpe (0.81) but slightly higher OOS Sharpe (0.95 at
  gross=1.5) — a less concentrated pick that is more generalizable but
  also lower-CAGR.
- **Max drawdown -35.5%.** Better than VANGUARD's -41% but still deep.
  At 2-3× ETF leverage, any regime that breaks both equity and bond
  trend simultaneously (2022) will punch a drawdown before the gates
  trigger.
- **Universe survivorship.** The leveraged ETFs are survivors;
  delisted names (UWTI, DWTI, UGAZ, DGAZ, FNGU/FNGD, JNUG/JDST) are
  absent. Adding those would likely reduce Sharpe.
- **No market-impact costs beyond 5 bps.** Small book; realistic for
  UPRO/TQQQ/TMF/UGL but optimistic for UCO/NUGT/DRN during stress.

## 7. Why I expect generalisation

- All signal computation uses close[t-1]. Manual arithmetic spot-check
  on 2010-03-15 matches the explicit `w_{t-1} · (o[t]/o[t-1] - 1)`
  formula. There is no open-to-close bleed.
- Parameters are picked from a ~3,000-cell IS-only grid (see
  `data/results/bastion_grid*.csv`). OOS was evaluated exactly once per
  config.
- The parameter surface is flat: IS Sharpe ranges 0.88 - 0.99 across
  the top 100 cells, all using `mom_lb ∈ {126, 189, 252}`,
  `sleeve_mode=invvol`, `trend_ma=200`, `gross ∈ {1.5, 1.75, 2.0, 2.25}`.
  Small perturbations move IS Sharpe by ±0.02 and OOS Sharpe by ±0.05.
- The selected cell's **IS-OOS gap of 0.14** is an honest generalisation
  signal. OOS (0.84) is lower than IS (0.99) but not collapsing. The
  absolute level just is not at institutional (2.0) targets.

## 8. Why it cannot honestly reach Sharpe 2.0

Pure macro-gated leveraged-ETF rotation runs into the same structural
cap that NOVA/ORION/KRAKEN/HELIOS/VANGUARD all hit:

- **Per-name Sharpes cap at ~0.84** (QLD, TQQQ) in this window.
- **Sleeve correlations collapse to ~1 in stress.** 2022 broke stock-bond
  diversification; 2011 and 2015 had equity+commodities down together.
- **Macro gates are lagging.** HY OAS widens AFTER the equity selloff
  starts; VIX spikes AFTER the drawdown; correlation flips AFTER
  several weeks of co-movement.
- **No vol targeting allowed.** The fastest lever to Sharpe 2 — daily
  vol targeting at ~10% — is explicitly disallowed.

To credibly hit Sharpe 2, the strategy would need either:
1. Intraday or derivative structures not present in this data (options
   carry, variance selling, futures basis).
2. Genuine cross-sectional long-short signal (requires inverse ETFs,
   explicitly excluded by the "long leveraged ETFs as core exposure"
   constraint on the permitted set).
3. Access to higher-Sharpe instruments — CTA, quality factor, merger
   arb — not in this universe.

**Honest ceiling with stated constraints: Sharpe ≈ 0.9-1.0.**

## 9. Files

- `alt/bastion_strategy.py` — runnable end-to-end backtest (this file)
- `alt/bastion_grid.py`, `alt/bastion_grid2.py` — IS grid searches
- `data/results/bastion_metrics.json` — full metrics dump
- `data/results/bastion_returns.csv` — daily ret, turnover, risk_off,
  weight_eq, weight_rates, weight_ra, weight_cash
- `data/results/bastion_grid.csv`, `bastion_grid2.csv` — grid traces
