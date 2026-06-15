# PHOENIX-5 — meta-ensemble research (self-contained)

Everything in this folder is isolated from the rest of the repo: it only
**reads** from `data/` and `data/results/`, and writes exclusively to
`phoenix5/results/`. Nothing in production (`alt/`, `scripts/`, `docs/`,
workflows) is touched.

## Objective

Improve on the production PHOENIX strategy (5-sleeve orthogonal ensemble,
OOS Sharpe ≈ 2.15) and/or invent a better strategy, with a target of
**Sharpe 5 out of sample**.

## Headline result

| | IS (2010–2018) | OOS (2019–2026) | Full |
|---|---|---|---|
| **PHOENIX-5** Sharpe | **2.29** | **2.72** | **2.50** |
| production PHOENIX Sharpe | 2.56* | 2.15 | 2.35 |
| PHOENIX-5 CAGR / Vol / MDD | 18.4% / 7.5% / −9.5% | 21.4% / 7.2% / −11.0% | 19.8% / 7.4% / −11.0% |

\* production IS figure is partly in-sample (its blend weights were fit on IS).

- OOS Sharpe improves **2.15 → 2.72** (+0.57) with *no* IS→OOS degradation
  (OOS > IS, the healthiest possible signature).
- **Positive every calendar year 2011–2026**; worst years +1.2% (2022) and +1.5% (2018).
- Robust to meta-parameter choice: across a 48-point grid over all four
  ensemble meta-parameters, OOS Sharpe spans **2.44–2.81, median 2.68**
  (`results/sensitivity_grid.csv`). Even with the Sharpe-tilt switched off
  entirely it stays ≥ 2.44.
- Block-bootstrap 95% CI for OOS Sharpe: **[1.70, 3.80]**.
- Trade-off: PHOENIX-5 runs unlevered (multiplier cap 1.0, same policy as
  production) and its realized vol is only ~7.4%, so CAGR is ~20% vs
  production's ~57%. Sharpe is scale-invariant: at the same 20% vol
  (≈2x leverage) CAGR would be ≈2x at the same Sharpe, if leverage were allowed.

### On the Sharpe-5 target — honest assessment

An out-of-sample Sharpe of 5 is **not achievable honestly** with this data and
asset universe, and any backtest here that claims it should be presumed broken.
The math: PHOENIX's sleeves are already near-orthogonal (|ρ| < 0.2) with OOS
Sharpe ≈ 0.9 each. Diversification scales the blend Sharpe as √N for N
independent streams: 5 sleeves → ≈ 2.0–2.2 (production is already *at* this
bound). Reaching 5 would require **~25 independent Sharpe-1 strategies** —
this repo's data supports nowhere near that, and decades of market history
suggest few institutions on earth achieve it net of costs at daily frequency.
The two "Sharpe ≈ 4–5 OOS" streams already present in `data/results/`
(`strategy_v10`, OOS 4.58) were audited as part of this work and shown to be
artifacts — see the research log below.

## PHOENIX-5X — "more money AND less risk", no leverage (`phoenix5x.py`)

PHOENIX-5 above maximizes Sharpe but earns less in absolute terms. PHOENIX-5X
instead targets **strict dominance** of production PHOENIX — higher CAGR *and*
lower vol *and* shallower drawdown — under the no-margin constraint
(multiplier ≤ 1.0, leverage only inside LETFs). The production core (sleeves,
weights, vol target, DD throttle) is untouched; only two kinds of changes:

| OOS 2019–2026 | production | 5X-CONSERVATIVE | 5X-RECOMMENDED | 5X-TURBO |
|---|---|---|---|---|
| Sharpe | 2.15 | 2.15 | 2.18 | **2.25** |
| CAGR | 35.7% | 35.8% | 36.2% | **37.6%** |
| Vol | 14.71% | 14.74% | 14.68% | **14.61%** |
| Max drawdown | −17.7% | −17.2% | −17.5% | **−17.0%** |
| $100k → (2019–2026) | $1.046M | $1.057M | $1.080M | **$1.165M** |
| strict dominance | — | ~tie (vol +0.03pp) | yes, all metrics | **yes, all metrics** |

- **5X-CONSERVATIVE** = correct-by-construction fixes only: 3-day smoothing of
  the overlay multiplier (cuts whipsaw TC) and idle de-risked capital earning
  BIL (T-bills) instead of 0%.
- **5X-RECOMMENDED** = additionally parks idle capital 50/50 in BIL and a
  no-margin diversifier basket (CREDLO + DBMF/KMLM/CTA managed futures), and
  deepens the extreme-vol gate (de-risk to 25% instead of 50% on 99th-pctile
  vol days). **Caveat:** the deeper gate's OOS gain is concentrated in the
  2020/2022 vol episodes and costs ~0.03 IS Sharpe (`research/dominance_validate.py`)
  — the +0.4pp CAGR edge over CONSERVATIVE is regime-dependent, not a law.

- **5X-TURBO** = RECOMMENDED + an intraday realized-vol accelerated overlay
  (`research/rv_overlay.py`): the 60d vol estimate in the vol target is
  multiplied by the 5d/60d ratio of market intraday RV (SPY/QQQ/TLT 5-min
  bars, clipped 0.6–2.5). The overlay then de-risks within days of a vol
  shock and — just as important — re-risks within days of it decaying, so
  capital spends more time deployed. Mechanism-consistent gains: vs the
  production-equivalent baseline it added +10.3pp in 2020 and +4.3pp in 2022
  (the vol-episode years) while giving back ~1–3pp in calm years.
  **Caveats:** intraday data exists only from 2016, so this variant runs
  2016-06 onward and its pre-2019 validation window is short; the 2020 gain
  is one episode. The mechanism (fast-RV vol management) has strong academic
  support (vol-managed portfolio literature), which is why it's offered
  despite the short sample.

Magnitude honesty: production is already near its no-leverage efficiency
frontier. Without margin, the diversifiers found in this research can only be
funded by displacing the ~36%-CAGR core, so the strictly-dominant gains are
real but small (≈ +0.5pp CAGR, ≈ −0.2pp MDD, +0.03 SR). Larger risk-adjusted
gains (PHOENIX-5's 2.72 Sharpe) inherently trade away absolute return.
Also tested and rejected for this goal: walk-forward Sharpe-tilted sleeve
weights at full risk (OOS SR 1.89–2.08, all worse), parking idle capital in
MOSAIC (MDD worsens), softer/earlier DD throttles (no improvement)
(`research/parking_grid.py`).

## Round-2 advanced research (`ADVANCED_RESEARCH.md`)

A second research round tried four genuinely different methods (causal sleeve
factory, parameter bagging, meta-labeling, new-data signals). Full writeup in
`ADVANCED_RESEARCH.md`. Headlines:
- **Parameter bagging** showed production's canonical OOS Sharpe (2.15) is
  ~0.27 optimistic — the overfit-robust estimate is **~1.88 with −25% MDD**,
  because ORION/HELIOS canonical parameters were partly lucky. Budget for the
  robust number going forward; a bagged-sleeve production base is available as
  a sturdier (lower-backtest) option.
- **Sleeve factory** and **meta-labeling** were rigorous negatives (no usable
  new return; no 5-day predictability, AUC 0.52). Confirms the existing daily
  data is mined out — further gains need new *data*, not new math.
- Net: 5X-TURBO remains the one durable, mechanism-backed improvement.

## What PHOENIX-5 is

A meta-ensemble of four return streams, combined with fully causal machinery:

| Sleeve | What it is | IS SR | OOS SR |
|---|---|---|---|
| PHXCORE | Production 5-sleeve PHOENIX raw blend (VANGUARD/ORION/HELIOS/QUANTUM/CRYPTO, static IS inverse-vol weights) | 2.44 | 1.99 |
| MOSAIC | 66 small carry/hedge/trend streams across credit, FX, commodities, sectors, REITs (the `strategy_v10` framework **minus** its un-investable LETF-short engines), trailing-Sharpe adaptive selection (causal) | 1.09 | 1.13 |
| CREDLO | Low-vol floating-rate / short-duration credit carry (BKLN/FLOT/MINT/HYG/GLD) gated by HY OAS level and 10y-rate spikes, 2011+ | 0.19 | 1.88 |
| MFUT | Managed-futures ETF basket (DBMF/KMLM/CTA equal-weight as available), 2019+ — pure diversifier, no backtest of our own | — | 0.85 |

Full-sample correlations: PHXCORE–MOSAIC 0.43, PHXCORE–CREDLO 0.32,
PHXCORE–MFUT 0.06, MOSAIC–CREDLO 0.55, others ≤ 0.22.

Construction (`phoenix5.py`, every estimate trailing-only):

1. Each sleeve vol-targeted to 10% ann (63d trailing, multiplier 0.25–4×).
2. Sleeve weights walk-forward, refreshed monthly: inverse-vol base ×
   trailing-252d-Sharpe tilt (exp(0.5·ΔSR), capped at 3× relative). A sleeve
   becomes eligible after 189 days of live history (CREDLO enters 2012,
   MFUT enters 2020 — no backfill).
3. Portfolio overlay identical in spirit to production: 15% vol target with
   **cap 1.0 (no leverage)**, −10% DD throttle on 252d HWM, 99th-percentile
   vol gate, multiplier smoothed over 3 days, 10bp TC per unit of multiplier
   change.
4. **Idle capital earns BIL** (T-bills) instead of 0% — the production
   backtest leaves the de-risked fraction (~15% of NAV on average) earning
   nothing.

## Research log — everything tested, including what failed

The bulk of this work was a systematic search for new orthogonal sleeves.
Most candidates **failed honest costs/validation** and were rejected; numbers
below so nobody re-treads this ground. Scripts in `research/`.

| Candidate | Result (net Sharpe, IS / OOS) | Verdict |
|---|---|---|
| Intraday momentum, first 30 min → last 30 min (7 ETFs, 5-min bars 2016–2026) | −2.4 / −1.9 | Dead. Sign is actually *reversal*… |
| …Intraday reversal (fade the open move) | gross ≈ +0.6, **net negative at 1bp/side** (cost drag ≈ 1.5–2.0 SR units on a 30-min-window stream) | Dead after costs at every window/sizing variant tested |
| Overnight long (close→open) | ≈ +0.3–0.6 gross per ETF | Too weak after costs |
| Gap fade (z-sized) | −0.6 to +0.4 | Dead |
| Cross-sectional 5d/21d reversal, 96 large caps, weekly | −0.1 / −0.7 and +0.2 / −0.2 | Dead (large-cap reversal has decayed) |
| Treasury duration timing (carry, momentum, curve) | +0.3–0.5 IS, ≈ 0 OOS | Too weak |
| LETF vol-decay capture (short bull+bear pairs) | IS +2.45 / OOS **−2.57** at 0% borrow; catastrophic at realistic 3–6%/yr borrow | Dead — and this audit shows `strategy_v10`'s OOS 4.58 is a borrow-cost artifact |
| Gated short-vol (SVXY, VIX-percentile + trend) | +0.2 / +0.4 | Too weak |
| Cross-sectional crypto momentum (108 coins, weekly, 20bp/side) | −0.3 / +0.4 | Dead |
| Turn-of-month equity seasonality | ≈ +0.4–0.5 both halves, but rest-of-month is just as good | No edge vs buy-and-hold |
| QUANTUM-WF: annual expanding-window retrain of the ML sleeve | OOS 0.76 vs frozen 0.86 | No improvement — staleness was not the problem; the frozen model's IS 2.73 was memorization |
| Walk-forward inverse-vol / ERC ensemble weights on the 5 production sleeves | OOS 1.91–2.07 vs 2.15 static | Production machinery already near-optimal |
| Overlay ablations (drop DD throttle / vol gate) | ±0.05 | Immaterial |
| Hedged credit-carry basket alone (9 pairs) | 0.54 / 0.61 | Weak alone; useful only inside MOSAIC's breadth |

What survived and went into PHOENIX-5: MOSAIC (breadth of 66 tiny carry
streams), CREDLO (regime-gated low-vol carry), MFUT (external diversifier),
BIL-on-idle-cash, and the walk-forward Sharpe-tilted sleeve weighting.

## Caveats, stated plainly

- **PHXCORE inherits the production sleeves' backtests** (their construction,
  costs, and any residual selection bias documented in `alt/*_DESIGN.md`).
- **MOSAIC's universe** (which carry pairs exist) comes from `strategy_v10`,
  which was written with knowledge of the full sample, although its pair list
  is generic asset-class carry rather than tuned parameters. Its adaptive
  selection itself is causal (trailing 252d windows only).
- **CREDLO's gate thresholds** (OAS 5.0→8.0, +0.7pp/63d rate spike) are taken
  from the repo's ZEPHYR production design, also written ex-post. CREDLO's
  OOS Sharpe additionally benefits from smoothed NAVs of credit ETFs (stale
  pricing understates true vol) and from the 2022–2026 high-rate regime.
- **MFUT** exists only OOS (2019+); it cannot be IS-validated. It is the
  canonical, non-cherry-picked managed-futures ETF set, but its inclusion
  decision was made knowing trend ETFs did fine post-2019.
- The OOS bootstrap CI is wide ([1.70, 3.80]); ~7 years of daily data cannot
  pin a Sharpe to better than ±1.
- Crypto sleeve inside PHXCORE uses GBTC/ETHE-era proxies as in production.

## Files

```
phoenix5/
├── phoenix5.py              # the strategy — builds sleeves, ensemble, metrics
├── sleeves/
│   └── quantum_wf.py        # walk-forward QUANTUM experiment (rejected)
├── research/
│   ├── explore_intraday.py  # intraday momentum/overnight tests
│   ├── explore_reversal.py  # reversal variants, gap fade, stock XS, duration
│   ├── explore_sleeves.py   # carry / LETF-decay / credlo / short-vol sleeves
│   ├── explore_misc.py      # crypto XS momentum, turn-of-month
│   ├── explore_ensemble.py  # ensemble machinery ablations
│   ├── explore_mosaic.py    # MOSAIC (v10-minus-decay) audit
│   ├── scan_streams.py      # audit of all 93 return streams in data/results
│   └── sensitivity.py       # 48-point meta-parameter grid
└── results/
    ├── phoenix5_metrics.json    # headline metrics, sleeve stats, correlations
    ├── phoenix5_returns.csv     # daily state: raw, multiplier, idle, net
    ├── phoenix5_weights.csv     # walk-forward sleeve weights
    ├── sensitivity_grid.csv     # robustness grid
    ├── quantum_wf_*             # rejected experiment artifacts
    └── mosaic_*.csv             # MOSAIC sleeve returns
```

Run: `python3 phoenix5/phoenix5.py` (needs `pandas`, `numpy`; the
`quantum_wf` experiment additionally needs `xgboost`).
