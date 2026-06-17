# HF underwater-avoidance stock picker

A stock-selection model built to one objective, in the user's words:

> "buying a stock that after it's purchased is not below the purchase price
> often or at all."

This is **not** a maximum-return model. It optimizes *entry quality*: pick
names that, after you buy them, spend little or no time below what you paid.
The headline metric is the purchase-weighted **underwater fraction** (share of
held days the close sits below the entry price), with the worst dip and the
end-of-horizon hit-rate as supporting metrics.

The state-of-the-art HuggingFace model used here is **Amazon Chronos-Bolt**
(`amazon/chronos-bolt-small`), a T5-based probabilistic time-series foundation
model. The win comes from using its **quantile (distributional) forecast**, not
a point forecast — see "Why this isn't the old Chronos experiment" below.

## TL;DR result

On the point-in-time S&P 500 (monthly, next-open execution, 2010–2026), ranked
by how little the picks go underwater:

| arm (k=3, horizon 126d) | underwater_frac ↓ | ever_underwater | mean_max_dip | hit_rate_end | mean_end_ret |
|---|---|---|---|---|---|
| **composite (low-vol + Chronos q10)** | **0.339** | 0.900 | −0.061 | **0.733** | +0.062 |
| low-vol only | 0.366 | 0.905 | −0.061 | 0.689 | +0.048 |
| Chronos q10 margin only | 0.387 | 0.915 | −0.102 | 0.628 | +0.070 |
| random control | 0.400 | 0.926 | −0.111 | 0.648 | +0.071 |
| momentum 12-1 (repo's *return* champ) | 0.421 | 0.915 | −0.170 | 0.607 | +0.143 |

The **composite** — a cross-sectional rank-average of trailing low-volatility
and the Chronos q10 downside-margin — is the best underwater-avoider at **all
nine** tested (k ∈ {1,3,5}) × (horizon ∈ {63,126,252d}) combinations: less time
underwater than low-vol alone, with a *higher* hit-rate and *higher* return.
Adding the HF model is a real, consistent improvement (paired t = −2.0 to −2.5
on time-underwater vs low-vol at the 126–252d horizons).

Momentum — which the repo's SUMMIT strategy rightly rides for *returns* — is one
of the **worst** entry-quality signals: its picks dip roughly 3× deeper below
entry. The objectives genuinely diverge.

## The signals

All scores are causal (row *d* uses closes only through *d*'s close) and
selection executes at the next open.

**Baselines (`baselines.py`)** — transparent yardsticks: `low_vol`
(−trailing 126d vol), `trend_quality` (trailing Sharpe), `downside_trend`
(Sortino-style), `trend_smoothness` (signed R² of log-price vs time),
`low_maxdd`, `self_underwater`, `mom_12_1`, `ret_126`, `random`.

**Chronos-Bolt (`chronos_signal.py`)** — for each candidate on each monthly
date, feed the trailing 256 closes, forecast 63 steps of price *quantiles*, and
read two signals off the predictive distribution:
- `chronos_q10margin` — the model's 10th-percentile 63d-ahead return
  (predicted shallow-case dip); the **stronger** of the two.
- `chronos_safety` = −predicted underwater fraction = −mean_t P(price_t < P0),
  where P(price_t < P0) is the interpolated CDF level at which today's price
  falls among step *t*'s forecast quantiles. This forecasts the objective
  directly but is the weaker signal (the probability saturates).

**Composite (`composite.py`)** — per-date rank-average of `low_vol` and
`chronos_q10margin`. No coefficients fit on outcomes (nothing to overfit to the
grid); the only choice — which two signals to average — is made on the IC
evidence below.

## Is the model actually skillful? (rank-IC, out-of-sample)

Spearman IC between each predictor and the *realized* underwater fraction across
all eligible candidates each month (horizon 126d). Chronos is pretrained and
never fit on this data, so this is genuinely out-of-sample. Lower (more
negative) = better safe-picker.

| predictor | IC vs underwater | t-stat | IC vs max-dip |
|---|---|---|---|
| low_vol | −0.083 | −5.6 | +0.29 |
| composite (low-vol + Chronos q10) | −0.078 | −5.5 | +0.27 |
| **chronos_q10margin** | **−0.060** | **−5.3** | +0.20 |
| trend_quality | −0.028 | −2.4 | +0.09 |
| mom_12_1 | −0.024 | −1.9 | +0.06 |
| chronos_safety | −0.023 | −3.0 | +0.05 |
| random | +0.003 | +0.8 | ~0 |
| low_maxdd | +0.078 | +5.4 | −0.25 (perverse) |

Takeaways:
- Trailing **low-volatility is the single strongest predictor** of underwater
  behavior — unsurprising, and a fair, hard baseline.
- **Chronos's q10 downside margin is a real, statistically robust #2** (t ≈ −5).
  It correlates +0.60 with low-vol (it *is* partly a volatility forecast) but
  retains ~40% independent variation — which is why averaging the two improves
  the *top-k picks* even though low-vol has the marginally better full
  cross-section IC.
- `low_maxdd` is **perversely** signed: names with the shallowest recent
  drawdown go underwater *more* (smooth recent run-ups mean-revert).

## Why this isn't the old Chronos experiment

`dca/research/results_chronos.md` tested Chronos and rejected it. That
experiment used the **median 42d forecast to rank expected return** and found it
added noise — median extrapolation is hopeless for large-cap direction. This
project differs on both axes:
1. **Different objective** — downside/underwater avoidance, not return ranking.
2. **Different use of the model** — the *quantile* forecast (q10 downside
   margin / distributional underwater probability), which is a volatility-and-
   tail signal the model *can* provide, not a directional bet it cannot.

The negative prior result stands for what it tested; this is a different question
and the model clears it.

## Live picks

`python live.py --k 10` runs Chronos on the latest close and prints the current
safest names. As of 2026-06-12 the top picks are low-volatility defensives
(utilities WEC/ATO/NI/DTE/AEE/FE, insurers L/AFL, REIT REG) — exactly the
profile the objective rewards.

## Also tested: L2GMOM network momentum (negative result)

A faithful implementation of the L2GMOM learnable-graph network-momentum model
(Pu et al., arXiv:2308.12212), adapted to this objective and benchmarked on the
same harness, is in `NETWORK_MOMENTUM.md` / `nm_*.py`. Short version: it has
weak genuine skill (IC ≈ −0.05) but **does not beat the plain low-vol factor**,
and the **learned graph makes it worse** than its own no-graph ablation — the
graph collapses to a near-complete averaging operator (10.9% same-sector edges,
~chance) and the model merely relearns "low-vol = safe". Network momentum is a
cross-asset *return* technique; it does not transfer to single-asset-class
downside avoidance. Kept as a documented negative control.

## Honest limitations

- **"Never underwater" is mostly a horizon artifact.** Over 126–252d, ~90% of
  *any* arm's buys close below entry on at least one day — that's market noise.
  The controllable, meaningful quantities are *how much* time underwater, *how
  deep*, and the end hit-rate; the composite improves all three.
- **The edge over plain low-vol is incremental** (~3pp less time underwater,
  +4pp hit-rate, +1.5pp return), though consistent across every cell and
  significant at longer horizons. Chronos helps; it does not transform the
  problem.
- **Defensive tilt has a cost in raw return.** This book deliberately trades the
  right tail for entry stability; for maximum compounding see SUMMIT.
- CPU-only, `chronos-bolt-small`, 256d context, 63d forecast. A larger model /
  GPU was not required for the result and was not used.

## Reproduce

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chronos-forecasting pandas pyarrow scipy scikit-learn

cd hf_picker
python chronos_signal.py --start 2010-01-01   # ~5 min CPU, caches scores
python compare.py --k 3 --horizon 126          # the scorecard above
python evaluate.py --horizon 126               # the IC table
python live.py --k 10                          # today's safest picks
```

## Files

- `data.py` — PIT S&P 500 panel + eligibility.
- `objective.py` — underwater metrics + next-open selection/evaluation.
- `baselines.py` — transparent causal candidate signals.
- `chronos_signal.py` — Chronos-Bolt probabilistic downside signal (cached).
- `composite.py` — leakage-free rank-composite.
- `evaluate.py` — rank-IC of predictors vs realized underwater.
- `compare.py` — full arm scorecard + IC, the decision document.
- `live.py` — current top-k safest picks.
