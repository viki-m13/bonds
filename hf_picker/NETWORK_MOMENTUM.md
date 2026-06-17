# L2GMOM network-momentum on the underwater objective — tested, not advanced

Faithful implementation of **L2GMOM** (Pu, Roberts, Zohren, Dong,
*"Learning to Learn Financial Networks for Optimising Momentum Strategies"*,
arXiv:2308.12212), adapted from its native return/Sharpe goal to this project's
**underwater-avoidance** objective, then judged on the same harness as the
low-vol + Chronos composite.

## What was built (faithful to the paper)

- **8 momentum features** (`nm_features.py`), the Baz/Lim set: volatility-scaled
  returns over 1/21/63/126/252d and three normalised MACD signals
  {(8,24),(16,48),(32,96)}. Plus **2 risk features** (trailing 126d vol and
  downside deviation), because the objective is downside, not return.
- **L2G layer** (`nm_model.py`): the paper's Algorithm 1 — `L` unrolled
  primal-dual-splitting steps of Kalofolias graph-from-smooth-signals learning,
  with the regularisation hyper-parameters (α, β) and per-layer step sizes (γ)
  made **learnable** (the "learning to learn" part).
- **L2GMOM forward** (Eq. 10): `A = L2G(U)`, normalise `Ã = D^-1/2 A D^-1/2`,
  signal `y = Ã (U θ) + b` — each asset's score is a graph-neighbour-weighted
  linear combination of features ("network momentum").
- **Ablation `LinearMom`**: `y = U θ + b`, identical features, no graph — so any
  difference is attributable to the learned network, not the features or label.
- **Training** (`nm_train.py`): label = realised underwater fraction over the
  next 126d, cross-sectionally standardised; expanding-window walk-forward with
  leakage control (a model predicting after cutoff C trains only on dates whose
  126d label closed before C); 20% validation tail, early stopping.

## Result (k=3, horizon 126d, 2011+, identical dates/eligibility)

| arm | underwater_frac ↓ | mean_max_dip | hit_rate_end | IC vs realised underwater | t |
|---|---|---|---|---|---|
| **composite (low-vol + Chronos)** | **0.342** | −0.062 | **0.721** | −0.079 | −5.4 |
| low_vol | 0.365 | −0.062 | 0.679 | **−0.085** | −5.5 |
| linmom (features, no graph) | 0.378 | −0.084 | 0.670 | −0.055 | −4.3 |
| **l2gmom (features + learned graph)** | 0.382 | −0.097 | 0.672 | −0.045 | −5.8 |
| chronos_q10margin | 0.388 | −0.103 | 0.622 | −0.060 | −5.1 |
| mom_12_1 | 0.419 | −0.170 | 0.613 | −0.026 | −2.0 |
| random | 0.406 | −0.113 | 0.638 | +0.003 | +0.8 |

**The network-momentum model has genuine but weak skill (IC ≈ −0.05, t < −4),
beating naive momentum — but it does not beat the plain `low_vol` factor, and
the learned graph makes things WORSE, not better.** `linmom` (no graph) edges
out `l2gmom` (graph) on every metric.

## Why — three concrete diagnostics

1. **The learned graph collapses to a near-complete graph.** On 2026-06-12 it
   has mean degree 495 of 496 nodes (no isolated nodes, learned α=0.62 pushing
   connectivity up). A near-uniform `Ã` is just an averaging operator, so
   `Ã(Uθ)` smears each name's score toward the cross-sectional mean — it
   *dilutes* the (already thin) signal. That is exactly why the graph version's
   IC (−0.045) is *smaller in magnitude* than the no-graph linear model
   (−0.055), despite a very stable sign (t = −5.8).
2. **The graph is not economically meaningful here.** Only **10.9%** of its
   edges connect same-sector names, versus ~9.1% by chance — essentially no
   sector/economic structure was discovered.
3. **The model just relearns "low volatility = safe."** Trained feature weights
   load almost entirely on the risk features (`vol_126` −0.69, `dsdev_126`
   −0.42, mid-horizon return −0.32); the MACD signals — the heart of *momentum*
   — are ≈ 0. So the model rediscovers the `low_vol` factor, only noisier, which
   is why `low_vol` alone beats it.

## Verdict

Network momentum is a **return/Sharpe** technique whose published edge leans on
the breadth of a **cross-asset-class futures** panel (commodity↔equity↔FX↔rates
links). Pointed at downside avoidance inside a **single-asset-class large-cap
equity** universe, (a) the momentum features it propagates are weak for this
objective, (b) the only useful signal is the risk features it isn't really
about, and (c) the learned graph adds no economic structure and dilutes the
cross-section. This is consistent with the repo's earlier results that ML on
OHLCV features (LightGBM ranker, Chronos median forecast) does not beat simple
factors on this data.

**Not advanced.** The low-vol + Chronos-q10 composite remains the best
underwater-avoidance picker. The L2GMOM code is kept for reproducibility and as
a documented negative control.

## Reproduce

```bash
cd hf_picker
python nm_train.py --arm both      # ~4 min CPU; caches l2gmom_score / linmom_score
# scorecard + IC: see the snippet in the commit message / README
```

Files: `nm_features.py`, `nm_model.py` (L2G + L2GMOM + LinearMom), `nm_train.py`.
