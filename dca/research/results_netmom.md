# Network momentum (L2GMOM-style) as a DCA selector — tested, negative

**Idea (user-suggested):** Pu, Roberts, Zohren & Dong, *"Learning to Learn
Financial Networks for Optimising Momentum Strategies"* (L2GMOM, arXiv:2308.12212)
and the sibling *"Network Momentum across Asset Classes"* (arXiv:2308.11294).
A graph-learning model learns a **momentum-spillover network** among assets and
propagates each asset's momentum over it; reported Sharpe ≈ 1.5–1.74, ~22%/yr
**after volatility scaling**, on **64 long-short futures** (commodities, equities,
bonds, FX), 2000–2022.

**Question:** does network momentum help our mandate — *most profitable long-only
DCA stock picker vs QQQ-DCA*?

**Verdict: No (as a selector here).** A causal, non-ML cross-sectional version
dilutes the concentrated momentum edge and is monotonically *less* profitable.
At most it is a mild worst-case buffer, like the optional trim — a risk lever,
not a return lever.

---

## Why the literal paper is a poor fit (before testing)

Three structural mismatches, each tied to our own evidence
(`literature_review_cited.md`):

1. **It optimizes Sharpe on a vol-targeted long-short book — the raw-return-
   inferior kind vs a high-beta growth benchmark.** Volatility targeting caps
   exposure exactly when momentum is calm and trending (the QQQ bull), so it lags
   QQQ on *raw return* even while being Sharpe-superior (Frazzini-Pedersen; the
   same mechanism that made the panel reject vol-scaled/risk-adjusted selection,
   `results_ram.md`). 22%/yr vol-targeted ≠ more profitable than ZENITH (~25%/yr).
2. **Graph learning is ML cross-sectional prediction**, which keeps losing to a
   single momentum column in liquid large caps (Gu-Kelly-Xiu thin ex-microcaps;
   panel LightGBM OOS IC 0.002 and Chronos both lost — `results_ml.md`,
   `results_chronos.md`).
3. **Network momentum's documented edge is *across asset classes*** (real
   commodities↔bonds↔FX↔equity spillovers). Within one homogeneous S&P 500
   universe the "network" collapses toward sector/correlation structure that is
   far more efficiently arbitraged.

## The tractable test we ran

We extracted the genuinely novel, testable core — *rank a stock by its own
momentum plus its correlated neighbours' momentum* — as a **causal, leakage-safe,
non-ML** signal (so it survives the truncation audit, unlike a graph NN):

* At each month-end, build an adjacency from **trailing-126d return correlations**
  (uses only data through that close), keep each node's **top-10 positive**
  neighbours, row-normalise.
* Network momentum = `(1−λ)·own_momentum + λ·Σ neighbours' momentum` (one-hop
  spillover), forward-filled to daily, re-ranked, then the **same SUMMIT size
  tilt + regime switch + bear sleeve + k=1** as ZENITH. λ sweeps the spillover dose.

## Result (PIT S&P 500, 244 windows, k=1, vs QQQ-DCA)

| signal | full mult | win | median | p10 | worst | IS win | OOS win |
|---|---|---|---|---|---|---|---|
| **ZENITH (plain size-tilted mom)** | **25.7×** | 95% | **+43%** | +5% | −11% | 92% | 99% |
| netmom λ=0.15 | 24.9× | 95% | +42% | +6% | **−10%** | **93%** | 99% |
| netmom λ=0.30 | 24.6× | 95% | +39% | +5% | −10% | 92% | 99% |
| netmom λ=0.50 | 23.8× | 94% | +34% | +3% | −11% | 90% | 100% |

**Reading:** spillover is **monotone in the dose, the wrong way** — the more
network blending, the lower the multiple and median, with win-rate flat. It
behaves as a **correlation smoother / mild diversifier**: at λ=0.15 it shaves the
worst window −11% → −10% and lifts IS win 92% → 93% at a cost of ~0.8× terminal
multiple (+43% → +42% median). That is the same trade as the optional trim — a
small robustness buffer, not a profit improvement. By the project's recurring
law (*anything that dilutes the concentrated momentum edge costs return*),
network momentum is another dilution here.

## Bottom line

For the long-only, raw-return, beat-QQQ-DCA mandate, network momentum does **not**
beat plain size-tilted momentum on our PIT large-cap panel; it is at best a mild
worst-case buffer at low λ. Its real home is its native framing — a **long-short,
volatility-targeted, multi-asset-class managed-futures overlay**, where the
cross-class spillover premium is economically real and Sharpe is the objective.
That is a different strategy (and a different mandate) from ZENITH/SUMMIT; if a
managed-futures sleeve is ever wanted in this repo, this is the right reference
for it (the leveraged-ETF strategies in `alt/`, `apex/` are the closest existing
analogues). Tested, documented, not adopted.

*Sources:* L2GMOM https://arxiv.org/abs/2308.12212 · Network Momentum across
Asset Classes https://arxiv.org/abs/2308.11294 / SSRN
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4540651
