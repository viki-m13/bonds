# Cross-asset trend baselines — the honest floor for L2GMOM

Before building a learned-graph momentum GNN ([L2GMOM, arXiv:2308.12212](https://arxiv.org/abs/2308.12212)),
this establishes whether the **cheap, rule-based trend signals it sits on top of**
clear a worthwhile Sharpe on the assets we actually have. They are the floor:
the paper's whole value-add is *marginal* Sharpe over exactly these baselines.

Builder: [`baselines.py`](baselines.py) · results: [`baselines_results.json`](baselines_results.json).

## Setup

- **Universe:** 44 liquid, distinct ETFs across 8 asset classes (US/intl/sector
  equity, rates, credit, commodities, FX, real assets). No leveraged/inverse,
  no target-maturity ladders, no crypto. Common history **2004-09 → 2026-06**.
- **Signals (causal, next-day execution):** TSMOM (12-month return sign,
  Moskowitz-Ooi-Pedersen), Baz et al. vol-normalised **MACD** (3 timescales),
  **LinReg** (sign of 252-day log-price OLS slope), and an equal-weight **COMBO**.
- **Sizing:** each position scaled to 15% ex-ante vol (EWMA-60 vol), equal
  capital across active assets, then the **portfolio re-scaled to 15% vol**
  (matching the paper's "rescaled to 15% vol target" panel). Leverage capped 3×.
- **Costs:** bps on notional turnover (|Δ position weight|); reported at 0 / 3 bps.
- **Leakage:** signal known at close *t*, earns *t→t+1*; vol from trailing data.

## Results — 15% vol-target portfolio, net of 3 bps

| signal | Sharpe | ann ret | maxDD | 2008–14 | 2015–19 | 2020–26 | corr SPY |
|---|---:|---:|---:|---:|---:|---:|---:|
| TSMOM  | 0.63 | 9.5% | −34.8% | 0.64 | 0.42 | 0.44 | 0.03 |
| MACD   | 0.49 | 6.1% | −27.0% | 0.61 | 0.27 | 0.28 | 0.02 |
| LinReg | 0.66 | 9.9% | −29.6% | 0.54 | 0.59 | 0.50 | 0.02 |
| COMBO  | 0.64 | 8.8% | −30.1% | 0.63 | 0.48 | 0.44 | 0.03 |
| *Long-only SPY @15% vol* | *0.76* | *11.8%* | *−33.2%* | — | — | — | 1.00 |

(Gross / 0 bps Sharpes are ~0.57–0.74.)

## Verdict — do not build the GNN on this substrate

The pre-registered gate was: *"if the cheap baselines don't clear ~1.0 Sharpe
net of costs on our assets, a learned-graph GNN won't rescue it — stop."*

They land at **Sharpe ≈ 0.5–0.7 net**, i.e.:
1. **Below the 1.0 bar**, and **below buy-and-hold SPY (0.76)** — on our data the
   trend portfolio does not even beat a vol-targeted index.
2. **Decays after 2014** (≈0.6 in 2008–14 → 0.3–0.5 since) — the documented
   "trend drought," and the same front-loaded/era-dependent pattern this repo
   keeps finding in other strategies.
3. The paper's marginal lift over these baselines is ~+0.3–0.5 Sharpe — not
   enough to turn a fragile ~0.6 base into a durable, compelling number.

**One honest caveat that could change the picture:** the paper uses **64
continuous futures**, which are materially better trend instruments than ETF
proxies (no contango drag like USO, deeper/cleaner ags-FX-rates trends, lower
costs). The ~0.6 here is partly an ETF-substrate penalty. So the result is
"trend-on-ETFs is weak," not "trend is dead."

**The one genuinely useful finding:** correlation to SPY is **~0.02–0.03**. Even
at Sharpe ~0.6, this is a real *diversifier* — a small uncorrelated sleeve, not
a standalone QQQ-beater. That, not the headline Sharpe, is what cross-asset
trend is for.

## If we want to take it further

The only worthwhile next step is **real futures data** (Pinnacle/Norgate CLC or
similar) to reproduce the paper's actual substrate, then add the network signal.
On the ETF universe, further signal engineering (incl. the GNN) is not expected
to clear the bar. Stopped here per the gate.

*Not investment advice. Past performance does not predict future results.*
