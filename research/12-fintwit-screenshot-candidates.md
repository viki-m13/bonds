# Phase 3 supplement — Fintwit Screenshot Candidates (validated)

**Prepared:** 2026-06-15
Five papers/strategies surfaced from X/Twitter screenshots, validated to ground truth. None is a
retail-accessible high-Sharpe (>5) strategy; the pattern is identical to the rest of the survey — a real but
modest edge wrapped in a headline number or multiple.

| # | Item | Headline | Honest read |
|---|---|---|---|
| 1 | **Factor MAX** (Wang & Zeng, Dec 2025) | chart "2.26/2.20" | those are *cumulative-return* values, not Sharpe; net Sharpe ~0.3–0.6 |
| 2 | **X-Trend** (Wood-Kessler-Roberts-Zohren, Oxford-Man 2024) | "5×/10× the Sharpe of TSMOM" | base-effect artifact; gross ~2.7, **net ~1.0–1.8** (costs hard-set to 0) |
| 3 | **Insider & stealth trading w/ dynamic legal risk** (Qiao & Xia, 2026) | — | continuous-time *theory* paper, no backtest/Sharpe; models illegal activity |
| 4 | **"Intraday momentum" writeup** (IMG_7549) | Sharpe 1.33 (3.50 @ VIX>40) | the Zarattini "Beat the Market" paper — already in [`08`](08-accessible-high-sharpe-hunt.md); net ~0.4–1.1 |

---

## 1. Factor MAX and Predictable Factor Returns — Wang & Zeng (Dec 2025)

**Source:** [SSRN 6053114](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6053114) (working paper, not peer-reviewed) ·
[AlphaArchitect](https://alphaarchitect.com/factor-max/) · [Swedroe](https://larryswedroe.substack.com/p/factor-max-a-new-signal-for-predicting) ·
companion [Factor MAX in the Chinese Market (EFMA 2025 PDF)](https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/MAX.pdf)

**Mechanism (confirmed).** Sort **factors, not stocks**. Universe = **172 equity factors**; each month rank them
into quintiles by prior-month **MAX** (largest daily factor return). **Long high-MAX factors (Q5), short
low-MAX factors (Q1)**, monthly rebalance. Direction is the *opposite* of stock-level MAX (Bali-Cakici-Whitelaw):
high-MAX *factors* continue up (underreaction in low-attention factors), whereas high-MAX *stocks* reverse down
(lottery overreaction). It is a momentum-like *continuation* bet at the factor level.

**Performance (confirmed).** Spread = **0.32%/month, t ≈ 5.9** (~3.9%/yr gross); Q1 0.09% vs Q5 0.41%/mo;
risk-adjusted alpha 0.24%/mo (behavioral) to 0.37%/mo (q-factor); strongest in low-attention factors (0.42%/mo,
t=2.93); cumulative alpha "$9.58 per dollar" over 1963–2023.

**The chart numbers you screenshotted.** The Figure 1 "2.26 / 2.20" are **cumulative growth/abnormal-return
values, NOT Sharpe ratios** — nothing in the paper supports a Sharpe of ~2.2. The companion (China) paper's
actual figure is **~0.18 monthly / ~0.86 annualized Sharpe gross.**

**Validation.** (a) **Gross, and the authors admit cost exposure** — "monthly rebalancing across many factors
could be expensive." Two-layer turnover (re-rank 172 factors, each itself a long/short stock book) is the
killer. (b) **In-sample only** (full-sample 1963–2023 backtest; the China sample is mild corroboration, not
OOS). (c) **Multiple-testing risk** — the signal is a meta-transform built on top of an already-mined 172-factor
zoo; t=5.9 is not Deflated-Sharpe-adjusted (though it would likely survive a *gross* t-haircut). (d) **Likely
overlaps factor momentum / factor-return autocorrelation** (Ehsani-Linnainmaa); the "distinct behavioral
channel" claim is plausible but not dispositive.

**Capacity/accessibility:** **not investable** by a normal participant — requires simultaneously holding the
long/short legs of dozens of academic factor portfolios with shorting and monthly rotation. At best a modest
overlay signal for a large factor-replication shop (AQR/DFA-type) whose factor sleeves already exist.

**Verdict:** genuine, robust, *in-sample* factor-timing anomaly of **modest magnitude (~4%/yr gross), net Sharpe
~0.3–0.6, not investable, no basis for >2 let alone >5.** The screenshot's "2.2" is a cumulative return, not a
Sharpe.

---

## 2. X-Trend: Few-Shot Learning for Trend-Following — Wood, Kessler, Roberts, Zohren (Oxford-Man, 2024)

**Source:** [arXiv 2310.10500](https://arxiv.org/abs/2310.10500) (v2, Mar 2024) · [JFDS 6(2):88](https://www.pm-research.com/content/iijjfds/6/2/88) ·
[code](https://github.com/kieranjwood/x-trend)

**Mechanism (confirmed).** Encoder-decoder deep net with **4-head cross-attention** over a *context set* of
historical futures regimes (segmented by change-point detection); a target asset's current sequence attends over
those regimes to transfer trends and forecast next-day return. Supports few-shot (asset seen in training) and
zero-shot (asset never seen). Designed for fast adaptation after regime shifts (COVID-19).

**The headline vs the absolute numbers.** All Sharpes are **gross**, averaged over 10 seeds:

| | TSMOM (baseline) | Neural baseline | **X-Trend (best)** |
|---|---|---|---|
| Few-shot, 2018–2023 | **0.23** | 2.27 | **2.70** (+18.9% over neural) |
| Few-shot, 1995–2023 | 1.01 | 2.91 | 3.11 |
| Zero-shot, 2018–2023 | −0.26 | −0.11 | **0.47** |
| Zero-shot, 1995–2023 | 0.61 | 1.00 | 1.44 |

The "**10× / 5× the Sharpe of TSMOM**" is a **base-effect artifact**: 2.70 / 0.23 ≈ 10× only because TSMOM was a
near-broken 0.23 in 2018–2023 (over the full sample the multiple collapses to ~3×). X-Trend's edge over its *own
neural baseline* is just **+17–19%** (2.27 → 2.70) — the giant multiple is the neural family beating broken
TSMOM, not X-Trend's contribution.

**Validation.** (a) **Decisive: costs are hard-set to zero** — the paper quotes "we set C(i) = 0… focusing on
pure predictive power," with **no net-of-cost table anywhere.** The same group's earlier Deep Momentum Networks
([arXiv 1904.04912](https://arxiv.org/abs/1904.04912)) showed the deep-momentum edge **dies above ~2–3 bps** of
cost — and X-Trend omits the turnover/cost analysis its own authors did in 2019. (b) To its credit, **genuine
walk-forward OOS** (expanding 5-yr blocks). (c) **Grid/selection:** the +18.9% is the max over ~30
hyperparameter configs (mitigated somewhat by 10-seed averaging clustering at 2.3–2.7). (d) No live track record;
15% vol-target overlay applied post-hoc.

**Verdict:** well-constructed OOS study with a *real but small* (~+18%) architectural gain over a strong neural
baseline. Gross few-shot ~2.7 is credible; **net is unproven and plausibly ~1.0–1.8** given likely high turnover;
zero-shot (gross 0.47) is break-even-to-negative net. The "5×/10×" corresponds to **no absolute Sharpe in the
paper** and certainly not >5.

---

## 3. Insider and Stealth Trading with Dynamic Legal Risk — Qiao & Xia (2026)

A continuous-time stochastic-control (Kyle-model-descendant) **theory** paper: how an informed insider optimally
trades against a legal-risk penalty that depends on the discrepancy between trade price and fundamental value.
The figures are *equilibrium strategy surfaces*, **not a backtest or an equity curve — there is no Sharpe.** It
is microstructure theory, not a deployable strategy. Separately, the activity it models (trading on material
non-public information) is illegal, so there is nothing here to validate or act on as a trading strategy.

---

## 4. "Intraday momentum" writeup (IMG_7549) — already covered

The LLM-style bullet summary ("1,985% total return, 19.6% annualized, Sharpe 1.33, beta ≈ 0, Sharpe 3.50 when
VIX>40, net of commissions and slippage") is the **Zarattini/Barbon/Aziz "Beat the Market" SPY intraday momentum
paper** ([SSRN 4824172](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4824172)). Fully validated in
[`08-accessible-high-sharpe-hunt.md`](08-accessible-high-sharpe-hunt.md) §3: headline 1.33 net, but independent
replications collapse it to **~0.4–1.1** at a fair 1× leverage with real spreads (QuantConnect repro: 0.40 at
1×, fees 16.7% of return); the 3.50 is a crisis-only (VIX>40) sub-sample, not a standing Sharpe.

---

## Takeaway

Four distinct fintwit items, the same lesson as the entire survey: **two are real-but-modest edges (Factor MAX
~0.3–0.6 net, X-Trend ~1.0–1.8 net) whose headline numbers are a cumulative-return chart value and a base-effect
multiple respectively; one is theory (and illegal to act on); one is the already-debunked Zarattini paper.** None
is a net, out-of-sample, accessible Sharpe > 5 — consistent with the three-phase conclusion that such a thing
exists only in the unscalable corners (HFT, Medallion-scale breadth, tiny-capacity betting niches).
