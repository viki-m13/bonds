# Event-Driven Micro-Edges & ML Intraday Alpha — Skeptical Review

*Phase 2 supplement. Thesis: almost everything here is real-but-thin or real-but-dead. The cleanest
documented event edges (index rebalance, classic PEAD) have structurally **decayed to ~0** in liquid names.
Merger arb survives at Sharpe ~0.5 with a hidden short-put tail. SPAC cash+warrant arb is near-risk-free but
its excess-over-T-bills has compressed to ~0 in calm markets. The entire ML limit-order-book literature
reports **classification accuracy, not net P&L** — honest walk-forward papers print Sharpe ~0.3 (often
insignificant); the "Sharpe >10" papers are √-time annualization artifacts. Default prior on any pitch here:
**overfit or decayed until proven otherwise.***

---

## Area A — Event-driven micro-edges

### A1. Index rebalance / reconstitution arbitrage — MOSTLY DEAD
Buy S&P 500 additions / short deletions ahead of forced index-fund flows; unwind after the effective date.
**Greenwood & Sammon, "The Disappearing Index Effect" (NBER w30748):** addition abnormal return
**7.4% (1990s) → 5.2% (2000s) → 1.0%, insignificant (2010s)**; deletion **−16.1% → −12.4% → −0.6%,
insignificant** — and this decay happened *despite* indexed AUM growing (opposite of demand-pressure
theory), because of MidCap migration offsetting net demand, better liquidity provision around predictable
pre-announced events, and arbitrageur front-running. The Russell 2000 reconstitution variant (historically
+4.6% in the window) decayed similarly after the 2003 banding change and longer pre-announcement lead time.
- [NBER w30748](https://www.nber.org/papers/w30748) · [HBS PDF](https://www.hbs.edu/ris/Publication%20Files/23-025_563e45c6-df92-4d9c-ae05-608d4d0acab1.pdf) · [Alpha Architect](https://alphaarchitect.com/disappearing-index-effect/) · [Russell SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=303279)

**Verdict: honest net Sharpe today ~0–0.2.** The single most clearly-documented "anomaly went to zero as it
got famous" case. (Consistent with Phase 1 [`01`](01-statistical-arbitrage.md) §4.)

### A2. Post-Earnings-Announcement Drift (PEAD) — DEAD in liquid equities
Long top-SUE decile / short bottom, hold ~60 days. Classic Bernard-Thomas ≈19% annualized (1974–85). The
decay is decisive: **Martineau (2021) "Rest in Peace PEAD"** — non-existent for large stocks since ~2006,
prices fully reflect the surprise *on announcement day*. **Columbia/CEASA:** hedge-portfolio abnormal return
6.24%/quarter (t=8.49) in 1974 → indistinguishable from zero after 2017, partly from a genuine decline in
earnings-surprise persistence (so unlikely to return). Clean liquid-universe replication: CAGR 2.6%, **Sharpe
0.50** and decaying. Intraday PEAD is captured by HFT liquidity providers at the quote → **~0 or negative
net** for non-colocated traders.
- [Martineau SSRN 3111607](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111607) · [Columbia/CEASA PDF](https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/PEAD_Declined_over_time.pdf) · [Chordia-Subrahmanyam-Tong](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2029057)

**Verdict: ~0.1–0.3 in a liquid universe, ~0 large cap, negative intraday.** Any fat post-2010 PEAD backtest
is microcap-contaminated, cost-blind, or look-ahead.

### A3. Earnings Announcement Premium (Frazzini & Lamont) — most durable, low-Sharpe
Long expected-announcers / short non-announcers, monthly. >60 bps/month, ~7–18%/yr, consistent since 1927,
largest in large caps. But it's a **high-turnover attention/beta tilt**, not market-neutral alpha, and partly
fair compensation for announcement risk. Net of turnover, gross ~0.5–0.8 drops materially.
- [NBER w13090](https://www.nber.org/papers/w13090) · [AQR](https://www.aqr.com/library/working-papers/the-earnings-announcement-premium-and-trading-volume)

**Verdict: ~0.3–0.5 as a large-cap announcement-month tilt** — the most durable of the three (a risk/attention
premium, not a decaying autocorrelation).

### A4. Merger / risk arbitrage — real, Sharpe ~0.5 with a hidden short-put tail
**Mitchell & Pulvino (2001), 4,750 deals 1963–98** — the canonical, unusually honest paper:

| | Frictionless (VWRA) | **Realistic, net of costs (RAIM)** | Market |
|---|---|---|---|
| Compound annual return | 16.05% | **10.64%** | 12.24% |
| Annual std dev | 9.29% | 7.74% | 15.08% |
| **Sharpe** | **1.06** | **0.57** | 0.40 |

The headline 1.06 is **not investable** — the cost/impact/illiquidity-adjusted number is **0.57**. Returns are
"similar to selling uncovered index put options": beta ~0 in normal/up markets but **jumps to ~0.50 when the
market falls >4%** — negative skew, real left tail, acknowledged by the authors. (Matches Phase 1
[`04`](04-futures-cta-arbitrage.md) §4.)
- [Mitchell-Pulvino full text](https://tevgeniou.github.io/EquityRiskFactors/bibliography/RiskArbitrage.pdf) · [Alpha Architect screen](https://alphaarchitect.com/a-quantitative-strategy-for-enhancing-merger-arbitrage/)

**Verdict: ~0.5–0.6, and that understates risk** (artificially low vol = the calm of a put-seller between
crashes). Anyone quoting merger-arb Sharpe >0.8 is using gross/frictionless numbers or a pre-crash window.

### A5. SPAC arbitrage ("T-bill + free warrant") — genuinely near-risk-free, but the excess has compressed
Buy SPAC common ≤ trust value (~$10 + accrued interest); at the merger vote, **redeem for full trust value
while keeping the warrant**. Downside floored by the T-bill trust; upside a free OTM call. **Klausner-Ohlrogge-
Ruan "A Sober Look at SPACs":** the *arbitrageur* earned **11.6% annualized on a risk-free basis** (2019–20),
while the *deSPAC buy-and-hold* investor lost **−34.9% mean / −65.3% median at 12 months** (median SPAC holds
only $6.67 cash/share after dilution). **Gahng-Ritter-Zhang (RFS 2023):** the redeem-and-keep-warrant package
returned **9.3% on average** (convertible-bond-like). **AQR (2024):** the edge = "spread-to-trust"; it exceeded
10% annualized in the 2008–09 crunch, went *negative* in the 2020–21 mania, and is **~0 in calm post-2021
markets** (current ~8% yield-to-trust is mostly just the T-bill rate).
- [Klausner-Ohlrogge-Ruan](https://www.ecgi.global/sites/default/files/working_papers/documents/klausnerohlroggeruanfinal.pdf) · [Gahng-Ritter-Zhang SSRN 3775847](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3775847) · [AQR "Are SPACs Still Alive?"](https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Are-SPACs-Still-Alive.pdf)

**Verdict: the only genuine near-risk-free edge in the set — but the EXCESS over T-bills (the actual alpha)
has compressed to ~0 in calm markets and only reopens in liquidity crunches.** The 23.9% boom-era figure is
non-repeatable warrant beta. Avoid the deSPAC long side — that's the documented −35% to −65% loser.

---

## Area B — ML intraday alpha (limit-order-book / microstructure)

**The core finding: the literature reports accuracy, not net P&L.** DeepLOB (Zhang, Zohren, Roberts 2019),
the most-cited paper, reports ~78–84% accuracy / 77–83% F1 on FI-2010 mid-price direction — **with no
transaction costs, no spread crossing, no queue model, no Sharpe.** The repo is classification-only; it
contains **zero evidence of net profitability.**

**Three walls between accuracy and profit:**
1. **√-time annualization inflation:** a per-interval Sharpe of 0.03 on minute bars × √(252×390) ≈ 313×
   becomes "Sharpe 10" on paper. **Any HFT Sharpe >10 is almost always this artifact** (Lo 2002).
2. **Cost/spread wall:** tick-ahead edges are sub-basis-point; you pay ~half-spread each way.
3. **Queue position / adverse selection:** backtests assume mid/touch fills; reality is back-of-FIFO-queue,
   filled right before the price moves against you. Plus FI-2010 label leakage (whole-series z-score
   normalization → look-ahead; smoothed-mid labels aren't tradeable events).

**The honest papers (cite these against any shiny claim):**
- **LOBFrame / Briola et al.:** *"high forecasting power does not necessarily correspond to actionable
  trading signals"* — MCC collapses to ~random (0.01–0.11) on small-tick stocks, i.e. exactly where the
  spread constrains you. [arXiv 2403.09267](https://arxiv.org/abs/2403.09267)
- **LOBCAST benchmark:** audited 15 SOTA models — *"all models exhibit a significant performance drop when
  exposed to new data."* [arXiv 2308.01915](https://arxiv.org/abs/2308.01915)
- **Most honest walk-forward intraday paper found:** "Interpretable Hypothesis-Driven Trading: A Rigorous
  Walk-Forward Validation Framework" (2025) — 5 microstructure patterns, 100 US equities, 2015–2024, 34
  walk-forward periods, realistic costs → annualized return **0.55%, Sharpe 0.33, max DD −2.76%, p=0.34
  (insignificant).** A transparently *negative* aggregate finding. [arXiv 2512.12924](https://arxiv.org/abs/2512.12924)
- **The statistical hammer:** Deflated Sharpe Ratio + "Pseudo-Mathematics…" — expected max Sharpe grows
  ~√(2·ln N) from pure noise across N trials; deep nets with thousands of architecture trials on FI-2010 (5
  stocks, 10 days) are a multiple-testing machine. See [`06`](06-validation-methodology.md).
- **Numerai** is the honest-OOS counterpoint: obfuscated features, live OOS scoring, originality-neutralized.
  Individual model live correlations are tiny (single-digit % CORR); only the crowd meta-model is tradeable —
  the realistic picture of ML equity alpha: small, crowded, decaying.

**Verdict (Area B): there is NO credible, reproducible, walk-forward, net-of-cost intraday Sharpe in the
public LOB literature that survives scrutiny.** Best classifiers say nothing about P&L; honest walk-forward
prints Sharpe ~0.3 (often insignificant); "Sharpe >10" claims are √-time artifacts. For non-colocated
participants, honest net intraday Sharpe is **≤0**. Default prior on "ML intraday, Sharpe >3, no cost/queue
model": **false.**

---

## Master scorecard

| Edge | Headline | Honest net Sharpe | Accessible? | Status |
|---|---|---|---|---|
| S&P 500 index rebalance | add +7.4%→+1.0%; del −16%→−0.6% | **~0–0.2** | Retail | **Decayed to ~0** |
| Russell reconstitution | +4.6% window | ~0–0.2 | Retail | **Mostly decayed** |
| Classic SUE-PEAD | ~15–19% gross | **~0.1–0.3 liquid; ~0 large cap** | Retail | **Dead in liquid names** |
| Intraday PEAD | — | **~0 / negative** | HFT-only | **Gone (HFT captures it)** |
| Earnings announcement premium | >60 bps/mo | **~0.3–0.5** | Retail tilt | **Durable but low-Sharpe** |
| Merger arb (realistic) | 10.6% net, SR 0.57 | **~0.5–0.6 (short-put tail)** | Cash deals retail | **Alive, thin, crash-correlated** |
| SPAC cash+warrant arb | 9.3–11.6% near-risk-free | high *when spread exists*, ~0 excess in calm | Inst. allocation edge | **Compressed post-2021; reopens in crises** |
| deSPAC buy-and-hold | −35% to −65%/yr | strongly negative | — | **The loser side — avoid** |
| ML-LOB (DeepLOB etc.) | 78% F1 FI-2010 | **≤0 net (non-colo); fantasy if >10** | HFT-only | **No honest net edge published** |
| Honest walk-forward ML | Sharpe 0.33, p=0.34 | **~0.3, insignificant** | Retail | **Honest = near-zero** |

**Three flags for anyone pitching these:** (1) any merger-arb Sharpe >0.8 hides the down-market short-put
beta; (2) any SPAC return that's really 2020–21 warrant appreciation masquerading as repeatable edge; (3) any
ML intraday Sharpe quoted without a queue model, costs, Deflated Sharpe, and walk-forward is overfit until
proven otherwise. **None of these reaches Sharpe 5; the durable ones cluster at 0.3–0.6 net** — consistent
with the whole survey.
