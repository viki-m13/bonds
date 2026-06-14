# Risk-Managed Short-Vol & Crypto Basis — Can You "Fix the Tail" Into a High Sharpe?

*Phase 2 supplement. Both short-vol and crypto carry are negatively-skewed carry trades — Sharpe
systematically overstates quality because the crash hasn't happened in-sample. The question here: do
tail-hedged / risk-managed variants rescue a high Sharpe? Answer: no — they buy a smaller left tail at a
roughly Sharpe-neutral-to-negative cost. The honest deployable numbers are ~0.8–1.0 (short-vol) and
~1.0–1.8 (crypto basis, post-FTX, decayed), with a fat residual left tail.*

---

## Area A — Tail-hedged / risk-managed short volatility

**Central finding: a genuine convex tail hedge (long OTM puts / long VIX calls) does NOT raise net Sharpe —
it lowers it.** It is drawdown insurance, not alpha. The cleanest proof is the CBOE/Bondarenko benchmark
pair (Jun 1986–Dec 2018):

| Index | Mechanism | Sharpe | Return | MaxDD |
|---|---|---|---|---|
| **PUT** | Naive cash-secured put-write, NO hedge | **0.65** | 9.54% | −32.7% |
| **PPUT** | Same put-write + 5% OTM long-put tail hedge | **0.33** | 6.64% | (smaller) |

Adding the long-put tail hedge **roughly halved the Sharpe (0.65 → 0.33).** Independently, AQR's standalone
rolling long-put has geometric mean **−6.4%, Sharpe −0.61, maxDD −92%** (1985–2020) — a brutal standalone
bleed. Tail hedging only helps via **portfolio-level risk-budget reallocation** (Universa-style: ~3.3%
Universa + 96.7% S&P beat the S&P over 2008–2018), never as standalone alpha.
- [Bondarenko CBOE PUT/PPUT](https://cdn.cboe.com/resources/education/research_publications/PutWriteCBOE19_v14_by_Prof_Oleg_Bondarenko_as_of_June_14.pdf) · [AQR Tail Risk: Put vs Trend](https://images.aqr.com/-/media/AQR/Documents/Insights/White-Papers/AQR-Tail-Risk-Hedging-Contrasting-Put-and-Trend-Strategies.pdf)

**What the "hedge" actually did (before→after Sharpe map):**

| Approach | Naive Sharpe | "Hedged" Sharpe | What the hedge did |
|---|---|---|---|
| XIV inverse-VIX 2010–17 | ~1.0–1.2 realized | **died −96% in a day** | nothing — no hedge; Sharpe was a lie |
| CBOE PUT → PPUT | 0.65 | **0.33** | true long-put tail hedge → **halves Sharpe** |
| Sepp filtered + delta-hedged SPX puts | ~0.6 | **~0.8–1.0** | hedged **beta/delta + regime filter**, NOT tail |
| AQR covered-call (BXM) | 0.37 | 0.52 | hedged **equity-reversal leg**, not vol tail |
| Eurex OTM vs ATM strangle | 0.978 (ATM) | 1.428 (OTM) | cheaper **short** strike, not a long hedge |
| Quantpedia long-VIXY VRP-timed overlay | 0.56 | 1.19 | *buying* vol (opposite of short vol), timed |

**The strongest credible candidate — Artur Sepp (filtered + delta-hedged short vol, 2005–2017).** Unhedged:
sell ATM SPX puts ~0.6, sell front VIX futures ~0.3 with **maxDD −80%+**. The filtered + delta-hedged version
**roughly doubles Sharpe to ~0.8–1.0, drives beta 0.5→~0, cuts drawdowns ~50%** — but the lift comes from a
**vol-regime filter + delta-hedging the equity leg, NOT from buying convex protection.** Sepp's own convexity
work warns genuine long-tail hedges carry "strongly negative overall performance."
- [Sepp](https://artursepp.com/2017/09/20/allocation-to-systematic-volatility-strategies-using-vix-futures-sp-500-index-puts-and-delta-hedged-long-short-strategies/) · [Quantpedia VRP](https://quantpedia.com/strategies/volatility-risk-premium-effect) · [Eurex short-vol](https://www.eurex.com/resource/blob/4341476/925095f00ceded85924b5357aef5a0a5/data/Short%20Volatility%20Strategies.pdf)

**Area A verdict:** Do not pay for naive long-put/VIX-call tail hedges expecting higher Sharpe — they lower
it. Best deployable risk-managed short vol = Sepp's filtered + delta-hedged book, **honest net ~0.8–1.0.**
Any standalone short-vol Sharpe >~1.0 quoted in-sample should be assumed fraudulent until shown surviving a
Feb-2018/Mar-2020 path net of margin and cost. (Reinforces [`03-volatility-options.md`](03-volatility-options.md).)

---

## Area B — Crypto basis / funding-rate arbitrage (post-FTX reality)

**Central finding: the eye-catching Sharpes (7–58) are real numbers from real papers, but every one is
in-sample, gross, and tail-blind** — compensation for exactly the risk they don't model (exchange
insolvency, stablecoin de-peg, deleveraging cascades). The honest post-FTX net Sharpe a small fund can
capture is **~1.0–1.8**, with a rare but catastrophic left tail.

- **The "fantasy" source — Werapun et al. (2025):** Sharpe Drift 58.40, BitMEX 22.21, ApolloX 18.09, Binance
  4.42 — but the sample is a **~6-month bull window (Aug 2023–Feb 2024) with persistently positive funding
  that excludes FTX entirely.** The same paper elsewhere reports a *contradictory* set (Drift 23.55, Binance
  −7.34, BitMEX −7.93) — proof the figures are period-conditional, not stable. **Reject at face value.**
  [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2096720925000818)
- **The honest anchor — He, Manela, Ross, von Wachter, "Fundamentals of Perpetual Futures" (arXiv 2212.06888):**
  Bitcoin perp arb **Sharpe ~1.8 under high retail trading costs; ~3.5 for zero-fee market makers.**
  Futures-spot deviations 60–90%/yr but **decay ~11%/yr** as markets mature. Spans COVID and FTX. This is
  the number to anchor on — net of fees, before counterparty tail, and shrinking. [arXiv 2212.06888](https://arxiv.org/abs/2212.06888)
- **BIS Working Paper 1087 "Crypto Carry":** average annualized carry **≈7% p.a.** (Apr 2019–Jul 2024),
  occasionally >40%. Debunks "risk-free arbitrage": the futures leg has severe drawdowns; no cross-margining
  → margin calls → forced liquidation; and **high carry forecasts future price crashes.** The ~7% gross
  average is what's actually harvestable; the 40%+ spikes are short-lived and coincide with stress.
  [BIS WP1087](https://www.bis.org/publ/work1087.pdf)
- **Practitioner ground-truth — Liquibit Market Neutral fund (arbing crypto since 2014, >$100m AUM):** their
  2022 headline return was **NEGATIVE — their only counterparty loss since 2014 — due to FTX**; ~+2% net
  excluding FTX (still worst year ever). A 10-year specialist running this exact trade had its worst year — a
  *loss* — precisely from counterparty failure, which no Sharpe captures. Mitigation: ~half assets
  off-exchange via custody (Copper ClearLoop). [HFJ](https://thehedgefundjournal.com/liquibit-market-neutral-crypto-strategy-traditional-trading/)

**Structural decay:** annualized basis 15–25% (2021) → 8–12% (2022–23) → **<4% (2024+), below the US T-bill
rate** on majors. **Why Sharpe 20–58 is fantasy:** short in-sample windows (no losing periods), gross of fees
(same trade: 3.5 zero-fee → 1.8 retail), counterparty risk unpriced (on-exchange collateral can go to zero),
survivorship (FTX would have shown a great Sharpe right up to −100%).

**Area B verdict:** an operationally-competent small fund running multi-exchange funding + basis with
off-exchange custody can realistically target **net Sharpe ~1.0–1.8, ~8–15% net APR in normal conditions —
with a fat, rare left tail (−50% to −100% of on-exchange collateral in an FTX-type event).** Anyone quoting
20+ is selling an in-sample, FTX-excluded backtest. (Reinforces [`05-crypto-niche.md`](05-crypto-niche.md).)

---

## Combining the two to chase > 5 — why it fails

Neither standalone gets near 5 (Area A best ~1.0, Area B best ~1.8), and **stacking them does NOT diversify —
it concentrates tail risk.** Both are negatively-skewed carry trades that blow up in the *same* "everyone
delevers at once" macro state: the Feb-2018 vol spike and the 2022 crypto deleverage are the same liquidity
event. Their left tails are correlated, so a combined backtest showing >3 has hidden, correlated tail risk.

The legitimate route to a high portfolio Sharpe is **breadth across genuinely uncorrelated, positive-or-
symmetric-skew engines** (short-horizon stat-arb, low-skew trend, market-making rebate capture) where
independent bets compound via diversification — *not* levering two skewed carry trades whose tails coincide.
This is the same conclusion as the ρ-ceiling math in [`08`](08-accessible-high-sharpe-hunt.md): you reach a
high Sharpe only with many *uncorrelated* bets, and correlated carry trades are the opposite of that.

*Open gap: one relevant SSRN paper (Iyer, "Shorting Volatility," 5464595) is Cloudflare-gated; its backtest
tables could not be verified and are not relied on here.*
