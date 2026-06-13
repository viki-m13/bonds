# Crypto & Niche Markets — Validated, Skeptical Survey

**Bias:** Adversarial. Crypto Sharpe ratios are systematically overstated because the dominant risk
(exchange blowup, depeg, smart-contract loss) is a fat left tail that rarely shows up in the return
volatility the Sharpe denominator measures.

## Executive summary

| Strategy | Reported Sharpe | Net? | Live? | Decayed? | Verdict |
|---|---|---|---|---|---|
| Crypto carry / cash-and-carry (BTC) | **12.8** BTC, 7.0 ETH | gross-ish | live edge, academic measure | severely — 6.45 (2020-25) → 4.06 (2024) → negative (2025) | **Real but Sharpe wildly understates tail risk** |
| Funding-rate perp arb (practitioner) | ~3–5 implied | net pre-tail | live | yes, ~25%→<5% APY | **Partially valid; counterparty risk dominates** |
| Triangular / indirect arb (intra-exchange) | 9–14 bps/trade, ~95% win | net | live (HFT only) | yes | **Valid only for co-located HFT** |
| Cross-exchange arb | high historically | often gross | edge mostly gone | yes | **Mostly decayed** |
| MEV arbitrage / sandwich | $675M+ cumulative | net of gas | live | margins compressed | **Real P&L, not a Sharpe-style strategy** |
| AMM LP / Uniswap V3 | often **negative** | net | live | n/a | **Mostly invalid for passive LPs (LVR)** |
| Crypto cross-sectional momentum (Liu-Tsyvinski) | ~1–2 | gross | academic | partial | **Valid, modest, not high-Sharpe** |
| Crypto time-series momentum / trend | ~1.0–1.3 | gross | live | stable | **Valid, modest** |
| Stat-arb / cointegration pairs | ~1.4–1.5 | net-ish | backtest | likely | **Modest, fragile** |
| Stablecoin/DeFi delta-neutral (Ethena) | high in bull funding | net pre-depeg | live | cyclical (20%+→~4%) | **Valid yield, depeg/funding tail** |
| Prediction-market / sports arb | $40M+ extracted | net | live | competitive | **Real, capacity-limited, operational risk** |

---

## 1. Crypto carry / cash-and-carry (perp-spot & futures basis)

**Mechanism.** Long spot (or staked) asset, short the perpetual or dated future. Capture the basis:
positive funding paid by leveraged longs (perps) or the futures premium decaying to spot. Delta-neutral.

**The headline "Sharpe >5" claim.** BIS Working Paper 1087 / "The Crypto Carry Trade" reports annualized
Sharpe ratios of **12.8 for Bitcoin** and **7.0 for Ether**. Sample Aug 2011–June 2022, capturing the
heavy-volume 2021 premium spike (3-month BTC futures hit a **~40% annualized premium** early 2021). Average
carry ~7–8%/yr, sometimes >40%. A later/practitioner measure: annualized crypto-carry Sharpe **6.45
(2020–2025)**, falling to **4.06 (2024)** and **negative (2025)** — a textbook decay signature.
- [BIS WP1087](https://www.bis.org/publ/work1087.pdf) · [CMU/Christin PDF](https://www.andrew.cmu.edu/user/azj/files/CarryTrade.v1.0.pdf) · [CEPR VoxEU](https://cepr.org/voxeu/columns/crypto-carry-market-segmentation-and-price-distortions-digital-asset-markets) · [arXiv 2510.14435](https://arxiv.org/pdf/2510.14435)

**Why the 12.8 is the single most misleading number in crypto.** The strategy's market risk is near zero,
so volatility is tiny and Sharpe is huge — but the dominant risk is **exchange insolvency**. Anyone running
BTC cash-and-carry with collateral on **FTX in Nov 2022 lost principal regardless of how delta-neutral they
were**. The Sharpe denominator never captured this. Also: short-leg liquidation/ADL risk, funding-sign
flips. Capacity bounded by open interest and position limits; crowding compresses the basis fast (2021's
40% premium → "25% in early 2024 → under 5%").
- [arbitragescanner](https://arbitragescanner.io/blog/crypto-funding-rate-arbitrage-guide) · [Boros/Pendle](https://medium.com/boros-fi/cross-exchange-funding-rate-arbitrage-a-fixed-yield-strategy-through-boros-c9e828b61215)

**Verdict.** **Real edge, but the 12.8 Sharpe measures a tail-free window of a strategy whose true risk is a
non-Gaussian exchange-blowup left tail.** A carry/yield strategy with embedded counterparty default risk,
not a Sharpe-12 free lunch — and the edge had decayed to negative by 2025.

---

## 2. Funding-rate arbitrage (practitioner / DeFi delta-neutral)

Same mechanism, framed as harvesting perpetual funding (APR ≈ funding_rate × 1,095). Practitioner sources
cite **8–20% APY** at low realized vol (implied Sharpe ~3–5 in good regimes). 2024 average funding: BTC
~11%, ETH ~12.6%, with **−0.05%** prints in bear phases. After taker fees, hedge rebalancing, and
occasional negative-funding bleed, net is meaningfully below gross ("if rates turn negative 72+ hours,
positions transition from income to bleeding cost"). Funding can flip sign within one interval; crowding
accelerates compression; same FTX-style counterparty tail.
- [Amberdata guide](https://blog.amberdata.io/the-ultimate-guide-to-funding-rate-arbitrage-amberdata) · [blofin](https://blofin.com/en/academy/education/delta-neutral-crypto-strategies)

**Verdict.** Partially valid — a legitimate low-vol carry strategy, but the high implied Sharpe ignores
funding-flip and counterparty tails, and the edge is cyclical/decaying.

---

## 3. Triangular & cross-exchange arbitrage

**Triangular (intra-exchange).** Makarov/Schoar-style "Indirect Internal Conversions": triangular trades
profitable **94.97%** of the time, equal-weighted **net return 9.3 bps**, ROC 9.8 bps; indirect conversions
93.92% profitable, **11.8 bps net**, ROC 14.4 bps. Per-trade microstructure edges, not annualized Sharpe —
only viable for co-located HFT. **Cross-exchange.** Early crypto had large cross-venue deviations
("Kimchi premium" era); these have **significantly narrowed** as markets matured. Net profit is gutted by
**withdrawal/transfer latency** (you can't move BTC between exchanges in milliseconds), withdrawal fees, and
the price moving in-flight.
- [arXiv 2002.12274](https://arxiv.org/pdf/2002.12274) · [Finance Research Letters / ScienceDirect](https://www.sciencedirect.com/science/article/pii/S154461232401537X)

**Verdict.** Valid only as a latency/HFT business. Per-trade edge real (9–14 bps, ~95% win) but tiny;
aggregate Sharpe depends entirely on fill rate and infrastructure. Cross-exchange version largely decayed.

---

## 4. MEV (sandwich, arbitrage, backrunning, liquidations)

Reorder/insert transactions in a block. Atomic DEX arbitrage (riskless within a block), sandwiching
(front+back-run a victim swap), liquidation backrunning. **>$675M extracted on Ethereum before Sept 2022**;
the bot "jaredfromsubway.eth" executed ~238,000 attacks on 100,000+ victims for ~$6M; a single Curve
front-run netted >$1M ETH. Net of gas and (post-Flashbots) priority bids to builders/validators. **Not a
Sharpe strategy** — it is an operational/infrastructure business (mempool monitoring, builder relationships,
gas-bidding wars), winner-take-most, with margins compressing as value migrated to builders under
proposer-builder separation.
- [arXiv 2405.17944](https://arxiv.org/abs/2405.17944) · [arXiv 2206.04185](https://arxiv.org/pdf/2206.04185) · [CoW DAO](https://cow.fi/learn/what-are-mev-bots-and-how-do-they-make-money)

**Verdict.** Real, large P&L; not a packageable high-Sharpe strategy. Sandwiching is extractive/adversarial
and increasingly mitigated (private order flow, MEV-protected RPCs).

---

## 5. AMM / Uniswap V3 liquidity provision

Provide liquidity, earn swap fees; bear impermanent loss / LVR. Across analyzed pools, Uniswap V3 generated
**$199.3M in fees but $260.1M in impermanent loss**, leaving **~49.5% of LPs with negative returns**.
**Loss-Versus-Rebalancing (Milionis, Moallemi, Roughgarden, Zhang):** instantaneous LVR ≈ **σ²/8**; a
5%-daily-vol ETH-USDC pool loses ~3.125 bps/day ≈ ~11%/yr to arbitrageurs. A pool needs to turn over ~10%
of TVL daily for 30 bps fees to cover LVR.
- [arXiv 2208.06046 (LVR)](https://arxiv.org/pdf/2208.06046) · [Uniswap blog](https://blog.uniswap.org/fee-returns) · [Atis E / LVR](https://atise.medium.com/liquidity-provider-strategies-for-uniswap-v3-loss-versus-rebalancing-lvr-ee0ffdf1f937)

**Verdict.** Mostly **invalid for passive LPs** — negative risk-adjusted returns for ~half. Only viable with
active hedging (short the rebalancing portfolio), tight actively-managed ranges, JIT liquidity, or
MEV-integration. No credible high Sharpe for passive LPing.

---

## 6. Crypto cross-sectional momentum & factors (Liu-Tsyvinski-Wu)

Liu, Tsyvinski & Wu, "Common Risk Factors in Cryptocurrency," *J. Finance* 2022: a **three-factor model
(market, size, momentum)** captures the cross-section; **ten characteristics form significant long-short
strategies**. Liu & Tsyvinski (2021): crypto returns/vol are an order of magnitude higher than equities, but
**Sharpe ratios are broadly comparable** to the stock market (~1, not >3).
- [SSRN 3379131](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3379131) · [J.Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119)

**Verdict.** Valid and well-replicated, but **NOT a high-Sharpe strategy** (~1, gross, before crypto's brutal
trading costs and short-borrow constraints). Honest academic work that explicitly does *not* claim Sharpe >3.

---

## 7. Crypto time-series momentum / trend

Daily TS-momentum on top-5 caps generates significant positive profits over short lookbacks; Bitcoin trend
with a trailing-stop exit improved **Sharpe to ~1.07** (Calmar 0.87). Cross-asset TS-momentum benchmarks
(Quantpedia): gross Sharpe ~1.17–1.60.
- [Quantpedia TS momentum](https://quantpedia.com/strategies/time-series-momentum-effect) · [Quantpedia Bitcoin trend](https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/)

**Verdict.** Valid, modest (Sharpe ~1). Robust and live-tradeable, but nowhere near >3.

---

## 8. Crypto statistical arbitrage / cointegration pairs

Pairs study across **209 cryptoassets, Aug 2021–Jan 2024**, identified 229 cointegrated pairs; best
risk-return strategy delivered **average annual Sharpe ~1.53 per pair**, but **median max drawdown 29%**.
Other backtests: Sharpe 1.42, return 96.6%, DD −31.98%.
- [EUR thesis](https://thesis.eur.nl/pub/67552/Thesis-Pairs-trading-.pdf) · [Digital Alpha](https://medium.com/digital-alpha-research/using-a-pairs-trading-statistical-arbitrage-approach-on-digital-assets-e29b10c6c651)

**Verdict.** Modest and fragile. Sharpe ~1.5 gross, sensitive to parameter estimation, deep drawdowns,
frequent pair re-selection. Net Sharpe likely <1 after crypto fees.

---

## 9. Stablecoin / DeFi delta-neutral yield (Ethena USDe/sUSDe)

Tokenized funding-rate carry — long staked ETH/BTC collateral, short equivalent perps; staking yield +
funding accrues to sUSDe. APY ranged from low single digits to **>30%** since early-2024 launch; **>20%
sustained** in the 2024 bull, **~3.72%** by early 2026. Negative funding absorbed by a Reserve Fund so
principal isn't lost — **until** a sustained negative regime depletes it and pressures the peg. Same
funding-flip + counterparty (CEX hedge) tail, **plus stablecoin depeg risk and smart-contract risk**. Low
day-to-day vol → flattering Sharpe that ignores a depeg/insolvency left tail.
- [eco.com Ethena](https://eco.com/support/en/articles/15254002-ethena-usde-and-susde-2026-delta-neutral-yield) · [Llama Risk](https://www.llamarisk.com/research/ethena-drawdown-methodology-v2)

**Verdict.** Valid productized carry yield, cyclical (20%+→~4%), with a depeg/funding/counterparty tail that
any Sharpe figure understates.

---

## 10. Niche markets

**Prediction markets / sports arb.** Academic (IMDEA Networks) documented **>$40M in arbitrage profits
extracted from Polymarket, Apr 2024–Apr 2025** (YES+NO < $1, combinatorial, cross-market vs sportsbooks).
Real but **capacity-limited** (thin order books), with operational/settlement and platform/regulatory risk;
no clean Sharpe published.
- [arXiv 2508.03474](https://arxiv.org/abs/2508.03474) · [QuantVPS](https://www.quantvps.com/blog/cross-market-arbitrage-polymarket)

**Electricity / power spreads & weather derivatives.** Stat-arb exists between PJM power and Henry Hub gas
futures, and across day-ahead/intraday/balancing markets; mature stat-arb desks target Sharpe ≥1.5, DD
10–15%. Weather derivatives are primarily hedging instruments (illiquid, OTC), not a documented high-Sharpe
strategy.
- [ScienceDirect power stat-arb](https://www.sciencedirect.com/science/article/pii/S2352467723000310) · [MDPI energy futures](https://www.mdpi.com/1911-8074/12/1/14)

**Verdict.** Real niche edges, modest Sharpe (~1.5) where measured, capacity-constrained; no credible >3 found.

---

## Cross-cutting conclusions

1. **The only genuine ">5 Sharpe" claims are crypto carry/basis (12.8 BTC, 7.0 ETH)** — near-real edges, but
   the Sharpe is structurally misleading: a delta-neutral cash flow has tiny return-vol, so any positive carry
   produces a huge Sharpe **while the dominant risk — exchange insolvency (FTX), liquidation, depeg — lives
   entirely in an unmeasured left tail.** For crypto delta-neutral strategies, tail/operational risk dwarfs
   market risk, so Sharpe systematically overstates quality.
2. **Decay is documented and severe** (carry Sharpe 6.45 → 4.06 → negative; basis APY 25% → <5%;
   cross-exchange spreads collapsed). The "Sharpe 5–10 in 2020-21" arbs have largely matured away.
3. **Everything with a legitimate, robustly-measured Sharpe is modest (~1–1.5):** momentum (Liu-Tsyvinski),
   trend (~1.1), stat-arb pairs (~1.5), power stat-arb (~1.5). None exceed 3 net.
4. **MEV and arb bots make real money but aren't Sharpe-style strategies** — latency/infrastructure
   businesses with winner-take-most dynamics and compressing margins.
5. **Passive Uniswap LPing is value-destructive** for ~half of LPs (LVR ≈ σ²/8).

**Bottom line.** Be deeply skeptical of any crypto "Sharpe >5." Where the number is real (carry/basis), it
measures a tail-blind delta-neutral cash flow whose true risk is exchange-blowup and depeg — and even that
edge had decayed to negative by 2025. No strategy surveyed offers a durable, net-of-cost, tail-honest Sharpe
above ~2.
