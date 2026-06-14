# What actually produces high Sharpe on Hyperliquid — multi-source research

Synthesis of a thorough, multi-source sweep (academic/SSRN/arXiv, quant blogs,
crypto/FinTwit, Reddit/forums, MetaTrader/ICT/TradingView) commissioned to answer:
**is a genuine, honest, net-of-fee Sharpe ~3 reachable with technical/price-action
trading on Hyperliquid perps, and via what?** Every number below is a *claim with a
named caveat*; HL fees are ~4.5 bps taker / 1.5 bps maker base (rebate only at top
tiers), hourly funding.

## The one-paragraph answer

A real, repeatable, **net Sharpe ~3 exists almost only in (a) passive market-making
with top-tier maker rebates + an adverse-selection signal, and (b) funding/basis
carry during favorable regimes** — both capacity- and regime-constrained. For a
strategy that **takes liquidity** (any bot that crosses the spread), the credible,
cost-aware ceiling is **~1.5–2.4**, delivered by **vol-scaled trend-following on
majors + funding carry**, and it **decays** (post-2024 institutional crowding has
compressed every crypto edge — the research is unanimous on this). Our own
validation lands at **~1.1 (recent/HL era) to ~1.2 (full sample)**, consistent with
the lower, more honest end of the literature once you restrict to the genuinely
tradeable 2023→now window. **A taker price-action Sharpe 3 is not real here.**

## Method-by-method, with the most credible net Sharpe and source

| method | credible NET Sharpe | taker-viable? | verdict |
|---|---|---|---|
| **Market-making** (Avellaneda-Stoikov + OFI/microprice filter, funding-aware) | **~2.9** (Hyperliquid HLP vault, audited on-chain) | **maker-only**, rebate + infra | The only real ~3; needs top rebate tier, inventory control, adverse-selection signal. Matches our VELOCITY result exactly: huge gross, dies the instant you take. |
| **Funding / basis carry** (cash-and-carry, cross-sectional funding) | **6.45 full-sample → 4.06 (2024) → NEGATIVE (2025)** (Borri-Liu-Tsyvinski-Wu, arXiv 2510.14435); XS carry ~0.74 (SSRN 4666425) | **yes** | Real but non-stationary, tail-heavy (pennies in front of a steamroller); basis compressed 25%→<5%. Our trend-filtered carry sleeve = 0.92. |
| **Vol-scaled time-series trend** on majors | **1.7** (Grayscale BTC 20/100 MA), **1.83–2.41** (arXiv 2602.11708, costs incl.), Sortino 3.83 (XBTO) | **yes** | The most robust taker-viable family; low turnover. Higher numbers lean on pre-2022 mega-bull years. Our PULSE = 1.2 full / 0.75 HL era. |
| **Cross-sectional momentum** | ~2.6%/wk gross (arXiv 2510.14435) but **OOS −2.35%/yr, MDD>75%, dies at 125bps** (Starkiller) | majors only | Cost-fragile; the "2+ Sharpe" vendor claims (unravel.finance) are marketing (their own footnote: "<2", no costs). |
| **Cross-sectional / residual reversal (daily)** | blog ~2.0 OOS / 1.5 after double-fees (Lui, Medium) | marginal | **Did NOT replicate on our data** at HL taker: raw/residual reversal nets ~0–0.3 Sharpe, broad universe. Short-borrow on 400+ alts is the unmodeled killer. |
| **Liquidation-cascade fade** | PF 2.5–2.9 on SOL/ETH, walk-forward (Curupira); BTC fails | marginal | bps-thin edge, <5-min scalp, needs L2/liquidation data + maker fills; HL's transparent on-chain data is a genuine advantage but it's HFT-adjacent. |
| **OFI / microprice / order-flow** | standalone Sharpe **0.12** (Dean Markwick), "costs eat you alive" | **no (maker overlay only)** | Real contemporaneous signal; only monetizable as the adverse-selection filter that lifts market-making from ~1 to ~3. |
| **Opening-Range Breakout (ORB)** | **2.81 in US EQUITIES** (Zarattini, 5-min, RelVol "stocks in play") | — | Real in equities but driven by relative-volume selection + rare monster winners (17–24% win rate), no slippage modeled. **Does NOT transfer to crypto 24/7**: our crypto-ORB test was IS −5.2 / OOS +8.5 — a pure regime artifact. |
| **VWAP reclaim / bands** | none credible; 713% gross → **−97%** at 0.1%/trade | no | Destroyed by fees + whipsaw; "session" is ambiguous in 24/7 crypto (extra overfit DOF). Mostly course-seller content. |
| **London/session breakout** | ~50% win, expectancy ≤ spread (independent backtests) | no | Mostly lore; net-zero on majors after costs. |
| **ICT / Smart-Money-Concepts** (order blocks, FVG, liquidity sweeps, Power-of-3) | **none** (90-trade vendor scripts; one descriptive forex paper that is near-tautological) | — | Unfalsifiable narrative + curve-fit screenshots + course sellers. No pre-registered, OOS, cost-inclusive test exists. The *liquidity-sweep mechanism* is real microstructure; the ICT ritual adds no proven edge. |
| **Cross-exchange / latency / triangular arb** | no defensible net taker Sharpe | no (HFT/colocated) | Makarov-Schoar "$1bn" is gross cross-region, gated by transfer latency/capital controls; normal cross-exchange dispersion <5bps < fees. Triangular arbitraged away. |
| **Intraday seasonality / funding-settlement** | economically tiny; "Sharpe 7.75" overnight claim = overfit | overlay only | Periodicity is statistically real but below costs; use only to size up MM/carry in active windows. |
| **Volatility / variance-risk-premium** (short straddles) | ~1.15, **catastrophic left tail** | maker on options | Sharpe flatters it by ignoring the tail; not a clean Sharpe-3 path; needs options infra. |

## What the research says to actually do (and what it doesn't)

- **For a genuine ~3:** run a **passive market maker** with maker rebates + an OFI/
  imbalance adverse-selection filter + funding-aware quoting + strict inventory
  control. This is exactly the VELOCITY finding (gross alpha is huge; it lives
  inside the spread). It is an infrastructure/rebate game, validated only with
  L2/queue simulation — not a taker bot.
- **For a robust ~1.5–2 at taker:** vol-scaled **trend on majors + funding carry**,
  vol-targeted, regime-filtered. This is what we built (PULSE + trend-filtered
  carry, blend ~1.1–1.2). The literature's 2.0–2.4 leans on pre-2022 regimes and
  "adaptive construction"; honestly discounted to the tradeable 2023→now era it is
  ~1.1–1.3 — which is what we measure.
- **Myths / avoid (taker, standalone):** ICT/SMC, VWAP systems, session/London
  breakouts, cross-exchange/triangular arb, funding-flush "buy the dip", any
  Twitter-screenshot or course-seller "70–80% win rate". The crypto-ORB that
  *looked* like Sharpe 3.48 on 60 days was an IS −5 / OOS +8 regime mirage.

## Honest bottom line for the brief

A validated, honest **Sharpe 3 from taker price-action does not exist** on the data
and venue available. The deployable, validated number is **~1.1–1.3 net** (trend +
trend-filtered carry, HL fees+funding, IS/OOS-stable). The only credible route to
~3 is **market-making with rebates** (VELOCITY's gross alpha is real but maker-only;
proving it needs live L2/queue-fill data, not 1-min bars). Everything in between
(~1.7–2.4) is either pre-2024-regime-dependent or carry that has already decayed.

## What we tested ourselves (not just read) — all confirm the verdict

On real data (Coinbase 1-min / daily crypto, HL fees+funding, IS/OOS):
- **Crypto ORB** (opening-range breakout): full-sample Sharpe 3.48 but **IS −5.2 / OOS +8.5** — a 60-day trend-regime mirage, not an edge.
- **Daily cross-sectional residual & raw reversal** (broad universe): nets **~0–0.3** at HL taker — the blog "Sharpe 2" does NOT replicate.
- **Mechanical liquidity-sweep reversal** (the one falsifiable ICT kernel), tested
  vs the controls the research prescribed: **−9.7 bps/trade, 32% win, IS −8.9 /
  OOS −10.5, and it does NOT beat random-entry (−7.7) or naive-breakout-fade
  (−7.7).** It is *worse* than random once costs are paid — ICT/SMC falsified.
- **Intraday momentum / VWAP-reclaim / VWAP-bands**: negative-to-catastrophic net.
- **What DID validate:** vol-scaled trend (PULSE, ~1.2 full / 0.75 HL) + trend-
  filtered funding carry (0.92 HL) → blend **~1.1–1.3**, IS/OOS-stable. That is the
  honest deployable number; everything claiming taker Sharpe 3 was a mirage.

### Primary sources
- Borri, Liu, Tsyvinski, Wu — *Cryptocurrency as an Investable Asset Class* (arXiv 2510.14435): carry 6.45→neg, CMOM 2.6%/wk.
- *Systematic Trend-Following w/ Adaptive Portfolio Construction* (arXiv 2602.11708): net trend Sharpe 1.83–2.41.
- Grayscale — *The Trend is Your Friend* (BTC 20/100 MA, Sharpe 1.7). AQR — *Time Series Momentum*.
- Albers et al. — *The Market Maker's Dilemma* (arXiv 2502.18625); *Funding-Aware Optimal MM for Perp DEXs* (arXiv 2605.06405); Hyperliquid HLP ~2.89 (secondary).
- Dean Markwick — *Order Flow Imbalance* (Sharpe 0.12). Starkiller Capital — XS momentum OOS negative.
- Zarattini, Barbon, Aziz — *A Profitable Day Trading Strategy (ORB Stocks-in-Play)* (SSRN 4729284, Sharpe 2.81 equities).
- Yang & Malik (arXiv 2405.15461, pairs Sharpe 0.68 full-cycle); Lui (Medium, daily XS reversal ~2.0); Curupira (liquidation-cascade walk-forward).
- Han, Kang, Ryu (SSRN 4675565): XS momentum/reversal insignificant net of costs.
