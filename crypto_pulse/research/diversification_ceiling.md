# The diversification ceiling — what it actually takes to reach Sharpe 3, and why a retail book can't

This is the quantitative answer to "keep pushing for Sharpe 3," triangulated from
(a) the portfolio math, (b) our own out-of-sample backtests, and (c) what real
funds deliver net. It supersedes the hand-wavy "not retail-accessible" claim with
numbers.

## 1. The math — Sharpe 3 is a BREADTH problem, not a signal problem

For K books each at Sharpe S with average pairwise correlation ρ, the combined
(risk-weighted) Sharpe is

    S_combined = S · √K / √(1 + (K−1)·ρ)

In the ideal ρ→0 case this is just **S·√K**. Equivalently (Grinold's Fundamental
Law) IR = IC·√breadth — you raise Sharpe by adding *independent bets*, not by
sharpening one signal. To hit 3:

| per-book Sharpe S | uncorrelated books K needed (ρ=0) |
|---|---|
| 1.0 | 9 |
| 0.7 | ~18 |
| 0.5 | 36 |

And with even mild correlation ρ=0.2 the denominator √(1+(K−1)·0.2) caps the
combined Sharpe at **S/√ρ ≈ 2.2·S** no matter how many books you add. So you need
*both* many books *and* genuinely near-zero correlation. This is exactly the
pod-shop model: Millennium/Citadel run hundreds of weakly-correlated pods, cap
each pod's risk, and the *fund* harvests the diversified top (plus leverage). It
is a headcount-and-infrastructure achievement, not one magic alpha.

## 2. Our own evidence — the books we can actually build

- **Crypto is ~single-factor.** `ensemble.py`: a 7-signal price-action ensemble
  (Sharpe +0.50, OOS negative) is *beaten* by the parsimonious 2-sleeve
  trend+carry (+1.11), because 5 of the 7 signals are redundant trend exposure.
  Adding correlated crypto signals adds variance, not Sharpe → within crypto we
  have ~2–3 independent bets, capping the book at ~1.1–1.4 net.
- **Cross-asset trend is genuinely uncorrelated but weak.** `cross_asset_book.py`:
  a risk-parity trend book over EQUITY/BONDS/COMMOD/FX ETFs scores **Sharpe +0.38
  over its full 1998–2026 history** (per-class: COMMOD +0.57, BONDS +0.19, EQUITY
  +0.13, FX +0.02) and was *negative* over 2023–24 (the well-documented bad trend
  regime). Its correlation to the crypto book is **+0.03** — real diversification —
  but Sharpe-optimal combination keeps the crypto book at **0.90** and assigns the
  weak cross-asset book ~0 weight on this window. **An uncorrelated book only
  helps if it has positive expectancy comparable to what it's diluting.**

Our +0.38 cross-asset number lands *exactly* where the literature says it should,
which is the honesty check that our backtests aren't inflated.

## 3. What real funds actually DELIVER net (the reality anchor)

| Program | Net Sharpe (full cycle) | Source |
|---|---|---|
| Diversified TSMOM (Hurst-Ooi-Pedersen, 1985–2012) | **1.79 gross → ~1.0 net** of fees+costs | docs.lhpedersen.com |
| TSMOM per single market (Century paper, 1880–2016) | ~0.4 avg gross | Hurst-Ooi-Pedersen JPM 2017 |
| Diversified global CARRY factor (KMPV, JFE 2018) | **1.10 gross** (single-class avg 0.74) | nber.org w19325 |
| AQR 4-style composite (val+mom+carry+def) | 1.74 **gross** ("unlikely achievable in practice") | AQR Investing with Style |
| AQR Managed Futures (AQMIX) | ~0.3–0.5 net | funds.aqr.com |
| SG CTA Index / Barclay BTOP50 | **0.56 / 0.64** net | cmegroup.com |
| AQR Style Premia (QSPIX) — *the canonical multi-style stack* | **target 0.70, delivered 0.41 live** | AQR / Morningstar |
| Managed-futures ETFs DBMF/KMLM/CTA (fees 0.75–0.90%) | ~0.4–0.7 since inception | stockanalysis.com |
| Carver, diversified 100+ instrument trend+carry futures | **~1.0 net, "rarely > 1.0"** | Advanced Futures Trading Strategies |
| AQR forward-looking NET assumption for trend | **0.4** | A Century of Evidence |

Note the pattern: the canonical diversified-trend backtest is **1.79 gross** but only **~1.0 net** and real funds deliver **0.27–0.88**. Carry adds a genuine low-correlation stream (carry–momentum corr **+0.18**), and the *gross* 4-style stack reaches 1.74 — but AQR themselves call >1.7 "unlikely achievable in practice," and Style Premia delivered **0.41 live vs a 0.70 target**.

Backtest→live degradation is ~50% (factor-of-2): McLean & Pontiff find returns
~26% lower OOS / ~58% lower post-publication; Harvey & Liu raise the significance
bar to t≈3. Carver: single-instrument ≈0.4, consistent live Sharpe >1.0 "rarely
achieved," and backtested 2–3 are "far too optimistic… caused by over-fitting."

## 4. Synthesis — the honest ceiling

- A **clean gross backtest** of a diversified multi-style multi-asset book can show
  ~1.0–1.5. After the ~50% live haircut, the **honest live ceiling for a
  diversified cross-asset retail systematic book is ~0.4–0.7 net**; single styles
  ~0.3–0.5. The largest real funds (AQMIX, BTOP50 0.64, QSPIX 0.41 live) bracket
  this exactly.
- Our validated **crypto 3-sleeve book at ~1.1 net (OOS 1.07)** is therefore
  already at the *high end* of what's realistically achievable for a small trader —
  because crypto is an inefficient, strongly-trending market where the per-book
  Sharpe is unusually good. It is NOT improved by (a) more correlated crypto
  signals or (b) bolting on a weak uncorrelated CTA book.
- **Sharpe 3 OOS net needs ~18 genuinely-uncorrelated books each at Sharpe ~0.7,
  with leverage and infrastructure to run them.** That is the pod-shop business
  model, not a signal we can discover. For us it remains out of reach as a taker;
  the maker/HFT route that *could* reach it we already falsified on real HL L2.

## 5. The crypto-native uncorrelated candidates (what actually has a pulse)

Traditional CTA trend is uncorrelated to crypto but too weak (net ~0.4) to lift a
1.1 book. The research ranked the crypto-native streams that are BOTH taker-viable
AND uncorrelated to trend+carry — these are the only real "extra sleeve" candidates:

| candidate | taker net Sharpe (honest) | corr to our book | capacity | catch |
|---|---|---|---|---|
| **Funding+OI extremes / liquidation-cascade fade** | ~0.3–0.8, **event-gated** | low | moderate | rare events, not always-on; "crowded gets more crowded" tail |
| **Long-tail funding-rate dispersion** (HL 4%/hr cap vs CEX) | ~1–2 on alts, ~0 on majors | low | **small ($10k–$500k/trade)**, crowds fast | secretly maker-only on majors; only long-tail survives taker fees |
| **Crypto VRP / short vol** (Deribit, not HL) | ~0.5–1.0 | low in calm, **+ in the tail** | ample | deep negative skew; not a clean diversifier; not on HL |
| On-chain flows / stablecoin / MVRV / sentiment | ~0.0–0.3 daily | low | high | mostly endogenous/lagging; label look-ahead bias; vendor hype |
| Cross-asset CTA trend (ETF) | ~0.4 net | **+0.03 (cleanest)** | high | too weak to add to 1.1 on this window |

The standout is the **funding+OI liquidation fade**: large per-event moves dwarf the
4.5 bps taker cost, it's mechanically driven (forced liquidations, not narrative),
and it's genuinely uncorrelated to trend+carry. It's an *event* sleeve (a handful
of high-conviction trades per quarter), so it diversifies the book's return timing
even if its standalone Sharpe is modest. **Long-tail funding dispersion** is the
other real one but is capacity-tiny and crowds quickly.

**Bottom line:** keep the crypto 3-sleeve (~1.1, OOS 1.07) as the core — it is
already at the high end of what's realistically achievable for a small taker,
because crypto is an unusually inefficient, strongly-trending market. The only
honest ways to *nudge* the portfolio higher are (i) add the funding+OI
liquidation-fade event sleeve (uncorrelated, taker-viable; needs forward funding+OI
recording to build), (ii) optionally a small long-tail funding-dispersion harvest,
and (iii) leverage on the diversified whole (scales return, not Sharpe). Chasing a
single price-action signal to Sharpe 3 is mathematically the wrong tree: 3 needs
~18 uncorrelated 0.7-Sharpe books plus leverage and infrastructure — the pod-shop
business model, not a discoverable alpha.
