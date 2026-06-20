# The honest hunt for Sharpe 3 on Hyperliquid — full record

This folder is the complete, honest record of a long brief: **find/invent a
trading strategy with a genuine out-of-sample, net-of-cost Sharpe of 3, deployable
on Hyperliquid — however necessary.**

## The answer (triangulated three independent ways)

**A durable, honest, OOS, net-of-cost Sharpe of 3 is not reachable for a retail
taker.** It is gated by infrastructure (maker queue priority + exchange-volume
rebates + colocated µs latency) or by breadth (hundreds of uncorrelated pods +
leverage) — neither retail-accessible. Full proof in
[`research/SHARPE3_VERDICT.md`](research/SHARPE3_VERDICT.md).

1. **Diversification math** — combined Sharpe `S·√K/√(1+(K-1)ρ)` is hard-capped at
   `S/√ρ`. Our measured numbers (5 uncorrelated sleeves, S≈0.64, ρ≈0.09) cap at
   **2.2**; reaching 3 needs ~18–22 genuinely-uncorrelated sleeves of this quality,
   which a single-factor crypto market can't supply.
   ([`research/diversification_ceiling.md`](research/diversification_ceiling.md))
2. **Costs vs. alpha (empirical, on real HL L2)** — order-book imbalance predicts
   short-horizon returns with **IC 0.24** but the move (~1 bp) is smaller than the
   spread+fee (~5–9 bp) → −4.3 bps net = **maker-only**. Confirmed by Pindza 2026
   (crypto microstructure taker net Sharpe −10 to −52) and DeepLOB's own authors.
   ([`research/microstructure_alpha.md`](research/microstructure_alpha.md),
   [`research/hft_microstructure_landscape.md`](research/hft_microstructure_landscape.md))
3. **Execution/latency** — Sharpe 3+ lives in HFT market-making (rebates + queue +
   µs latency) we falsified on real fills, or in pod-shop breadth. Even real HFT
   firms median ~1.6 *gross* (Baron-Brogaard-Hagströmer-Kirilenko).

## What we DID build — the deployable book (~1.5–1.6 honest)

The **grand stack**: 6 genuinely-uncorrelated sleeves (mean ρ≈0), net of real HL
funding + 4.5 bps taker, vol-targeted.
([`grand_stack.py`](grand_stack.py), [`research/grand_stack.md`](research/grand_stack.md))

| sleeve | archetype | role |
|---|---|---|
| TREND, ACCEL | directional momentum | core |
| CARRY | cross-sectional funding | core |
| BAB | low-beta factor | diversifier |
| SQUEEZE | volatility breakout | diversifier |
| FUNDFADE | funding-extreme event fade | **ρ=−0.29 decorrelator** |

**Sharpe ~1.5 daily (IS 1.32 / OOS 1.63) → ~1.6 weekly, maxDD <9%, Calmar ~3.3.**
Up honestly from the 1.1 starting point. Full-HL-universe validation (230 crypto
perps + HIP-3 equities/indices/commodities/FX/bonds) confirms the cross-asset
vehicles are genuinely uncorrelated (mean |corr| 0.08) but too weak to add — the
max-Sharpe book stays crypto-dominated.
([`hl_full_universe.py`](hl_full_universe.py),
[`research/hl_full_universe.md`](research/hl_full_universe.md))

### Leverage & CAGR (Kelly) — the return dial
Sharpe is leverage-invariant; leverage is purely the CAGR/risk dial, and HL's
caps (40× BTC → 3–5× alts) are not binding — drawdown is.
([`kelly_cagr.py`](kelly_cagr.py), [`breadth_leverage.py`](breadth_leverage.py))

| leverage | Sharpe | CAGR | maxDD |
|---|---|---|---|
| 1× | ~1.5 | +21% | −6% |
| 3× (¼-Kelly) | ~1.5 | +69% | −18% |
| 5× (½-Kelly) | ~1.5 | +124% | −30% |
| ~10× (Kelly) | ~1.5 | +273% | −55% (not survivable) |

### Downside mitigation
Drawdown *depth* is set by leverage (Calmar ~invariant); overlays cut the *tail*
that causes liquidation. Best: a 10% negatively-correlated **crash-hedge** +
**drawdown-throttle** + hard **kill-switch**; the popular equity-curve filter
HURTS this book (snap-back drawdowns).
([`downside_overlays.py`](downside_overlays.py))

## The full search — what was tested and why it didn't beat ~1.5

Every archetype, validated OOS net of real costs (negative results kept honestly):

| archetype | result | file |
|---|---|---|
| daily trend / carry / order-flow | the core (~1.1–1.4) | `three_sleeve.py` |
| BAB / low-beta | +0.69, adds | `bab_sleeve.py` |
| max stack (8 candidates, 5 admitted) | 1.40 (OOS 1.38) | `max_stack.py` |
| crypto stat-arb (residual reversion) | taker-blocked | `stat_arb.py` |
| equity stat-arb (Avellaneda-Lee) | IS 1.24 / OOS 0.26 (decayed) | `stat_arb_equities.py` |
| equity factor ensemble | +0.42, uncorrelated but weak | `equity_ensemble.py` |
| cross-asset CTA trend (ETF/HIP-3) | ~0.4, weak on overlap | `cross_asset_book.py` |
| IBS mean-reversion (Quantitativo) | −0.08 (doesn't transfer to crypto) | `claimed_strategies.py` |
| rotational momentum (NDX) | +0.63 | `claimed_strategies.py` |
| overnight/time-of-day seasonality | real gross, taker-blocked | `overnight_seasonality.py` |
| microstructure / HFT (real HL L2) | IC 0.24 but maker-only | `microstructure_alpha.py` |
| MOSAIC ensemble / network overlay | parsimony beats kitchen-sink | `ensemble.py`, `network_overlay.py` |

Deep research syntheses (academic + GitHub + practitioner, all cited):
`prop_shop_landscape.md`, `hft_microstructure_landscape.md`,
`diversification_ceiling.md`, `SHARPE_INVESTIGATION.md`, `STRATEGY_RESEARCH.md`.

## Deployment
- [`live_signal.py`](live_signal.py) — emits today's target signed notional per coin.
- [`executor.py`](executor.py) — reconciles to live HL positions, **dry-run by
  default**, with gross-leverage / per-coin / funding / drawdown-kill gates.
- [`BOT_DEPLOYMENT.md`](BOT_DEPLOYMENT.md) — deployment checklist & risk controls.
- Forward data recorders: `record_l2.py`, `record_orderflow.py`, `record_hip3.py`.

## Reproduce
```bash
pip install -r ../requirements.txt
python grand_stack.py          # the deployable ~1.5 book
python hl_full_universe.py      # full-HL-universe max-Sharpe validation
python kelly_cagr.py            # leverage -> CAGR (Kelly)
python downside_overlays.py     # downside mitigation
```

*Research code, not investment advice. The point of this folder is the honesty: a
real, validated ~1.5-Sharpe book deployable on Hyperliquid, and a thorough,
multiply-confirmed demonstration of why a stable 3 is an infrastructure/breadth
achievement out of reach for a retail taker — not a number to fabricate.*
