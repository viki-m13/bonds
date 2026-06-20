# Consolidation review: the `vol` repo (viki-m13/vol) HL strategy

Reviewed read-only (nothing in the vol repo touched). The vol repo runs a mature,
**live-on-Hyperliquid** strategy with serious execution-cost modeling and real
reconciliation. This documents what was learned and what was copied/consolidated.

## What the vol strategy is
An **adaptive volatility-channel breakout** (`crypto_strategy/strategy.py`): per-asset,
intraday (5-min bars), VWAP±band·sigma breakout, 10-hour signal eval, vol-targeted
sizing, plus a **drawdown-aware exposure scaler** (cut to 25% between 2% and 5% DD).
Production params: band_mult 1.5, vwap 4d, sigma 14h, rvol 1d, target 2%/day, max-lev 2×.
Universe: **TURBO-5** rotating 5-pair (BTC+ETH pinned + top-3 by 24h volume).

## Key cross-confirmations (their LIVE experience validates our assumptions)
1. **Real HL taker fee = 4.5 bps** — confirmed by their *live reconciliation* (CHANGELOG
   PR #66: "reconcile showed real HL taker tier is 0.045%"). Exactly our assumption.
2. **The edge is MAKER-dependent; taker turnover kills it.** Their own
   `LIVE_DEPLOYMENT_MODELING.md`: the headline Sharpe 4–6 needs passive-maker
   execution on the liquid-5; as a taker the strategy is "net flat-to-negative" and
   its drawdown blows from −4% to −28%. **This is exactly our independent real-L2
   finding** (microstructure alpha is sub-cost for a taker; maker-only).
3. **Rotating top-volume universe** = our PIT top-N-by-liquidity rule (mutual
   validation): both repos converged on "trade the most liquid names, re-ranked."
4. **Funding is the dominant residual cost after slippage** — matches our finding
   that funding > fees for a held perp book.
5. **Slippage, not fees, is the destroyer** at high turnover (their break-even ≈ 9–10
   bps round-trip; the every-bar vol-target rebalances ~159–377×/yr). Reinforces our
   preference for LOW-turnover (daily/weekly) sleeves.

## What we reproduced on our data (`vol_channel.py`)
Ran their vol-channel breakout on our 27-coin hourly HL data (2024–2026), net of real
HL costs (params mapped 5-min→hourly):

| execution | Sharpe | vs grand-stack corr |
|---|---|---|
| gross (0 bps) | +0.98 | — |
| maker (1.5 bps) | +0.79 | +0.21 |
| **taker (4.5 bps)** | **+0.42** | +0.21 |

- It is a **genuinely different alpha** (intraday per-asset TS breakout) and **~uncorrelated**
  to our daily cross-sectional grand stack (corr **+0.21**) — so it *would* diversify.
- **But net-taker it is too weak (+0.42) to lift the book**: 60/40 blend = 1.26 vs 1.51
  grand-alone. Only its maker version (+0.79) would help, and our real-L2 study showed
  passive maker is adverse-selected at retail latency. **Same wall, reached independently
  from both repos.**
- Our hourly/2024-26 reproduction is necessarily weaker than their 5-min/8-year/
  liquid-5-maker headline; the point is the *taker* read on *our* data, which is honest.

## What's worth adopting (and what isn't)
- **ADOPT — the live-reconciliation discipline.** Their `verify.py` (backtest-formula
  parity + live slippage + equity drag) and `reconcile_live.py` (exchange ground truth:
  positions/fills/fees/funding via HL `info`) are exactly the right pre-deploy gate for
  our executor. Template to mirror, not Sharpe alpha.
- **ADOPT — confirmed cost model**: 4.5 bps taker / 1.5 bps maker, funding charged
  per-bar on |exposure|, turnover-based commission. We already use this; now externally
  validated by their live fills.
- **NOTE — the DD-scaling sizer**: their claimed "Sharpe 1.5→3–5" from DD-scaling is
  the one claim to treat with caution (it's a Calmar/vol-control device; we tested
  equity-curve/DD overlays on our book — modest help at best, equity-curve filter
  hurts). Not a free Sharpe multiplier.
- **DON'T consolidate the vol-channel as a taker sleeve** — it doesn't clear costs for
  us; it's a maker strategy. Kept `vol_channel.py` as a documented candidate for if/when
  a maker execution path is genuinely available.

## Bottom line
The vol repo is excellent and live, and reviewing it was high-value — but it does **not
raise our Sharpe**, because its high numbers live in the same maker/intraday regime we
already proved is gated for a retail taker. The most valuable takeaways are
*confirmations* (HL taker 4.5 bps, maker-dependence, top-volume universe, funding-as-
dominant-cost) and the *live-reconciliation tooling* to harden our own deployment.
