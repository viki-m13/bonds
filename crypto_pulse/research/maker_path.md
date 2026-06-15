# The maker path, tested on real Hyperliquid L2 data

The research kept saying the only honest route to Sharpe ~3 is **passive
market-making with maker rebates + an adverse-selection filter** (HLP ~2.9; "up
to 3.5 for zero-fee market makers"). So I built the real tool and tested it on
real data — not a daily-bar proxy.

## Tools (reusable)
- `record_l2.py` — records the HL WebSocket **l2Book (20 levels) + trades** to
  JSONL with receive timestamps. Forward data collection (L2 can't be
  backfilled).
- `maker_sim.py` — a **queue-aware** market-making simulator: posts small orders
  at best bid/ask, models **FIFO queue position** (a quote sits behind the
  resting size and fills only after aggressive trades consume the queue ahead of
  it — the part naive backtests skip), re-prices to back-of-queue on a new best
  (pessimistic, realistic), caps inventory, and attributes each fill to
  **spread captured + maker fee/rebate − adverse selection** (mid move against
  you over 10 s). An optional **order-book-imbalance filter** pulls the side the
  book leans against (the adverse-selection avoidance the literature requires).

## Result (24.7 min live sample, 8 liquid coins, 28,435 messages)

Per-fill economics, bps of notional:

| policy | fee tier | spread | fee | adverse | **NET/fill** |
|---|---|---|---|---|---|
| naive | rebate −0.3 | +0.22 | +0.30 | −2.56 | **−2.04** |
| imbalance-filtered (imb>0.15) | base 1.5 | +0.36 | −1.50 | −1.59 | −2.73 |
| imbalance-filtered (imb>0.15) | zero fee | +0.36 | 0.00 | −1.59 | −1.23 |
| imbalance-filtered (imb>0.15) | **rebate −0.3** | +0.36 | +0.30 | −1.59 | **−0.93** |

- **Passive MM loses at every fee tier, including the top −0.3 bps rebate.** On
  HL's liquid coins the captured spread (~0.2–0.4 bps) is far smaller than the
  adverse selection (~1.6–2.6 bps): you get filled precisely when the market is
  about to move against you.
- **The imbalance filter genuinely helps** — it cuts adverse selection ~40%
  (−2.56 → −1.59) and nearly doubles spread capture (+0.22 → +0.36) by avoiding
  the worst fills — but it is **not enough to flip the edge positive**.
- **No reliable wider-spread niche** in this sample: per-coin nets are mostly
  negative; the two positive coins (LTC, ETH) had 12 and 2 fills — noise.

## Honest verdict
Measured on real HL L2 with proper queue modeling, **the retail maker path does
not produce a positive edge** — passive quoting on liquid coins is dominated by
adverse selection even at the rebate tier retail can't reach anyway. A profitable
MM needs what retail lacks: **lower latency** (better queue position → fewer
picked-off fills than my pessimistic back-of-queue model), **deeper alpha** than
top-of-book imbalance, and the **HLP's liquidation-backstop + funding flows**.
This is the same conclusion the research reached, now confirmed empirically: the
Sharpe-3 maker edge is a pro-infrastructure game, structurally walled off from a
retail taker/maker bot.

## Caveats
- 25-minute sample, one time-of-day/regime → the *per-fill edge* is the robust
  read (thousands of fills); an annualized Sharpe from this is not trustworthy.
- The requote model is deliberately pessimistic (back-of-queue on every reprice);
  a low-latency MM gets better queue position — but that is exactly the pro-infra
  advantage retail doesn't have.
- Adverse-selection horizon = 10 s; a faster-unwinding MM would realize less of
  it, but also captures less spread per round-trip.

Reproduce: `python record_l2.py --secs 1500 --coins BTC,ETH,SOL,DOGE,AVAX,XRP,LINK,LTC`
then `python maker_sim.py`. (Recordings are git-ignored; collect your own — and
for a real validation, run it for days, not minutes.)
