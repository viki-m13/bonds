# 3-sleeve HL bot — deployment checklist & risk controls

Turning the validated `live_signal.py` targets into a safe Hyperliquid bot.
Read alongside `research/three_sleeve.md` (the numbers) and the HL mechanics in
`research/SHARPE_INVESTIGATION.md` / `research/prop_shop_landscape.md`.

## What it trades — and the honest expectation
The deployable book is **3-sleeve TREND + CARRY + ORDER-FLOW** (risk-weighted,
inverse-vol across sleeves), vol-targeted, on the ~57 HL-listed coins in our
daily panel. Validated net of real HL funding + 4.5 bps taker:

| book | Sharpe | IS (first 60%) | OOS (last 40%) | maxDD | Calmar |
|---|---|---|---|---|---|
| trend+carry (2-sleeve) | +0.86 | +0.91 | +0.79 | −12.4% | 0.91 |
| **+ order-flow (3-sleeve, risk-weighted)** | **+1.12** | **+1.15** | **+1.07** | −12.2% | 1.28 |

This is the **real ceiling for a taker-executed retail book: ~1.1–1.4 net.**
We proved on recorded HL L2 data that the maker/MM path (the only route to
Sharpe ~3) loses to adverse selection at our queue position
(`research/maker_path_to_sharpe3.md`, `prop_shop_landscape.md`). Deploy this for
a genuine, uncorrelated ~1.1 Sharpe sleeve — not for a number that isn't real.

The OHLC order-flow signal is a *proxy*; running `record_orderflow.py` forward
for a few weeks and swapping in real signed taker volume is the one clean upgrade
path to the higher end of the ~1.1–1.4 range.

## Architecture (implemented in `executor.py`, dry-run by default)
1. **Signal (`live_signal.py`).** Once per UTC day after the daily close, compute
   target signed notional per coin from `three_sleeve.book_weights` (TREND +
   CARRY + ORDER-FLOW, inverse-vol risk-weighted, vol-targeted). Uses only closed
   daily bars; the vol-target scale is `.shift(1)`-lagged (causal).
2. **Reconciler (`executor.reconcile`).** Pull `clearinghouseState`; diff vs
   targets to per-coin deltas. Pure/inspectable — no network, no signing — so it
   unit-tests cleanly.
3. **Executor (`executor.main`).** ALO/limit one `SLIP_LIMIT_BPS` inside mid
   (maker) for the deltas; **reduce-only** on any shrink/flip; round size to the
   asset's `szDecimals`; skip deltas under **$10**. DRY-RUN prints orders; `--live`
   signs via the hyperliquid SDK (`pip install hyperliquid-python-sdk eth-account`)
   only when `HL_ACCOUNT` + `HL_SECRET_KEY` are set.
4. **Monitor.** Track fills, funding accrual (hourly), equity, gross/net
   leverage, drawdown; pass `--high-water` so the kill-switch can trip.

Run order: `python executor.py` (signal-only) → `HL_ACCOUNT=0x.. python
executor.py` (dry-run vs live state) → `--live` only after the gate below.

## Hard risk limits (the bot must enforce, not just the signal)
- **Gross leverage cap** (`MAX_GROSS_LEVERAGE`, default 3x) — refuse to exceed,
  even if the vol-target scale asks for more. The book runs ~1.0–1.4x gross at a
  20% vol target, so this is a wide safety rail, not a binding constraint.
- **Per-coin notional cap** (`PER_COIN_CAP_FRAC`, 15% equity) AND of live L2 book
  depth (cap order size so expected impact < a few bps; size from `l2Book`).
- **Cross-margin, majors only at higher size.** Use cross margin; keep mid/long-
  tail coins small (their HL max leverage is 3–10x and books are thin). The
  signal is market-neutral-ish (longs ≈ shorts), which limits beta but NOT the
  correlated-crash risk — size for the everything-dumps-together day.
- **Drawdown kill-switch.** Flatten and pause if account drawdown breaches a set
  threshold (default 25%, ~2x the backtest maxDD).
- **Funding guard.** If a coin's funding is extreme (e.g. > 5 bps/hr against the
  position), reduce or skip — funding, not fees, is a dominant holding cost, and
  the CARRY sleeve already tilts toward collecting it.
- **Staleness/halt guard.** Check `exchangeStatus`; if the L1 is degraded or the
  daily candle is missing, do nothing (hold), never trade on stale data.

## Operational gotchas (from the HL mechanics review)
- Funding accrues **hourly on oracle-price notional**; reconcile it — it is a
  main cost of holding, not the 4.5 bps taker fee.
- TP/SL triggers fire on **mark price** (blends external CEX feeds), so avoid
  resting stops you don't want fired by an off-book wick; manage exits in code.
- **Address-based rate limit** (≈1 request per 1 USDC traded, 10k initial
  buffer) — batch and throttle; a low-volume bot can exhaust it by spamming.
- Large positions silently lower max leverage / raise maintenance margin via
  **margin tiers** — size-check before submitting.
- Re-pull `meta`/`userFees` at startup for live max-leverage and your actual
  fee tier; don't hardcode.
- The signal universe is the daily panel; a production bot should pull daily
  candles from the HL `candleSnapshot` (interval `1d`) API for the same coins so
  the live data source matches the backtest math.

## Pre-deploy gate
- [ ] Paper-trade the reconciler+executor against testnet for ≥2 weeks; confirm
      realized slippage matches the modeled ~few bps on majors.
- [ ] Confirm realized funding tracks the backtest's funding series.
- [ ] Verify the kill-switch and per-coin caps trip correctly in a dry run.
- [ ] (Optional but recommended) run `record_orderflow.py` forward in parallel so
      the order-flow sleeve can be upgraded from the OHLC proxy to real signed
      taker volume before scaling capital.
- [ ] Start at a small fraction of intended capital and a **10% vol target**
      (lowest leverage) before scaling.

*Not investment advice. The honest expected edge is ~1.1–1.4 net Sharpe, not 3.
Deploy real capital only after the review the user asked for and a successful
paper-trading period.*
