# PULSE-HL bot — deployment checklist & risk controls

Turning the validated `live_signal.py` targets into a safe Hyperliquid bot.
Read alongside `research/hl_validation.md` (the numbers) and the HL mechanics in
`research/SHARPE_INVESTIGATION.md`.

## Architecture (implemented in `executor.py`, dry-run by default)
1. **Signal (`live_signal.py`).** Once per UTC day after the daily close, compute
   target signed notional per coin (PULSE daily; `LONG_ONLY` toggle). Uses only
   closed daily bars.
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
  even if the vol-target scale asks for more. PULSE runs ~0.3–0.7x gross at
  10–20% vol, so this is a wide safety rail, not a binding constraint.
- **Per-coin notional cap** as a fraction of equity AND of live L2 book depth
  (cap order size so expected impact < a few bps; size from `l2Book`).
- **Cross-margin, majors only at higher size.** Use cross margin; keep mid/long-
  tail coins small (their HL max leverage is 3–10x and books are thin).
- **Drawdown kill-switch.** Flatten and pause if account drawdown breaches a set
  threshold (e.g. 2x the backtest maxDD, ~25%).
- **Funding guard.** If a coin's funding is extreme (e.g. > X bps/hr against the
  position), reduce or skip — funding, not fees, is the dominant holding cost.
- **Staleness/halt guard.** Check `exchangeStatus`; if the L1 is degraded or the
  daily candle is missing, do nothing (hold), never trade on stale data.

## Operational gotchas (from the HL mechanics review)
- Funding accrues **hourly on oracle-price notional**; reconcile it — it is the
  main cost of holding, not the 4.5 bps taker fee.
- TP/SL triggers fire on **mark price** (blends external CEX feeds), so avoid
  resting stops you don't want fired by an off-book wick; manage exits in code.
- **Address-based rate limit** (≈1 request per 1 USDC traded, 10k initial
  buffer) — batch and throttle; a low-volume bot can exhaust it by spamming.
- Large positions silently lower max leverage / raise maintenance margin via
  **margin tiers** — size-check before submitting.
- Re-pull `meta`/`userFees` at startup for live max-leverage and your actual
  fee tier; don't hardcode.

## Pre-deploy gate
- [ ] Paper-trade the reconciler+executor against testnet for ≥2 weeks; confirm
      realized slippage matches the modeled ~few bps on majors.
- [ ] Confirm realized funding tracks the backtest's funding series.
- [ ] Verify the kill-switch and per-coin caps trip correctly in a dry run.
- [ ] Start at a small fraction of intended capital and a **10% vol target**
      (lowest leverage) before scaling.

*Not investment advice. Deploy real capital only after the review the user
asked for and a successful paper-trading period.*
