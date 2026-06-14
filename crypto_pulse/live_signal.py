"""PULSE-HL live signal generator — emits target perp positions for a bot.

Computes today's PULSE target weights on the Hyperliquid-listed universe from
the latest daily closes, scales to the configured account vol target, and prints
target *signed notional* per coin (long > 0, short < 0). A bot turns these into
reduce-only/limit orders; this module is the brain, not the executor.

Causality / live use: run after each UTC daily close; weights use only closed
daily bars. Re-fetch the HL `meta` at runtime for per-asset max leverage and
round sizes to asset tick/lot before sending (see research/SHARPE_INVESTIGATION
.md and the HL mechanics notes). Funding accrues hourly on open notional — the
bot must reconcile it, but it does not change the target weights.

This reads the repo's daily panel (data/crypto) for the backtest-identical
signal. A production bot would instead pull daily candles from the HL/info API
(`candleSnapshot`, interval="1d") for the same coins, which yields the same
math; keep the lookbacks and the vol target in sync with validate_hl.py.
"""
import json
import os

import numpy as np
import pandas as pd

import validate_hl as v

# Account configuration (edit per deployment) -------------------------------
ACCOUNT_EQUITY_USD = 10_000.0
VOL_TARGET = 0.20            # annual; ~0.5x gross on HL, far from any liq.
TAKER_BPS = v.TAKER_BPS
MAX_GROSS_LEVERAGE = 3.0     # hard cap the bot will never exceed
MIN_ORDER_USD = 12.0        # HL min notional is $10; pad it


def current_targets():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    net, comp, w = v.run(C, V, H, L, F, vol_target=VOL_TARGET, funding=True)
    # the per-day scale is embedded in comp via gross_lev; recover today's scale
    last = C.index[-1]
    raw_w = w.loc[last]                                   # gross-1 weights
    # today's vol-target scale = today's gross_lev / sum|raw_w|
    gl_today = comp["gross_lev"].loc[last]
    scale = gl_today / max(raw_w.abs().sum(), 1e-9)
    scale = min(scale, MAX_GROSS_LEVERAGE / max(raw_w.abs().sum(), 1e-9))
    target_w = (raw_w * scale).dropna()
    notional = (target_w * ACCOUNT_EQUITY_USD)
    notional = notional[notional.abs() >= MIN_ORDER_USD].sort_values()
    return last, notional, float(target_w.abs().sum())


def main():
    asof, notional, gross = current_targets()
    print(f"PULSE-HL targets as of {asof.date()} | equity "
          f"${ACCOUNT_EQUITY_USD:,.0f} | vol target {VOL_TARGET:.0%} | "
          f"gross leverage {gross:.2f}x | {len(notional)} positions")
    print(f"{'coin':6s} {'side':5s} {'notional$':>12s} {'weight':>8s}")
    for c, n in notional.items():
        print(f"{c:6s} {'LONG' if n > 0 else 'SHORT':5s} {n:>12,.0f} "
              f"{n / ACCOUNT_EQUITY_USD:>+8.2%}")
    out = {"asof": str(asof.date()), "equity_usd": ACCOUNT_EQUITY_USD,
           "vol_target": VOL_TARGET, "gross_leverage": gross,
           "targets_usd": {c: round(float(n), 2) for c, n in notional.items()}}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                     "live_targets.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("[written]", p)


if __name__ == "__main__":
    main()
