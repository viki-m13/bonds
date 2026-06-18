"""3-SLEEVE live signal generator — emits target perp positions for a bot.

Computes today's TREND + CARRY + ORDER-FLOW target weights (the validated,
deployable book; see research/three_sleeve.md) on the Hyperliquid-listed
universe from the latest daily closes, risk-weights the sleeves by inverse
realized PnL vol, scales to the configured account vol target, and prints
target *signed notional* per coin (long > 0, short < 0). A bot turns these into
reduce-only/limit orders; this module is the brain, not the executor.

Causality / live use: run after each UTC daily close; weights use only closed
daily bars (the book_weights scale is .shift(1)-lagged). Re-fetch the HL `meta`
at runtime for per-asset max leverage and round sizes to asset tick/lot before
sending. Funding accrues hourly on open notional — the bot reconciles it, but it
does not change the target weights.

This reads the repo's daily panel (data/crypto) for the backtest-identical
signal. A production bot would instead pull daily candles from the HL/info API
(`candleSnapshot`, interval="1d") for the same coins, which yields the same
math; keep the lookbacks and the vol target in sync with three_sleeve.py.
"""
import json
import os

import numpy as np
import pandas as pd

import validate_hl as v
import three_sleeve as ts

# Account configuration (edit per deployment) -------------------------------
ACCOUNT_EQUITY_USD = 10_000.0
VOL_TARGET = 0.20            # annual; ~1x gross on HL, far from any liq.
TAKER_BPS = v.TAKER_BPS
MAX_GROSS_LEVERAGE = 3.0     # hard cap the bot will never exceed
MIN_ORDER_USD = 12.0        # HL min notional is $10; pad it


def current_targets():
    """Today's 3-sleeve target signed notional per coin. Risk weights are
    estimated on all history up to today (causal); the vol-target scale is the
    book_weights scale, already lagged. Identical math to the backtest's
    risk-weighted 3-sleeve book."""
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    W_target, scale, rw = ts.book_weights(C, V, H, L, F, vol_target=VOL_TARGET)
    last = C.index[-1]
    sc = float(scale.loc[last])
    if not np.isfinite(sc):
        sc = float(scale.dropna().iloc[-1]) if scale.notna().any() else 1.0
    raw_w = W_target.loc[last]                            # combined gross weights
    # cap account gross leverage
    sc = min(sc, MAX_GROSS_LEVERAGE / max(raw_w.abs().sum(), 1e-9))
    target_w = (raw_w * sc).dropna()
    notional = (target_w * ACCOUNT_EQUITY_USD)
    notional = notional[notional.abs() >= MIN_ORDER_USD].sort_values()
    return last, notional, float(target_w.abs().sum()), rw


def main():
    asof, notional, gross, rw = current_targets()
    print(f"3-SLEEVE targets as of {asof.date()} | equity "
          f"${ACCOUNT_EQUITY_USD:,.0f} | vol target {VOL_TARGET:.0%} | "
          f"gross leverage {gross:.2f}x | {len(notional)} positions")
    print("risk weights: " + ", ".join(f"{k} {rw[k]:.0%}" for k in rw.index))
    print(f"{'coin':6s} {'side':5s} {'notional$':>12s} {'weight':>8s}")
    for c, n in notional.items():
        print(f"{c:6s} {'LONG' if n > 0 else 'SHORT':5s} {n:>12,.0f} "
              f"{n / ACCOUNT_EQUITY_USD:>+8.2%}")
    out = {"asof": str(asof.date()), "book": "3-sleeve TREND+CARRY+ORDERFLOW",
           "equity_usd": ACCOUNT_EQUITY_USD, "vol_target": VOL_TARGET,
           "gross_leverage": gross,
           "risk_weights": {k: round(float(rw[k]), 3) for k in rw.index},
           "targets_usd": {c: round(float(n), 2) for c, n in notional.items()}}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                     "live_targets.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("[written]", p)


if __name__ == "__main__":
    main()
