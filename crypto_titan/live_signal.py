"""CRYPTO-TITAN live signal generator.

Outputs the WEEKLY rebalance instructions: target weights for each coin
based on the strategy's current state, mapped to manually-executable
positions on OKX (leveraged tokens) or Hyperliquid (perp positions).

Usage:
    python live_signal.py                  # show this week's target
    python live_signal.py --capital 10000  # show absolute USD sizes
    python live_signal.py --venue hyperliquid  # Hyperliquid perp instructions
    python live_signal.py --venue okx          # OKX leveraged-token mapping (default)

Run on Wednesday close (or Tuesday after-close) each week. The output
shows what to BUY/SELL/HOLD relative to last week's positions.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from util import (load_prices, load_macro, ALL_COINS, OUT)
import sleeves as SV
import strategy as ST


# OKX leveraged-token mapping. OKX has 3× leveraged tokens for major coins
# (e.g. BTC3L for long, BTC3S for short). When the strategy's leverage is
# below 1.5×, use spot. Between 1.5×-3× use the leveraged token.
OKX_LEVERAGED = {
    "BTC": ("BTC", "BTC3L", "BTC3S"),  # spot, 3× long, 3× short
    "ETH": ("ETH", "ETH3L", "ETH3S"),
    "SOL": ("SOL", "SOL3L", "SOL3S"),
    "BNB": ("BNB", "BNB3L", "BNB3S"),
    "DOGE": ("DOGE", "DOGE3L", "DOGE3S"),
    "XRP": ("XRP", "XRP3L", "XRP3S"),
    "LTC": ("LTC", "LTC3L", "LTC3S"),
}


def load_prior_weights() -> pd.Series | None:
    fp = OUT / "live_prior_weights.json"
    if not fp.exists():
        return None
    d = json.loads(fp.read_text())
    return pd.Series(d.get("weights", {}))


def save_prior_weights(target: pd.Series, asof: pd.Timestamp) -> None:
    fp = OUT / "live_prior_weights.json"
    fp.write_text(json.dumps({
        "asof": str(asof.date()),
        "weights": target.to_dict(),
    }, indent=2))


def compute_target_weights() -> tuple[pd.Series, pd.Timestamp, float]:
    """Re-run the strategy and return THIS WEEK's target weights.

    Returns:
        target_weights: Series indexed by coin, values are notional weights
                        (sum of abs values can exceed 1.0 with leverage)
        asof:           timestamp of the latest signal date
        leverage:       implied gross leverage (sum of absolute weights)
    """
    cp = load_prices()
    macro = load_macro(cp.index)
    sleeves = SV.build_all(cp, macro)
    _net, W_eff = ST.build_portfolio(cp, sleeves)
    # Latest non-zero row
    last_row = W_eff.iloc[-1]
    target = last_row[last_row.abs() > 1e-4].sort_values(key=lambda x: -x.abs())
    asof = W_eff.index[-1]
    leverage = float(target.abs().sum())
    return target, asof, leverage


def format_okx(target: pd.Series, capital: float | None) -> str:
    """Render target as OKX-friendly trade instructions.

    Each weight maps to either spot (low conviction / 1×) or a 3×
    leveraged token (high conviction / >1.5×).
    """
    lines = []
    lines.append(f"{'COIN':<8} {'WEIGHT':>8} {'INSTR':<24} {'NOTIONAL':>14}")
    lines.append("-" * 60)
    for coin, w in target.items():
        if coin not in OKX_LEVERAGED:
            # No 3× token available — use spot only
            ticker = coin
            instr = f"BUY {w*100:+.1f}% SPOT"
        else:
            spot, lev_long, lev_short = OKX_LEVERAGED[coin]
            if abs(w) < 1.5:
                ticker = spot
                side = "BUY" if w > 0 else "SELL"
                instr = f"{side} {abs(w)*100:.1f}% {spot} SPOT"
            else:
                # Use 3× leveraged token; effective weight = w / 3
                if w > 0:
                    ticker = lev_long
                    instr = f"BUY {abs(w)/3*100:.1f}% {lev_long}"
                else:
                    ticker = lev_short
                    instr = f"BUY {abs(w)/3*100:.1f}% {lev_short}"
        notional_str = f"${capital * abs(w):,.0f}" if capital else ""
        lines.append(f"{coin:<8} {w:+.3f}   {instr:<24} {notional_str:>14}")
    return "\n".join(lines)


def format_hyperliquid(target: pd.Series, capital: float | None) -> str:
    """Render target as Hyperliquid perp instructions.

    On Hyperliquid every coin position is just a perp with set leverage.
    Each row: coin, side, position size at base 1× (multiply by leverage).
    """
    lines = []
    lines.append(f"{'COIN':<8} {'WEIGHT':>8} {'SIDE':<5} {'SIZE %':>8} {'NOTIONAL':>14}")
    lines.append("-" * 56)
    for coin, w in target.items():
        side = "LONG" if w > 0 else "SHORT"
        notional = f"${capital * abs(w):,.0f}" if capital else ""
        lines.append(f"{coin:<8} {w:+.3f}   {side:<5} {abs(w)*100:>6.1f}%  {notional:>14}")
    return "\n".join(lines)


def diff_against_prior(target: pd.Series, prior: pd.Series | None) -> str:
    if prior is None:
        return "(no prior week recorded — first run, all positions are NEW)"
    all_coins = sorted(set(target.index) | set(prior.index))
    lines = []
    lines.append(f"{'COIN':<8} {'PRIOR':>8} {'TARGET':>8} {'DELTA':>8} {'ACTION':<10}")
    lines.append("-" * 50)
    any_change = False
    for c in all_coins:
        prev = prior.get(c, 0.0)
        new = target.get(c, 0.0)
        d = new - prev
        if abs(d) < 0.005:
            continue  # < 0.5% change — skip
        any_change = True
        if abs(prev) < 0.005:
            action = "OPEN"
        elif abs(new) < 0.005:
            action = "CLOSE"
        elif np.sign(prev) != np.sign(new):
            action = "FLIP"
        elif d > 0:
            action = "ADD"
        else:
            action = "TRIM"
        lines.append(f"{c:<8} {prev:>+7.3f} {new:>+7.3f} {d:>+7.3f}  {action}")
    if not any_change:
        lines.append("  (no positions changed > 0.5% — HOLD all current positions)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capital", type=float, default=None,
                     help="account capital in USD (for notional sizing)")
    ap.add_argument("--venue", choices=["okx", "hyperliquid"], default="okx")
    ap.add_argument("--no-save", action="store_true",
                     help="don't persist this week's weights as 'prior'")
    args = ap.parse_args()

    print("=" * 60)
    print("CRYPTO-TITAN — Weekly Rebalance Signal")
    print("=" * 60)

    target, asof, leverage = compute_target_weights()
    print(f"\nSignal as-of: {asof.date()}")
    print(f"Total gross exposure (leverage): {leverage:.2f}×")
    print(f"Net long/short: long {target[target>0].sum():.2f} / "
          f"short {-target[target<0].sum():.2f}")
    print(f"Number of positions: {len(target)}")

    print(f"\n--- Target positions ({args.venue.upper()}) ---")
    if args.venue == "okx":
        print(format_okx(target, args.capital))
    else:
        print(format_hyperliquid(target, args.capital))

    print(f"\n--- Trades vs last week ---")
    prior = load_prior_weights()
    print(diff_against_prior(target, prior))

    if not args.no_save:
        save_prior_weights(target, asof)
        print(f"\n[Saved this week's weights as 'prior' for next run.]")

    print("\n" + "=" * 60)
    print("EXECUTION NOTES:")
    print(f"  * Run again on the next Wednesday (weekly rebalance).")
    print(f"  * Use {leverage:.2f}× total leverage; the strategy already accounts for it.")
    print(f"  * Funding cost on perps will reduce returns ~{leverage-1.0:.1f} × 4 bps/day")
    print(f"    when long-heavy (≈ {(leverage-1.0)*15:.0f}% annual on lever portion).")
    print(f"  * If a position size is < 1% of capital, skip it — TC eats it.")


if __name__ == "__main__":
    main()
