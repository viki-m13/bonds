"""Print current buy recommendations from the validated rule ensemble.

Usage:
    python recommend.py [--lookback N] [--no-refresh]

Shows stocks that triggered any production rule within the last N trading
days (default 5), excluding stocks that already signaled in the prior 21
trading days (matching how the validation counted signals).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from config import HORIZON, RULES_FILE
from data import load_prices
from features import compute_panel, describe_rule, rule_mask
from rules import dedup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookback", type=int, default=5,
                    help="show signals from the last N trading days")
    ap.add_argument("--no-refresh", action="store_true",
                    help="use cached prices instead of downloading")
    args = ap.parse_args()

    if not RULES_FILE.exists():
        raise SystemExit("selected_rules.json not found - run validate.py first")
    spec = json.loads(RULES_FILE.read_text())
    rules: list[list[str]] = spec["rules"]

    prices = load_prices(refresh=not args.no_refresh)
    panel = compute_panel(prices)
    idx = panel.index

    union = np.zeros(panel.close.shape, dtype=bool)
    per_rule = {}
    for rule in rules:
        m = rule_mask(panel, rule)
        per_rule[tuple(rule)] = m
        union |= m
    deduped = dedup(union)

    start = len(idx) - args.lookback
    rows, cols = np.nonzero(deduped[start:, :])
    rows += start

    oos = spec["oos_pooled"]
    print(f"\nwinrate30 recommendations  (data through {idx[-1].date()})")
    print(f"Validated out-of-sample: {oos['hits']}/{oos['n']} signals positive "
          f"after {HORIZON} trading days = {100 * oos['rate']:.1f}% "
          f"(95% lower bound {100 * oos['wilson_lb']:.1f}%)")
    print("-" * 78)

    if len(rows) == 0:
        print(f"No new signals in the last {args.lookback} trading days.")
        print("This is normal - the rules only fire when conditions are met. "
              "Re-run regularly (e.g. daily after the close).")
    else:
        print(f"{'Ticker':<8}{'Signal date':<14}{'Close':>10}{'RSI':>7}"
              f"{'Off 52w high':>14}  Rule")
        for r, c in sorted(zip(rows, cols), key=lambda t: (-t[0], t[1])):
            matched = next(
                " + ".join(rule) for rule in rules
                if per_rule[tuple(rule)][r, c]
            )
            print(f"{panel.tickers[c]:<8}{str(idx[r].date()):<14}"
                  f"{panel.close[r, c]:>10.2f}{panel.extras['rsi'][r, c]:>7.1f}"
                  f"{100 * panel.extras['dd'][r, c]:>13.1f}%  {matched}")
        print()
        print("Hold horizon: ~30 calendar days (21 trading days) per signal.")
    print("-" * 78)
    print("Rules in force:")
    for rule in rules:
        print(f"  - {describe_rule(rule)}")
    print("\nNot investment advice. Past hit rates do not guarantee future "
          "results;\nsignals are correlated in time, so a bad market month "
          "hits many at once.")


if __name__ == "__main__":
    main()
