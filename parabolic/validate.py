"""Run IGNITION (and the falsifiers / a momentum baseline) through the repo's
standard DCA grid harness (dca/protocol.evaluate_signal) for apples-to-apples
comparability with SUMMIT, ROTATOR and the momentum baselines.

NOTE on objective mismatch (stated up front, honestly): the DCA grid rewards a
*never-sell, beat-QQQ/SPY* compounding objective. IGNITION is a high-variance
right-tail (parabolic-capture) signal — it is NOT engineered to beat QQQ-DCA and
is not expected to. This script exists to (a) confirm IGNITION clears the random
control on the same grid, and (b) quantify the gap vs the compounding strategies,
not to claim a DCA win.

Run from parabolic/:  python validate.py        (writes research/validate.md,
scorecards land in dca/research/scorecards/PARABOLIC_*.json)
"""
import os
import sys

import numpy as np
import pandas as pd

DCA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dca")
sys.path.insert(0, DCA)
import data as dca_data  # noqa: E402
import protocol  # noqa: E402
import features as feat  # noqa: E402
import strategy as strat  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                   "validate.md")


def momentum_baseline(P, F):
    """9-1 momentum, members only — the repo's best pure-momentum control."""
    member = P["member"] & P["close"].notna()
    mom = (P["close"].shift(21) / P["close"].shift(189) - 1)
    return mom.where(member)


def main():
    P = dca_data.build_panel()
    F = feat.build_features(P)

    signals = {
        "PARABOLIC_ignition": strat.ignition_score(P, F),
        "PARABOLIC_ignition_beta": strat.VARIANTS["ignition_beta"](P, F),
        "PARABOLIC_practitioner": strat.practitioner_breakout_score(P, F),
        "PARABOLIC_pure_energy": strat.pure_energy_score(P, F),
        "PARABOLIC_mom91": momentum_baseline(P, F),
    }

    lines = ["# IGNITION on the standard DCA grid (vs QQQ / SPY / random)\n"]
    lines.append("Harness: dca/protocol.evaluate_signal — biweekly never-sell "
                 "DCA, 244-window grid + 8 regimes, 5 bps/trade. `win_qqq`/"
                 "`win_spy` = share of windows beating QQQ-/SPY-DCA; `med_vs_qqq`"
                 " = median excess final multiple; `full_mult` = whole-period "
                 "money multiple. The DCA objective is compounding, NOT parabolic "
                 "capture — read alongside backtest.md.\n")

    # random control on the same grid (k=10) for the survivorship floor
    rc = protocol.random_control(k=10, n_draws=20)
    rc_mult = rc.groupby("window").apply(
        lambda d: (d["mult"]).median(), include_groups=False)
    grid_only = [w for w in rc_mult.index if "_" in w and w.rsplit("_", 1)[-1]
                 in ("3", "5", "10", "end")]
    bench = protocol.get_shared()
    qqq = protocol._bench_grid(10, 0, 1000.0, 5.0)[1]["qqq"]
    rc_win_qqq = np.mean([rc_mult[w] > qqq[w] for w in grid_only])

    hdr = ("| signal | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | "
           "full_mult |")
    lines.append(hdr)
    lines.append("|" + "---|" * 6)
    for name, sc in signals.items():
        card = protocol.evaluate_signal(sc, name, k=10, quiet=True)
        lines.append(
            f"| {name} | {card['win_qqq']:.0%} | {card['win_spy']:.0%} | "
            f"{card['med_vs_qqq']:+.1%} | {card['worst_vs_qqq']:+.1%} | "
            f"{card['full_mult']:.1f}x |")
    lines.append(f"| random-pick (k=10, control) | {rc_win_qqq:.0%} | — | — | "
                 "— | — |")
    lines.append("")
    lines.append("Reading: IGNITION should clear the random-pick floor on "
                 "`win_qqq` and beat SPY in most windows, while trailing QQQ "
                 "(the high-variance tail objective costs compounding "
                 "consistency). The practitioner-breakout and pure-energy "
                 "variants are included for contrast.\n")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[written] {OUT}")


if __name__ == "__main__":
    main()
