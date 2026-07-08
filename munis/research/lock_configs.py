"""Lock one config per strategy family from the in-sample grid, by a fixed
rule, so the OOS run cannot be cherry-picked.

Rule (pre-specified): among configs of a family with n >= MIN_FILLS
in-sample, pick the one with the highest excess return vs its matched
random control. If no config clears MIN_FILLS, the family is dropped.

Writes research/locked_configs.json. Run once, after `run_backtest.py is`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
IS_GRID = HERE / "results" / "is_grid.csv"
LOCKED = HERE / "locked_configs.json"
MIN_FILLS = 30


def main() -> None:
    df = pd.read_csv(IS_GRID)
    df = df[df["n"] >= MIN_FILLS]
    locked = []
    for family, g in df.groupby("family"):
        best = g.sort_values("excess_vs_control", ascending=False).iloc[0]
        locked.append({
            "family": family,
            "params": json.loads(best["params"]),
            "is_n": int(best["n"]),
            "is_excess_vs_control": float(best["excess_vs_control"]),
            "is_excess_p_boot": float(best.get("excess_p_boot", float("nan"))),
        })
    LOCKED.write_text(json.dumps(locked, indent=2))
    print(f"locked {len(locked)} configs -> {LOCKED}")
    print(json.dumps(locked, indent=2))


if __name__ == "__main__":
    main()
