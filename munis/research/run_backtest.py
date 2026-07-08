"""Staged backtest runner.

  python munis/research/run_backtest.py is    # explore grids in-sample only
  python munis/research/run_backtest.py oos   # run LOCKED configs out-of-sample

Discipline: the `oos` stage refuses to run unless locked_configs.json
exists, and it evaluates exactly those configs. IS = entries through
2022-12-31; OOS = entries 2023-01-01..2026-07-07; the survivorship-free
window (2025-07-08 onward, matching the universe scan window) is also
reported separately in the OOS stage.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
import panel as panel_mod  # noqa: E402
from strategies import FACTORIES, GRIDS, MIN_HOLDS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCKED = Path(__file__).resolve().parent / "locked_configs.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

IS_LO = pd.Timestamp("2012-01-01")
IS_HI = bt.IS_END
OOS_LO = pd.Timestamp("2023-01-01")
OOS_HI = bt.OOS_END
SURV_FREE_LO = pd.Timestamp("2025-07-08")


def load_bonds():
    print("loading panel ...", flush=True)
    pnl = panel_mod.load()
    uni = pd.read_csv(ROOT / "data" / "universe" / "universe.csv.gz")
    coupons = uni.set_index("six")["coupon"]
    coupons = coupons[~coupons.index.duplicated()]
    print("preparing per-bond frames ...", flush=True)
    return bt.prepare(pnl, coupons)


def eval_config(bonds, family: str, params: dict, lo, hi) -> dict:
    fn = FACTORIES[family](**params)
    fills = bt.run_signal(bonds, fn, min_hold=MIN_HOLDS[family],
                          date_lo=lo, date_hi=hi)
    ctl = bt.matched_random_control(bonds, fills,
                                    min_hold=MIN_HOLDS[family])
    row = bt.summarize(fills, f"{family} {params}", control=ctl)
    row["family"] = family
    row["params"] = json.dumps(params)
    return row


def stage_is(bonds) -> None:
    rows = []
    for family, grid in GRIDS.items():
        for params in grid:
            print(f"IS: {family} {params}", flush=True)
            rows.append(eval_config(bonds, family, params, IS_LO, IS_HI))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "is_grid.csv", index=False)
    print(df.to_string(index=False))


def stage_oos(bonds) -> None:
    if not LOCKED.exists():
        sys.exit("locked_configs.json missing — lock IS choices first")
    locked = json.loads(LOCKED.read_text())
    rows = []
    for entry in locked:
        family, params = entry["family"], entry["params"]
        for label, lo, hi in [
            ("OOS 2023-01..2026-07", OOS_LO, OOS_HI),
            ("survivorship-free 2025-07-08..", SURV_FREE_LO, OOS_HI),
        ]:
            print(f"OOS: {family} {params} [{label}]", flush=True)
            row = eval_config(bonds, family, params, lo, hi)
            row["window"] = label
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "oos_results.csv", index=False)
    print(df.to_string(index=False))


def main() -> None:
    stage = sys.argv[1] if len(sys.argv) > 1 else "is"
    if not panel_mod.PANEL_PATH.exists():
        panel_mod.build()
    bonds = load_bonds()
    print(f"{len(bonds)} bonds prepared", flush=True)
    if stage == "is":
        stage_is(bonds)
    elif stage == "oos":
        stage_oos(bonds)
    else:
        sys.exit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
