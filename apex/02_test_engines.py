"""APEX — Phase 2: run each engine individually and report metrics."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import util
import engines

OUT = Path("/home/user/bonds/data/apex")


def run_engine(engine_fn, name: str, op, cp, rc):
    w = engine_fn(op, cp)
    r, state = util.apply_weights(w, rc, util.tc_map())
    print(f"\n=== {name} ===")
    util.summarize(r, "FULL")
    util.summarize(util.regime_slice(r, "2005-01-01", "2018-12-31"), "IS 2005-2018")
    util.summarize(util.regime_slice(r, util.OOS_START, "2026-12-31"), "OOS 2019+")
    util.summarize(util.regime_slice(r, "2000-01-01", "2008-12-31"), "pre-2008 env")
    util.summarize(util.regime_slice(r, "2007-01-01", "2009-12-31"), "GFC 2007-09")
    util.summarize(util.regime_slice(r, "2020-01-01", "2020-12-31"), "COVID 2020")
    util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "Rate hike 2022")
    return r, w, state


def main():
    op, cp = util.load_prices()
    rc = cp.pct_change()

    returns = {}
    weights = {}
    for name, fn in [
        ("BETA",   engines.engine_beta),
        ("ROT",    engines.engine_rot),
        ("BOND",   engines.engine_bond),
        ("GOLD",   engines.engine_gold),
        ("VRP",    engines.engine_vrp),
        ("CRED",   engines.engine_cred),
        ("SECTOR", engines.engine_sector),
    ]:
        r, w, _ = run_engine(fn, name, op, cp, rc)
        returns[name] = r
        weights[name] = w

    R = pd.DataFrame(returns).dropna(how="all")
    print("\nEngine return correlations (full sample):")
    print(R.corr().round(2))
    print("\nEngine return correlations (IS 2005-2018):")
    print(R.loc[:util.IS_END].corr().round(2))

    R.to_csv(OUT / "engine_returns.csv")
    print(f"\nSaved → {OUT/'engine_returns.csv'}")


if __name__ == "__main__":
    main()
