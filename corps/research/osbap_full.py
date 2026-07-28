"""Bias-free corporate KEYSTONE backtest on the FULL OSBAP universe
(all 55,545 bonds with >=20 trading days — no top-N / survivorship pre-filter).

Buy at ask, sell at bid, signal on clean price. Point-in-time liquidity gate
(>=8 active days in trailing 90) is the only tradability screen, applied at
trade time. Bonds that default/mature and stop trading are included; an open
position exits at the last available bid (loss taken honestly).

Writes corps/research/osbap_results.json.
  python corps/research/osbap_full.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "munis" / "research"))
import backtest as bt          # noqa: E402
from strategies import FACTORIES  # noqa: E402

PANEL = ROOT / "data" / "panel_osbap_full.parquet"
OUT = ROOT / "research" / "osbap_results.json"
MIN_HOLD, MAX_HOLD = 365, 455
DATA_END = pd.Timestamp("2025-03-31")


def load():
    print("loading full panel ...", flush=True)
    p = pd.read_parquet(PANEL, columns=["six", "date", "mid", "s_px", "p_px", "ytw"])
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    print(f"{p['six'].nunique()} bonds, {len(p):,} bond-days; preparing ...", flush=True)
    return bt.prepare(p, coup)


def evalcfg(bonds, disc, lo, hi, label, draws=15):
    fn = FACTORIES["price_discount"](discount=disc)
    fills = bt.run_signal(bonds, fn, min_hold=MIN_HOLD, max_hold=MAX_HOLD,
                          date_lo=lo, date_hi=hi)
    ctl = bt.matched_random_control(bonds, fills, min_hold=MIN_HOLD,
                                    max_hold=MAX_HOLD, n_draws=draws)
    s = bt.summarize(fills, label, control=ctl)
    print(f"  {label:26} n={s.get('n',0):6} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+.2f}% "
          f"excess={s.get('excess_vs_control',0)*100:+.2f}% "
          f"p={s.get('excess_p_boot',1):.3f} stale={s.get('stale_share',0)*100:.0f}%",
          flush=True)
    return s


def main():
    bonds = load()
    res = {}

    print("threshold sweep (full sample):", flush=True)
    res["threshold"] = [evalcfg(bonds, d, pd.Timestamp("2002-01-01"),
                                DATA_END - pd.Timedelta(days=MAX_HOLD),
                                f">={d}pt")
                        for d in (1.0, 2.0, 3.0, 4.0)]

    print("IS/OOS (>=3pt):", flush=True)
    res["is"] = evalcfg(bonds, 3.0, pd.Timestamp("2002-01-01"),
                        pd.Timestamp("2015-12-31"), "IS 2002-2015")
    res["oos"] = evalcfg(bonds, 3.0, pd.Timestamp("2016-01-01"),
                         DATA_END - pd.Timedelta(days=MAX_HOLD), "OOS 2016-2025")

    print("by era (>=3pt):", flush=True)
    eras = [("2004-2007", "2004-01-01", "2007-12-31"),
            ("2008-2009 GFC", "2008-01-01", "2009-12-31"),
            ("2010-2015", "2010-01-01", "2015-12-31"),
            ("2016-2019", "2016-01-01", "2019-12-31"),
            ("2020 COVID", "2020-01-01", "2020-12-31"),
            ("2021-2023", "2021-01-01", "2023-12-31")]
    res["era"] = []
    for name, lo, hi in eras:
        s = evalcfg(bonds, 3.0, pd.Timestamp(lo),
                    min(pd.Timestamp(hi), DATA_END - pd.Timedelta(days=MAX_HOLD)),
                    name)
        s["era"] = name
        res["era"].append(s)

    # survivorship diagnostic: stale-exit rate (bond stopped trading) already
    # tracked as stale_share; report it explicitly on the >=3pt full run.
    OUT.write_text(json.dumps(res, default=float))
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
