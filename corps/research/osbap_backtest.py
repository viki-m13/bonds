"""Corporate KEYSTONE on the free OSBAP daily TRACE panel (2002-2025).

Reuses the muni backtest engine unchanged. Fills: buy at ask (prc_ask -> s_px),
sell at bid (prc_bid -> p_px); signal on the clean price (pr -> mid). Carry
proxied by each bond's median yield (nets out of the matched-control excess).

  python corps/research/osbap_backtest.py sweep   # threshold + horizon scan
  python corps/research/osbap_backtest.py full     # IS/OOS/era for >=3pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "munis" / "research"))
import backtest as bt          # noqa: E402  (shared validated engine)
from strategies import FACTORIES  # noqa: E402

PANEL = ROOT / "data" / "panel_osbap.parquet"
MIN_HOLD, MAX_HOLD = 365, 455
DATA_END = pd.Timestamp("2025-03-31")


def load_bonds():
    print("loading panel ...", flush=True)
    p = pd.read_parquet(PANEL)
    coup = p.groupby("six")["ytw"].median().clip(1, 12)   # carry proxy
    print(f"{p['six'].nunique()} bonds, {len(p):,} bond-days; preparing ...",
          flush=True)
    return bt.prepare(p, coup)


def horizon_curve(bonds):
    """Unconditional forward customer round-trip (buy ask -> sell bid + carry)."""
    print("horizon curve ...", flush=True)
    hz = [30, 90, 180, 365, 730]
    res = {h: [] for h in hz}
    for six, g in bonds.items():
        a = bt._arr(six, g)
        c = a.coupon
        for i in range(a.n):
            if np.isnan(a.s_px[i]):
                continue
            for h in hz:
                ex = bt._exit_for(a, int(a.day[i]), h, h + 90)
                if ex:
                    xd, xp, _ = ex
                    hold = xd - int(a.day[i])
                    res[h].append((xp - a.s_px[i] + c / 100 / 365 * hold * 100)
                                  / a.s_px[i])
    for h in hz:
        arr = np.array(res[h])
        print(f"  {h:4}d: mean {arr.mean()*100:+.2f}%  win {(arr>0).mean()*100:.0f}%"
              f"  n={len(arr)}")


def evalcfg(bonds, disc, lo, hi, label):
    fn = FACTORIES["price_discount"](discount=disc)
    fills = bt.run_signal(bonds, fn, min_hold=MIN_HOLD, max_hold=MAX_HOLD,
                          date_lo=lo, date_hi=hi)
    ctl = bt.matched_random_control(bonds, fills, min_hold=MIN_HOLD,
                                    max_hold=MAX_HOLD)
    s = bt.summarize(fills, label, control=ctl)
    print(f"  {label:32} n={s['n']:5} win={s['win_rate']*100:3.0f}% "
          f"mean={s['mean_ret']*100:+.2f}% excess={s['excess_vs_control']*100:+.2f}% "
          f"p={s['excess_p_boot']:.3f}", flush=True)
    return s


def sweep(bonds):
    horizon_curve(bonds)
    print("\nthreshold sweep (full sample):")
    for d in (1.0, 2.0, 3.0, 4.0):
        evalcfg(bonds, d, pd.Timestamp("2002-01-01"),
                DATA_END - pd.Timedelta(days=MAX_HOLD), f"price_discount>={d}")


def full(bonds):
    print("IS 2002-2015 / OOS 2016-2025 (>=3pt):")
    evalcfg(bonds, 3.0, pd.Timestamp("2002-01-01"), pd.Timestamp("2015-12-31"),
            "IS 2002-2015")
    evalcfg(bonds, 3.0, pd.Timestamp("2016-01-01"),
            DATA_END - pd.Timedelta(days=MAX_HOLD), "OOS 2016-2025")
    print("\nby era (>=3pt):")
    eras = [("2004-2007", "2004-01-01", "2007-12-31"),
            ("2008-2009 GFC", "2008-01-01", "2009-12-31"),
            ("2010-2015", "2010-01-01", "2015-12-31"),
            ("2016-2019", "2016-01-01", "2019-12-31"),
            ("2020 COVID", "2020-01-01", "2020-12-31"),
            ("2021-2023", "2021-01-01", "2023-12-31")]
    for name, lo, hi in eras:
        evalcfg(bonds, 3.0, pd.Timestamp(lo),
                min(pd.Timestamp(hi), DATA_END - pd.Timedelta(days=MAX_HOLD)),
                name)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    b = load_bonds()
    (sweep if cmd == "sweep" else full)(b)
