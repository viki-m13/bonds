"""Second improvement round — robust avenues (depth, dynamic exit, hold),
each judged strictly on OUT-OF-SAMPLE excess vs the base. Rejected overlays
(regime/credit) from round 1 are not re-run.

Custom dynamic-recovery exit: sell at the first bid once the bond's clean
price recovers back to its trailing-60d median (reversion complete), after a
21-day minimum, else the 455-day hard stop. Economically matched to the thesis
(exit when the dislocation has reverted) — may raise return and cut holding
risk. Uses only past data for the recovery test.

  python corps/research/osbap_improve2.py
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
sys.path.insert(0, str(ROOT / "research"))
from panel_io import load_full  # noqa: E402
from strategies import FACTORIES  # noqa: E402

MAXH = 455
DATA_END = pd.Timestamp("2025-03-31")
IS = (pd.Timestamp("2002-01-01"), pd.Timestamp("2015-12-31"))
OOS = (pd.Timestamp("2016-01-01"), DATA_END - pd.Timedelta(days=MAXH))


def load():
    p = load_full(columns=["six", "date", "mid", "s_px", "p_px", "ytw"])
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    print(f"{p['six'].nunique()} bonds; preparing ...", flush=True)
    return bt.prepare(p, coup)


def ev(bonds, disc, lo, hi, label, min_hold=365, max_hold=MAXH):
    fn = FACTORIES["price_discount"](discount=disc)
    fills = bt.run_signal(bonds, fn, min_hold=min_hold, max_hold=max_hold,
                          date_lo=lo, date_hi=hi)
    ctl = bt.matched_random_control(bonds, fills, min_hold=min_hold,
                                    max_hold=max_hold, n_draws=15)
    s = bt.summarize(fills, label, control=ctl)
    print(f"  {label:34} n={s.get('n',0):6} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+.2f}% hold={s.get('mean_hold',0):.0f}d "
          f"excess={s.get('excess_vs_control',0)*100:+.2f}% p={s.get('excess_p_boot',1):.3f}",
          flush=True)
    return s


def run_dynamic(bonds, disc, window, lo, hi, min_hold=21, hard=MAXH):
    """Base entry; exit on recovery to trailing median (or hard stop)."""
    lo_d = np.datetime64(lo, "D").astype(np.int64)
    hi_d = np.datetime64(hi, "D").astype(np.int64)
    fills = []
    for six, g in bonds.items():
        a = bt._arr(six, g)
        s = g.set_index("date")
        med = s["mid"].rolling(f"{window}D", min_periods=5).median().shift(1).to_numpy()
        sig = ((g["s_px"].to_numpy() - med) <= -disc) & a.elig
        idx = np.flatnonzero(sig & ~np.isnan(med))
        last_exit = -10**9
        mid = g["mid"].to_numpy()
        for i in idx:
            sd = a.day[i]
            j = np.searchsorted(a.s_day, sd, side="right")
            if j >= len(a.s_day) or a.s_day[j] - sd > 7:
                continue
            ed = int(a.s_day[j]); ep = float(a.s_px_at[j])
            if ed < lo_d or ed > hi_d or ed - last_exit < 30:
                continue
            # recovery target = trailing median at entry
            tgt = med[i]
            # scan bid prints from ed+min_hold..ed+hard; exit at first bid whose
            # same-day mid >= tgt (recovered), else last bid (hard stop)
            lo_b = ed + min_hold; hi_b = ed + hard
            k0 = np.searchsorted(a.p_day, lo_b, "left")
            exit_day = exit_px = None
            for k in range(k0, len(a.p_day)):
                if a.p_day[k] > hi_b:
                    break
                # find that day's mid
                di = np.searchsorted(a.day, a.p_day[k])
                if di < a.n and a.day[di] == a.p_day[k] and mid[di] >= tgt:
                    exit_day = int(a.p_day[k]); exit_px = float(a.p_px_at[k]); break
            if exit_day is None:  # no recovery -> hard stop at last bid <= hi_b
                kk = np.searchsorted(a.p_day, hi_b, "right") - 1
                if kk < 0:
                    continue
                exit_day = hi_b; exit_px = float(a.p_px_at[kk])
            fills.append(bt.Fill(six, pd.Timestamp(ed, unit="D"), ep,
                                 pd.Timestamp(exit_day, unit="D"), exit_px,
                                 a.coupon, exit_day == hi_b))
            last_exit = exit_day
    ctl = bt.matched_random_control(bonds, fills, min_hold=min_hold,
                                    max_hold=hard, n_draws=15)
    s = bt.summarize(fills, "dynamic", control=ctl)
    return s


def main():
    bonds = load()

    print("\n[A] Depth robustness (>=3 vs >=4 pt):", flush=True)
    for tag, lo, hi in [("IS", *IS), ("OOS", *OOS)]:
        ev(bonds, 3.0, lo, hi, f"{tag} >=3pt")
        ev(bonds, 4.0, lo, hi, f"{tag} >=4pt")

    print("\n[B] Hold-period sensitivity (>=3pt):", flush=True)
    for mh in (270, 365, 455):
        ev(bonds, 3.0, *OOS, f"OOS hold~{mh}d", min_hold=mh,
           max_hold=mh + 90)

    print("\n[C] Dynamic recovery-exit vs fixed 1yr (>=3pt):", flush=True)
    for tag, lo, hi in [("IS", *IS), ("OOS", *OOS)]:
        d = run_dynamic(bonds, 3.0, 60, lo, hi)
        print(f"  {tag} dynamic-exit                 n={d.get('n',0):6} "
              f"win={d.get('win_rate',0)*100:3.0f}% mean={d.get('mean_ret',0)*100:+.2f}% "
              f"hold={d.get('mean_hold',0):.0f}d excess={d.get('excess_vs_control',0)*100:+.2f}% "
              f"p={d.get('excess_p_boot',1):.3f}", flush=True)


if __name__ == "__main__":
    main()
