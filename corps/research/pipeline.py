"""Corporate-bond dislocation-reversion pipeline.

Reuses the *identical* honest backtest engine validated on 3,085 munis /
5.0M prints (munis/research/backtest.py) — corporate TRACE has the same
normalized trade schema (date, price, ytw, par, side ∈ {S customer-buy,
P customer-sell, D inter-dealer}), so the engine, the matched random-entry
control, and the IS/OOS machinery carry over unchanged.

Corporate-specific notes:
  * Coupon accrual needs a coupon-by-CUSIP reference join (TRACE trades do
    not carry the coupon). The headline metric is *excess vs a matched
    control in the same bond*, which nets out coupon almost entirely; a
    default coupon is used only for absolute-return reporting until a
    reference file is supplied.
  * Corporate spreads are wider and credit dispersion richer than munis,
    which — hypothesis, to be tested on the data — should strengthen a
    dislocation-reversion signal.

Entry points:
  build_panel()  -> daily per-bond panel from corps/data/trades
  run(stage)     -> 'is' | 'oos' backtest of the price_discount family
  selftest()     -> synthetic end-to-end check (no data / creds needed)

Usage:
  python corps/research/pipeline.py selftest
  python corps/research/pipeline.py panel
  python corps/research/pipeline.py is
  python corps/research/pipeline.py oos
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MUNIS_RESEARCH = ROOT.parent / "munis" / "research"
sys.path.insert(0, str(MUNIS_RESEARCH))
import backtest as bt          # noqa: E402  (the shared, validated engine)
import panel as muni_panel     # noqa: E402  (shared per-bond aggregation)
from strategies import FACTORIES, MIN_HOLDS, MAX_HOLDS  # noqa: E402

TRADES = ROOT / "data" / "trades"
PANEL = ROOT / "data" / "panel_daily.parquet"
DEFAULT_COUPON = 5.0            # corporate IG-ish default until ref join
DATA_END = None                # set from panel


def build_panel() -> pd.DataFrame:
    files = sorted(glob.glob(str(TRADES / "*.csv.gz")))
    if not files:
        sys.exit("no corp trade files — run corps/scripts/download_trades.py "
                 "download (needs FINRA API credentials).")
    parts = []
    for i, f in enumerate(files):
        # reuse the muni per-bond aggregator; it keys on file stem as `six`
        p = muni_panel._one(f)
        if p is not None:
            parts.append(p)
        if i % 200 == 0:
            print(f"panel {i}/{len(files)}", flush=True)
    panel = pd.concat(parts, ignore_index=True).sort_values(["six", "date"])
    panel.to_parquet(PANEL)
    print(f"panel: {len(panel):,} bond-days, {panel['six'].nunique()} bonds "
          f"-> {PANEL}", flush=True)
    return panel


def _bonds(panel: pd.DataFrame):
    coupons = pd.Series(DEFAULT_COUPON, index=panel["six"].unique())
    return bt.prepare(panel, coupons)


def run(stage: str) -> None:
    panel = pd.read_parquet(PANEL)
    bonds = _bonds(panel)
    end = panel["date"].max()
    fn = FACTORIES["price_discount"](discount=3.0)
    mh = MAX_HOLDS["price_discount"]
    if stage == "is":
        lo, hi = pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")
    else:
        lo, hi = pd.Timestamp("2021-01-01"), end - pd.Timedelta(days=mh)
    fills = bt.run_signal(bonds, fn, min_hold=MIN_HOLDS["price_discount"],
                          date_lo=lo, date_hi=hi, max_hold=mh)
    ctl = bt.matched_random_control(bonds, fills,
                                    min_hold=MIN_HOLDS["price_discount"],
                                    max_hold=mh)
    s = bt.summarize(fills, f"corp price_discount {stage}", control=ctl)
    for k, v in s.items():
        print(f"  {k}: {v}")


def selftest() -> None:
    """Synthetic corp panel with a planted dislocation-reversion pattern,
    long hold — verifies the engine wiring end-to-end without data/creds."""
    dates = pd.date_range("2015-01-01", "2022-06-30", freq="5D")
    rng = np.random.default_rng(0)
    rows = []
    for cusip in [f"BOND{i:03d}" for i in range(40)]:
        base = 100.0
        for k, d in enumerate(dates):
            base += rng.normal(0, 0.15)
            # periodic forced-seller dip that reverts over the next ~year
            dip = -4.0 if (k % 60 == 30) else 0.0
            mid = base + dip
            rows.append(dict(six=cusip, date=d, mid=mid,
                             s_px=mid + 0.6, p_px=mid - 0.6,
                             d_px=mid, s_par=25000, p_par=25000,
                             s_n=2, p_n=2, d_n=1, ytw=5.0 - (mid - 100) * 0.05,
                             n_all=5))
    panel = pd.DataFrame(rows)
    bonds = bt.prepare(panel, pd.Series(DEFAULT_COUPON,
                                        index=panel["six"].unique()))
    fn = FACTORIES["price_discount"](discount=3.0)
    fills = bt.run_signal(bonds, fn, min_hold=365, max_hold=455,
                          date_lo=pd.Timestamp("2015-01-01"),
                          date_hi=pd.Timestamp("2021-03-01"))
    ctl = bt.matched_random_control(bonds, fills, min_hold=365, max_hold=455)
    s = bt.summarize(fills, "corp selftest", control=ctl)
    print("SELFTEST (synthetic corp data):")
    for k in ("n", "win_rate", "mean_ret", "excess_vs_control", "excess_p_boot"):
        print(f"  {k}: {s.get(k)}")
    assert s["n"] > 0, "engine produced no fills"
    print("OK — corp pipeline wiring verified end-to-end.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "panel":
        build_panel()
    elif cmd in ("is", "oos"):
        run(cmd)
    else:
        sys.exit(f"unknown command {cmd}")
