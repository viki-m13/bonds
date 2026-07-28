"""Improve the corporate dislocation strategy — OOS-validated, overfitting-aware.

Two economically-motivated overlays on the base price-dislocation signal, both
using only past data:

  A. Market regime gate. Don't buy dislocations while the WHOLE market is
     deteriorating. Gate on market credit-spread MOMENTUM (not level): stand
     down when the cross-sectional median credit spread is rising fast. This
     targets the slow-burn GFC (spreads rose for a year) without killing sharp
     V-shaped stress like COVID (spreads spiked then fell, re-admitting entries).

  B. Per-bond credit filter. Distinguish a LIQUIDITY dislocation (price fell,
     credit spread stable = forced selling → reverts) from CREDIT deterioration
     (price fell because the spread blew out → may default). Require the bond's
     own credit spread not to have widened beyond its trailing median.

Each variant is evaluated IS (2002-2015) and OOS (2016-2025) with the matched
control, so only improvements that hold OUT-OF-SAMPLE are kept.

  python corps/research/osbap_improve.py
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
sys.path.insert(0, str(ROOT / 'research'))
from panel_io import load_full  # noqa: E402

PANEL = ROOT / "data" / "panel_osbap_full.parquet"
MKT = ROOT / "data" / "market_credit_index.parquet"
MIN_HOLD, MAX_HOLD = 365, 455
DATA_END = pd.Timestamp("2025-03-31")


def load():
    cols=['six', 'date', 'mid', 's_px', 'p_px', 'ytw', 'cs']
    p = load_full(columns=cols)
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    print(f"{p['six'].nunique()} bonds; preparing ...", flush=True)
    bonds = bt.prepare(p, coup)
    mkt = pd.read_parquet(MKT)
    # market cs 20-trading-day momentum; stress when rising fast (>10% over ~1mo)
    mkt = mkt.sort_index()
    mkt["mom"] = mkt["mkt_cs"] / mkt["mkt_cs"].shift(20) - 1
    stress = (mkt["mom"] > 0.10)
    # map date -> "risk-off" using only past info (already trailing); shift 1 day
    stress = stress.shift(1).fillna(False)
    stress_map = {d.normalize(): bool(v) for d, v in stress.items()}
    return bonds, stress_map


def sig_base(discount=3.0, window=60):
    def fn(g):
        s = g.set_index("date")
        med = s["mid"].rolling(f"{window}D", min_periods=5).median().shift(1)
        return ((s["s_px"] - med) <= -discount).reset_index(drop=True).fillna(False)
    return fn


def sig_regime(stress_map, discount=3.0, window=60):
    base = sig_base(discount, window)
    def fn(g):
        b = base(g)
        ok = ~g["date"].dt.normalize().map(stress_map).fillna(False).to_numpy()
        return pd.Series(b.to_numpy() & ok)
    return fn


def sig_credit(discount=3.0, window=60, cs_tol=0.25):
    """Price dislocated but credit spread NOT blown out beyond trailing median."""
    def fn(g):
        s = g.set_index("date")
        med = s["mid"].rolling(f"{window}D", min_periods=5).median().shift(1)
        price_disl = (s["s_px"] - med) <= -discount
        cs_med = s["cs"].rolling(f"{window}D", min_periods=5).median().shift(1)
        credit_ok = s["cs"] <= cs_med * (1 + cs_tol)   # spread not blowing out
        return (price_disl & credit_ok).reset_index(drop=True).fillna(False)
    return fn


def sig_both(stress_map, discount=3.0, window=60, cs_tol=0.25):
    reg = sig_regime(stress_map, discount, window)
    cred = sig_credit(discount, window, cs_tol)
    def fn(g):
        return pd.Series(reg(g).to_numpy() & cred(g).to_numpy())
    return fn


def ev(bonds, fn, lo, hi, label):
    fills = bt.run_signal(bonds, fn, min_hold=MIN_HOLD, max_hold=MAX_HOLD,
                          date_lo=lo, date_hi=hi)
    ctl = bt.matched_random_control(bonds, fills, min_hold=MIN_HOLD,
                                    max_hold=MAX_HOLD, n_draws=15)
    s = bt.summarize(fills, label, control=ctl)
    print(f"  {label:24} n={s.get('n',0):6} win={s.get('win_rate',0)*100:3.0f}% "
          f"mean={s.get('mean_ret',0)*100:+.2f}% excess={s.get('excess_vs_control',0)*100:+.2f}% "
          f"p={s.get('excess_p_boot',1):.3f}", flush=True)
    return s


def main():
    bonds, stress = load()
    IS = (pd.Timestamp("2002-01-01"), pd.Timestamp("2015-12-31"))
    OOS = (pd.Timestamp("2016-01-01"), DATA_END - pd.Timedelta(days=MAX_HOLD))
    GFC = (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-12-31"))
    COVID = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"))
    variants = {
        "base": sig_base(),
        "regime": sig_regime(stress),
        "credit": sig_credit(),
        "regime+credit": sig_both(stress),
    }
    out = {}
    for name, fn in variants.items():
        print(f"\n=== {name} ===", flush=True)
        out[name] = {
            "IS": ev(bonds, fn, *IS, "IS 2002-2015"),
            "OOS": ev(bonds, fn, *OOS, "OOS 2016-2025"),
            "GFC": ev(bonds, fn, *GFC, "GFC 2008-2009"),
            "COVID": ev(bonds, fn, *COVID, "COVID 2020"),
        }
    (ROOT / "research" / "osbap_improve_results.json").write_text(
        json.dumps(out, default=float))
    print("\nwrote osbap_improve_results.json")


if __name__ == "__main__":
    main()
