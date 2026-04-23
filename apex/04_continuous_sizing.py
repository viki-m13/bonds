"""APEX — Continuous position sizing experiments.

Key idea: instead of binary trend-on/off, use continuous sizing:
  position = sign(trend) * min(1, target_vol / realized_vol)

Plus multi-timeframe trend consensus (63/126/252d momentum avg), risk-off
shifts to bonds or gold (diversifying).

Experiments:
  E1: Continuous TQQQ sizing (short=cash, long=sized)
  E2: Continuous TQQQ + 50% TMF base in risk-off periods (barbell)
  E3: Continuous sleeves of (TQQQ, TMF, UGL) all vol-targeted, summed
  E4: TSMOM consensus: sum of 63d, 126d, 252d momentum normalized signals
      across 10 LETFs, long each proportional to +signal, size by vol.
      Sum long-only, normalize to 1.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import util


def multi_tsmom(cp: pd.DataFrame, universe: list[str],
                lookbacks: list[int] = [63, 126, 252],
                target_sleeve_vol: float = 0.10,
                vol_win: int = 60,
                rebal: int = 1) -> pd.Series:
    """Multi-timeframe TSMOM with per-asset vol targeting, long-only, normalized.

    For each asset a:
        consensus_a = sum_h [ sign(ret(h)) ] / len(lookbacks)    ∈ [-1,1]
        raw_size_a  = max(0, consensus_a) * target_sleeve_vol / rv_a
    Long-only weight: normalize raw_size_a to sum ≤ 1.
    If sum is 0, allocate to cash.
    """
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    rv = p.pct_change().rolling(vol_win).std() * np.sqrt(util.DPY)

    # Consensus score
    cons = None
    for L in lookbacks:
        r = p.pct_change(L)
        s = np.sign(r).fillna(0.0)
        cons = s if cons is None else cons + s
    cons = cons / len(lookbacks)   # ∈ [-1, 1]

    # Long-only raw size (no shorting)
    long_cons = cons.clip(lower=0.0)
    raw_w = long_cons * target_sleeve_vol / rv.replace(0, np.nan)
    raw_w = raw_w.fillna(0.0)

    # Normalize to sum ≤ 1
    s = raw_w.sum(axis=1)
    scale = np.minimum(1.0, 1.0 / s.replace(0, np.nan)).fillna(0.0)
    w = raw_w.mul(scale, axis=0)

    # Cash residual
    cash_w = (1 - w.sum(axis=1)).clip(lower=0.0)

    # Monthly rebal (optional)
    if rebal > 1:
        mask = pd.Series(range(len(cp.index)), index=cp.index)
        is_rebal = mask % rebal == 0
        w_full = pd.DataFrame(np.nan, index=cp.index, columns=w.columns)
        w_full[is_rebal] = w[is_rebal]
        w = w_full.ffill().fillna(0.0)
        cash_w = (1 - w.sum(axis=1)).clip(lower=0.0)

    # Compute portfolio return (weights[t] earn ret[t+1])
    rets = p.pct_change()
    cash_ret = cp.get("SHY", cp.get("BIL")).pct_change()
    portfolio = (w.shift(1).fillna(0.0) * rets.fillna(0.0)).sum(axis=1) + \
                cash_w.shift(1).fillna(1.0) * cash_ret.fillna(0.0)

    # Transaction cost (TC bps per unit weight change)
    tc_map = util.tc_map()
    tc_vec = pd.Series({c: tc_map.get(c, 5.0) for c in w.columns})
    dw = w.diff().abs().fillna(w.abs())
    tc_drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)

    return portfolio - tc_drag


def multi_tsmom_with_dd_and_voltarget(cp: pd.DataFrame, universe: list[str],
                                       target_sleeve_vol: float = 0.10,
                                       target_port_vol: float = 0.15,
                                       dd_floor: float = -0.15) -> pd.Series:
    r = multi_tsmom(cp, universe, target_sleeve_vol=target_sleeve_vol)
    # DD throttle
    c = (1 + r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    m = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)
    r = r * m
    # Daily vol target
    rv = r.rolling(60).std() * np.sqrt(util.DPY)
    vm = (target_port_vol / rv).clip(lower=0.25, upper=1.5).shift(1).fillna(1.0)
    return r * vm


def main():
    op, cp = util.load_prices()

    # --- E4: Multi-TSMOM across a broad LETF universe ---
    universe = ["UPRO", "TQQQ", "TECL", "FAS", "SOXL", "EDC", "YINN",
                "TMF", "UBT", "TYD", "UGL", "UCO", "DRN"]
    print("=" * 100)
    print("Variant E4: Multi-timeframe TSMOM (63/126/252d) across 13 LETFs, sleeve-vol 10%, daily rebal")
    r = multi_tsmom(cp, universe, target_sleeve_vol=0.10)
    util.summarize(r, "base")

    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31")),
                        ("COVID", ("2020-01-01", "2020-12-31")),
                        ("2022RH", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(r, s, e), f"  {lbl}")

    print("\n" + "=" * 100)
    print("Variant E5: Multi-TSMOM + DD throttle + vol target 15%")
    r = multi_tsmom_with_dd_and_voltarget(cp, universe, target_sleeve_vol=0.10, target_port_vol=0.15)
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31")),
                        ("COVID", ("2020-01-01", "2020-12-31")),
                        ("2022RH", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(r, s, e), f"  {lbl}")

    # --- Target vol sweep ---
    print("\nVariant E5 target-vol sweep:")
    for tv in (0.10, 0.15, 0.20, 0.25):
        r = multi_tsmom_with_dd_and_voltarget(cp, universe,
                                               target_sleeve_vol=0.10, target_port_vol=tv)
        util.summarize(r, f"  tv={tv}")

    # --- Sleeve-vol sweep ---
    print("\nVariant E5 sleeve-vol sweep (port-vol=15%):")
    for sv in (0.05, 0.08, 0.10, 0.12, 0.15):
        r = multi_tsmom_with_dd_and_voltarget(cp, universe,
                                               target_sleeve_vol=sv, target_port_vol=0.15)
        util.summarize(r, f"  sv={sv}")

    # --- Lookback robustness ---
    print("\nLookback robustness (one lookback only, port-vol=15%):")
    for lbs in ([63], [126], [252], [63,126], [126,252], [63,252], [63,126,252]):
        r = multi_tsmom(cp, universe, lookbacks=lbs, target_sleeve_vol=0.10)
        # Apply vol target
        rv = r.rolling(60).std() * np.sqrt(util.DPY)
        vm = (0.15 / rv).clip(lower=0.25, upper=1.5).shift(1).fillna(1.0)
        util.summarize(r * vm, f"  lbs={lbs}")


if __name__ == "__main__":
    main()
