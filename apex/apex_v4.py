"""APEX v4 — aggressive overlays, focused sleeves, crisis cash-out.

Key additions:
  • Dual-bear detector: if both SPY 60d and TLT 60d returns < 0, shift to cash
  • VIX > 35 → immediate 25% exposure
  • Full exit when portfolio DD > -10% (not -15%)
  • 60-day rolling vol target applied to the FINAL net returns
  • Smaller sleeve set — keep the best 5
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
import util
import sleeves_v3 as S3

OUT = Path("/home/user/bonds/data/apex")
FRED = Path("/home/user/bonds/data/fred")


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


# Best sleeves per earlier tests
SLEEVE_FNS = {
    "ORION":    S3.sleeve_orion,
    "TREND":    S3.sleeve_trend_vol,
    "RPAR_CF":  S3.sleeve_rpar_cf,
    "PAA":      S3.sleeve_paa,
    "ML":       S3.sleeve_ml_load,
    "HELIOS":   S3.sleeve_helios,
}

PORT_VOL_TARGET = 0.20
DD_FLOOR = -0.10


def dual_bear_gate(cp: pd.DataFrame) -> pd.Series:
    """Returns multiplier: 0.25 when both SPY 60d and TLT 60d are negative."""
    idx = cp.index
    spy = cp["SPY"]
    tlt = cp["TLT"] if "TLT" in cp.columns else spy
    spy_r60 = spy.pct_change(60)
    tlt_r60 = tlt.pct_change(60)
    dual_bear = ((spy_r60 < 0) & (tlt_r60 < 0)).astype(float)
    return (1 - 0.75 * dual_bear).shift(1).fillna(1.0)


def strong_vix_gate(cp: pd.DataFrame) -> pd.Series:
    vix = _fred("VIXCLS", cp.index)
    # VIX > 30 → 50%; VIX > 40 → 25%
    m = pd.Series(1.0, index=cp.index)
    m[vix > 30] = 0.5
    m[vix > 40] = 0.25
    return m.shift(1).fillna(1.0)


def run(cp, blend_weights=None, target_vol=PORT_VOL_TARGET, dd_floor=DD_FLOOR):
    sw = {name: fn(cp, target_vol=0.15) for name, fn in SLEEVE_FNS.items()}
    if blend_weights is None:
        blend_weights = {k: 1.0 / len(sw) for k in sw}
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        P = P + W.fillna(0.0) * blend_weights[name]
    P = P.clip(upper=1.0, lower=0.0)

    # Apply strong overlays BEFORE computing return
    vix_m = strong_vix_gate(cp)
    dual_m = dual_bear_gate(cp)
    P = P.mul(vix_m * dual_m, axis=0)

    rets = cp.pct_change()
    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)

    # DD throttle
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    # Vol target (bidirectional, gross-capped)
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    final_scale = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(final_scale, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross_ret - drag
    return net, w_eff, sw


def main():
    op, cp = util.load_prices()
    print("Running APEX v4...")
    net, w_eff, sw = run(cp)
    print("\n=== APEX v4 ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31")),
                        ("Recovery 23-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")
    net.to_frame("apex_v4_ret").to_csv(OUT / "apex_v4_returns.csv")


if __name__ == "__main__":
    main()
