"""APEX v3 — Phoenix-inspired multi-sleeve strategy.

Design follows the Phoenix recipe with LETF-focused sleeves:
  S1 VANGUARD — 189d momentum on {QLD,UGL,TMF,TYD}, VIX gate
  S2 ORION    — RISK (top-3 eq-LETFs) + SAFE (top-2 bond-gold), weekly
  S3 HELIOS   — 189d mom on unlevered, expressed via 3x LETF, VIX gate
  S4 PAA      — accelerating multi-horizon momentum across LETFs
  S5 TREND    — per-LETF 200MA trend, inv-vol sized
  S6 RPAR_CF  — risk parity UPRO/TMF/UGL with TMF-correlation filter
  S7 MREV     — mean-reversion on pullbacks (only uptrend), 5d hold
  S8 CAL      — turn-of-month seasonal
  S9 ML       — XGBoost rank-IC ML sleeve (loaded from ml_v2_weights.csv)

Each sleeve scaled to 15% vol. Portfolio-level overlays:
  • Vol-regime gate: scale book to 25% when VIX > 99th pct(3y)
  • DD throttle: floor -12%
  • Vol target: 25% bidirectional, gross-capped at 1.0
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

SLEEVE_FNS = {
    "VANGUARD":  S3.sleeve_vanguard,
    "ORION":     S3.sleeve_orion,
    "HELIOS":    S3.sleeve_helios,
    "PAA":       S3.sleeve_paa,
    "TREND":     S3.sleeve_trend_vol,
    "RPAR_CF":   S3.sleeve_rpar_cf,
    "MREV":      S3.sleeve_mrev_winners,
    "CALENDAR":  S3.sleeve_calendar,
    "ML":        S3.sleeve_ml_load,
}

SLEEVE_VOL_TARGET = 0.15
PORT_VOL_TARGET = 0.25
DD_FLOOR = -0.12


def build(cp: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    sleeve_weights = {}
    for name, fn in SLEEVE_FNS.items():
        sleeve_weights[name] = fn(cp, target_vol=SLEEVE_VOL_TARGET)
    return sleeve_weights


def blend_and_overlay(sleeve_weights: dict, cp: pd.DataFrame,
                      blend_weights: dict | None = None) -> tuple[pd.Series, pd.DataFrame, dict]:
    """Blend sleeves → apply DD throttle + vol target + vol-regime gate."""
    if blend_weights is None:
        blend_weights = {k: 1.0 / len(sleeve_weights) for k in sleeve_weights}

    # Weighted sum of weight frames
    first = next(iter(sleeve_weights.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sleeve_weights.items():
        P = P + W.fillna(0.0) * blend_weights[name]
    P = P.clip(upper=1.0, lower=0.0)

    # Vol-regime gate (on SPY VIX)
    gates = S3.macro_gates(cp)
    vix_gate = gates["vix_ok"].shift(1).fillna(1.0)
    # When VIX_ok is 0, scale to 25%
    vix_mult = (vix_gate + (1 - vix_gate) * 0.25)
    P = P.mul(vix_mult, axis=0)

    # Raw portfolio return
    rets = cp.pct_change()
    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)

    # DD throttle
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / DD_FLOOR).clip(0, 1).shift(1).fillna(1.0)

    # Vol target (scale UP constrained by gross cap)
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (PORT_VOL_TARGET / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    final_scale = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(final_scale, axis=0)

    # TC
    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross_ret - drag

    state = pd.DataFrame({
        "raw_ret": raw_r, "dd_mult": dd_mult, "vol_mult": vol_mult,
        "vix_gate": vix_gate, "total_mult": total_mult,
        "gross_ret": gross_ret, "tc_drag": drag, "net_ret": net,
    })
    return net, w_eff, state


def main():
    op, cp = util.load_prices()
    print("Building v3 sleeves...")
    sw = build(cp)

    # Report sleeve metrics
    sleeve_rets = {}
    print(f"\n{'Sleeve':15s}   {'SR':>6}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}")
    for name, W in sw.items():
        r = S3._weights_to_returns(W, cp)
        sleeve_rets[name] = r
        m = util.metrics(r)
        print(f"  {name:15s}  {m['sharpe']:>6.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>6.1f}%  {m['mdd']*100:>6.1f}%")

    R = pd.DataFrame(sleeve_rets).fillna(0.0)
    print(f"\nIS correlations:")
    print(R.loc[:"2018-12-31"].corr().round(2))

    # IS-fit inverse-variance blend
    is_std = R.loc[:"2018-12-31"].std().replace(0, np.nan)
    iv = (1.0 / is_std).fillna(0)
    iv = iv / iv.sum()
    iv_dict = iv.to_dict()

    # Also try equal weight
    ew_dict = {k: 1.0 / len(R.columns) for k in R.columns}

    print(f"\nBlend weights (IS-fit inv-var):")
    for k, v in iv_dict.items():
        print(f"  {k:12s} {v:.3f}")

    # EW
    net_ew, _, _ = blend_and_overlay(sw, cp, blend_weights=ew_dict)
    net_iv, _, _ = blend_and_overlay(sw, cp, blend_weights=iv_dict)

    print(f"\n=== EW BLEND ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31")),
                        ("Recovery 23-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net_ew, s, e), f"  {lbl}")

    print(f"\n=== IV BLEND ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31")),
                        ("Recovery 23-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net_iv, s, e), f"  {lbl}")

    # Save
    R.to_csv(OUT / "sleeve_returns_v3.csv")
    net_iv.to_frame("apex_net_ret").to_csv(OUT / "apex_v3_returns.csv")


if __name__ == "__main__":
    main()
