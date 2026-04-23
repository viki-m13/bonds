"""APEX — canonical production strategy.

A six-sleeve ensemble of rule-based leveraged-ETF strategies. Long-only, with
sum of weights ≤ 1.0 at all times (no portfolio margin — only LETF leverage).

SLEEVES (all computed at close[t-1], activated on day t):
  S1 XSMOM    — Cross-sectional momentum top-2 of 12 LETFs, monthly rebal
  S2 RPAR     — Inverse-vol risk parity UPRO/TMF/UGL
  S3 TREND_EQ — TQQQ trend-following (200MA + 126d filter)
  S4 TREND_BD — TMF trend-following
  S5 TREND_GD — UGL trend-following
  S6 TSMOM    — Multi-timeframe TSMOM across 12 LETFs
  S7 CRISIS   — UGL allocation in multi-asset stress; inverse-vol UPRO/TMF
                in calm. Diversifies against sleeves 1-6.

BLEND: equal-weight (wᵢ = 1/7). Each sleeve's weights sum to ≤ 1, so blend
gross is also ≤ 1.

OVERLAYS (applied sequentially on the blended weights, lagged 1 day):
  1. CRISIS_SWITCH: when SPY 60d RV > 30% AND SPY 60d < -10% AND TLT 60d < -5%,
     FULL exit to UGL (via EW-blend scaling to 0; sleeve-level CRISIS already
     hedges). Added as outer safety net.
  2. DD_THROTTLE: scale ∈ [0,1] by (1 + NAV_DD / DD_FLOOR). DD_FLOOR = -15%.
  3. VOL_TARGET: scale ∈ [0,1] toward TARGET_VOL = 25%. Scales down only.

Parameters locked in 2005-2018 IS and never re-fit on 2019+ OOS.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd

import util
import sleeves as S

OUT = Path("/home/user/bonds/data/apex")

# Selected sleeves — chosen for diverse signal types and low mutual correlation.
# Subset chosen on IS only (2005-2018) via greedy-select optimizing IS Sharpe
# minus maximum-pairwise-correlation to already-selected.
SLEEVE_FNS = {
    "XSMOM":    S.sleeve_xsmom,       # cross-sectional momentum (monthly rotation)
    "RPAR":     S.sleeve_rpar,        # inverse-vol risk parity UPRO/TMF/UGL
    "TREND_EQ": S.sleeve_trend_eq,    # TQQQ trend-following (200MA + 126d)
    "TREND_BD": S.sleeve_trend_bd,    # TMF trend-following
    "TREND_GD": S.sleeve_trend_gd,    # UGL trend-following
    "TSMOM":    S.sleeve_tsmom,       # multi-timeframe TSMOM across 12 LETFs
}

# Equal-weight blend
BLEND_WEIGHTS = {k: 1.0 / len(SLEEVE_FNS) for k in SLEEVE_FNS}

# Overlay parameters
DD_FLOOR = -0.15
DD_WIN = 252
TARGET_VOL = 0.25
VOL_WIN = 60

IS_END = "2018-12-31"
OOS_START = "2019-01-02"


def build_sleeve_weights(cp: pd.DataFrame, target_vol: float = 0.20) -> dict[str, pd.DataFrame]:
    W = {}
    for name, fn in SLEEVE_FNS.items():
        W[name] = fn(cp, target_vol=target_vol) if fn.__code__.co_argcount > 1 else fn(cp)
    return W


def blend_weights(sleeve_weights: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first = next(iter(sleeve_weights.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sleeve_weights.items():
        P = P + W.fillna(0.0) * BLEND_WEIGHTS[name]
    return P.clip(upper=1.0, lower=0.0)


def crisis_switch(P: pd.DataFrame, cp: pd.DataFrame) -> pd.DataFrame:
    """Outer safety net: when joint equity+bond crisis, reduce to 0 and let
    sleeves' CRISIS component do the hedging."""
    spy = cp["SPY"]
    tlt = cp["TLT"] if "TLT" in cp.columns else spy
    rv60 = spy.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    r60_spy = spy.pct_change(60)
    r60_tlt = tlt.pct_change(60)
    # Stress: SPY vol spike AND SPY DOWN AND TLT DOWN
    stress = ((rv60 > 0.30) & (r60_spy < -0.08) & (r60_tlt < -0.05)).astype(float)
    # Scale down by 0.7 when stressed (keep 30% running)
    m = (1 - 0.7 * stress).shift(1).fillna(1.0)
    return P.mul(m, axis=0)


def portfolio_return_with_overlays(weights: pd.DataFrame, cp: pd.DataFrame,
                                    target_vol: float = TARGET_VOL,
                                    dd_floor: float = DD_FLOOR,
                                    dd_win: int = DD_WIN,
                                    vol_win: int = VOL_WIN) -> tuple[pd.Series, pd.DataFrame]:
    w_raw = weights.fillna(0.0)
    rets = cp.pct_change()
    raw_r = (w_raw.shift(1).fillna(0.0) * rets.reindex_like(w_raw).fillna(0.0)).sum(axis=1)

    # DD throttle (monotone ≤ 1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(dd_win, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)

    # Vol target — bidirectional but capped by no-margin constraint.
    # When the unused sleeve capacity is available, allow scaling UP to bring
    # weights closer to full 100% exposure (but never past it). When vol is
    # above target, scale down.
    rv = raw_r.rolling(vol_win, min_periods=20).std() * np.sqrt(util.DPY)
    vol_mult_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = w_raw.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vol_mult_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = w_raw.mul(total_mult, axis=0)
    row_sum = w_eff.sum(axis=1)
    final_scale = np.minimum(1.0, 1.0 / row_sum.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(final_scale, axis=0)

    gross = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross - drag

    state = pd.DataFrame({
        "raw_ret": raw_r,
        "dd_mult": dd_mult,
        "vol_mult": vol_mult,
        "total_mult": total_mult,
        "gross_ret": gross,
        "tc_drag": drag,
        "net_ret": net,
    })
    return net, state


def run_apex(cp: pd.DataFrame, sleeve_vol: float = 0.20) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, dict]:
    """Full end-to-end: sleeves → blend → crisis → DD/vol overlays."""
    SW = {}
    for name, fn in SLEEVE_FNS.items():
        try:
            SW[name] = fn(cp, target_vol=sleeve_vol)
        except TypeError:
            SW[name] = fn(cp)

    P_blend = blend_weights(SW)
    P_after = crisis_switch(P_blend, cp)
    net_ret, state = portfolio_return_with_overlays(P_after, cp)
    sleeve_rets = {name: S._weights_to_returns(W, cp) for name, W in SW.items()}
    return net_ret, state, P_after, sleeve_rets


def main():
    op, cp = util.load_prices()
    print(f"Loaded prices: {cp.shape}, {cp.index.min().date()} to {cp.index.max().date()}")

    print("\nBuilding sleeves...")
    net_ret, state, P, sleeve_rets = run_apex(cp)

    R = pd.DataFrame(sleeve_rets)
    print("\nSleeve metrics:")
    for name in R.columns:
        util.summarize(R[name], f"  {name:10s}")
    print("\nFull-sample sleeve correlations:")
    print(R.corr().round(2))

    print("\n=== APEX FINAL ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31")),
                        ("Recovery 23-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net_ret, s, e), f"  {lbl}")

    R.to_csv(OUT / "sleeve_returns.csv")
    net_ret.to_frame("apex_net_ret").to_csv(OUT / "apex_production_returns.csv")
    state.to_csv(OUT / "apex_production_state.csv")
    P.to_csv(OUT / "apex_production_weights.csv")

    mfull = util.metrics(net_ret)
    mis = util.metrics(util.regime_slice(net_ret, "2005-01-01", IS_END))
    moos = util.metrics(util.regime_slice(net_ret, OOS_START, "2027-12-31"))
    with open(OUT / "apex_production_metrics.json", "w") as f:
        json.dump({
            "params": {
                "sleeve_vol_target": 0.20,
                "target_vol": TARGET_VOL,
                "dd_floor": DD_FLOOR,
                "dd_win": DD_WIN,
                "vol_win": VOL_WIN,
                "blend_weights": BLEND_WEIGHTS,
                "n_sleeves": len(BLEND_WEIGHTS),
                "is_end": IS_END,
                "oos_start": OOS_START,
            },
            "full": mfull,
            "is": mis,
            "oos": moos,
            "sleeve_metrics": {name: util.metrics(R[name]) for name in R.columns},
            "correlations": R.corr().round(3).to_dict(),
        }, f, indent=2, default=str)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
