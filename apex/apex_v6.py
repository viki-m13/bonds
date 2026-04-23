"""APEX v6 — Phoenix-faithful multi-sleeve ensemble.

Sleeves chosen to minimize mutual correlation while keeping each SR >= 0.5:
  V1  MOM_LEV     — monthly 189d mom on 4-LETF defensive basket
  V2  RISK_SAFE   — weekly RISK+SAFE split
  V3  SECTOR      — weekly sector-LETF rotation
  V4  ML5         — XGBoost 5d-horizon
  V5  ML63        — XGBoost 63d-horizon
  V6  SHORT_MR    — RSI(2) dip-buy in uptrends
  V7  CALENDAR    — turn-of-month equity tilt
  V9  VOL_REGIME  — low-RV equity exposure

Drop V8_CURVE (negative OOS Sharpe, broken).

Blend: inverse-variance weights fit on IS (2005-2018) only.

Portfolio overlays (Phoenix-style):
  • DD throttle (linear ramp to -12% floor)
  • Vol target 22% (bidirectional, gross-capped at 1.0 — no margin)
  • Vol-regime gate (halve exposure when SPY 60d RV > 99th pct of 504d)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_v6 as S

OUT = Path("/home/user/bonds/data/apex")

SLEEVES = {
    "V1_MOM_LEV":   S.s_v1_mom_lev,
    "V2_RISK_SAFE": S.s_v2_risk_safe,
    "V3_SECTOR":    S.s_v3_sector,
    "V4_ML5":       S.s_v4_ml5,
    "V5_ML63":      S.s_v5_ml63,
    "V6_SHORT_MR":  S.s_v6_short_mr,
    "V7_CALENDAR":  S.s_v7_calendar,
    "V9_VOL_REG":   S.s_v9_vol_regime,
}

IS_END = "2018-12-31"
PORT_VOL = 0.22
DD_FLOOR = -0.12


def compute_weights(cp):
    sw = {name: fn(cp, target_vol=0.18) for name, fn in SLEEVES.items()}
    return sw


def compute_returns(sw, cp):
    return {name: S._weights_to_ret(W, cp) for name, W in sw.items()}


def inv_var_blend_weights(R: pd.DataFrame, is_end: str = IS_END) -> pd.Series:
    """Inverse-variance weights fit on IS."""
    is_R = R.loc[:is_end].dropna(how="all")
    var = is_R.var().replace(0, np.nan)
    iv = 1.0 / var
    iv = iv / iv.sum()
    return iv.fillna(0)


def apply_portfolio_overlays(P: pd.DataFrame, cp: pd.DataFrame,
                              target_vol=PORT_VOL, dd_floor=DD_FLOOR):
    """Phoenix-style overlays: vol-regime gate, DD throttle, vol target."""
    rets = cp.pct_change()

    # Vol-regime gate: halve exposure when SPY 60d RV > 99th pct of 504d
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = regime_ok + (1 - regime_ok) * 0.5   # half during vol spike
    P = P.mul(regime_mult.shift(1).fillna(1.0), axis=0)

    # Raw portfolio return
    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)

    # DD throttle
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)

    # Vol target (bidirectional, gross-capped at 1.0)
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    # Net return with TC
    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross_ret - drag

    state = pd.DataFrame({
        "raw_ret": raw_r, "regime_mult": regime_mult, "dd_mult": dd_mult,
        "vol_mult": vol_mult, "total_mult": total_mult,
        "gross_ret": gross_ret, "tc_drag": drag, "net_ret": net,
    })
    return net, w_eff, state


def build_blend(sw, cp, blend_weights):
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        if name in blend_weights:
            P = P + W.fillna(0.0) * blend_weights[name]
    return P.clip(upper=1.0, lower=0.0)


def main():
    op, cp = util.load_prices()
    print("Building v6 sleeves...")
    sw = compute_weights(cp)
    rets_sleeve = compute_returns(sw, cp)
    R = pd.DataFrame(rets_sleeve).fillna(0.0)

    # Report sleeve metrics
    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS SR':>7}")
    for name in R.columns:
        m = util.metrics(R[name])
        om = util.metrics(util.regime_slice(R[name], "2019-01-02", "2027-12-31"))
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>7.2f}")

    # Inverse-variance blend weights on IS
    iv = inv_var_blend_weights(R)
    print(f"\nIS inverse-variance blend weights:")
    for k, v in iv.sort_values(ascending=False).items():
        print(f"  {k:15s}  {v:.3f}")

    # Build portfolio with IV weights
    P_iv = build_blend(sw, cp, iv.to_dict())
    net_iv, w_iv, state_iv = apply_portfolio_overlays(P_iv, cp)

    # Also try EW blend
    ew = {k: 1.0 / len(SLEEVES) for k in SLEEVES}
    P_ew = build_blend(sw, cp, ew)
    net_ew, _, _ = apply_portfolio_overlays(P_ew, cp)

    # Try SR-weighted blend on IS
    is_R = R.loc[:IS_END]
    is_sr = (is_R.mean() / is_R.std()).fillna(0).clip(lower=0.1)
    srw = (is_sr / is_sr.sum()).to_dict()
    P_sr = build_blend(sw, cp, srw)
    net_sr, _, _ = apply_portfolio_overlays(P_sr, cp)

    # Compare
    print(f"\n=== EW BLEND (1/N) ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(net_ew, s, e), f"  {lbl}")

    print(f"\n=== IV BLEND (inverse-variance) ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(net_iv, s, e), f"  {lbl}")

    print(f"\n=== SR-WEIGHTED BLEND ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(net_sr, s, e), f"  {lbl}")

    # Save best: IV
    net_iv.to_frame("apex_v6_ret").to_csv(OUT / "apex_v6_returns.csv")
    w_iv.to_csv(OUT / "apex_v6_weights.csv")
    R.to_csv(OUT / "apex_v6_sleeve_returns.csv")
    (OUT / "apex_v6_meta.json").write_text(json.dumps({
        "blend_weights": iv.to_dict(),
        "target_vol": PORT_VOL,
        "dd_floor": DD_FLOOR,
        "is_end": IS_END,
    }, indent=2))
    print(f"\nSaved v6 artifacts to {OUT}")


if __name__ == "__main__":
    main()
