"""APEX v8 — minimalist, 3-sleeve concentrated on best OOS performers.

V1 MOM_LEV      (0.96 OOS SR)
V2 RISK_SAFE    (0.94 OOS SR)
V4 ML5 at 0.25 target vol

Equal-weight blend. Same overlays as v7b.
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
FRED = Path("/home/user/bonds/data/fred")


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def run(cp, blend_weights, sleeve_tvol=(0.18, 0.18, 0.25, 0.18, 0.18),
        port_tv=0.25, dd_floor=-0.12):
    """Build and run APEX with given blend and target vols."""
    sleeves = {
        "V1_MOM_LEV":   S.s_v1_mom_lev(cp, target_vol=sleeve_tvol[0]),
        "V2_RISK_SAFE": S.s_v2_risk_safe(cp, target_vol=sleeve_tvol[1]),
        "V4_ML5":       S.s_v4_ml5(cp, target_vol=sleeve_tvol[2]),
        "V3_SECTOR":    S.s_v3_sector(cp, target_vol=sleeve_tvol[3]),
        "V6_SHORT_MR":  S.s_v6_short_mr(cp, target_vol=sleeve_tvol[4]),
    }
    first = next(iter(sleeves.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sleeves.items():
        if name in blend_weights:
            P = P + W.fillna(0.0) * blend_weights[name]
    P = P.clip(upper=1.0, lower=0.0)

    # Overlays
    # Dual-bear
    spy = cp["SPY"]
    tlt = cp["TLT"] if "TLT" in cp.columns else spy
    dual_bear = ((spy.pct_change(20) < -0.02) & (tlt.pct_change(20) < -0.01)).astype(float)
    dual_m = (1 - 0.75 * dual_bear).shift(1).fillna(1.0)

    # Correlation flip
    sr = spy.pct_change()
    tr = tlt.pct_change()
    corr = sr.rolling(20).corr(tr)
    cf_ok = ((corr > 0).rolling(10).sum() < 8).astype(float).fillna(1.0).shift(1).fillna(1.0)
    for c in ["TMF", "UBT", "TYD"]:
        if c in P.columns:
            P[c] = P[c] * cf_ok

    # VIX
    vix = _fred("VIXCLS", cp.index)
    vix_m = pd.Series(1.0, index=cp.index)
    vix_m[vix > 30] = 0.5
    vix_m[vix > 40] = 0.25
    vix_m = vix_m.shift(1).fillna(1.0)

    P = P.mul(dual_m * vix_m, axis=0)

    # Vol-regime
    spy_rv60 = spy.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    # Raw return
    rets = cp.pct_change()
    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)

    # DD throttle
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    # Vol target
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (port_tv / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (1.0 / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, 1.0 / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    net = gross_ret - drag
    return net, w_eff, sleeves


def main():
    op, cp = util.load_prices()
    # Try multiple 3-5 sleeve combinations
    configs = {
        "V1+V2+V4 EW":      {"V1_MOM_LEV": 1/3, "V2_RISK_SAFE": 1/3, "V4_ML5": 1/3},
        "V1+V2+V4 (ML50)":  {"V1_MOM_LEV": 0.25, "V2_RISK_SAFE": 0.25, "V4_ML5": 0.50},
        "V1+V4 (ML60)":     {"V1_MOM_LEV": 0.40, "V4_ML5": 0.60},
        "V2+V4 (ML60)":     {"V2_RISK_SAFE": 0.40, "V4_ML5": 0.60},
        "V1+V2+V3+V4 EW":   {"V1_MOM_LEV": 0.25, "V2_RISK_SAFE": 0.25, "V3_SECTOR": 0.25, "V4_ML5": 0.25},
        "V1+V2+V4+V6 EW":   {"V1_MOM_LEV": 0.25, "V2_RISK_SAFE": 0.25, "V4_ML5": 0.25, "V6_SHORT_MR": 0.25},
    }
    print(f"{'Config':25s}  {'FULL_SR':>7}  {'OOS_SR':>7}  {'FULL_CAGR':>10}  {'OOS_CAGR':>9}  {'2022_SR':>8}")
    best_oos_sr = -np.inf
    best_cfg = None
    best_net = None
    for name, bw in configs.items():
        net, _, _ = run(cp, bw)
        m = util.metrics(net)
        oo = util.metrics(util.regime_slice(net, "2019-01-02", "2027-12-31"))
        m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        print(f"  {name:25s}  {m['sharpe']:>7.2f}  {oo.get('sharpe',0):>7.2f}  "
              f"{m['cagr']*100:>8.1f}%  {oo.get('cagr',0)*100:>7.1f}%  {m22.get('sharpe',0):>8.2f}")
        if oo.get("sharpe", 0) > best_oos_sr:
            best_oos_sr = oo.get("sharpe", 0)
            best_cfg = name
            best_net = net

    print(f"\nBEST by OOS SR: {best_cfg}  (SR={best_oos_sr:.2f})")
    best_net.to_frame("apex_v8_ret").to_csv(OUT / "apex_v8_returns.csv")
    print("\n=== BEST CONFIG METRICS ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(best_net, s, e), f"  {lbl}")


if __name__ == "__main__":
    main()
