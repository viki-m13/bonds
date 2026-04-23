"""APEX v9 — Phoenix-exact clones + my ML + my orthogonal sleeves.

Sleeves (8 total):
  PX_VANGUARD  — Phoenix Vanguard exact (4-LETF basket, macro gate)
  PX_ORION     — Phoenix Orion exact (RISK+SAFE, weekly Wed)
  PX_HELIOS    — Phoenix Helios exact (unlevered → LETF, weekly Fri)
  PX_CRYPTO    — Phoenix-style BTC momentum sleeve (via TQQQ proxy)
  V3_SECTOR    — sector-LETF top-1 weekly (my addition)
  V4_ML5       — XGBoost 5d-horizon rank-IC
  V6_SHORT_MR  — RSI(2) dip-buy in uptrends

BLEND: inverse-variance fit on IS (2005-2018), same as Phoenix.
OVERLAYS: vol-regime gate, DD throttle, vol target (bidirectional, gross ≤ 1).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_v6 as SV6
import sleeves_phoenix_exact as PX

OUT = Path("/home/user/bonds/data/apex")
FRED = Path("/home/user/bonds/data/fred")

IS_START = "2005-01-03"
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


SLEEVE_BUILDERS = {
    "PX_VANGUARD": lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":    lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":   lambda cp: PX.sleeve_helios_exact(cp),
    "PX_CRYPTO":   lambda cp: PX.sleeve_crypto_exact(cp),
    "V3_SECTOR":   lambda cp: SV6.s_v3_sector(cp, target_vol=0.18),
    "V4_ML5":      lambda cp: SV6.s_v4_ml5(cp, target_vol=0.25),
    "V6_SHORT_MR": lambda cp: SV6.s_v6_short_mr(cp, target_vol=0.18),
}


def apply_sleeve_vol_scale(W, cp, target_vol=0.15):
    """Scale sleeve weights DOWN so realized vol ≤ target (never up past 1.0)."""
    rets = cp.pct_change()
    r = (W.shift(1).fillna(0.0) * rets.reindex_like(W).fillna(0.0)).sum(axis=1)
    rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target_vol / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
    return W.mul(m, axis=0)


def build_sleeves(cp):
    sw = {}
    for name, fn in SLEEVE_BUILDERS.items():
        W = fn(cp)
        # Vol-scale each sleeve to 15% ann (matching Phoenix's 15% target)
        W = apply_sleeve_vol_scale(W, cp, target_vol=0.15)
        sw[name] = W
    return sw


def inv_var_weights(R, is_end=IS_END):
    """Phoenix-style IS inverse-variance weights."""
    is_R = R.loc[:is_end].dropna(how="all")
    var = is_R.var().replace(0, np.nan)
    iv = 1.0 / var
    # Cap max weight at 40% of total
    iv = iv.clip(upper=iv.mean() * 4.0)
    iv = iv / iv.sum()
    return iv.fillna(0)


def blend(sleeves, cp, weights):
    first = next(iter(sleeves.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sleeves.items():
        if name in weights:
            P = P + W.fillna(0.0) * weights[name]
    return P.clip(upper=1.0, lower=0.0)


def portfolio_overlays(P, cp, target_vol=0.15, dd_floor=-0.10, dd_win=252, vol_win=60):
    """Phoenix-style overlays: vol-regime gate + DD throttle + daily vol target."""
    rets = cp.pct_change()

    # Vol-regime gate: halve when SPY 60d RV > 99th pct(504d)
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)

    # DD throttle
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(dd_win, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    # Vol target bidirectional but gross-capped at 1
    rv = raw_r.rolling(vol_win, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
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
    return net, w_eff


def main():
    op, cp = util.load_prices()
    print("Building v9 sleeves (Phoenix-exact + my orthogonal)...")
    sleeves = build_sleeves(cp)

    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS SR':>7}  {'2008 MDD':>9}")
    sleeve_rets = {}
    for name, W in sleeves.items():
        r = PX._weights_to_ret(W, cp)
        sleeve_rets[name] = r
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"mdd": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>7.2f}  "
              f"{m08.get('mdd',0)*100:>8.1f}%")

    R = pd.DataFrame(sleeve_rets).fillna(0.0)
    print("\nIS correlations:")
    print(R.loc[:IS_END].corr().round(2))
    print(f"\nAvg pairwise IS correlation: {(R.loc[:IS_END].corr().values.sum() - len(R.columns)) / (len(R.columns)**2 - len(R.columns)):.3f}")

    # Blend weights
    iv = inv_var_weights(R)
    print(f"\nInverse-variance IS blend weights:")
    for k, v in iv.sort_values(ascending=False).items():
        print(f"  {k:15s}  {v:.3f}")

    # Build and overlay
    P = blend(sleeves, cp, iv.to_dict())
    net, w_eff = portfolio_overlays(P, cp, target_vol=0.18, dd_floor=-0.10)

    print("\n=== APEX v9 FINAL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 05-18 synth", ("2005-01-01", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal year", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    # Also try EW
    ew_w = {k: 1.0 / len(sleeves) for k in sleeves}
    P_ew = blend(sleeves, cp, ew_w)
    net_ew, _ = portfolio_overlays(P_ew, cp, target_vol=0.18, dd_floor=-0.10)
    print("\n=== APEX v9 EW (equal-weight blend, same overlays) ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18 synth", ("2005-01-01", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("2008 cal year", ("2008-01-01", "2008-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(net_ew, s, e), f"  {lbl}")

    # Save
    R.to_csv(OUT / "apex_v9_sleeve_returns.csv")
    net.to_frame("apex_v9_ret").to_csv(OUT / "apex_v9_returns.csv")
    w_eff.to_csv(OUT / "apex_v9_weights.csv")
    net_ew.to_frame("apex_v9_ew_ret").to_csv(OUT / "apex_v9_ew_returns.csv")
    (OUT / "apex_v9_meta.json").write_text(json.dumps({
        "blend_weights": iv.to_dict(),
        "target_vol": 0.18,
        "dd_floor": -0.10,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
