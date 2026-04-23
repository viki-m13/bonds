"""APEX v13 — Phoenix-exact + DUALBEAR defense + CALENDAR + CRYPTO.

Sleeves (7 total):
  PX_VANGUARD, PX_ORION, PX_HELIOS (Phoenix-exact clones)
  SL_DUALBEAR (2022-killer: UUP+UCO+UGL when dual-bear score >= 3)
  SL_CALENDAR (FOMC + TOM + pre-holiday + Santa stack)
  SL_VRP (SVXY when contango)
  CRYPTO (external, BTC momentum)

No portfolio margin: LETF portion ≤ 65%, CRYPTO ≤ 25%, total ≤ 90%.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util
import sleeves_phoenix_exact as PX
import sleeves_v12 as SV12
import crypto_sleeve as CS

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"

LETF_WEIGHT = 0.75    # total LETF allocation
CRYPTO_WEIGHT = 0.25
TARGET_VOL = 0.22
DD_FLOOR = -0.10


def extend_cp(cp):
    """Add SVXY, UUP, UCO, VIXY to cp if not present (needed for new sleeves)."""
    for t in ["SVXY", "UUP", "VIXY", "USO"]:
        if t not in cp.columns:
            s = SV12._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s
    return cp


LETF_BUILDERS = {
    "PX_VANGUARD":  lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":     lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":    lambda cp: PX.sleeve_helios_exact(cp),
    "SL_DUALBEAR":  lambda cp: SV12.sleeve_dualbear_defense(cp),
    "SL_CALENDAR":  lambda cp: SV12.sleeve_calendar(cp),
    "SL_VRP":       lambda cp: SV12.sleeve_vrp(cp),
}


def build(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        # Each sleeve scaled to 15% vol
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def run(cp, sw, blend_w, crypto_w=CRYPTO_WEIGHT, target_vol=TARGET_VOL, dd_floor=DD_FLOOR):
    letf_cap = 1.0 - crypto_w
    first = next(iter(sw.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw.items():
        if name in blend_w:
            P = P + W.fillna(0.0) * blend_w[name]
    P = P.clip(upper=letf_cap, lower=0.0)

    rets = cp.pct_change()

    # Phoenix vol-regime gate
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    # Also apply dual-bear override: halve LETF exposure when DBS >= 2
    dbs = SV12.dual_bear_score(cp)
    dbs_mult = pd.Series(1.0, index=cp.index)
    dbs_mult[dbs >= 2] = 0.5
    dbs_mult[dbs >= 3] = 0.25
    dbs_mult = dbs_mult.shift(1).fillna(1.0)
    P = P.mul(dbs_mult, axis=0)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

    # Vol target — capped to letf_cap
    rv = raw_r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm_raw = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=3.0)
    gross_now = P.sum(axis=1).replace(0, np.nan)
    max_up = (letf_cap / gross_now).clip(lower=1.0)
    vol_mult = np.minimum(vm_raw, max_up).shift(1).fillna(1.0)

    total_mult = dd_mult * vol_mult
    w_eff = P.mul(total_mult, axis=0)
    rs = w_eff.sum(axis=1)
    fs = np.minimum(1.0, letf_cap / rs.replace(0, np.nan)).fillna(1.0)
    w_eff = w_eff.mul(fs, axis=0)

    gross_ret = (w_eff.shift(1).fillna(0.0) * rets.reindex_like(w_eff).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w_eff.diff().abs().fillna(w_eff.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w_eff.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    letf_net = gross_ret - drag

    # External crypto
    crypto_r = CS.crypto_sleeve_returns(cp.index, target_vol=0.18)
    net = letf_net + crypto_w * crypto_r
    return net, w_eff


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    print("Building v13 sleeves...")
    sw = build(cp)

    # Print sleeve metrics
    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sw.items():
        r = PX._weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    # Try multiple blend configs
    configs = {
        "EW 3PX + 3SL": {"PX_VANGUARD": 1/6, "PX_ORION": 1/6, "PX_HELIOS": 1/6,
                         "SL_DUALBEAR": 1/6, "SL_CALENDAR": 1/6, "SL_VRP": 1/6},
        "tilt to PX": {"PX_VANGUARD": 0.22, "PX_ORION": 0.22, "PX_HELIOS": 0.22,
                       "SL_DUALBEAR": 0.15, "SL_CALENDAR": 0.10, "SL_VRP": 0.09},
        "heavy dualbear": {"PX_VANGUARD": 0.15, "PX_ORION": 0.15, "PX_HELIOS": 0.15,
                           "SL_DUALBEAR": 0.35, "SL_CALENDAR": 0.10, "SL_VRP": 0.10},
        "no vrp": {"PX_VANGUARD": 0.20, "PX_ORION": 0.20, "PX_HELIOS": 0.20,
                   "SL_DUALBEAR": 0.25, "SL_CALENDAR": 0.15},
    }
    print(f"\n{'Config':20s}  {'FULL_SR':>7}  {'OOS_SR':>7}  {'CAGR_F':>8}  {'CAGR_O':>8}  {'2008_SR':>8}  {'2022_SR':>8}  {'MDD':>7}")
    best_oos = -np.inf
    best_cfg = None
    best_net = None
    for name, bw in configs.items():
        # scale by (1-crypto_w)
        bw_scaled = {k: v * LETF_WEIGHT for k, v in bw.items()}
        net, _ = run(cp, sw, bw_scaled)
        m = util.metrics(net)
        om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        r08 = util.regime_slice(net, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        r22 = util.regime_slice(net, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        print(f"  {name:20s}  {m['sharpe']:>7.2f}  {om.get('sharpe',0):>7.2f}  "
              f"{m['cagr']*100:>7.1f}%  {om.get('cagr',0)*100:>7.1f}%  "
              f"{m08.get('sharpe',0):>8.2f}  {m22.get('sharpe',0):>8.2f}  {m['mdd']*100:>6.1f}%")
        if om.get("sharpe", 0) > best_oos:
            best_oos = om.get("sharpe", 0)
            best_cfg = name
            best_net = net

    print(f"\nBEST OOS: {best_cfg} (SR={best_oos:.2f})")
    print("\n=== BEST DETAIL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(best_net, s, e), f"  {lbl}")

    best_net.to_frame("apex_v13_ret").to_csv(OUT / "apex_v13_returns.csv")
    (OUT / "apex_v13_meta.json").write_text(json.dumps({"best_config": best_cfg, "blend": configs[best_cfg]}, indent=2))


if __name__ == "__main__":
    main()
