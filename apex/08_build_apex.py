"""APEX — Build the production sleeve ensemble.

Blend: equal-weight (so each sleeve contributes equally to portfolio risk).
Because each sleeve is vol-targeted to 10%, equal weights give ~10%/√N vol
baseline, and we then apply DD throttle and final vol target.
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

SLEEVE_FNS = {
    "TSMOM":    S.sleeve_tsmom,
    "XSMOM":    S.sleeve_xsmom,
    "RPAR":     S.sleeve_rpar,
    "TREND_EQ": S.sleeve_trend_eq,
    "TREND_BD": S.sleeve_trend_bd,
    "TREND_GD": S.sleeve_trend_gd,
    "CREDIT":   S.sleeve_credit,
    "VOLREG":   S.sleeve_volreg,
}

SLEEVE_VOL_TARGET = 0.10
PORT_VOL_TARGET = 0.20
DD_FLOOR = -0.15


def finalize(r, target_vol=PORT_VOL_TARGET, dd_floor=DD_FLOOR):
    c = (1 + r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    mdd = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)
    r2 = r * mdd
    rv = r2.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.5).shift(1).fillna(1.0)
    return r2 * vm


def main():
    op, cp = util.load_prices()

    print("Building 8 sleeves (each vol-targeted to 10% ann)...\n")
    sleeve_rets = {}
    for name, fn in SLEEVE_FNS.items():
        r = fn(cp, target_vol=SLEEVE_VOL_TARGET)
        sleeve_rets[name] = r
        util.summarize(r, f"  {name}")

    R = pd.DataFrame(sleeve_rets).fillna(0.0)

    print("\nFull-sample correlations:")
    print(R.corr().round(2))

    # --- 3 blend schemes ---
    schemes = {}
    # EW
    schemes["EW"] = R.mean(axis=1)
    # Inv-variance IS
    is_vol = R.loc[:"2018-12-31"].std().replace(0, np.nan)
    iv = 1.0 / is_vol
    iv = iv / iv.sum()
    schemes["IV"] = (R * iv).sum(axis=1)
    # IS Sharpe-weighted (ex-post)
    is_mu = R.loc[:"2018-12-31"].mean() * util.DPY
    sr = (is_mu / (is_vol * np.sqrt(util.DPY))).clip(lower=0.1)
    sw = sr / sr.sum()
    schemes["SR"] = (R * sw).sum(axis=1)

    print("\nBlend weights (IS):")
    print(f"  IV: {iv.round(3).to_dict()}")
    print(f"  SR: {sw.round(3).to_dict()}")

    for name, blend in schemes.items():
        print(f"\n--- Scheme {name} ---")
        util.summarize(blend, f"  pre-final")
        rf = finalize(blend)
        print("  After DD throttle + vol target:")
        for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                            ("IS 05-18", ("2005-01-01", "2018-12-31")),
                            ("OOS 19+", ("2019-01-02", "2027-12-31")),
                            ("pre-08", ("2000-01-01", "2008-12-31")),
                            ("2022RH", ("2022-01-01", "2022-12-31"))]:
            util.summarize(util.regime_slice(rf, s, e), f"    {lbl}")

    # Final: pick best scheme
    best_r = schemes["EW"]
    rf = finalize(best_r)
    print("\n=== FINAL APEX (EW blend, port 20% tv, dd -15%) ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("RateHike 22", ("2022-01-01", "2022-12-31")),
                        ("Recovery 23-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(rf, s, e), f"  {lbl}")

    R.to_csv(OUT / "sleeve_returns.csv")
    rf.to_frame("apex_net_ret").to_csv(OUT / "apex_returns.csv")
    with open(OUT / "apex_meta.json", "w") as f:
        json.dump({
            "sleeve_vol_target": SLEEVE_VOL_TARGET,
            "port_vol_target": PORT_VOL_TARGET,
            "dd_floor": DD_FLOOR,
            "blend_scheme": "EW",
        }, f, indent=2)


if __name__ == "__main__":
    main()
