"""APEX FINAL v3 — Phoenix-exact sleeves + BTC crypto + overlays.

BEST CONFIG FROM v11 SWEEP:
  6 LETF sleeves (Phoenix-exact clones + V3_SECTOR + V4_ML5 + V6_SHORT_MR), each
  equal-weighted at (1-crypto_weight)/6 of capital. Each sleeve has Phoenix
  composite macro gate (HY proxy, VIX, curve, SPY 200MA) and internal vol-scale
  to 15%.
  + 35% capital allocated to a BTC sleeve (63d momentum + SPY 200MA + VIX < 30
  gates, vol-scaled to 18%).

OVERLAYS on LETF portion (Phoenix-style):
  • Vol-regime gate: halve exposure when SPY 60d RV > 99th pct(504d)
  • DD throttle: floor -10%
  • Vol target: 22% ann (bidirectional, LETF gross cap = 1 - crypto_weight = 65%)

ALL within no-margin: LETF gross ≤ 65%, crypto ≤ 35%, total ≤ 100%.
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
import crypto_sleeve as CS
from apex_v11 import run_portfolio, build, ALL_SLEEVES

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"

CRYPTO_WEIGHT = 0.35
TARGET_VOL = 0.22
DD_FLOOR = -0.10


def main():
    op, cp = util.load_prices()
    print("Building APEX FINAL (Phoenix-exact + crypto)...")
    sw = build(cp)

    # EW blend weights × (1-crypto_weight)
    n = len(sw)
    bw = {k: (1 - CRYPTO_WEIGHT) / n for k in sw}

    net, w_eff = run_portfolio(cp, sw, bw, CRYPTO_WEIGHT, TARGET_VOL, DD_FLOOR)

    # Individual sleeve metrics
    from apex_v10 import sleeve_rets
    sr_dict = sleeve_rets(sw, cp)
    crypto_r = CS.crypto_sleeve_returns(cp.index)
    sr_dict["CRYPTO"] = crypto_r
    R = pd.DataFrame(sr_dict).fillna(0.0)

    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS_SR':>7}  {'2008_MDD':>9}")
    for name in R.columns:
        m = util.metrics(R[name])
        om = util.metrics(util.regime_slice(R[name], OOS_START, "2027-12-31"))
        r08 = util.regime_slice(R[name], "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"mdd": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>7.2f}  "
              f"{m08.get('mdd',0)*100:>8.1f}%")
    print(f"\nIS correlations (including CRYPTO):")
    print(R.loc[:IS_END].corr().round(2))
    print(f"\nAvg pairwise IS corr: {((R.loc[:IS_END].corr().values.sum() - len(R.columns)) / (len(R.columns)**2 - len(R.columns))):.3f}")

    print("\n=== APEX FINAL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal year", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    # Save
    net.to_frame("apex_net_ret").to_csv(OUT / "apex_final_returns.csv")
    w_eff.to_csv(OUT / "apex_final_weights.csv")
    R.to_csv(OUT / "apex_final_sleeve_returns.csv")
    meta = {
        "version": "v11_final",
        "sleeves": list(sw.keys()) + ["CRYPTO"],
        "blend_weights_letf": bw,
        "crypto_weight": CRYPTO_WEIGHT,
        "target_vol": TARGET_VOL,
        "dd_floor": DD_FLOOR,
        "sleeve_vol": 0.15,
    }
    (OUT / "apex_final_meta.json").write_text(json.dumps(meta, indent=2, default=str))

    metrics = {}
    for lbl, (s, e) in [("full", ("1999-01-01", "2027-12-31")),
                        ("is", ("2005-01-01", IS_END)),
                        ("oos", (OOS_START, "2027-12-31")),
                        ("pre08", ("2000-01-01", "2008-12-31")),
                        ("gfc", ("2007-01-01", "2009-12-31")),
                        ("covid", ("2020-01-01", "2020-12-31")),
                        ("ratehike22", ("2022-01-01", "2022-12-31")),
                        ("recovery2324", ("2023-01-01", "2024-12-31"))]:
        metrics[lbl] = util.metrics(util.regime_slice(net, s, e))
    (OUT / "apex_final_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
