"""APEX FINAL v2 — locked production strategy.

Configuration chosen after extensive v1-v8 exploration:
  5-sleeve equal-weight blend, Phoenix-style portfolio overlays.

SLEEVES (each targeting 18% individual vol, ML5 at 28%):
  V1 MOM_LEV     — monthly 189d mom on {QLD, UGL, TMF, TYD} (Vanguard clone)
  V2 RISK_SAFE   — weekly RISK+SAFE split across 11 LETFs (Orion clone)
  V3 SECTOR      — weekly top-1 sector LETF by 63d mom (market-filtered)
  V4 ML5         — XGBoost 5d-horizon on price+macro features (rank-IC CV)
  V6 SHORT_MR    — RSI(2)<5 dip-buy in uptrends, 3-day hold

PORTFOLIO OVERLAYS (Phoenix-style):
  • Vol-regime gate: halve exposure when SPY 60d RV > 99th pct(504d)
  • DD throttle: linear ramp to -12% floor
  • Vol target: 22% annualised (bidirectional, gross-capped at 1.0)

ALL WEIGHTS sum to <= 1 — no portfolio margin. Only LETF leverage.

RESULTS:
  Full:    Sharpe 0.93, CAGR 18.1%, MDD -42.8%
  IS:      Sharpe 1.10, CAGR 24.2%
  OOS 19+: Sharpe 0.97, CAGR 21.9%, MDD -42.8%
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
from apex_v6 import apply_portfolio_overlays

OUT = Path("/home/user/bonds/data/apex")

SLEEVES_CONFIG = [
    ("V1_MOM_LEV",   S.s_v1_mom_lev,    0.18),
    ("V2_RISK_SAFE", S.s_v2_risk_safe,  0.18),
    ("V3_SECTOR",    S.s_v3_sector,     0.18),
    ("V4_ML5",       S.s_v4_ml5,        0.28),
    ("V6_SHORT_MR",  S.s_v6_short_mr,   0.18),
]
PORT_VOL = 0.22
DD_FLOOR = -0.12


def build():
    op, cp = util.load_prices()
    sleeves = {name: fn(cp, target_vol=tv) for name, fn, tv in SLEEVES_CONFIG}
    # Equal-weight blend
    first = next(iter(sleeves.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    w_each = 1.0 / len(sleeves)
    for W in sleeves.values():
        P = P + W.fillna(0.0) * w_each
    P = P.clip(upper=1.0, lower=0.0)
    # Overlays
    net, w_eff, state = apply_portfolio_overlays(P, cp, target_vol=PORT_VOL, dd_floor=DD_FLOOR)
    sleeve_rets = {name: S._weights_to_ret(W, cp) for name, W in sleeves.items()}
    return cp, sleeves, sleeve_rets, net, w_eff, state


def main():
    cp, sleeves, sleeve_rets, net, w_eff, state = build()

    # Sleeve metrics
    R = pd.DataFrame(sleeve_rets)
    print("Sleeve metrics:")
    print(f"  {'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS SR':>7}")
    for name in R.columns:
        m = util.metrics(R[name])
        om = util.metrics(util.regime_slice(R[name], "2019-01-02", "2027-12-31"))
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>7.2f}")

    print("\nIS sleeve correlations:")
    print(R.loc[:"2018-12-31"].corr().round(2))

    # Windows
    print("\n=== APEX FINAL ===")
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    # Save artifacts
    net.to_frame("apex_net_ret").to_csv(OUT / "apex_final_returns.csv")
    w_eff.to_csv(OUT / "apex_final_weights.csv")
    state.to_csv(OUT / "apex_final_state.csv")
    R.to_csv(OUT / "apex_final_sleeve_returns.csv")

    meta = {
        "version": "v7_final",
        "sleeves": [{"name": n, "target_vol": tv} for n, _, tv in SLEEVES_CONFIG],
        "blend": "equal_weight",
        "port_vol_target": PORT_VOL,
        "dd_floor": DD_FLOOR,
    }
    (OUT / "apex_final_meta.json").write_text(json.dumps(meta, indent=2))

    # Metrics JSON
    metrics = {}
    for lbl, (s, e) in [("full", ("1999-01-01", "2027-12-31")),
                        ("is", ("2005-01-01", "2018-12-31")),
                        ("oos", ("2019-01-02", "2027-12-31")),
                        ("pre08", ("2000-01-01", "2008-12-31")),
                        ("gfc", ("2007-01-01", "2009-12-31")),
                        ("covid", ("2020-01-01", "2020-12-31")),
                        ("ratehike22", ("2022-01-01", "2022-12-31")),
                        ("recovery2324", ("2023-01-01", "2024-12-31"))]:
        metrics[lbl] = util.metrics(util.regime_slice(net, s, e))
    (OUT / "apex_final_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    print(f"\nSaved final artifacts to {OUT}")


if __name__ == "__main__":
    main()
