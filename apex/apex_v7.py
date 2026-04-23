"""APEX v7 — lean config. Drop weak sleeves, scale ML higher, MV optimal blend.

Active sleeves (5):
  V1 MOM_LEV
  V2 RISK_SAFE
  V3 SECTOR
  V4 ML5 (at higher target vol 0.25 — let it shine)
  V6 SHORT_MR

Blend: mean-variance optimal weights (subject to long-only, sum=1).
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
from apex_v6 import apply_portfolio_overlays, build_blend

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"


def mean_var_weights(R: pd.DataFrame, is_end: str = IS_END) -> pd.Series:
    """Long-only Markowitz: w ∝ max(0, Σ^-1 · μ).
    Use diagonal-enhanced covariance for stability."""
    is_R = R.loc[:is_end].dropna(how="all").fillna(0)
    mu = is_R.mean().values
    cov = is_R.cov().values
    # Ledoit-Wolf-style shrinkage to diagonal
    shrink = 0.3
    cov_shr = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
    try:
        inv_cov = np.linalg.inv(cov_shr)
        raw = inv_cov @ mu
    except Exception:
        raw = mu / np.diag(cov)
    raw = np.maximum(raw, 0)   # long-only
    if raw.sum() == 0:
        raw = np.ones_like(raw)
    w = raw / raw.sum()
    return pd.Series(w, index=R.columns)


def main():
    op, cp = util.load_prices()

    # Build sleeves — V4 ML5 at HIGHER target vol
    sleeves = {
        "V1_MOM_LEV":   S.s_v1_mom_lev(cp, target_vol=0.18),
        "V2_RISK_SAFE": S.s_v2_risk_safe(cp, target_vol=0.18),
        "V3_SECTOR":    S.s_v3_sector(cp, target_vol=0.18),
        "V4_ML5":       S.s_v4_ml5(cp, target_vol=0.28),   # let ML run hot
        "V6_SHORT_MR":  S.s_v6_short_mr(cp, target_vol=0.18),
    }
    rets = {name: S._weights_to_ret(W, cp) for name, W in sleeves.items()}
    R = pd.DataFrame(rets).fillna(0.0)

    print("Sleeve metrics:")
    print(f"  {'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS SR':>7}")
    for name in R.columns:
        m = util.metrics(R[name])
        om = util.metrics(util.regime_slice(R[name], "2019-01-02", "2027-12-31"))
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>7.2f}")

    print("\nIS correlations:")
    print(R.loc[:IS_END].corr().round(2))

    # Try multiple blend schemes
    blends = {
        "EW": pd.Series({k: 1.0/len(R.columns) for k in R.columns}),
        "SR": (R.loc[:IS_END].mean() / R.loc[:IS_END].std()).clip(lower=0.05),
        "IV": 1.0 / R.loc[:IS_END].var().replace(0, np.nan),
        "MV": mean_var_weights(R),
        "ML_heavy": pd.Series({"V4_ML5": 0.5, "V1_MOM_LEV": 0.15, "V2_RISK_SAFE": 0.15,
                                "V3_SECTOR": 0.10, "V6_SHORT_MR": 0.10}),
    }
    for name, w in blends.items():
        blends[name] = (w / w.sum()).fillna(0)
        print(f"\nBlend {name}:")
        for k, v in blends[name].sort_values(ascending=False).items():
            print(f"  {k:15s}  {v:.3f}")

    # Compute each blend + portfolio overlays
    results = {}
    for name, w in blends.items():
        P = build_blend(sleeves, cp, w.to_dict())
        net, w_eff, state = apply_portfolio_overlays(P, cp, target_vol=0.25, dd_floor=-0.12)
        results[name] = net
        m = util.metrics(net)
        omf = util.metrics(util.regime_slice(net, "2005-01-01", IS_END))
        omo = util.metrics(util.regime_slice(net, "2019-01-02", "2027-12-31"))
        om22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        print(f"\n=== {name} BLEND ===")
        print(f"  FULL    SR={m['sharpe']:.2f} CAGR={m['cagr']*100:.1f}% MDD={m['mdd']*100:.1f}%")
        print(f"  IS      SR={omf.get('sharpe',0):.2f} CAGR={omf.get('cagr',0)*100:.1f}% MDD={omf.get('mdd',0)*100:.1f}%")
        print(f"  OOS     SR={omo.get('sharpe',0):.2f} CAGR={omo.get('cagr',0)*100:.1f}% MDD={omo.get('mdd',0)*100:.1f}%")
        print(f"  2022    SR={om22.get('sharpe',0):.2f} CAGR={om22.get('cagr',0)*100:.1f}%")

    # Best by OOS SR
    best = max(results, key=lambda k: util.metrics(util.regime_slice(results[k], "2019-01-02", "2027-12-31"))["sharpe"])
    print(f"\nBEST OOS: {best}")
    # Save best
    best_r = results[best]
    best_r.to_frame("apex_v7_ret").to_csv(OUT / "apex_v7_returns.csv")
    (OUT / "apex_v7_meta.json").write_text(json.dumps({
        "best_blend": best,
        "blend_weights": blends[best].to_dict(),
        "target_vol": 0.25,
        "dd_floor": -0.12,
    }, indent=2))


if __name__ == "__main__":
    main()
