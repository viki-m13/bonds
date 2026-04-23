"""Test each v6 sleeve individually — we want SR >=0.8 and corr <0.3."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import util
import sleeves_v6 as S

OUT = Path("/home/user/bonds/data/apex")

SLEEVES = {
    "V1_MOM_LEV":  S.s_v1_mom_lev,
    "V2_RISK_SAFE": S.s_v2_risk_safe,
    "V3_SECTOR":    S.s_v3_sector,
    "V4_ML5":      S.s_v4_ml5,
    "V5_ML63":     S.s_v5_ml63,
    "V6_SHORT_MR": S.s_v6_short_mr,
    "V7_CALENDAR": S.s_v7_calendar,
    "V8_CURVE":    S.s_v8_curve,
    "V9_VOL_REG":  S.s_v9_vol_regime,
}


def main():
    op, cp = util.load_prices()
    rets = {}
    weights = {}
    for name, fn in SLEEVES.items():
        W = fn(cp, target_vol=0.18)
        r = S._weights_to_ret(W, cp)
        rets[name] = r
        weights[name] = W
        m = util.metrics(r)
        print(f"  {name:15s}  SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>5.1f}%  Vol={m['vol']*100:>5.1f}%  MDD={m['mdd']*100:>6.1f}%  NAV={m['nav']:>6.1f}x")
    R = pd.DataFrame(rets)
    print("\nFull-sample correlations:")
    print(R.corr().round(2))
    print("\nIS (05-18) correlations:")
    print(R.loc[:"2018-12-31"].corr().round(2))
    # OOS metrics
    print("\nOOS (2019+) metrics:")
    for name in R.columns:
        oos = util.regime_slice(R[name], "2019-01-02", "2027-12-31")
        m = util.metrics(oos)
        print(f"  {name:15s}  SR={m.get('sharpe',0):>5.2f}  CAGR={m.get('cagr',0)*100:>5.1f}%  MDD={m.get('mdd',0)*100:>6.1f}%")
    R.to_csv(OUT / "v6_sleeve_returns.csv")


if __name__ == "__main__":
    main()
