"""Test the new v15 sleeves individually."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import util
import sleeves_v15 as SV15


def main():
    op, cp = util.load_prices()
    # Extend cp with inverse ETFs + crypto
    for t in ["SH", "PSQ", "SDS", "TBF", "SQQQ", "SPXU"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    # Sleeve 1: Inverse
    W_inv = SV15.sleeve_inverse(cp)
    r_inv = SV15._weights_to_ret(W_inv, cp)

    # Sleeve 2: Multi-crypto
    r_crypto = SV15.multi_crypto_returns(cp.index, target_vol=0.20)

    # Sleeve 3: Walk-forward ML (slow, skip for now)
    # W_wf = SV15.sleeve_wf_ml(cp)
    # r_wf = SV15._weights_to_ret(W_wf, cp)

    rets = {
        "INVERSE": r_inv,
        "MULTI_CRYPTO": r_crypto,
    }
    print(f"{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, r in rets.items():
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    R = pd.DataFrame(rets).fillna(0.0)
    print("\nCorrelation:")
    print(R.corr().round(2))

    # Save
    R.to_csv("/home/user/bonds/data/apex/v15_sleeve_returns.csv")


if __name__ == "__main__":
    main()
