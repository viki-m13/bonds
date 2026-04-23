"""Test Phoenix-exact sleeves individually."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import util
import sleeves_phoenix_exact as PX

OUT = Path("/home/user/bonds/data/apex")


def main():
    op, cp = util.load_prices()

    sleeves = {
        "PX_VANGUARD": PX.sleeve_vanguard_exact(cp),
        "PX_ORION":    PX.sleeve_orion_exact(cp),
        "PX_HELIOS":   PX.sleeve_helios_exact(cp),
        "PX_CRYPTO":   PX.sleeve_crypto_exact(cp),
    }

    # Macro gate stats
    gate = PX.compute_macro_gate(cp)
    print("Macro gate participation distribution:")
    print(gate.describe())
    print(f"  Days at 0% participation: {(gate < 0.01).sum()} / {len(gate)} ({(gate<0.01).mean()*100:.1f}%)")
    print(f"  Days at 100%: {(gate > 0.99).sum()} / {len(gate)} ({(gate>0.99).mean()*100:.1f}%)")
    print(f"  2008 gate mean: {gate.loc['2008-01-01':'2008-12-31'].mean():.3f}")
    print(f"  2022 gate mean: {gate.loc['2022-01-01':'2022-12-31'].mean():.3f}")
    print(f"  2020 Mar (COVID) gate mean: {gate.loc['2020-03-01':'2020-04-30'].mean():.3f}")

    rets = {}
    print(f"\n{'Sleeve':15s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'2008 MDD':>9}")
    for name, W in sleeves.items():
        r = PX._weights_to_ret(W, cp)
        rets[name] = r
        m = util.metrics(r)
        # 2008 specifically
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"mdd": 0}
        print(f"  {name:15s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {m08.get('mdd',0)*100:>8.1f}%")

    R = pd.DataFrame(rets).fillna(0.0)
    print("\nIS (2005-2018) correlations:")
    print(R.loc[:"2018-12-31"].corr().round(2))

    print("\nOOS (2019+) metrics:")
    for name in R.columns:
        r_oos = util.regime_slice(R[name], "2019-01-02", "2027-12-31")
        m = util.metrics(r_oos)
        print(f"  {name:15s}  SR={m.get('sharpe',0):>5.2f}  CAGR={m.get('cagr',0)*100:>5.1f}%  MDD={m.get('mdd',0)*100:>6.1f}%")

    print("\n2008 calendar year (critical stress):")
    for name in R.columns:
        r08 = util.regime_slice(R[name], "2008-01-01", "2008-12-31")
        m = util.metrics(r08)
        print(f"  {name:15s}  SR={m.get('sharpe',0):>5.2f}  CAGR={m.get('cagr',0)*100:>5.1f}%  MDD={m.get('mdd',0)*100:>6.1f}%")

    R.to_csv(OUT / "px_sleeve_returns.csv")


if __name__ == "__main__":
    main()
