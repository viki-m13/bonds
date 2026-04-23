"""Test each v12 sleeve individually."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import util
import sleeves_v12 as SV12

OUT = Path("/home/user/bonds/data/apex")


def main():
    op, cp = util.load_prices()
    # Need to extend cp universe to include SVXY, UUP, DBC if missing
    # Load them into cp if not present
    for t in ["SVXY", "UUP", "DBC", "VIXY", "USO"]:
        if t not in cp.columns:
            s = SV12._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    sleeves = {
        "SL_DUALBEAR":    SV12.sleeve_dualbear_defense(cp),
        "SL_REGIME":      SV12.sleeve_regime(cp),
        "SL_CALENDAR":    SV12.sleeve_calendar(cp),
        "SL_VRP":         SV12.sleeve_vrp(cp),
        "SL_DISPERSION":  SV12.sleeve_dispersion_mom(cp),
        "SL_CPPI":        SV12.sleeve_cppi(cp),
    }

    print(f"{'Sleeve':18s}  {'SR':>5}  {'CAGR':>7}  {'Vol':>6}  {'MDD':>7}  {'OOS':>5}  {'2008':>7}  {'2022':>7}")
    rets = {}
    for name, W in sleeves.items():
        r = SV12._weights_to_ret(W, cp)
        rets[name] = r
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
        r08 = util.regime_slice(r, "2008-01-01", "2008-12-31")
        m08 = util.metrics(r08) if len(r08) > 20 else {"sharpe": 0}
        r22 = util.regime_slice(r, "2022-01-01", "2022-12-31")
        m22 = util.metrics(r22) if len(r22) > 20 else {"sharpe": 0}
        print(f"  {name:18s}  {m['sharpe']:>5.2f}  {m['cagr']*100:>6.1f}%  "
              f"{m['vol']*100:>5.1f}%  {m['mdd']*100:>6.1f}%  {om.get('sharpe',0):>5.2f}  "
              f"{m08.get('sharpe',0):>7.2f}  {m22.get('sharpe',0):>7.2f}")

    R = pd.DataFrame(rets).fillna(0.0)
    print("\nIS correlations (05-18):")
    print(R.loc[:"2018-12-31"].corr().round(2))

    # Regime distribution
    print("\nRegime distribution (whole sample):")
    regime = SV12.classify_regime(cp)
    print(regime.value_counts(normalize=True).round(3))

    # Dual-bear score distribution
    dbs = SV12.dual_bear_score(cp)
    print("\nDual-bear score distribution:")
    print(dbs.value_counts(normalize=True).sort_index().round(3))
    print(f"DBS >= 3 days: {(dbs >= 3).sum()} ({(dbs>=3).mean()*100:.1f}%)")
    print(f"DBS >= 3 in 2022: {(dbs.loc['2022-01-01':'2022-12-31'] >= 3).mean()*100:.1f}%")
    print(f"DBS >= 3 in 2008: {(dbs.loc['2008-01-01':'2008-12-31'] >= 3).mean()*100:.1f}%")

    R.to_csv(OUT / "v12_sleeve_returns.csv")


if __name__ == "__main__":
    main()
