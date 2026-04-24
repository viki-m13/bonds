"""APEX v20 — 25 LETF sleeves, truly orthogonal mix."""
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
import sleeves_v15 as SV15
import sleeves_v16 as SV16
import sleeves_v17 as SV17
import sleeves_v18 as SV18
import sleeves_v19 as SV19
import sleeves_v20 as SV20
from apex_v13 import extend_cp
from apex_v17 import run_v17, greedy_select

OUT = Path("/home/user/bonds/data/apex")
IS_END = "2018-12-31"
OOS_START = "2019-01-02"


LETF_BUILDERS = {
    "PX_VANGUARD":       lambda cp: PX.sleeve_vanguard_exact(cp),
    "PX_ORION":          lambda cp: PX.sleeve_orion_exact(cp),
    "PX_HELIOS":         lambda cp: PX.sleeve_helios_exact(cp),
    "SL_DUALBEAR":       lambda cp: SV12.sleeve_dualbear_defense(cp),
    "SL_CALENDAR":       lambda cp: SV12.sleeve_calendar(cp),
    "SL_VRP":            lambda cp: SV12.sleeve_vrp(cp),
    "SL_INVERSE":        lambda cp: SV15.sleeve_inverse(cp),
    "SL_CRASH_CONTRA":   lambda cp: SV16.sleeve_crash_contrarian(cp),
    "SL_REAL_YIELD":     lambda cp: SV16.sleeve_real_yield(cp),
    "SL_CROSS_DECORR":   lambda cp: SV16.sleeve_cross_decorr(cp),
    "SL_PCA":            lambda cp: SV17.sleeve_pca(cp),
    "SL_VOL_OF_VOL":     lambda cp: SV17.sleeve_vol_of_vol(cp),
    "SL_BREAKOUT52":     lambda cp: SV18.sleeve_breakout52(cp),
    "SL_HMM":            lambda cp: SV18.sleeve_hmm(cp),
    "SL_DIVERGENCE":     lambda cp: SV18.sleeve_divergence(cp),
    "SL_VOLZ_MOM":       lambda cp: SV19.sleeve_volz_mom(cp),
    "SL_RANGE_BREAK":    lambda cp: SV19.sleeve_range_break(cp),
    "SL_VIX_EXPAND":     lambda cp: SV19.sleeve_vix_expand(cp),
    "SL_RATE_MOM":       lambda cp: SV20.sleeve_rate_momentum(cp),
    "SL_USD_STRENGTH":   lambda cp: SV20.sleeve_usd_strength(cp),
    "SL_SMART_MONEY":    lambda cp: SV20.sleeve_smart_money(cp),
    "SL_CREDIT_TIGHT":   lambda cp: SV20.sleeve_credit_tight(cp),
    "SL_COMMOD_REGIME":  lambda cp: SV20.sleeve_commodity_regime(cp),
    "SL_BOND_FLIGHT":    lambda cp: SV20.sleeve_bond_flight(cp),
}


def build_v20(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF", "UUP", "DBC", "HYG"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    print(f"Building v20 sleeves ({len(LETF_BUILDERS)} LETF sleeves)...")
    sw = build_v20(cp)

    # Correlation analysis
    rets = {name: PX._weights_to_ret(W, cp) for name, W in sw.items()}
    R = pd.DataFrame(rets).fillna(0.0)
    Corr = R.corr()

    # Avg correlation per sleeve
    avg_corr = {}
    for name in R.columns:
        others = [c for c in R.columns if c != name]
        avg_corr[name] = abs(Corr.loc[name, others]).mean()
    print(f"\nAverage sleeve correlation: {np.mean(list(avg_corr.values())):.3f}")

    print(f"\n{'Sleeve':20s}  {'SR':>5}  {'OOS':>5}  {'2022':>7}  {'2008':>7}  {'AvgCorr':>7}")
    for name, W in sw.items():
        r = rets[name]
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(r, "2022-01-01", "2022-12-31"))
        m08 = util.metrics(util.regime_slice(r, "2008-01-01", "2008-12-31"))
        print(f"  {name:20s}  {m['sharpe']:>5.2f}  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}  "
              f"{avg_corr[name]:>7.3f}")

    # Configurations
    configs = []
    # All 24 EW
    for cw in [0.45, 0.50, 0.55, 0.60]:
        net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=0.18)
        m = util.metrics(net)
        om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        configs.append({"sel": "all24", "cw": cw,
                        "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                        "oos_cagr": om.get("cagr", 0), "y22": m22.get("sharpe", 0),
                        "mdd": m["mdd"], "net": net})

    # Drop lowest-OOS sleeves (those below 0.3 OOS SR)
    is_sr_vec = R.loc[:IS_END].mean() / R.loc[:IS_END].std() * np.sqrt(util.DPY)
    strong = [k for k in sw.keys() if is_sr_vec[k] > 0.2]
    print(f"\nStrong sleeves (IS SR > 0.2): {len(strong)}: {strong}")
    for cw in [0.45, 0.50, 0.55]:
        net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=0.20, selected_sleeves=strong)
        m = util.metrics(net)
        om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        configs.append({"sel": "strong", "cw": cw,
                        "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                        "oos_cagr": om.get("cagr", 0), "y22": m22.get("sharpe", 0),
                        "mdd": m["mdd"], "net": net})

    # Top-12 by IS SR
    top12 = is_sr_vec.sort_values(ascending=False).head(12).index.tolist()
    print(f"\nTop-12 by IS SR: {top12}")
    for cw in [0.45, 0.50, 0.55]:
        net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=0.20, selected_sleeves=top12)
        m = util.metrics(net)
        om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        configs.append({"sel": "top12", "cw": cw,
                        "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                        "oos_cagr": om.get("cagr", 0), "y22": m22.get("sharpe", 0),
                        "mdd": m["mdd"], "net": net})

    configs.sort(key=lambda r: -r["oos_sr"])
    print(f"\n{'sel':8s} {'cw':>5}  {'FULL':>5}  {'OOS':>5}  {'OOS_CAGR':>8}  {'2022':>6}  {'MDD':>6}")
    for r in configs[:15]:
        print(f"  {r['sel']:8s}  {r['cw']:.2f}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['oos_cagr']*100:>6.1f}%  {r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%")

    best = configs[0]
    net = best["net"]
    print(f"\nBEST: {best['sel']} cw={best['cw']} OOS SR {best['oos_sr']:.2f}")
    print("\n=== BEST DETAIL ===")
    for lbl, (s, e) in [("FULL 99-26", ("1999-01-01", "2027-12-31")),
                        ("Phoenix window 10-26", ("2010-03-11", "2027-12-31")),
                        ("IS 10-18", ("2010-03-11", IS_END)),
                        ("OOS 19+", (OOS_START, "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("2008 cal", ("2008-01-01", "2008-12-31")),
                        ("GFC 07-09", ("2007-01-01", "2009-12-31")),
                        ("COVID 20", ("2020-01-01", "2020-12-31")),
                        ("2022", ("2022-01-01", "2022-12-31")),
                        ("2023-24", ("2023-01-01", "2024-12-31")),
                        ("2025+", ("2025-01-01", "2027-12-31"))]:
        util.summarize(util.regime_slice(net, s, e), f"  {lbl}")

    net.to_frame("apex_v20_ret").to_csv(OUT / "apex_v20_returns.csv")
    (OUT / "apex_v20_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"},
        "sleeves": list(sw.keys()),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
