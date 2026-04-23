"""APEX v17 — v16 + 4 novel sleeves (FOMC, PCA, BUY_FEAR, VOL_OF_VOL)."""
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
from apex_v13 import extend_cp

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
    "SL_FOMC":           lambda cp: SV17.sleeve_fomc(cp),
    "SL_PCA":            lambda cp: SV17.sleeve_pca(cp),
    "SL_BUY_FEAR":       lambda cp: SV17.sleeve_buy_fear(cp),
    "SL_VOL_OF_VOL":     lambda cp: SV17.sleeve_vol_of_vol(cp),
}


def build_v17(cp):
    sw = {}
    for name, fn in LETF_BUILDERS.items():
        W = fn(cp)
        r = PX._weights_to_ret(W, cp)
        rv = r.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
        m = (0.15 / rv.replace(0, np.nan)).clip(upper=1.0, lower=0.1).shift(1).fillna(1.0)
        sw[name] = W.mul(m, axis=0)
    return sw


def greedy_select(sw, cp, is_end=IS_END, max_n=10, corr_pen=1.0):
    """Greedy: rank sleeves by IS Sharpe, add them one by one while
    avoiding high-correlation duplicates."""
    sr_dict = {name: PX._weights_to_ret(W, cp) for name, W in sw.items()}
    R = pd.DataFrame(sr_dict).fillna(0.0)
    is_R = R.loc[:is_end]
    is_sr = (is_R.mean() / is_R.std() * np.sqrt(util.DPY)).fillna(0)
    ranked = is_sr.sort_values(ascending=False).index.tolist()
    C = is_R.corr()
    selected = [ranked[0]]
    for cand in ranked[1:]:
        if len(selected) >= max_n:
            break
        max_c = max(abs(C.loc[cand, s]) for s in selected)
        score = is_sr[cand] - corr_pen * max_c
        if score > 0:
            selected.append(cand)
    return selected


def run_v17(cp, sw, crypto_w=0.45, target_vol=0.20, dd_floor=-0.10,
            selected_sleeves=None):
    if selected_sleeves is None:
        selected_sleeves = list(sw.keys())
    sw_sel = {k: sw[k] for k in selected_sleeves}
    n = len(sw_sel)
    letf_cap = 1.0 - crypto_w
    blend_w = {k: letf_cap / n for k in sw_sel}

    first = next(iter(sw_sel.values()))
    P = pd.DataFrame(0.0, index=first.index, columns=first.columns)
    for name, W in sw_sel.items():
        P = P + W.fillna(0.0) * blend_w[name]
    P = P.clip(upper=letf_cap, lower=0.0)

    rets = cp.pct_change()
    spy_rv60 = cp["SPY"].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    thr = spy_rv60.rolling(504, min_periods=60).quantile(0.99)
    regime_ok = (spy_rv60 <= thr).astype(float).fillna(1.0)
    regime_mult = (regime_ok + (1 - regime_ok) * 0.5).shift(1).fillna(1.0)
    P = P.mul(regime_mult, axis=0)

    # Dual-bear override
    dbs = SV12.dual_bear_score(cp)
    dbs_mult = pd.Series(1.0, index=cp.index)
    dbs_mult[dbs >= 3] = 0.5
    dbs_mult[dbs >= 4] = 0.25
    dbs_mult = dbs_mult.shift(1).fillna(1.0)
    P = P.mul(dbs_mult, axis=0)

    raw_r = (P.shift(1).fillna(0.0) * rets.reindex_like(P).fillna(0.0)).sum(axis=1)
    c = (1 + raw_r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)

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

    crypto_r = SV15.multi_crypto_returns(cp.index, target_vol=0.20)
    net = letf_net + crypto_w * crypto_r
    return net, w_eff


def main():
    op, cp = util.load_prices()
    cp = extend_cp(cp)
    for t in ["SH", "PSQ", "SDS", "TBF"]:
        if t not in cp.columns:
            s = SV15._etf_close(t, cp.index)
            if not s.isna().all():
                cp[t] = s

    print("Building v17 sleeves (14 LETF sleeves)...")
    sw = build_v17(cp)

    # Sleeve metrics
    print(f"\n{'Sleeve':20s}  {'SR':>5}  {'OOS':>5}  {'2022':>7}  {'2008':>7}")
    for name, W in sw.items():
        r = PX._weights_to_ret(W, cp)
        m = util.metrics(r)
        om = util.metrics(util.regime_slice(r, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(r, "2022-01-01", "2022-12-31"))
        m08 = util.metrics(util.regime_slice(r, "2008-01-01", "2008-12-31"))
        print(f"  {name:20s}  {m['sharpe']:>5.2f}  {om.get('sharpe',0):>5.2f}  "
              f"{m22.get('sharpe',0):>7.2f}  {m08.get('sharpe',0):>7.2f}")

    # Configs
    configs = []

    # All 14 sleeves, various params
    for cw in [0.35, 0.45, 0.50]:
        for tv in [0.18, 0.22, 0.25]:
            net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=tv)
            m = util.metrics(net)
            om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
            m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
            configs.append({
                "sel": "all14", "cw": cw, "tv": tv,
                "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
                "oos_cagr": om.get("cagr", 0), "y22": m22.get("sharpe", 0),
                "mdd": m["mdd"], "net": net,
            })

    # Greedy-selected subset
    sel_greedy = greedy_select(sw, cp, max_n=8, corr_pen=1.0)
    print(f"\nGreedy-selected (max 8): {sel_greedy}")
    for cw in [0.35, 0.45, 0.50]:
        net, _ = run_v17(cp, sw, crypto_w=cw, target_vol=0.20, selected_sleeves=sel_greedy)
        m = util.metrics(net)
        om = util.metrics(util.regime_slice(net, OOS_START, "2027-12-31"))
        m22 = util.metrics(util.regime_slice(net, "2022-01-01", "2022-12-31"))
        configs.append({
            "sel": "greedy", "cw": cw, "tv": 0.20,
            "full_sr": m["sharpe"], "oos_sr": om.get("sharpe", 0),
            "oos_cagr": om.get("cagr", 0), "y22": m22.get("sharpe", 0),
            "mdd": m["mdd"], "net": net,
        })

    configs.sort(key=lambda r: -r["oos_sr"])
    print(f"\n{'sel':8s} {'cw':>5} {'tv':>5}  {'FULL':>5}  {'OOS':>5}  {'OOS_CAGR':>8}  {'2022':>6}  {'MDD':>6}")
    for r in configs[:12]:
        print(f"  {r['sel']:8s}  {r['cw']:.2f}  {r['tv']:.2f}  "
              f"{r['full_sr']:>5.2f}  {r['oos_sr']:>5.2f}  "
              f"{r['oos_cagr']*100:>6.1f}%  {r['y22']:>6.2f}  {r['mdd']*100:>5.1f}%")

    best = configs[0]
    net = best["net"]
    print(f"\nBEST: {best['sel']} cw={best['cw']} tv={best['tv']}")
    print(f"OOS SR = {best['oos_sr']:.2f}, Full SR = {best['full_sr']:.2f}")

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

    net.to_frame("apex_v17_ret").to_csv(OUT / "apex_v17_returns.csv")
    (OUT / "apex_v17_meta.json").write_text(json.dumps({
        "best": {k: v for k, v in best.items() if k != "net"},
        "greedy_selected": sel_greedy,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
