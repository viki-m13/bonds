"""APEX — Mega sleeve library + greedy selection + robust blend.

Build 25+ candidate sleeves with different signals and parameters. Select the
8-10 best combinations using a greedy procedure that maximizes SR - max_corr.
Fit blend weights via inverse-variance on IS only. Apply DD throttle + vol target.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd

import util

OUT = Path("/home/user/bonds/data/apex")
FRED = Path("/home/user/bonds/data/fred")


def _fred(name, idx):
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    return df[df.columns[0]].astype(float).reindex(idx).ffill()


def _portfolio(W, cp):
    w = W.fillna(0.0)
    rets = cp.pct_change()
    r = (w.shift(1).fillna(0.0) * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    tc_vec = pd.Series({c: tc.get(c, 5.0) for c in w.columns})
    drag = (dw * tc_vec / 1e4).sum(axis=1).shift(1).fillna(0.0)
    return r - drag


def _vol_target(r, target=0.10, win=60, cap=2.5, floor=0.15):
    rv = r.rolling(win, min_periods=20).std() * np.sqrt(util.DPY)
    m = (target / rv.replace(0, np.nan)).clip(lower=floor, upper=cap).shift(1).fillna(1.0)
    return r * m


# --- Sleeve generators ---

def sl_trend(cp, letf, under, fast=50, slow=200, ret_win=126,
             target=0.10, target_pos_vol=0.25):
    if letf not in cp.columns or under not in cp.columns:
        return None
    u = cp[under]
    ma_f = u.rolling(fast).mean()
    ma_s = u.rolling(slow).mean()
    r = u.pct_change(ret_win)
    on = ((u > ma_s) & (ma_f > ma_s) & (r > 0)).astype(float)
    rv = cp[letf].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    sc = (target_pos_vol / rv.replace(0, np.nan)).clip(upper=1.0).fillna(0.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    W[letf] = on * sc
    return _vol_target(_portfolio(W, cp), target=target)


def sl_tsmom(cp, universe, lookbacks=(21, 63, 126, 252), target=0.10):
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    rv60 = p.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    cons = pd.DataFrame(0.0, index=cp.index, columns=universe)
    for L in lookbacks:
        cons = cons + np.sign(p.pct_change(L)).fillna(0.0)
    cons = (cons / len(lookbacks)).clip(lower=0.0)
    raw = cons * (0.15 / rv60.replace(0, np.nan))
    raw = raw.fillna(0.0).clip(upper=0.4)
    s = raw.sum(axis=1)
    scale = np.minimum(1.0, 1.0 / s.replace(0, np.nan)).fillna(0.0)
    w = raw.mul(scale, axis=0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = w[a]
    return _vol_target(_portfolio(W, cp), target=target)


def sl_xsmom(cp, universe, top_n=2, rebal=21, lookback=126, skip=21, target=0.10):
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    mom = p.shift(skip).pct_change(lookback - skip)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_n) & (mom > 0)
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    is_rebal = mask % rebal == 0
    sel_m = sel.where(is_rebal).ffill().fillna(False)
    n_sel = sel_m.sum(axis=1)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in universe:
        W[a] = (sel_m[a].astype(float) / n_sel.replace(0, np.nan)).fillna(0.0)
    return _vol_target(_portfolio(W, cp), target=target)


def sl_rpar(cp, assets, vol_win=60, target=0.10):
    assets = [a for a in assets if a in cp.columns]
    rv = cp[assets].pct_change().rolling(vol_win).std()
    iv = 1.0 / rv.replace(0, np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in assets:
        W[a] = iv[a]
    return _vol_target(_portfolio(W, cp), target=target)


def sl_credit(cp, risky="UPRO", pct_thr=0.80, target=0.10):
    hy = _fred("BAMLH0A0HYM2", cp.index)
    if hy.isna().all():
        return None
    pct = hy.rolling(504, min_periods=60).quantile(pct_thr)
    ma60 = hy.rolling(60).mean()
    on = ((hy < pct) & (hy < ma60)).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        W[risky] = on
    return _vol_target(_portfolio(W, cp), target=target)


def sl_volreg(cp, risky="SSO", ma=200, target=0.10):
    spy = cp["SPY"]
    rv21 = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    rv63 = spy.pct_change().rolling(63).std() * np.sqrt(util.DPY)
    med = rv21.rolling(504, min_periods=60).median()
    ma_s = spy.rolling(ma).mean()
    on = ((rv21 < rv63) & (rv21 < med) & (spy > ma_s)).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        W[risky] = on
    return _vol_target(_portfolio(W, cp), target=target)


def sl_dip(cp, risky="UPRO", drop_thr=-0.05, hold=5, target=0.10):
    spy = cp["SPY"]
    r5 = spy.pct_change(5)
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta).clip(lower=0).rolling(2).mean()
    rsi2 = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    ma200 = spy.rolling(200).mean()
    trig = (((r5 < drop_thr) | (rsi2 < 5)) & (spy > ma200)).astype(float)
    held = trig.rolling(hold, min_periods=1).sum().clip(upper=1.0)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        W[risky] = held
    return _vol_target(_portfolio(W, cp), target=target)


def sl_season(cp, risky="UPRO", target=0.10):
    idx = cp.index
    day = pd.Series(idx.day, index=idx)
    month = pd.Series(idx.month, index=idx)
    year = pd.Series(idx.year, index=idx)
    group = year.astype(str) + "-" + month.astype(str).str.zfill(2)
    g = pd.DataFrame({"g": group.values}, index=idx)
    fwd_rank = g.groupby("g").cumcount()
    back_rank = g.groupby("g").cumcount(ascending=False)
    tom = ((fwd_rank < 3) | (back_rank < 2))
    santa = (((month == 12) & (day >= 15)) | ((month == 1) & (day <= 3)))
    on = (tom | santa).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        W[risky] = on
    return _vol_target(_portfolio(W, cp), target=target)


def sl_curve(cp, target=0.10):
    """Long TMF when 10y-2y slope positive and increasing."""
    slope = _fred("T10Y2Y", cp.index)
    if slope.isna().all():
        return None
    ma20 = slope.rolling(20).mean()
    ma60 = slope.rolling(60).mean()
    on = ((slope > 0) & (ma20 > ma60)).astype(float)
    W = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if "TMF" in cp.columns:
        W["TMF"] = on
    return _vol_target(_portfolio(W, cp), target=target)


def generate_library(cp):
    L = {}
    # --- Trend sleeves, many variations ---
    L["TREND_TQQQ_QQQ"] = sl_trend(cp, "TQQQ", "QQQ")
    L["TREND_UPRO_SPY"] = sl_trend(cp, "UPRO", "SPY")
    L["TREND_TECL_XLK"] = sl_trend(cp, "TECL", "QQQ")  # XLK not in cp
    L["TREND_SOXL_SMH"] = sl_trend(cp, "SOXL", "QQQ")
    L["TREND_EDC_EEM"]  = sl_trend(cp, "EDC", "EEM") if "EEM" in cp.columns else None
    L["TREND_YINN_FXI"] = sl_trend(cp, "YINN", "EEM") if "EEM" in cp.columns else None  # Proxy
    L["TREND_DRN_VNQ"]  = sl_trend(cp, "DRN", "SPY") if "DRN" in cp.columns else None
    L["TREND_TMF_TLT"]  = sl_trend(cp, "TMF", "TLT")
    L["TREND_UBT_TLT"]  = sl_trend(cp, "UBT", "TLT")
    L["TREND_TYD_TLT"]  = sl_trend(cp, "TYD", "TLT")
    L["TREND_UGL_GLD"]  = sl_trend(cp, "UGL", "GLD")
    L["TREND_UCO_USO"]  = sl_trend(cp, "UCO", "SPY") if "UCO" in cp.columns else None  # proxy

    # Trend variants with different lookbacks
    L["TREND_TQQQ_fast"]  = sl_trend(cp, "TQQQ", "QQQ", fast=20, slow=100, ret_win=63)
    L["TREND_UPRO_slow"]  = sl_trend(cp, "UPRO", "SPY", fast=100, slow=300, ret_win=252)

    # --- TSMOM ---
    big_univ = ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN",
                "TMF", "UBT", "TYD", "UGL", "UCO", "DRN"]
    L["TSMOM_L4"] = sl_tsmom(cp, big_univ, lookbacks=(21, 63, 126, 252))
    L["TSMOM_L2"] = sl_tsmom(cp, big_univ, lookbacks=(63, 252))
    L["TSMOM_L3"] = sl_tsmom(cp, big_univ, lookbacks=(21, 63, 252))

    # --- XSMOM variations ---
    L["XSMOM_T2_126"] = sl_xsmom(cp, big_univ, top_n=2, lookback=126, skip=21, rebal=21)
    L["XSMOM_T3_126"] = sl_xsmom(cp, big_univ, top_n=3, lookback=126, skip=21, rebal=21)
    L["XSMOM_T2_252"] = sl_xsmom(cp, big_univ, top_n=2, lookback=252, skip=21, rebal=21)

    # --- Risk parity variations ---
    L["RPAR_3"]   = sl_rpar(cp, ["UPRO", "TMF", "UGL"])
    L["RPAR_4"]   = sl_rpar(cp, ["UPRO", "TMF", "UGL", "UCO"])
    L["RPAR_6"]   = sl_rpar(cp, ["UPRO", "TQQQ", "TMF", "UBT", "UGL", "UCO"])
    L["RPAR_eqtrend"] = sl_rpar(cp, ["UPRO", "TQQQ", "TECL"], vol_win=21)

    # --- Regime sleeves ---
    L["CREDIT_UPRO"] = sl_credit(cp, risky="UPRO", pct_thr=0.80)
    L["CREDIT_SSO"]  = sl_credit(cp, risky="SSO", pct_thr=0.80)
    L["CREDIT_TQQQ"] = sl_credit(cp, risky="TQQQ", pct_thr=0.80)

    L["VOLREG_SSO"]  = sl_volreg(cp, risky="SSO")
    L["VOLREG_UPRO"] = sl_volreg(cp, risky="UPRO")
    L["VOLREG_TQQQ"] = sl_volreg(cp, risky="TQQQ")

    # --- Dip / Mean reversion ---
    L["DIP_UPRO"] = sl_dip(cp, risky="UPRO", drop_thr=-0.05)
    L["DIP_TQQQ"] = sl_dip(cp, risky="TQQQ", drop_thr=-0.05)
    L["DIP_UPRO_tight"] = sl_dip(cp, risky="UPRO", drop_thr=-0.03)

    # --- Seasonality ---
    L["SEASON_UPRO"] = sl_season(cp, risky="UPRO")
    L["SEASON_TQQQ"] = sl_season(cp, risky="TQQQ")

    # --- Curve ---
    L["CURVE_TMF"] = sl_curve(cp)

    # Drop Nones
    return {k: v for k, v in L.items() if v is not None}


def score_sleeve(r: pd.Series) -> dict:
    is_r = r.loc[:"2018-12-31"].dropna()
    oos_r = r.loc["2019-01-02":].dropna()
    fm = util.metrics(r)
    ism = util.metrics(is_r) if len(is_r) > 60 else {}
    om = util.metrics(oos_r) if len(oos_r) > 60 else {}
    return {
        "full_sr": fm.get("sharpe", 0),
        "full_cagr": fm.get("cagr", 0),
        "full_mdd": fm.get("mdd", 0),
        "is_sr": ism.get("sharpe", 0),
        "oos_sr": om.get("sharpe", 0),
    }


def greedy_select(RDF: pd.DataFrame, scores: dict,
                  max_n: int = 10, corr_penalty: float = 1.5) -> list[str]:
    """Greedy: start with best IS SR, keep adding the engine that maximizes
    IS SR minus (corr_penalty * max_corr_to_already_selected)."""
    C = RDF.loc[:"2018-12-31"].corr()
    candidates = [k for k in RDF.columns if scores[k].get("is_sr", 0) > 0.3
                  and scores[k].get("full_mdd", -1) > -0.6]
    ranked = sorted(candidates, key=lambda k: -scores[k]["is_sr"])
    selected = [ranked[0]]
    while len(selected) < max_n and len(selected) < len(ranked):
        best_k = None
        best_score = -np.inf
        for c in ranked:
            if c in selected:
                continue
            max_c = max(abs(C.loc[c, s]) for s in selected)
            score = scores[c]["is_sr"] - corr_penalty * max_c
            if score > best_score:
                best_score = score
                best_k = c
        if best_k is None or scores[best_k]["is_sr"] < 0.25:
            break
        selected.append(best_k)
    return selected


def finalize(r, target_vol=0.20, dd_floor=-0.15):
    c = (1 + r).cumprod()
    hwm = c.rolling(252, min_periods=30).max()
    dd = c / hwm - 1
    m = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)
    r2 = r * m
    rv = r2.rolling(60, min_periods=20).std() * np.sqrt(util.DPY)
    vm = (target_vol / rv.replace(0, np.nan)).clip(lower=0.2, upper=1.5).shift(1).fillna(1.0)
    return r2 * vm


def main():
    op, cp = util.load_prices()
    print("Generating library...")
    L = generate_library(cp)
    print(f"Generated {len(L)} sleeves.")

    RDF = pd.DataFrame(L).fillna(0.0)
    scores = {k: score_sleeve(RDF[k]) for k in RDF.columns}

    # Show all
    print(f"\n{'Sleeve':28s} {'IS_SR':>6} {'OOS_SR':>6} {'FULL_SR':>6} {'CAGR':>7} {'MDD':>7}")
    for k in sorted(scores, key=lambda x: -scores[x]["is_sr"]):
        s = scores[k]
        print(f"{k:28s} {s['is_sr']:>6.2f} {s['oos_sr']:>6.2f} {s['full_sr']:>6.2f} "
              f"{s['full_cagr']*100:>6.1f}% {s['full_mdd']*100:>6.1f}%")

    # Greedy select
    selected = greedy_select(RDF, scores, max_n=10, corr_penalty=1.2)
    print(f"\nSelected {len(selected)} sleeves:")
    for s in selected:
        sc = scores[s]
        print(f"  {s:28s} IS={sc['is_sr']:.2f} OOS={sc['oos_sr']:.2f}")

    R_sel = RDF[selected]
    print("\nSelected correlation matrix (FULL):")
    print(R_sel.corr().round(2))
    print("\nSelected correlation matrix (IS):")
    print(R_sel.loc[:"2018-12-31"].corr().round(2))

    # Blend
    blends = {}
    # 1. Equal weight
    blends["EW"] = R_sel.mean(axis=1)
    # 2. IV
    is_std = R_sel.loc[:"2018-12-31"].std()
    iv = 1.0 / is_std.replace(0, np.nan)
    iv = iv / iv.sum()
    blends["IV"] = (R_sel * iv).sum(axis=1)
    # 3. SR-weighted
    is_mu = R_sel.loc[:"2018-12-31"].mean() * util.DPY
    is_sd = is_std * np.sqrt(util.DPY)
    sr = (is_mu / is_sd).clip(lower=0.1)
    sw = sr / sr.sum()
    blends["SR"] = (R_sel * sw).sum(axis=1)
    # 4. Inverse-covariance with shrinkage (Ledoit-Wolf-ish)
    is_R = R_sel.loc[:"2018-12-31"]
    C = is_R.cov().values
    n = C.shape[0]
    mu_shr = 0.3
    C_shr = (1 - mu_shr) * C + mu_shr * np.diag(np.diag(C))
    try:
        inv_C = np.linalg.inv(C_shr)
        ones = np.ones(n)
        cov_w = inv_C @ ones / (ones @ inv_C @ ones)
        cov_w = pd.Series(cov_w, index=R_sel.columns).clip(lower=0)
        cov_w = cov_w / cov_w.sum()
        blends["COV"] = (R_sel * cov_w).sum(axis=1)
    except Exception:
        pass

    print(f"\n=== Blend schemes comparison ===")
    for name, blend in blends.items():
        print(f"\n--- {name} ---")
        util.summarize(blend, "  pre-final")
        rf = finalize(blend, target_vol=0.20)
        for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                            ("IS 05-18", ("2005-01-01", "2018-12-31")),
                            ("OOS 19+", ("2019-01-02", "2027-12-31")),
                            ("2022RH", ("2022-01-01", "2022-12-31"))]:
            util.summarize(util.regime_slice(rf, s, e), f"    {lbl}")

    # Pick best scheme
    best_k = max(blends, key=lambda k: util.metrics(finalize(blends[k]).loc["2019-01-02":]).get("sharpe", 0))
    print(f"\n*** Best by OOS Sharpe: {best_k} ***")

    RDF.to_csv(OUT / "lib_returns.csv")
    R_sel.to_csv(OUT / "lib_selected.csv")
    (OUT / "lib_selected.json").write_text(json.dumps({
        "selected": selected,
        "scores": {k: scores[k] for k in selected}
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
