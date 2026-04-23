"""APEX — Big experiment: exhaustively test many simple engines, find the
uncorrelated ones with highest IS Sharpe, optimize the blend.

Approach:
  1. Generate ~30 simple engines parametrized by (asset, signal, lookback).
  2. Compute IS (2005-2018) metrics for each.
  3. Rank by IS Sharpe. Pick top 8-10 with low mutual correlation.
  4. Inverse-variance blend → vol-target → DD-throttle → final.
  5. Report IS/OOS metrics.
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


def _weights_with_cash(w, cp):
    cash = "SHY" if "SHY" in cp.columns else "BIL"
    w = w.copy()
    used = w.sum(axis=1)
    # When used < 1, stuff goes to cash
    w[cash] = (1 - used.clip(upper=1.0)).clip(lower=0.0)
    return w


def _portfolio_return(w, rets, cp):
    """Compute simple portfolio return with TC drag."""
    w = _weights_with_cash(w.fillna(0.0), cp)
    w_eff = w.shift(1).fillna(0.0)
    r = (w_eff * rets.reindex_like(w).fillna(0.0)).sum(axis=1)
    tc = util.tc_map()
    dw = w.diff().abs().fillna(w.abs())
    drag = sum(dw[c] * tc.get(c, 5.0) / 1e4 for c in w.columns if c in tc or True)
    drag = drag.shift(1).fillna(0.0)
    return r - drag


def eng_trend(letf: str, under: str, cp: pd.DataFrame,
              fast: int = 50, slow: int = 200, ret_win: int = 126) -> pd.DataFrame:
    """Binary trend: long LETF if underlying trend-on; cash otherwise."""
    u = cp[under]
    ma_s = u.rolling(slow).mean()
    ma_f = u.rolling(fast).mean()
    r = u.pct_change(ret_win)
    on = ((u > ma_s) & (ma_f > ma_s) & (r > 0)).astype(float)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w[letf] = on
    return w


def eng_trend_voltarget(letf: str, under: str, cp: pd.DataFrame,
                        target: float = 0.20,
                        fast: int = 50, slow: int = 200, ret_win: int = 126) -> pd.DataFrame:
    """Trend-on + vol scaling: weight = min(1, target_vol / LETF_vol)."""
    u = cp[under]
    ma_s = u.rolling(slow).mean()
    ma_f = u.rolling(fast).mean()
    r = u.pct_change(ret_win)
    on = ((u > ma_s) & (ma_f > ma_s) & (r > 0)).astype(float)
    rv = cp[letf].pct_change().rolling(60).std() * np.sqrt(util.DPY)
    sc = (target / rv).clip(upper=1.0).fillna(0.0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w[letf] = on * sc
    return w


def eng_dm_topn(universe: list[str], cp: pd.DataFrame,
                top_n: int = 2, lookback: int = 126, skip: int = 21) -> pd.DataFrame:
    """Dual momentum top-N (positive momentum required), equal-weight."""
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    mom = p.shift(skip).pct_change(lookback - skip)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_n) & (mom > 0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    n = sel.sum(axis=1)
    for a in universe:
        w[a] = (sel[a].astype(float) / n.replace(0, np.nan)).fillna(0.0)
    return w


def eng_carry_hedge(cp: pd.DataFrame) -> pd.DataFrame:
    """50% UPRO + 50% TMF rebalanced (HFEA classic)."""
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["UPRO"] = 0.5
    w["TMF"] = 0.5
    return w


def eng_min_variance_dynamic(cp: pd.DataFrame,
                              assets: list[str] = None, lookback: int = 60,
                              max_weight: float = 0.5) -> pd.DataFrame:
    """Minimum-variance portfolio from inverse-vol weighting (diagonal cov)."""
    if assets is None:
        assets = ["UPRO", "TMF", "UGL"]
    assets = [a for a in assets if a in cp.columns]
    rv = cp[assets].pct_change().rolling(lookback).std()
    iv = 1.0 / rv.replace(0, np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    iv = iv.clip(upper=max_weight)
    iv = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in assets:
        w[a] = iv[a]
    return w


def eng_cred_regime(cp: pd.DataFrame, risky: str = "UPRO") -> pd.DataFrame:
    """Long `risky` when HY spread below 504d median and 60d-trailing decline."""
    hy = _fred("BAMLH0A0HYM2", cp.index)
    med = hy.rolling(504, min_periods=60).median()
    ma60 = hy.rolling(60).mean()
    on = ((hy < med) & (hy < ma60)).astype(float)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        w[risky] = on
    return w


def eng_vix_regime(cp: pd.DataFrame, risky: str = "SSO") -> pd.DataFrame:
    """Long `risky` when SPY 21d RV < 15% AND RV < RV_63d.

    Proxy for VIX term-structure: low + decreasing realized vol ~ VIX in
    contango → short-vol / long-risk works.
    """
    spy = cp["SPY"]
    rv21 = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    rv63 = spy.pct_change().rolling(63).std() * np.sqrt(util.DPY)
    on = ((rv21 < 0.15) & (rv21 < rv63)).astype(float)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        w[risky] = on
    return w


def eng_season(cp: pd.DataFrame, risky: str = "UPRO", days: int = 5) -> pd.DataFrame:
    """Turn-of-month effect: long equity last 2 trading days + first 3 of month."""
    idx = cp.index
    s = pd.Series(0.0, index=idx)
    for date in idx:
        # Is this in the "turn of month" window?
        day_of_month = date.day
        # last 2 days: use month-end group
        # For simplicity: days 1-3 or 28+
        if day_of_month <= 3 or day_of_month >= 27:
            s[date] = 1.0
    w = pd.DataFrame(0.0, index=idx, columns=cp.columns)
    if risky in cp.columns:
        w[risky] = s
    return w


def eng_inv_vol_trio(cp: pd.DataFrame, sleeves: list[str] = None,
                     vol_win: int = 60) -> pd.DataFrame:
    """Inverse-vol across arbitrary sleeves (dynamic risk parity)."""
    if sleeves is None:
        sleeves = ["UPRO", "TMF", "UGL"]
    sleeves = [a for a in sleeves if a in cp.columns]
    rv = cp[sleeves].pct_change().rolling(vol_win).std()
    iv = 1.0 / rv.replace(0, np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    for a in sleeves:
        w[a] = iv[a].fillna(0.0)
    return w


def eng_mrev_dip(cp: pd.DataFrame, risky: str = "UPRO",
                 drop_thr: float = -0.05, hold: int = 5) -> pd.DataFrame:
    """After a sharp 5d drop in SPY (< drop_thr), buy risky for `hold` days."""
    r5 = cp["SPY"].pct_change(5)
    trig = (r5 < drop_thr).astype(float)
    held = trig.rolling(hold, min_periods=1).sum().clip(upper=1.0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    if risky in cp.columns:
        w[risky] = held
    return w


def all_engines(cp):
    engines = {}
    # Trend variants
    for letf, under in [("TQQQ", "QQQ"), ("UPRO", "SPY"),
                         ("TECL", "XLK") if "XLK" in cp.columns else ("TECL", "QQQ"),
                         ("SOXL", "SMH") if "SMH" in cp.columns else ("SOXL", "QQQ"),
                         ("EDC", "EEM"),
                         ("TMF", "TLT"),
                         ("TYD", "IEF"),
                         ("UGL", "GLD"),
                         ("DRN", "VNQ") if "VNQ" in cp.columns else ("DRN", "SPY")]:
        if letf in cp.columns and under in cp.columns:
            engines[f"TR_{letf}"] = lambda cp, l=letf, u=under: eng_trend(l, u, cp)
            engines[f"TRV_{letf}"] = lambda cp, l=letf, u=under: eng_trend_voltarget(l, u, cp, target=0.20)

    # Dual-momentum across subsets
    engines["DM_big"] = lambda cp: eng_dm_topn(
        ["TQQQ", "UPRO", "TECL", "EDC", "TMF", "UGL", "TYD"], cp, top_n=2)
    engines["DM_small"] = lambda cp: eng_dm_topn(
        ["TQQQ", "UPRO", "TMF", "UGL"], cp, top_n=2)
    engines["DM_top1"] = lambda cp: eng_dm_topn(
        ["TQQQ", "UPRO", "TMF", "UGL"], cp, top_n=1)

    # Credit regime
    engines["CRED_UPRO"] = lambda cp: eng_cred_regime(cp, risky="UPRO")
    engines["CRED_SSO"] = lambda cp: eng_cred_regime(cp, risky="SSO")

    # Vix-like regime
    engines["VIX_SSO"] = lambda cp: eng_vix_regime(cp, risky="SSO")
    engines["VIX_UPRO"] = lambda cp: eng_vix_regime(cp, risky="UPRO")

    # Seasonality
    engines["SEAS_UPRO"] = lambda cp: eng_season(cp, risky="UPRO")
    engines["SEAS_SSO"] = lambda cp: eng_season(cp, risky="SSO")
    engines["SEAS_QQQ"] = lambda cp: eng_season(cp, risky="QQQ")

    # Mean-reversion dip
    engines["MR_UPRO"] = lambda cp: eng_mrev_dip(cp, risky="UPRO")
    engines["MR_SSO"] = lambda cp: eng_mrev_dip(cp, risky="SSO")

    # Inverse-vol trio (multiple variants)
    engines["IVT_UPRO_TMF_UGL"] = lambda cp: eng_inv_vol_trio(cp, sleeves=["UPRO", "TMF", "UGL"])
    engines["IVT_TQQQ_TMF_UGL"] = lambda cp: eng_inv_vol_trio(cp, sleeves=["TQQQ", "TMF", "UGL"])
    engines["IVT_6"] = lambda cp: eng_inv_vol_trio(cp, sleeves=["TQQQ", "UPRO", "TMF", "UBT", "UGL", "TYD"])

    # Buy-and-hold benchmarks
    engines["HFEA_UPRO_TMF"] = lambda cp: eng_carry_hedge(cp)
    return engines


def main():
    op, cp = util.load_prices()
    rets = cp.pct_change()

    engines = all_engines(cp)
    print(f"Generated {len(engines)} engines")

    R = {}
    for name, fn in engines.items():
        try:
            w = fn(cp)
            r = _portfolio_return(w, rets, cp)
            R[name] = r
        except Exception as e:
            print(f"  {name}: ERR {e}")

    RDF = pd.DataFrame(R).dropna(how="all")
    meta = {}
    for name in RDF.columns:
        r = RDF[name]
        meta[name] = {
            "full": util.metrics(r),
            "is": util.metrics(util.regime_slice(r, "2005-01-01", "2018-12-31")),
            "oos": util.metrics(util.regime_slice(r, util.OOS_START, "2027-12-31")),
        }

    # Rank by IS Sharpe
    ranked = sorted(meta.items(), key=lambda kv: -kv[1]["is"].get("sharpe", 0))
    print("\nTop 20 by IS Sharpe:")
    print(f"{'Engine':25s} {'IS_SR':>6} {'OOS_SR':>6} {'Full_SR':>6} {'CAGR':>7} {'MDD':>7}")
    for n, m in ranked[:20]:
        print(f"{n:25s} {m['is'].get('sharpe',0):>6.2f} {m['oos'].get('sharpe',0):>6.2f} "
              f"{m['full'].get('sharpe',0):>6.2f} {m['full'].get('cagr',0)*100:>6.1f}% "
              f"{m['full'].get('mdd',0)*100:>6.1f}%")

    # Save
    RDF.to_csv(OUT / "big_exp_returns.csv")
    with open(OUT / "big_exp_metrics.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
