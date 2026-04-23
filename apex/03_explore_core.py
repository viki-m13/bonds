"""APEX — Explore the 'right' core design.

Build a single strong, vol-targeted strategy as the APEX core. Test variants:

  A. 100% TQQQ trend (simple 200d MA filter, cash otherwise) with vol target
  B. 100% TQQQ trend + TMF hedge when risk-off
  C. Dual-momentum top-1 across (TQQQ, UPRO, TMF, UGL, cash)
  D. Same as C but top-2
  E. Inverse-vol weighted sleeve of (TQQQ-trend, TMF-trend, UGL-trend, cash)

Each with:
  - Daily vol targeting to various targets
  - DD throttle at -15%
  - Transaction costs modeled
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import util


def trend_on(s: pd.Series, fast: int = 50, slow: int = 200,
             ret_win: int = 126) -> pd.Series:
    ma_s = s.rolling(slow).mean()
    ma_f = s.rolling(fast).mean()
    r = s.pct_change(ret_win)
    return ((s > ma_s) & (ma_f > ma_s) & (r > 0)).astype(float)


def apply_dd_throttle(r: pd.Series, floor: float = -0.15, win: int = 252) -> pd.Series:
    c = (1 + r).cumprod()
    hwm = c.rolling(win, min_periods=30).max()
    dd = c / hwm - 1
    m = (1 + dd / floor).clip(0, 1).shift(1).fillna(1.0)
    return r * m


def daily_vol_target(r: pd.Series, target: float = 0.15,
                     win: int = 60, cap: float = 1.5, floor: float = 0.25) -> pd.Series:
    rv = r.rolling(win).std() * np.sqrt(util.DPY)
    m = (target / rv).clip(lower=floor, upper=cap).shift(1).fillna(1.0)
    return r * m


def variant_single_trend(cp: pd.DataFrame, letf: str, under: str,
                          target_vol: float = 0.15) -> pd.Series:
    """Hold LETF when underlying is in uptrend (trend filter); cash otherwise."""
    u = cp[under]
    on = trend_on(u).shift(1).fillna(0.0)   # signal at close[t-1] → ret[t]
    r_letf = cp[letf].pct_change().fillna(0.0)
    r_cash = cp["SHY" if "SHY" in cp.columns else "BIL"].pct_change().fillna(0.0)
    r = on * r_letf + (1 - on) * r_cash
    r = apply_dd_throttle(r, -0.15)
    r = daily_vol_target(r, target_vol)
    return r


def variant_dualmom_top1(cp: pd.DataFrame, universe: list[str],
                         target_vol: float = 0.15,
                         lookback: int = 126) -> pd.Series:
    """Each day: pick the single highest 126d return asset with ret>0.
    Else cash."""
    p = cp[universe]
    mom = p.pct_change(lookback)
    mom_filled = mom.dropna(how="all")
    # Use numpy for robust argmax
    best_idx = mom_filled.values.argmax(axis=1)
    best = pd.Series(mom_filled.columns[best_idx], index=mom_filled.index)
    best = best.reindex(cp.index)
    best_mom = mom.max(axis=1)
    has_winner = (best_mom > 0).fillna(False) & best.notna()
    rets = p.pct_change()
    # Pick best asset's return
    picked = pd.Series(0.0, index=cp.index)
    for a in universe:
        mask = (best == a) & has_winner
        picked += mask.shift(1).fillna(False).astype(float) * rets[a].fillna(0.0)
    cash_ret = cp["SHY" if "SHY" in cp.columns else "BIL"].pct_change().fillna(0.0)
    picked += (~has_winner.shift(1).fillna(False)).astype(float) * cash_ret
    picked = apply_dd_throttle(picked, -0.15)
    picked = daily_vol_target(picked, target_vol)
    return picked


def variant_dualmom_topn(cp: pd.DataFrame, universe: list[str],
                          top_n: int = 2, target_vol: float = 0.15,
                          lookback: int = 126, rebal: int = 21) -> pd.Series:
    """Top-N equal weight; monthly rebal."""
    p = cp[universe]
    mom = p.pct_change(lookback)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_n) & (mom > 0)
    # Resample to monthly: use last day of each month's signal, carry forward
    mask = pd.Series(range(len(cp.index)), index=cp.index)
    rebal_days = mask % rebal == 0
    w_monthly = sel.where(rebal_days).ffill().fillna(False)
    # Normalize
    ns = w_monthly.sum(axis=1)
    w = w_monthly.astype(float).div(ns.replace(0, np.nan), axis=0).fillna(0.0)
    rets = p.pct_change()
    pr = (w.shift(1).fillna(0.0) * rets.fillna(0.0)).sum(axis=1)
    # Cash if nothing selected
    cash_ret = cp["SHY" if "SHY" in cp.columns else "BIL"].pct_change().fillna(0.0)
    cash_w = (w.sum(axis=1) == 0).astype(float)
    pr += cash_w.shift(1).fillna(1.0) * cash_ret
    pr = apply_dd_throttle(pr, -0.15)
    pr = daily_vol_target(pr, target_vol)
    return pr


def variant_invvol_trend_sleeves(cp: pd.DataFrame, sleeves: list[tuple[str, str]],
                                  target_vol: float = 0.15) -> pd.Series:
    """Each sleeve: trend-filtered LETF or cash. Blend by inverse-vol."""
    sleeve_returns = {}
    for letf, under in sleeves:
        u = cp[under]
        on = trend_on(u).shift(1).fillna(0.0)
        r = on * cp[letf].pct_change().fillna(0.0) + (1 - on) * cp["SHY"].pct_change().fillna(0.0)
        sleeve_returns[f"{letf}_trend"] = r
    R = pd.DataFrame(sleeve_returns)
    # Inverse-vol blend (lagged)
    vol60 = R.rolling(60).std()
    iv = 1.0 / vol60.replace(0, np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0)
    iv = iv.shift(1).fillna(1.0 / len(sleeves))
    pr = (R * iv).sum(axis=1)
    pr = apply_dd_throttle(pr, -0.15)
    pr = daily_vol_target(pr, target_vol)
    return pr


def main():
    op, cp = util.load_prices()

    print("=" * 100)
    print("Variant A: Single TQQQ trend + cash")
    for tv in (0.10, 0.15, 0.20, 0.25):
        r = variant_single_trend(cp, "TQQQ", "QQQ", target_vol=tv)
        util.summarize(r, f"  target_vol={tv}")

    print("\n" + "=" * 100)
    print("Variant A2: Single UPRO trend + cash")
    for tv in (0.10, 0.15, 0.20, 0.25):
        r = variant_single_trend(cp, "UPRO", "SPY", target_vol=tv)
        util.summarize(r, f"  target_vol={tv}")

    print("\n" + "=" * 100)
    print("Variant C: Top-1 dual-mom across {TQQQ,UPRO,TMF,UGL,SHY}")
    universe = ["TQQQ", "UPRO", "TMF", "UGL", "SHY"]
    for tv in (0.10, 0.15, 0.20, 0.25):
        r = variant_dualmom_top1(cp, universe, target_vol=tv)
        util.summarize(r, f"  target_vol={tv}")

    print("\n" + "=" * 100)
    print("Variant D: Top-2 dual-mom across {TQQQ,UPRO,TECL,EDC,TMF,UGL,UBT}, monthly rebal")
    universe = ["TQQQ", "UPRO", "TECL", "EDC", "TMF", "UGL", "UBT"]
    for tv in (0.10, 0.15, 0.20, 0.25):
        r = variant_dualmom_topn(cp, universe, top_n=2, target_vol=tv)
        util.summarize(r, f"  target_vol={tv}")

    print("\n" + "=" * 100)
    print("Variant E: Inverse-vol trend sleeves (TQQQ, UPRO, TMF, UGL, TECL, EDC)")
    sleeves = [("TQQQ", "QQQ"), ("UPRO", "SPY"), ("TMF", "TLT"),
               ("UGL", "GLD"), ("TECL", "XLK"), ("EDC", "EEM")]
    for tv in (0.10, 0.15, 0.20, 0.25):
        r = variant_invvol_trend_sleeves(cp, sleeves, target_vol=tv)
        util.summarize(r, f"  target_vol={tv}")

    # Show the best
    print("\n" + "=" * 100)
    print("Best variant E @ 0.15 — by window:")
    r = variant_invvol_trend_sleeves(cp, sleeves, target_vol=0.15)
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31")),
                        ("COVID", ("2020-01-01", "2020-12-31")),
                        ("2022RH", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(r, s, e), f"  {lbl}")


if __name__ == "__main__":
    main()
