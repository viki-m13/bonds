"""MERIDIAN — exploration of additional truly orthogonal sleeves."""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from meridian_strategy import (panel, load_fred, load_etf, metrics,
                                 backtest_o2o, backtest_overnight,
                                 monthly_dates, hold_at,
                                 IS_START, IS_END, OOS_START, RES, TC_BPS)


# ============================================================================
# Additional sleeve candidates
# ============================================================================
def sleeve_quality_div():
    """Quality dividend: SCHD + DVY + VIG inverse-vol, gated by SPY trend."""
    UNI = ["SCHD", "DVY", "VIG"]
    o, c = panel(UNI + ["BIL", "SPY"])
    cl = c.shift(1)
    rets60 = cl[UNI].pct_change().rolling(60).std()
    iv = 1.0 / rets60
    w = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    spy = cl["SPY"]
    on = (spy > spy.rolling(200).mean()).astype(float)
    w = w.mul(on, axis=0)
    w["BIL"] = (1 - w[UNI].sum(axis=1)).clip(lower=0)
    w["SPY"] = 0.0
    held = hold_at(o.index, w[UNI + ["BIL", "SPY"]], monthly_dates(o.index))
    return backtest_o2o(held, o)


def sleeve_intl_rotation():
    """International equity rotation: pick top-2 by 6mo momentum among EFA, EEM, INDA, FXI, EWJ."""
    UNI = ["EFA", "EEM", "FXI", "EWJ"]
    o, c = panel(UNI + ["BIL"])
    cl = c.shift(1)
    momo = cl[UNI].pct_change(126)
    rk = momo.rank(axis=1, ascending=False, method="first")
    pick = (rk <= 2) & (momo > 0)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.astype(float).div(n, axis=0).fillna(0.0)
    w["BIL"] = (1 - w[UNI].sum(axis=1)).clip(lower=0)
    held = hold_at(o.index, w, monthly_dates(o.index))
    return backtest_o2o(held, o)


def sleeve_term_premium():
    """Pure term-premium: long TLT only when 200d trend up AND vol low."""
    UNI = ["TLT", "EDV"]
    o, c = panel(UNI + ["BIL", "SPY"])
    cl = c.shift(1)
    sma200 = cl["TLT"].rolling(200).mean()
    momo60 = cl["TLT"].pct_change(60)
    eligible = (cl["TLT"] > sma200) & (momo60 > 0)

    rets60 = cl[UNI].pct_change().rolling(60).std()
    iv = 1.0 / rets60
    w = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    w = w.mul(eligible.astype(float), axis=0)
    w["BIL"] = (1 - w[UNI].sum(axis=1)).clip(lower=0)
    w["SPY"] = 0.0
    held = hold_at(o.index, w[UNI + ["BIL", "SPY"]], monthly_dates(o.index))
    return backtest_o2o(held, o)


def sleeve_commodity_trend():
    """Commodity trend: top-1 among GLD, SLV, DBC, USO by 90d momentum."""
    UNI = ["GLD", "SLV", "DBC", "USO"]
    o, c = panel(UNI + ["BIL"])
    cl = c.shift(1)
    momo = cl[UNI].pct_change(90)
    rk = momo.rank(axis=1, ascending=False, method="first")
    pick = (rk <= 1) & (momo > 0)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.astype(float).div(n, axis=0).fillna(0.0)
    w["BIL"] = (1 - w[UNI].sum(axis=1)).clip(lower=0)
    held = hold_at(o.index, w, monthly_dates(o.index))
    return backtest_o2o(held, o)


def sleeve_dispersion():
    """Sector-dispersion conditional momentum: only engage when sector
    cross-sectional dispersion (60d std of 60d returns) is in top half of 252d.
    Trade momentum top-2."""
    SECTORS = ["XLK", "XLY", "XLP", "XLU", "XLV", "XLE", "XLF", "XLI", "XLB"]
    o, c = panel(SECTORS + ["SPY", "BIL"])
    cl = c.shift(1)
    momo60 = cl[SECTORS].pct_change(60)
    cs_dispersion = momo60.std(axis=1)
    disp_pct = cs_dispersion.rolling(252).rank(pct=True).shift(1).fillna(0.5)
    engage = (disp_pct > 0.5).astype(float)

    momo126 = cl[SECTORS].pct_change(126)
    rk = momo126.rank(axis=1, ascending=False, method="first")
    pick = (rk <= 2) & (momo126 > 0)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.astype(float).div(n, axis=0).fillna(0.0)
    w = w.mul(engage, axis=0)
    spy = cl["SPY"]
    on_spy = (spy > spy.rolling(200).mean()).astype(float)
    w = w.mul(on_spy, axis=0)
    w["BIL"] = (1 - w[SECTORS].sum(axis=1)).clip(lower=0)
    w["SPY"] = 0.0
    held = hold_at(o.index, w[SECTORS + ["BIL", "SPY"]], monthly_dates(o.index))
    return backtest_o2o(held, o)


def sleeve_size_factor():
    """Long IWM/MDY when small-cap relative momentum positive vs SPY."""
    UNI = ["IWM"]
    o, c = panel(UNI + ["SPY", "BIL"])
    cl = c.shift(1)
    rel = cl["IWM"] / cl["SPY"]
    rel_sma = rel.rolling(126).mean()
    iwm_above_sma = (cl["IWM"] > cl["IWM"].rolling(200).mean())
    rel_uptrend = (rel > rel_sma)
    on = (iwm_above_sma & rel_uptrend).astype(float)
    w = pd.DataFrame(0.0, index=o.index, columns=UNI + ["SPY", "BIL"])
    w["IWM"] = on * 1.0
    w["BIL"] = 1 - w["IWM"]
    held = hold_at(o.index, w, monthly_dates(o.index))
    return backtest_o2o(held, o)


def sleeve_tom():
    """Turn-of-month effect: long SPY in last 4 days + first 3 days of month."""
    o, c = panel(["SPY", "BIL"])
    idx = o.index
    # Identify last 4 days of month and first 3 days of next
    month = pd.Series(idx.to_period("M"), index=idx)
    is_last_3 = pd.Series(False, index=idx)
    is_first_3 = pd.Series(False, index=idx)
    for m, group in pd.Series(idx, index=idx).groupby(month):
        days = group.values
        if len(days) >= 6:
            for d in days[-4:]:
                is_last_3.loc[d] = True
            for d in days[:3]:
                is_first_3.loc[d] = True
    in_window = (is_last_3 | is_first_3).shift(0).fillna(False)

    w = pd.DataFrame(0.0, index=idx, columns=["SPY", "BIL"])
    w.loc[in_window, "SPY"] = 1.0
    w.loc[~in_window, "BIL"] = 1.0
    return backtest_o2o(w, o)


def sleeve_vix_meanrev():
    """When VIX 5d change > +5 pts (vol spike), allocate to BIL (defensive).
    When VIX has stabilized after spike (z-score returning to mean), reallocate
    to SPY for the rebound."""
    o, c = panel(["SPY", "BIL"])
    vix = load_fred("VIXCLS").reindex(o.index).ffill()
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    # Spike: z > 1.5; calming: z < 0.5 after a spike
    in_spike = (vix_z > 1.5).rolling(20).max()  # spike anywhere in last 20d
    calming = (vix_z < 0.5)
    rebound = (in_spike == 1) & calming
    rebound = rebound.shift(1).fillna(False).astype(float)
    w = pd.DataFrame(0.0, index=o.index, columns=["SPY", "BIL"])
    w["SPY"] = rebound * 1.0
    w["BIL"] = 1 - w["SPY"]
    return backtest_o2o(w, o)


def main():
    print("=" * 100)
    print("MERIDIAN — additional orthogonal sleeve candidates")
    print("=" * 100)

    sleeves = {}
    candidates = [
        ("QUAL_DIV",  sleeve_quality_div),
        ("INTL_ROT",  sleeve_intl_rotation),
        ("TERM_PREM", sleeve_term_premium),
        ("COMMOD_TR", sleeve_commodity_trend),
        ("DISPERSION", sleeve_dispersion),
        ("SIZE",      sleeve_size_factor),
        ("TOM",       sleeve_tom),
        ("VIX_REB",   sleeve_vix_meanrev),
    ]
    for name, fn in candidates:
        try:
            r = fn()
            sleeves[name] = r
            m_is = metrics(r.loc[IS_START:IS_END]); m_oos = metrics(r.loc[OOS_START:]); m_full = metrics(r.loc[IS_START:])
            print(f"  {name:12s}  IS Sh={m_is['sharpe']:5.2f} OOS Sh={m_oos['sharpe']:5.2f} "
                  f"FULL Sh={m_full['sharpe']:5.2f} CAGR={m_full['cagr']*100:5.1f}% "
                  f"Vol={m_full['vol']*100:5.1f}% MDD={m_full['mdd']*100:5.1f}%")
        except Exception as e:
            print(f"  {name:12s}  ERROR: {e}")

    df = pd.concat(sleeves, axis=1, sort=True).fillna(0.0)
    print("\nCorrelations among new candidates (full sample):")
    print(df.loc[IS_START:].corr().round(2).to_string())


if __name__ == "__main__":
    main()
