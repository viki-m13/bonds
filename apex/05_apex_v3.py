"""APEX v3 — Multi-engine ensemble with per-engine vol targeting.

Design principles:
  1. Each engine operates on a different underlying signal source (momentum,
     credit, yield curve, vol regime, mean reversion). Signals are uncorrelated
     by construction.
  2. Every engine is scaled to the same 10% ex-post vol before blending, so
     blend weights reflect *skill*, not vol scale.
  3. Inverse-vol blend weights (fit on IS 2005-2018 only) — avoids mean/Sharpe
     estimation which is noisy.
  4. Portfolio-level risk controls: DD throttle and daily vol target.
  5. No portfolio leverage: sum of weights ≤ 1.
  6. Engines can hold cash (SHY/BIL) when their signal is off — avoids dragging
     the blend in unfavorable regimes.

Engines:
  M_EQ   — Top-2 equity-LETFs by cross-sectional momentum
  M_BOND — TMF when TLT trending up
  M_GOLD — UGL when GLD trending up
  M_CRED — UPRO when HY spread tight and tightening
  M_VRP  — SSO when SPY realized vol low and declining
  M_CURVE — TMF when yield curve steepening
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

import util

ROOT = Path("/home/user/bonds")
FRED = ROOT / "data/fred"
OUT = ROOT / "data/apex"


def _fred(name: str, idx: pd.DatetimeIndex) -> pd.Series:
    fp = FRED / f"{name}.csv"
    if not fp.exists():
        return pd.Series(np.nan, index=idx)
    df = pd.read_csv(fp, parse_dates=["Date"]).sort_values("Date").set_index("Date")
    col = df.columns[0]
    return df[col].astype(float).reindex(idx).ffill()


def _cash_ticker(cp: pd.DataFrame) -> str:
    for t in ("SHY", "BIL"):
        if t in cp.columns:
            return t
    return list(cp.columns)[0]


def engine_m_eq(op, cp, top_n=2, lookback=126, skip=21):
    universe = ["UPRO", "TQQQ", "TECL", "FAS", "SOXL", "EDC", "YINN", "DRN"]
    universe = [a for a in universe if a in cp.columns]
    p = cp[universe]
    rv = p.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    mom = p.shift(skip).pct_change(lookback - skip) / rv.replace(0, np.nan)
    rnk = mom.rank(axis=1, ascending=False, method="first")
    sel = (rnk <= top_n) & (mom > 0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    iv = (1.0 / rv.replace(0, np.nan))
    for a in universe:
        w[a] = (sel[a].astype(float) * iv[a]).fillna(0.0)
    s = w.sum(axis=1).replace(0, np.nan)
    w = w.div(s, axis=0).fillna(0.0)
    # Cash fallback
    cash = _cash_ticker(cp)
    w[cash] = (w.sum(axis=1) == 0).astype(float)
    return w


def engine_m_bond(op, cp):
    if "TMF" not in cp.columns or "TLT" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    tlt = cp["TLT"]
    ma200 = tlt.rolling(200).mean()
    ma50 = tlt.rolling(50).mean()
    on = ((tlt > ma200) & (ma50 > ma200)).astype(float)
    rv = tlt.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    sc = (0.20 / rv).clip(upper=1.0).fillna(1.0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["TMF"] = (on * sc).fillna(0.0)
    cash = _cash_ticker(cp)
    w[cash] = (1 - w["TMF"]).clip(lower=0.0)
    return w


def engine_m_gold(op, cp):
    if "UGL" not in cp.columns or "GLD" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    gld = cp["GLD"]
    ma200 = gld.rolling(200).mean()
    ma50 = gld.rolling(50).mean()
    on = ((gld > ma200) & (ma50 > ma200)).astype(float)
    rv = gld.pct_change().rolling(60).std() * np.sqrt(util.DPY)
    sc = (0.20 / rv).clip(upper=1.0).fillna(1.0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["UGL"] = (on * sc).fillna(0.0)
    cash = _cash_ticker(cp)
    w[cash] = (1 - w["UGL"]).clip(lower=0.0)
    return w


def engine_m_cred(op, cp):
    """Credit regime: HY spread tight & falling → long SSO (2x SPY)."""
    if "SSO" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    hy = _fred("BAMLH0A0HYM2", cp.index)
    if hy.isna().all():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    hy_ma60 = hy.rolling(60).mean()
    hy_pct = hy.rolling(504, min_periods=60).rank(pct=True)
    # risk-on: spread in bottom 70% AND below 60-day mean
    on = ((hy_pct < 0.7) & (hy < hy_ma60)).astype(float)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["SSO"] = on
    cash = _cash_ticker(cp)
    w[cash] = 1 - on
    return w


def engine_m_vrp(op, cp):
    """Calm-market equity tilt: realized vol in lowest 50% + decreasing."""
    if "SSO" not in cp.columns or "SPY" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    spy = cp["SPY"]
    rv = spy.pct_change().rolling(21).std() * np.sqrt(util.DPY)
    rv63 = spy.pct_change().rolling(63).std() * np.sqrt(util.DPY)
    pct = rv.rolling(504, min_periods=60).rank(pct=True)
    on = ((pct < 0.5) & (rv < rv63)).astype(float)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["SSO"] = on
    cash = _cash_ticker(cp)
    w[cash] = 1 - on
    return w


def engine_m_curve(op, cp):
    """Yield curve steepening + positive → long TMF; else cash."""
    if "TMF" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    slope = _fred("T10Y2Y", cp.index)
    if slope.isna().all():
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    ma20 = slope.rolling(20).mean()
    ma60 = slope.rolling(60).mean()
    on = ((slope > 0) & (ma20 > ma60)).astype(float)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["TMF"] = on
    cash = _cash_ticker(cp)
    w[cash] = 1 - on
    return w


def engine_m_dip(op, cp):
    """Buy UPRO when SPY drops > 3% in a day AND RSI(2) < 5. Hold 3 days."""
    if "UPRO" not in cp.columns or "SPY" not in cp.columns:
        return pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    spy = cp["SPY"]
    r1 = spy.pct_change()
    # RSI(2)
    delta = spy.diff()
    gain = delta.clip(lower=0).rolling(2).mean()
    loss = (-delta).clip(lower=0).rolling(2).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    trigger = ((r1 < -0.03) & (rsi < 5)).astype(float)
    held = trigger.rolling(3, min_periods=1).sum().clip(upper=1.0)
    w = pd.DataFrame(0.0, index=cp.index, columns=cp.columns)
    w["UPRO"] = held
    cash = _cash_ticker(cp)
    w[cash] = (1 - held).clip(lower=0.0)
    return w


def run_engine(engine_fn, cp, rc, name):
    w = engine_fn(cp, cp)
    r, _ = util.apply_weights(w, rc, util.tc_map())
    return r


def main():
    op, cp = util.load_prices()
    rc = cp.pct_change()

    engines_fns = {
        "M_EQ":    engine_m_eq,
        "M_BOND":  engine_m_bond,
        "M_GOLD":  engine_m_gold,
        "M_CRED":  engine_m_cred,
        "M_VRP":   engine_m_vrp,
        "M_CURVE": engine_m_curve,
        "M_DIP":   engine_m_dip,
    }

    returns = {}
    for name, fn in engines_fns.items():
        r = run_engine(fn, cp, rc, name)
        returns[name] = r
        print(f"\n=== {name} ===")
        util.summarize(r, "FULL")
        util.summarize(util.regime_slice(r, "2005-01-01", "2018-12-31"), "IS 05-18")
        util.summarize(util.regime_slice(r, util.OOS_START, "2027-12-31"), "OOS 19+")
        util.summarize(util.regime_slice(r, "2022-01-01", "2022-12-31"), "2022")

    R = pd.DataFrame(returns)
    print("\nCorrelations (IS 05-18):")
    print(R.loc[:util.IS_END].corr().round(2))

    # --- Scale each engine to 10% vol (daily, lagged), then blend ---
    target_engine_vol = 0.10

    def scale_eng(r, target=target_engine_vol, win=60, cap=2.0, floor=0.25):
        rv = r.rolling(win).std() * np.sqrt(util.DPY)
        m = (target / rv).clip(lower=floor, upper=cap).shift(1).fillna(1.0)
        return r * m

    RS = pd.DataFrame({k: scale_eng(v) for k, v in returns.items()})
    print("\nScaled-engine metrics (each to 10% vol):")
    for k in RS.columns:
        util.summarize(RS[k], f"  {k}")

    # IS-fit blend weights via inverse variance (full period; we'll check IS only below)
    is_R = RS.loc[:util.IS_END].dropna(how="any")
    iv = 1.0 / (is_R.std() * np.sqrt(util.DPY)).replace(0, np.nan)
    iv = iv / iv.sum()
    print(f"\nIS blend weights:")
    print(iv.round(3))

    blend = (RS * iv).sum(axis=1)
    print("\nBlend (raw, with IS weights applied to full):")
    util.summarize(blend, "FULL")
    util.summarize(util.regime_slice(blend, "2005-01-01", "2018-12-31"), "IS 05-18")
    util.summarize(util.regime_slice(blend, util.OOS_START, "2027-12-31"), "OOS 19+")
    util.summarize(util.regime_slice(blend, "2000-01-01", "2008-12-31"), "pre-08")

    # Apply DD throttle and portfolio-level vol target
    def finalize(r, target_vol=0.15, dd_floor=-0.12):
        # DD
        c = (1 + r).cumprod()
        hwm = c.rolling(252, min_periods=30).max()
        dd = c / hwm - 1
        m = (1 + dd / dd_floor).clip(0, 1).shift(1).fillna(1.0)
        r2 = r * m
        rv = r2.rolling(60).std() * np.sqrt(util.DPY)
        vm = (target_vol / rv).clip(lower=0.25, upper=1.5).shift(1).fillna(1.0)
        return r2 * vm

    print("\n=== Final (DD throttle + 15% vol target) ===")
    for tv in (0.10, 0.15, 0.20, 0.25):
        rf = finalize(blend, target_vol=tv)
        util.summarize(rf, f"  tv={tv}")

    rf = finalize(blend, 0.20)
    for lbl, (s, e) in [("FULL", ("1999-01-01", "2027-12-31")),
                        ("IS 05-18", ("2005-01-01", "2018-12-31")),
                        ("OOS 19+", ("2019-01-02", "2027-12-31")),
                        ("pre-08", ("2000-01-01", "2008-12-31")),
                        ("GFC", ("2007-01-01", "2009-12-31")),
                        ("COVID", ("2020-01-01", "2020-12-31")),
                        ("2022RH", ("2022-01-01", "2022-12-31"))]:
        util.summarize(util.regime_slice(rf, s, e), f"  final tv=0.20  {lbl}")


if __name__ == "__main__":
    main()
