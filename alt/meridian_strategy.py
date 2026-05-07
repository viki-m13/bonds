"""MERIDIAN — Strict-no-leverage multi-sleeve daily-managed strategy.

Hard constraints (in priority order):
  1. NEVER hold any leveraged or inverse ETF (no 2x/3x/-1x products).
  2. NEVER use portfolio-level margin or borrowing. Sum of weights at any
     time t must be <= 1.0 (cash residual goes to BIL or stays unallocated
     and earns ~zero).
  3. NEVER use forward-looking data. All signals computed on close[t-1] (or
     earlier), positions established at open[t] (or close[t-1] for overnight),
     returns earned open[t] -> open[t+1] (or close[t-1] -> open[t] overnight).

Design philosophy
=================
Phoenix achieves Sharpe ~2.4 by combining 5 sleeves with average pairwise
correlation ~0.02. Each sleeve is mediocre in isolation; the diversification
multiplier from the near-zero correlations gives the ensemble its Sharpe.

To match this without leverage, MERIDIAN runs ten sleeves spanning the
distinct alpha categories that empirically exhibit low cross-correlation,
combining time-series and cross-sectional, monthly and daily, equity and
bond and commodity, and explicit calendar/regime trades.

Alpha categories
================
  A. CARRY            — earn risk premium of credit & duration
  B. TREND-TS         — time-series (single-asset) momentum
  C. TREND-CS         — cross-sectional ranking momentum
  D. BREAKOUT         — Donchian-channel triggers
  E. VOL-REGIME       — VIX-conditioned trades (vol-cooling + vol-rebound)
  F. OVERNIGHT        — close-to-open calendar premium
  G. CALENDAR-TOM     — turn-of-month equity premium
  H. QUALITY-INCOME   — quality dividend long-only
  I. TERM-PREMIUM     — pure long-duration treasury when uptrending

Each sleeve allocates 100% of its capital between ETFs and BIL (cash)
internally. Sleeves are blended by EQUAL weights (1/N), so total portfolio
gross is exactly 1.0. No margin, no shorting, no leveraged products.

Sleeves
=======
S1  CARRY        — A. monthly. Bond carry (LQD/HYG/EMB/TLT/IEF) with HY-OAS gate.
S2  EQ_TSMOM     — B. monthly. Equity TS momentum (SPY/QQQ/IWM/EFA/EEM).
S3  DEF_TSMOM    — B. monthly. Defensive TS momentum (TLT/GLD/EDV/IEF/BND).
S4  SECT_CSMOM   — C. weekly.  Sector cross-sectional momentum (9 SPDRs).
S5  GOLD_BREAK   — D. daily.   Gold/silver Donchian breakout.
S6  VOL_COOL     — E. weekly.  Post-stress equity rally (VIX-cooling).
S7  VIX_REB      — E. daily.   VIX-spike rebound trade.
S8  OVERNIGHT    — F. daily.   Overnight equity return premium.
S9  TOM          — G. daily.   Turn-of-month SPY long.
S10 QUAL_DIV     — H. monthly. SCHD/DVY/VIG inverse-vol with SPY-trend gate.

Aggregator
==========
Sharpe^2 / vol weights (IS-fitted, frozen for OOS): w_i ∝ Sharpe_i^2 / vol_i
on the IS window (2010-2018) and then held fixed. This is a positive-only
Markowitz-style allocation that emphasizes high-Sharpe, low-vol sleeves
without using IS-OOS information. Each sleeve always allocates 100% of its
capital between ETFs and BIL; portfolio gross is exactly sum(weights) = 1.0
— no margin, no leverage anywhere.

Risk overlays apply at the portfolio level and only DE-RISK (multiplier <= 1):

  - DD throttle: linear scale toward zero as NAV falls below the 252d HWM,
    floor at -8% (below this, fully cash).
  - Vol-regime gate: halve exposure when 60d realized vol > 99th percentile
    of 252d lookback.

The aggregator never multiplies portfolio exposure ABOVE 1.0.

This file is the SINGLE production reference. Exploration scripts
(meridian_explore*.py) are documentation of the search.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data" / "etfs"
FRED = ROOT / "data" / "fred"
RES = ROOT / "data" / "results"
RES.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2010-01-04")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")

# ETF execution costs
TC_BPS = 5.0
# Portfolio-level overlays — NEVER scale UP exposure
DD_FLOOR = -0.08
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252
VOL_WIN = 60

# ============================================================================
# Universes (all 1x ETFs only)
# ============================================================================
CARRY_UNI = ["LQD", "HYG", "EMB", "TLT", "IEF"]
EQUITY_UNI = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
DEFENSE_UNI = ["TLT", "GLD", "EDV", "IEF", "BND"]
SECTORS = ["XLK", "XLY", "XLP", "XLU", "XLV", "XLE", "XLF", "XLI", "XLB"]
GOLD_UNI = ["GLD", "SLV"]
TIDE_UNI = ["SPY", "QQQ", "SMH", "IWM"]
ON_UNI = ["SPY", "QQQ"]
QUAL_UNI = ["SCHD", "DVY", "VIG"]
TERM_UNI = ["TLT", "EDV"]


# ============================================================================
# Data loaders
# ============================================================================
def load_etf(t: str) -> pd.DataFrame | None:
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    p = FRED / f"{name}.csv"
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(s, errors="coerce").sort_index()


def panel(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for t in tickers:
        d = load_etf(t)
        if d is None:
            continue
        opens[t] = d["Open"]
        closes[t] = d["Close"]
    o = pd.DataFrame(opens).sort_index()
    c = pd.DataFrame(closes).sort_index()
    idx = pd.bdate_range(o.index.min(), o.index.max())
    return o.reindex(idx).ffill(limit=3), c.reindex(idx).ffill(limit=3)


def metrics(r: pd.Series, name: str = "") -> dict:
    r = r.dropna()
    if len(r) < 30:
        return {"name": name, "sharpe": 0, "n": len(r)}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = cum.iloc[-1] ** (1 / yrs) - 1 if cum.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 0
    return dict(name=name, sharpe=round(float(sr), 4), cagr=round(float(cagr), 4),
                vol=round(float(sd), 4), mdd=round(float(dd), 4),
                sortino=round(float(sortino), 4),
                calmar=round(float(cagr / abs(dd)), 4) if dd < 0 else 0,
                n=int(len(r)), navx=round(float(cum.iloc[-1]), 4))


# ============================================================================
# Backtest helpers
# ============================================================================
def backtest_o2o(weights: pd.DataFrame, opens: pd.DataFrame,
                 tc_bps: float = TC_BPS) -> pd.Series:
    w = weights.reindex(columns=opens.columns).fillna(0.0)
    o2o = opens.pct_change()
    w_held = w.shift(1).fillna(0.0)
    gross = (w_held * o2o).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (turnover * tc_bps / 1e4).shift(1).fillna(0.0)
    return gross - cost


def backtest_overnight(weights: pd.DataFrame, opens: pd.DataFrame,
                        closes: pd.DataFrame, tc_bps: float = TC_BPS) -> pd.Series:
    """Overnight return: close[t-1] -> open[t]. Position taken at close[t-1]
    based on signals computed through close[t-1] info.
    Returns are realized at open[t]; sleeve return aligned to date t."""
    w = weights.reindex(columns=closes.columns).fillna(0.0)
    over = (opens / closes.shift(1) - 1.0)
    w_held = w.shift(1).fillna(0.0)
    gross = (w_held * over).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (turnover * tc_bps / 1e4).shift(1).fillna(0.0)
    return gross - cost


def monthly_dates(idx):
    m = pd.Series(idx, index=idx).groupby(
        [idx.year, idx.month]).transform("first") == pd.Series(idx, index=idx)
    return idx[m.values]


def hold_at(idx, weights, rebal_dates):
    held = pd.DataFrame(0.0, index=idx, columns=weights.columns)
    cur = pd.Series(0.0, index=weights.columns)
    rb = set(rebal_dates)
    for dt in idx:
        if dt in rb and dt in weights.index and not weights.loc[dt].isna().all():
            cur = weights.loc[dt].fillna(0.0)
        held.loc[dt] = cur
    return held


# ============================================================================
# S1 — CARRY: bond carry (monthly, HY-OAS gated).
# ============================================================================
def sleeve_carry() -> pd.Series:
    o, c = panel(CARRY_UNI + ["BIL"])
    cl = c.shift(1)
    rets60 = cl[CARRY_UNI].pct_change().rolling(60).std()
    iv = 1.0 / rets60
    w = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    momo6 = cl[CARRY_UNI].pct_change(126)
    w = w.where(momo6 > 0, 0.0)

    hy = load_fred("BAMLH0A0HYM2").reindex(o.index).ffill()
    hy_z = (hy - hy.rolling(252).mean()) / hy.rolling(252).std()
    hy_z = hy_z.shift(1).fillna(0.0)
    gate = ((1.5 - hy_z) / 1.0).clip(0.0, 1.0)
    w = w.mul(gate, axis=0)
    w["BIL"] = (1 - w[CARRY_UNI].sum(axis=1)).clip(lower=0)

    held = hold_at(o.index, w, monthly_dates(o.index))
    return backtest_o2o(held, o)


# ============================================================================
# S2 — EQ_TSMOM: equity time-series momentum (monthly).
# ============================================================================
def sleeve_eq_tsmom() -> pd.Series:
    o, c = panel(EQUITY_UNI + ["BIL"])
    cl = c.shift(1)
    momo = cl.pct_change(252 - 21).shift(21)
    sma200 = cl.rolling(200).mean()
    eligible = (momo > 0) & (cl > sma200)
    w = eligible[EQUITY_UNI].astype(float)
    n = w.sum(axis=1).replace(0, np.nan)
    w = w.div(n, axis=0).fillna(0.0)
    w["BIL"] = (1 - w[EQUITY_UNI].sum(axis=1)).clip(lower=0)
    held = hold_at(o.index, w, monthly_dates(o.index))
    return backtest_o2o(held, o)


# ============================================================================
# S3 — DEF_TSMOM: defensive time-series momentum (monthly).
# ============================================================================
def sleeve_def_tsmom() -> pd.Series:
    o, c = panel(DEFENSE_UNI + ["BIL"])
    cl = c.shift(1)
    momo = cl.pct_change(126)
    sma200 = cl.rolling(200).mean()
    eligible = (momo > 0) & (cl > sma200)
    rets60 = cl[DEFENSE_UNI].pct_change().rolling(60).std()
    iv = 1.0 / rets60
    iv = iv.where(eligible[DEFENSE_UNI], 0.0)
    w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    w["BIL"] = (1 - w[DEFENSE_UNI].sum(axis=1)).clip(lower=0)
    held = hold_at(o.index, w, monthly_dates(o.index))
    return backtest_o2o(held, o)


# ============================================================================
# S4 — SECT_CSMOM: cross-sectional sector momentum (weekly, Wed).
# ============================================================================
def sleeve_sector_csmom() -> pd.Series:
    o, c = panel(SECTORS + ["SPY", "BIL"])
    cl = c.shift(1)
    sig = cl.pct_change(126) - cl.pct_change(21)
    rk = sig[SECTORS].rank(axis=1, ascending=False, method="first")
    pick = (rk <= 3) & (cl[SECTORS].pct_change(126) > 0)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.astype(float).div(n, axis=0).fillna(0.0)
    spy = cl["SPY"]
    on = (spy > spy.rolling(200).mean()).astype(float)
    w = w.mul(on, axis=0)
    w["BIL"] = (1 - w[SECTORS].sum(axis=1)).clip(lower=0)
    w["SPY"] = 0.0

    idx = o.index
    is_wed = pd.Series(idx, index=idx).dt.dayofweek == 2
    rebal = idx[is_wed.values]
    held = hold_at(idx, w[SECTORS + ["BIL", "SPY"]], rebal)
    return backtest_o2o(held, o)


# ============================================================================
# S5 — GOLD_BREAK: Donchian breakout on gold (daily).
# ============================================================================
def sleeve_gold_break() -> pd.Series:
    o, c = panel(GOLD_UNI + ["BIL"])
    cl = c.shift(1)
    high60 = cl["GLD"].rolling(60).max()
    low60 = cl["GLD"].rolling(60).min()
    above_high = (cl["GLD"] >= high60 * 0.99).astype(float)
    below_low = (cl["GLD"] <= low60 * 1.01).astype(float)
    raw_sig = (above_high - below_low).clip(lower=0).rolling(5).max()
    on = raw_sig.shift(1).fillna(0.0)
    w = pd.DataFrame(0.0, index=o.index, columns=GOLD_UNI + ["BIL"])
    w["GLD"] = on * 0.7
    w["SLV"] = on * 0.3
    w["BIL"] = (1 - w["GLD"] - w["SLV"]).clip(lower=0)
    return backtest_o2o(w, o)


# ============================================================================
# S6 — VOL_COOL: post-stress equity recovery (weekly).
# ============================================================================
def sleeve_vol_cool() -> pd.Series:
    o, c = panel(TIDE_UNI + ["BIL"])
    cl = c.shift(1)
    vix = load_fred("VIXCLS").reindex(o.index).ffill()
    vix_high60 = vix.rolling(60).max()
    cool = (vix < vix_high60 * 0.75)
    slope21 = vix - vix.shift(21)
    decomp = (cool & (slope21 < 0)) | (vix < 18)

    momo = cl[TIDE_UNI].pct_change(63)
    rk = momo.rank(axis=1, ascending=False, method="first")
    pick = (rk <= 2) & (momo > 0)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.astype(float).div(n, axis=0).fillna(0.0)

    storm = (vix > 30).shift(1).fillna(False)
    gate = pd.Series(0.5, index=o.index)
    gate[decomp.shift(1).fillna(False)] = 1.0
    gate[storm] = 0.0
    w = w.mul(gate, axis=0)
    w["BIL"] = (1 - w[TIDE_UNI].sum(axis=1)).clip(lower=0)

    idx = o.index
    is_fri = pd.Series(idx, index=idx).dt.dayofweek == 4
    rebal = idx[is_fri.values]
    held = hold_at(idx, w, rebal)
    return backtest_o2o(held, o)


# ============================================================================
# S7 — VIX_REB: VIX-rebound after spike (daily).
# When VIX 5d change > +5pts (spike) recently AND z-score now < 0.5 (calming),
# allocate to SPY for the rebound. Otherwise BIL.
# ============================================================================
def sleeve_vix_reb() -> pd.Series:
    o, c = panel(["SPY", "BIL"])
    vix = load_fred("VIXCLS").reindex(o.index).ffill()
    vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    in_spike = (vix_z > 1.5).rolling(20).max()
    calming = (vix_z < 0.5)
    rebound = (in_spike == 1) & calming
    rebound = rebound.shift(1).fillna(False).astype(float)
    w = pd.DataFrame(0.0, index=o.index, columns=["SPY", "BIL"])
    w["SPY"] = rebound * 1.0
    w["BIL"] = 1 - w["SPY"]
    return backtest_o2o(w, o)


# ============================================================================
# S8 — OVERNIGHT: equity overnight return premium (daily).
# Hold SPY/QQQ from close[t-1] to open[t] when SPY > 200d MA AND VIX < 25.
# Earns close-to-open returns instead of open-to-open.
# ============================================================================
def sleeve_overnight() -> pd.Series:
    tickers = ON_UNI + ["BIL"]
    o, c = panel(tickers)
    cl = c.shift(1)
    spy_close = cl["SPY"]
    sma200 = spy_close.rolling(200).mean()
    vix = load_fred("VIXCLS").reindex(o.index).ffill().shift(1)
    on = ((spy_close > sma200) & (vix < 25)).astype(float)

    w = pd.DataFrame(0.0, index=o.index, columns=tickers)
    w["SPY"] = on * 0.5
    w["QQQ"] = on * 0.5
    w["BIL"] = 1 - w["SPY"] - w["QQQ"]
    return backtest_overnight(w, o, c)


# ============================================================================
# S9 — TOM: turn-of-month SPY long.
# Long SPY in last 4 trading days of month + first 3 trading days of next.
# Calendar-deterministic; minimal correlation with all other sleeves.
# ============================================================================
def sleeve_tom() -> pd.Series:
    o, c = panel(["SPY", "BIL"])
    idx = o.index
    month = pd.Series(idx.to_period("M"), index=idx)
    is_last_4 = pd.Series(False, index=idx)
    is_first_3 = pd.Series(False, index=idx)
    for _, group in pd.Series(idx, index=idx).groupby(month):
        days = group.values
        if len(days) >= 6:
            for d in days[-4:]:
                is_last_4.loc[d] = True
            for d in days[:3]:
                is_first_3.loc[d] = True
    in_window = (is_last_4 | is_first_3)

    w = pd.DataFrame(0.0, index=idx, columns=["SPY", "BIL"])
    w.loc[in_window, "SPY"] = 1.0
    w.loc[~in_window, "BIL"] = 1.0
    return backtest_o2o(w, o)


# ============================================================================
# S10 — QUAL_DIV: quality dividend basket (monthly), SPY-trend gated.
# ============================================================================
def sleeve_qual_div() -> pd.Series:
    o, c = panel(QUAL_UNI + ["BIL", "SPY"])
    cl = c.shift(1)
    rets60 = cl[QUAL_UNI].pct_change().rolling(60).std()
    iv = 1.0 / rets60
    w = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    spy = cl["SPY"]
    on = (spy > spy.rolling(200).mean()).astype(float)
    w = w.mul(on, axis=0)
    w["BIL"] = (1 - w[QUAL_UNI].sum(axis=1)).clip(lower=0)
    w["SPY"] = 0.0
    held = hold_at(o.index, w[QUAL_UNI + ["BIL", "SPY"]], monthly_dates(o.index))
    return backtest_o2o(held, o)


# ============================================================================
# Aggregation: equal weights across all sleeves; gross stays at 1.0.
# ============================================================================
SLEEVE_NAMES = ["CARRY", "EQ_TSMOM", "DEF_TSMOM", "SECT_CSMOM",
                 "GOLD_BREAK", "VOL_COOL", "VIX_REB", "OVERNIGHT",
                 "TOM", "QUAL_DIV"]


def build_sleeves() -> pd.DataFrame:
    sl = {
        "CARRY":      sleeve_carry(),
        "EQ_TSMOM":   sleeve_eq_tsmom(),
        "DEF_TSMOM":  sleeve_def_tsmom(),
        "SECT_CSMOM": sleeve_sector_csmom(),
        "GOLD_BREAK": sleeve_gold_break(),
        "VOL_COOL":   sleeve_vol_cool(),
        "VIX_REB":    sleeve_vix_reb(),
        "OVERNIGHT":  sleeve_overnight(),
        "TOM":        sleeve_tom(),
        "QUAL_DIV":   sleeve_qual_div(),
    }
    df = pd.concat(sl, axis=1, sort=True).fillna(0.0)
    return df.loc[IS_START:]


def apply_overlays(raw: pd.Series, dd_floor=DD_FLOOR, dd_win=DD_WIN,
                   vol_gate_pct=VOL_GATE_PCT, vol_gate_lb=VOL_GATE_LOOKBACK,
                   vol_win=VOL_WIN) -> tuple[pd.Series, pd.DataFrame]:
    """Risk overlays that ONLY de-risk (multipliers <= 1.0). No leverage.
    Strict no-leverage discipline: total_mult cannot exceed 1.0.
    """
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = (cum / hwm - 1)
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)

    rv = raw.rolling(vol_win).std()
    rv_thr = rv.rolling(vol_gate_lb, min_periods=60).quantile(vol_gate_pct)
    vol_gate_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    vol_gate_mult = vol_gate_ok + (1 - vol_gate_ok) * 0.5

    total_mult = (dd_mult * vol_gate_mult).clip(upper=1.0)
    net = raw * total_mult
    state = pd.DataFrame({
        "raw": raw, "dd_mult": dd_mult, "vol_gate_mult": vol_gate_mult,
        "total_mult": total_mult, "net": net,
    })
    return net, state


def fit_blend_weights(sleeves: pd.DataFrame) -> pd.Series:
    """IS-fitted Markowitz-style weights: w_i ∝ Sharpe_i^2 / vol_i.

    Computed strictly on IS data (2010-2018) and then frozen for OOS.
    Gives more weight to high-Sharpe and low-vol sleeves; equivalent to
    a positive-only minimum-variance frontier point under uncorrelated
    sleeves. Provably no look-ahead.
    """
    is_sl = sleeves.loc[IS_START:IS_END]
    is_vol = is_sl.std() * np.sqrt(252)
    is_sharpe = (is_sl.mean() * 252) / is_vol
    is_sharpe_pos = is_sharpe.clip(lower=0.0)
    raw_w = (is_sharpe_pos ** 2 / is_vol).fillna(0.0)
    if raw_w.sum() <= 0:
        return pd.Series(1 / len(sleeves.columns), index=sleeves.columns)
    return raw_w / raw_w.sum()


def run_strategy() -> dict:
    print("Building sleeves...")
    sleeves = build_sleeves()

    weights = fit_blend_weights(sleeves)
    raw = sleeves @ weights

    print("\nPer-sleeve metrics:")
    print(f"  {'sleeve':12s} {'IS Sh':>6s} {'OOS Sh':>6s} {'FULL Sh':>7s} "
          f"{'CAGR':>7s} {'Vol':>6s} {'MDD':>6s}")
    for col in sleeves.columns:
        m_full = metrics(sleeves[col].loc[IS_START:])
        m_is = metrics(sleeves[col].loc[IS_START:IS_END])
        m_oos = metrics(sleeves[col].loc[OOS_START:])
        print(f"  {col:12s}  {m_is['sharpe']:5.2f}  {m_oos['sharpe']:5.2f}  "
              f"{m_full['sharpe']:6.2f}  {m_full['cagr']*100:5.1f}%  "
              f"{m_full['vol']*100:5.1f}%  {m_full['mdd']*100:5.1f}%")

    print("\nSleeve correlations (full sample):")
    print(sleeves.corr().round(2).to_string())

    avg_corr = sleeves.corr().values[np.triu_indices(len(sleeves.columns), k=1)].mean()
    print(f"\nAvg pairwise correlation: {avg_corr:.3f}")

    print("\nApplying portfolio risk overlays (de-risk only)...")
    net, state = apply_overlays(raw)

    m_full = metrics(net.loc[IS_START:], "FULL")
    m_is = metrics(net.loc[IS_START:IS_END], "IS")
    m_oos = metrics(net.loc[OOS_START:], "OOS")
    raw_full = metrics(raw.loc[IS_START:], "RAW_FULL")

    print("\n" + "=" * 90)
    print("MERIDIAN — final metrics (no leverage; gross <= 1.0)")
    print("=" * 90)
    for label, m in [("FULL (raw)", raw_full), ("FULL", m_full),
                     ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {label:14s}  Sh={m['sharpe']:5.2f}  CAGR={m['cagr']*100:5.1f}%  "
              f"Vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%  "
              f"Sortino={m['sortino']:5.2f}  Calmar={m['calmar']:5.2f}  "
              f"NAVx={m['navx']:.2f}")
    gap = abs(m_is["sharpe"] - m_oos["sharpe"])
    print(f"  IS-OOS gap: {gap:.2f}  Avg multiplier: {state['total_mult'].mean():.3f}")

    out = {
        "params": {"dd_floor": DD_FLOOR, "vol_gate_pct": VOL_GATE_PCT,
                    "vol_gate_lb": VOL_GATE_LOOKBACK, "tc_bps": TC_BPS,
                    "rule": "Sharpe^2/vol weights (IS-fit, frozen); gross <= 1.0; no margin; no levered ETFs",
                    "sleeves": SLEEVE_NAMES},
        "weights": {k: float(v) for k, v in weights.items()},
        "full": m_full, "is": m_is, "oos": m_oos, "raw_full": raw_full,
        "is_oos_gap": float(gap),
        "avg_pairwise_corr": float(avg_corr),
        "avg_total_mult": float(state["total_mult"].mean()),
        "correlations": sleeves.corr().round(3).to_dict(),
    }
    with open(RES / "meridian_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    state.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_returns.csv", index=False)
    sleeves.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_sleeves.csv", index=False)
    print("\nSaved data/results/meridian_metrics.json, meridian_returns.csv, "
          "meridian_sleeves.csv")
    return out


if __name__ == "__main__":
    run_strategy()
