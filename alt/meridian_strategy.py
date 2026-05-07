"""MERIDIAN — Strict-no-leverage daily-managed broad-universe tactical strategy.

Hard constraints (in priority order):
  1. NEVER hold any leveraged or inverse ETF (no 2x/3x/-1x products).
  2. NEVER use portfolio-level margin or borrowing. Sum of weights at any
     time t is bounded at 1.0 (cash residual goes to BIL).
  3. NEVER use forward-looking data. All signals computed on close[t-1];
     positions established at open[t]; returns earned open[t]→open[t+1].
  4. NO selection bias toward winners. The universe is fixed ex-ante to a
     broad set of 33 liquid 1x ETFs spanning every major asset class and
     region. Sector concentration emerges only from systematic momentum
     rules applied uniformly to all members.
  5. Daily-managed eligibility checks; weekly rebalance to keep turnover
     manageable.

Universe (fixed ex-ante by liquidity and inception)
===================================================
33 ETFs across asset classes:
  Broad equity (5)   : SPY, QQQ, IWM, EFA, EEM
  US sectors (9)     : XLK, XLY, XLP, XLU, XLV, XLE, XLF, XLI, XLB
  Sub-sectors (6)    : SMH, XBI, ITB, XHB, TAN, VNQ
  International (2)  : EWJ, FXI
  Treasuries (4)     : TLT, IEF, IEI, SHY
  Credit/TIPS (4)    : HYG, LQD, EMB, TIP
  Commodities (3)    : GLD, SLV, DBC

Strategy
========
Three rule-based sleeves applied uniformly across the entire universe.

  S1 COMPOSITE-MOMO   — top-2 by composite of (42d momo rank, 126d momo
                        rank, 126d/63d-vol risk-adjusted rank). Eligibility:
                        positive 6-month return AND > 200d SMA.
                        Daily check, weekly rebalance.

  S2 SECTOR-ROTATION  — top-3 cross-sectional sector momentum on 9 SPDRs;
                        SPY > 200d SMA gate. Daily check, weekly rebalance.

  S3 DEFENSIVE-RP     — inverse-vol blend of {TLT, GLD, IEF}, active only
                        when SPY < 200d SMA. Monthly rebalance.

Aggregator
==========
EQUAL weights (1/3 each). No IS-fitted blending.
Total portfolio gross is exactly 1.0. No margin.

Risk overlays at portfolio level (de-risk only, multipliers in [0, 1]):
  - Drawdown throttle: linear scale toward zero as NAV falls below the
    252d HWM, floor at -15%.
  - Vol-regime gate: halve exposure when 60d realized vol > 99th
    percentile of 252d trailing distribution.
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

TC_BPS = 5.0
DD_FLOOR = -0.15
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252
VOL_WIN = 60

BROAD_EQUITY = ["SPY", "QQQ", "IWM", "EFA", "EEM"]
SECTORS = ["XLK", "XLY", "XLP", "XLU", "XLV", "XLE", "XLF", "XLI", "XLB"]
SUB_SECTORS = ["SMH", "XBI", "ITB", "XHB", "TAN", "VNQ"]
INTL = ["EWJ", "FXI"]
TREASURY = ["TLT", "IEF", "IEI", "SHY"]
CREDIT_TIPS = ["HYG", "LQD", "EMB", "TIP"]
COMMODITY = ["GLD", "SLV", "DBC"]

UNIVERSE = (BROAD_EQUITY + SECTORS + SUB_SECTORS + INTL +
            TREASURY + CREDIT_TIPS + COMMODITY)
DEFENSIVE = ["TLT", "GLD", "IEF"]


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
    idx = pd.bdate_range(IS_START, c.index.max())
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


def backtest_o2o(weights: pd.DataFrame, opens: pd.DataFrame,
                 tc_bps: float = TC_BPS) -> tuple[pd.Series, pd.Series]:
    w = weights.reindex(columns=opens.columns).fillna(0.0)
    o2o = opens.pct_change()
    w_held = w.shift(1).fillna(0.0)
    gross = (w_held * o2o).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (turnover * tc_bps / 1e4).shift(1).fillna(0.0)
    return gross - cost, turnover


def hold_at(idx, weights, rebal_dates):
    held = pd.DataFrame(0.0, index=idx, columns=weights.columns)
    cur = pd.Series(0.0, index=weights.columns)
    rb = set(rebal_dates)
    for dt in idx:
        if dt in rb and dt in weights.index and not weights.loc[dt].isna().all():
            cur = weights.loc[dt].fillna(0.0)
        held.loc[dt] = cur
    return held


def weekly_rebalance(idx, weights, dow=2):
    is_dow = pd.Series(idx, index=idx).dt.dayofweek == dow
    rebal = idx[is_dow.values]
    return hold_at(idx, weights, rebal)


def monthly_rebalance(idx, weights):
    m = pd.Series(idx, index=idx).groupby(
        [idx.year, idx.month]).transform("first") == pd.Series(idx, index=idx)
    rebal = idx[m.values]
    return hold_at(idx, weights, rebal)


# ============================================================================
# S1 — COMPOSITE-MOMO (broad universe)
# Composite of 42d momentum rank, 126d momentum rank, 126d/63d-vol Sharpe rank.
# ============================================================================
def sleeve_composite_momo(top_k: int = 2) -> pd.Series:
    o, c = panel(UNIVERSE + ["BIL"])
    cl = c.shift(1)
    momo_short = cl[UNIVERSE].pct_change(42)
    momo_long = cl[UNIVERSE].pct_change(126)
    vol_lb = 63
    daily_rets = cl[UNIVERSE].pct_change()
    vol_63 = daily_rets.rolling(vol_lb).std()
    risk_adj = momo_long / vol_63

    # Rank-based composite (each component scored as percentile rank)
    sig = (momo_short.rank(axis=1, pct=True) +
           momo_long.rank(axis=1, pct=True) +
           risk_adj.rank(axis=1, pct=True)) / 3.0

    sma200 = cl[UNIVERSE].rolling(200).mean()
    eligible = (momo_long > 0) & (cl[UNIVERSE] > sma200)
    rk = sig.where(eligible).rank(axis=1, ascending=False, method="first")
    pick = (rk <= top_k).astype(float)
    n = pick.sum(axis=1).replace(0, np.nan)
    w_eq = pick.div(n, axis=0).fillna(0.0)

    weights = pd.DataFrame(0.0, index=o.index, columns=UNIVERSE + ["BIL"])
    for col in UNIVERSE:
        weights[col] = w_eq[col]
    weights["BIL"] = (1 - weights[UNIVERSE].sum(axis=1)).clip(lower=0)

    held = weekly_rebalance(o.index, weights, dow=2)
    ret, _ = backtest_o2o(held, o)
    return ret


# ============================================================================
# S2 — SECTOR-ROTATION (cross-sectional, 9 SPDRs)
# ============================================================================
def sleeve_sector_rotation(top_k: int = 3) -> pd.Series:
    o, c = panel(SECTORS + ["SPY", "BIL"])
    cl = c.shift(1)
    momo_3m = cl[SECTORS].pct_change(63)
    momo_6m = cl[SECTORS].pct_change(126)
    sig = (momo_3m + momo_6m) / 2.0
    sma200 = cl[SECTORS].rolling(200).mean()
    eligible = (momo_6m > 0) & (cl[SECTORS] > sma200)
    rk = sig.where(eligible).rank(axis=1, ascending=False, method="first")
    pick = (rk <= top_k).astype(float)
    n = pick.sum(axis=1).replace(0, np.nan)
    w_eq = pick.div(n, axis=0).fillna(0.0)

    spy = cl["SPY"]
    on = (spy > spy.rolling(200).mean()).astype(float)
    w_eq = w_eq.mul(on, axis=0)

    weights = pd.DataFrame(0.0, index=o.index, columns=SECTORS + ["SPY", "BIL"])
    for col in SECTORS:
        weights[col] = w_eq[col]
    weights["BIL"] = (1 - weights[SECTORS].sum(axis=1)).clip(lower=0)
    weights["SPY"] = 0.0
    held = weekly_rebalance(o.index, weights[SECTORS + ["BIL", "SPY"]], dow=2)
    ret, _ = backtest_o2o(held, o)
    return ret


# ============================================================================
# S3 — DEFENSIVE RISK-PARITY (active only when SPY < 200d SMA)
# ============================================================================
def sleeve_def_risk_parity() -> pd.Series:
    o, c = panel(DEFENSIVE + ["SPY", "BIL"])
    cl = c.shift(1)
    spy = cl["SPY"]
    risk_off = (spy < spy.rolling(200).mean()).astype(float)

    rets60 = cl[DEFENSIVE].pct_change().rolling(60).std()
    iv = 1.0 / rets60
    iv_w = iv.div(iv.sum(axis=1), axis=0).fillna(0.0)
    momo60 = cl[DEFENSIVE].pct_change(60)
    iv_w = iv_w.where(momo60 > 0, 0.0)

    weights = pd.DataFrame(0.0, index=o.index, columns=DEFENSIVE + ["SPY", "BIL"])
    for col in DEFENSIVE:
        weights[col] = iv_w[col].mul(risk_off, axis=0).fillna(0.0)
    weights["BIL"] = (1 - weights[DEFENSIVE].sum(axis=1)).clip(lower=0)
    weights["SPY"] = 0.0
    held = monthly_rebalance(o.index, weights[DEFENSIVE + ["BIL", "SPY"]])
    ret, _ = backtest_o2o(held, o)
    return ret


SLEEVE_NAMES = ["COMPOSITE_MOMO", "SECTOR_ROT", "DEF_RP"]
SLEEVE_WEIGHTS = pd.Series({"COMPOSITE_MOMO": 1/3, "SECTOR_ROT": 1/3, "DEF_RP": 1/3})


def build_sleeves() -> pd.DataFrame:
    sl = {
        "COMPOSITE_MOMO": sleeve_composite_momo(),
        "SECTOR_ROT": sleeve_sector_rotation(),
        "DEF_RP": sleeve_def_risk_parity(),
    }
    df = pd.concat(sl, axis=1, sort=True).fillna(0.0)
    return df.loc[IS_START:]


def apply_overlays(raw: pd.Series, dd_floor=DD_FLOOR, dd_win=DD_WIN,
                   vol_gate_pct=VOL_GATE_PCT, vol_gate_lb=VOL_GATE_LOOKBACK,
                   vol_win=VOL_WIN) -> tuple[pd.Series, pd.DataFrame]:
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


def run_strategy() -> dict:
    print(f"Universe: {len(UNIVERSE)} ETFs, no selection bias.")
    print("Building sleeves...")
    sleeves = build_sleeves()

    print("\nPer-sleeve metrics:")
    print(f"  {'sleeve':16s} {'IS Sh':>6s} {'OOS Sh':>6s} {'FULL Sh':>7s} "
          f"{'CAGR':>7s} {'Vol':>6s} {'MDD':>6s}")
    for col in sleeves.columns:
        m_full = metrics(sleeves[col].loc[IS_START:])
        m_is = metrics(sleeves[col].loc[IS_START:IS_END])
        m_oos = metrics(sleeves[col].loc[OOS_START:])
        print(f"  {col:16s}  {m_is['sharpe']:5.2f}  {m_oos['sharpe']:5.2f}  "
              f"{m_full['sharpe']:6.2f}  {m_full['cagr']*100:5.1f}%  "
              f"{m_full['vol']*100:5.1f}%  {m_full['mdd']*100:5.1f}%")

    print("\nSleeve correlations (full sample):")
    print(sleeves.corr().round(2).to_string())

    raw = sleeves @ SLEEVE_WEIGHTS

    print("\nApplying portfolio risk overlays (de-risk only)...")
    net, state = apply_overlays(raw)

    m_full = metrics(net.loc[IS_START:], "FULL")
    m_is = metrics(net.loc[IS_START:IS_END], "IS")
    m_oos = metrics(net.loc[OOS_START:], "OOS")
    raw_full = metrics(raw.loc[IS_START:], "RAW_FULL")

    print("\n" + "=" * 90)
    print("MERIDIAN — final metrics (no leverage; no margin; no levered ETFs;"
          " unbiased universe)")
    print("=" * 90)
    for label, m in [("FULL (raw)", raw_full), ("FULL", m_full),
                     ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {label:14s}  Sh={m['sharpe']:5.2f}  CAGR={m['cagr']*100:5.1f}%  "
              f"Vol={m['vol']*100:5.1f}%  MDD={m['mdd']*100:5.1f}%  "
              f"Sortino={m['sortino']:5.2f}  Calmar={m['calmar']:5.2f}  "
              f"NAVx={m['navx']:.2f}")
    gap = abs(m_is["sharpe"] - m_oos["sharpe"])
    print(f"  IS-OOS gap: {gap:.2f}  Avg de-risk multiplier: {state['total_mult'].mean():.3f}")

    out = {
        "params": {"dd_floor": DD_FLOOR, "vol_gate_pct": VOL_GATE_PCT,
                    "vol_gate_lb": VOL_GATE_LOOKBACK, "tc_bps": TC_BPS,
                    "rule": ("Equal-weighted 3-sleeve ensemble: COMPOSITE-MOMO "
                             "(top-2 by 42d/126d/risk-adj rank, broad 33-ETF), "
                             "SECTOR-ROT (top-3 sector cross-sectional with "
                             "SPY-trend gate), DEF-RP (inverse-vol "
                             "{TLT,GLD,IEF} when SPY<200dSMA)."),
                    "universe": UNIVERSE, "sleeves": SLEEVE_NAMES},
        "weights": SLEEVE_WEIGHTS.to_dict(),
        "full": m_full, "is": m_is, "oos": m_oos, "raw_full": raw_full,
        "is_oos_gap": float(gap),
        "avg_total_mult": float(state["total_mult"].mean()),
        "correlations": sleeves.corr().round(3).to_dict(),
    }
    with open(RES / "meridian_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    state.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_returns.csv", index=False)
    sleeves.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_sleeves.csv", index=False)
    print("\nSaved data/results/meridian_metrics.json, meridian_returns.csv, meridian_sleeves.csv")
    return out


if __name__ == "__main__":
    run_strategy()
