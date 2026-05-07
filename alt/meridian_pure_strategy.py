"""MERIDIAN-PURE — Pure long-only stock momentum top-3.

The simplest and most aggressive of the three MERIDIAN variants. Long-only
top-3 large-cap stocks by 6-month momentum, weekly rebalance.

Hard constraints:
  1. NO leveraged or inverse ETFs.
  2. NO portfolio margin or borrowing.
  3. NO forward-looking signals.
  4. Stock universe disclosed as survivorship-biased; 3% CAGR haircut applied.

Survivorship-bias accounting
The 90 stocks in `data/stocks/` are currently-listed S&P 500 large-caps
that survived to 2026. Concentrated top-3 amplifies survivorship bias —
we apply a conservative 3% CAGR haircut to the disclosed CAGR.

Strategy
========
Single sleeve:
  - Universe: 90 large-cap stocks (`data/stocks/` with 2010+ history)
  - Eligibility: 126-day return > 0
  - Position: top-3 equal-weight, weekly rebalance (Wed)
  - Cash residual: BIL

Risk overlays (de-risk only):
  - DD throttle: floor at -25% on 252d HWM
  - Vol-regime gate: halve exposure above 99th percentile of 60d realized vol

Performance
===========
  FULL  Sh=1.18  CAGR=33.0% (haircut: 30.0%)  MDD=-26.5%
  IS    Sh=1.30  CAGR=26.7%
  OOS   Sh=1.06  CAGR=42.8%

Why this matters: simpler than the composite ensemble, higher CAGR,
slightly lower Sharpe and higher MDD. For investors who prefer maximum
CAGR with one mental model.
"""
from __future__ import annotations
from pathlib import Path
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF = ROOT / "data" / "etfs"
STOCK = ROOT / "data" / "stocks"
RES = ROOT / "data" / "results"

IS_START = pd.Timestamp("2010-01-04")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")

TC_BPS = 3.0
DD_FLOOR = -0.25
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252
VOL_WIN = 60
SURVIVORSHIP_HAIRCUT_PCT = 3.0


def load_etf(t, folder="etfs"):
    base = ETF if folder == "etfs" else STOCK
    p = base / f"{t}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Open", "Close"]].astype(float)


STOCK_UNIVERSE = []
for f in sorted(os.listdir(STOCK)):
    if not f.endswith(".csv"): continue
    t = f.replace(".csv", "")
    df = load_etf(t, folder="stocks")
    if df is not None and df.index[0] <= IS_START:
        STOCK_UNIVERSE.append(t)


def metrics(r, name=""):
    r = r.dropna()
    if len(r) < 30: return {"name": name, "sharpe": 0}
    mu = r.mean() * 252; sd = r.std() * np.sqrt(252)
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


def run_strategy():
    opens_d, closes_d = {}, {}
    for t in STOCK_UNIVERSE + ["BIL"]:
        folder = "stocks" if t != "BIL" else "etfs"
        d = load_etf(t, folder=folder)
        if d is not None:
            opens_d[t] = d["Open"]; closes_d[t] = d["Close"]
    opens = pd.DataFrame(opens_d); closes = pd.DataFrame(closes_d)
    idx = pd.bdate_range(IS_START, closes.index.max())
    opens = opens.reindex(idx).ffill(limit=3); closes = closes.reindex(idx).ffill(limit=3)
    cl = closes.shift(1)

    # Top-3 by 126d momentum, weekly Wednesday rebalance
    momo = cl[STOCK_UNIVERSE].pct_change(126)
    eligible = momo > 0
    rk = momo.where(eligible).rank(axis=1, ascending=False, method="first")
    pick = (rk <= 3).astype(float)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.div(n, axis=0).fillna(0.0)
    weights = pd.DataFrame(0.0, index=idx, columns=opens.columns)
    for col in STOCK_UNIVERSE: weights[col] = w[col]
    weights["BIL"] = (1 - weights[STOCK_UNIVERSE].sum(axis=1)).clip(lower=0)

    is_wed = pd.Series(idx, index=idx).dt.dayofweek == 2
    held = weights.copy(); held[~is_wed.values] = np.nan
    held = held.ffill().fillna(0.0)
    o2o = opens.pct_change()
    held_lag = held.shift(1).fillna(0.0)
    raw = (held_lag * o2o.reindex(columns=held.columns)).sum(axis=1)
    tov = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (tov * TC_BPS / 1e4).shift(1).fillna(0.0)
    raw_ret = raw - cost

    # Apply overlays
    cum = (1 + raw_ret).cumprod()
    hwm = cum.rolling(DD_WIN, min_periods=30).max()
    dd = (cum / hwm - 1)
    dd_mult = (1.0 + dd / DD_FLOOR).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)
    rv = raw_ret.rolling(VOL_WIN).std()
    rv_thr = rv.rolling(VOL_GATE_LOOKBACK, min_periods=60).quantile(VOL_GATE_PCT)
    vol_gate_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    vg_mult = vol_gate_ok + (1 - vol_gate_ok) * 0.5
    total_mult = (dd_mult * vg_mult).clip(upper=1.0)
    net = raw_ret * total_mult

    state = pd.DataFrame({"raw": raw_ret, "dd_mult": dd_mult, "vol_gate_mult": vg_mult,
                          "total_mult": total_mult, "net": net})

    m_full = metrics(net.loc[IS_START:], "FULL")
    m_is = metrics(net.loc[IS_START:IS_END], "IS")
    m_oos = metrics(net.loc[OOS_START:], "OOS")
    m_raw = metrics(raw_ret.loc[IS_START:], "RAW")
    haircut = m_full["cagr"] - SURVIVORSHIP_HAIRCUT_PCT / 100.0

    print(f"MERIDIAN-PURE: top-3 stock momentum (126d, weekly), {len(STOCK_UNIVERSE)} stocks")
    print(f"⚠ Survivorship-biased universe — {SURVIVORSHIP_HAIRCUT_PCT}% CAGR haircut applied")
    print()
    for label, m in [("RAW", m_raw), ("FULL", m_full), ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {label:5s}: Sh={m['sharpe']:.2f} CAGR={m['cagr']*100:.1f}% Vol={m['vol']*100:.1f}% "
              f"MDD={m['mdd']*100:.1f}% Sortino={m['sortino']:.2f} Calmar={m['calmar']:.2f}")
    print(f"\nSurvivorship-haircut FULL CAGR: {haircut*100:.1f}%")

    out = {
        "params": {"top_k": 3, "lookback": 126, "rebal": "weekly_wed",
                    "tc_bps": TC_BPS, "dd_floor": DD_FLOOR, "vol_gate_pct": VOL_GATE_PCT,
                    "survivorship_haircut_pct": SURVIVORSHIP_HAIRCUT_PCT,
                    "stock_universe_size": len(STOCK_UNIVERSE),
                    "rule": "Pure long-only top-3 stock momentum, weekly rebal."},
        "full": m_full, "is": m_is, "oos": m_oos, "raw_full": m_raw,
        "cagr_haircut": float(haircut),
    }
    with open(RES / "meridian_pure_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    state.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_pure_returns.csv", index=False)
    return out


if __name__ == "__main__":
    run_strategy()
