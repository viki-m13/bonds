"""MERIDIAN-LEV — Standalone leveraged-ETF momentum strategy.

Strategy 3 of the MERIDIAN family. Standalone (does NOT use Phoenix).
Combines stock momentum + leveraged ETF momentum + Phoenix-like LETF
basket rotation, all under 1.0 gross with no margin.

Hard constraints:
  1. Leveraged ETFs ALLOWED (TQQQ, UPRO, TMF, SOXL, etc.).
  2. NO portfolio margin or borrowing — gross ≤ 1.0 every day.
  3. NO forward-looking signals.
  4. Stock universe disclosed as survivorship-biased; 3% CAGR haircut on
     stock portion.

Honest comparison to Phoenix
============================
Phoenix achieves Sharpe 2.39 / CAGR 38.5% on 2011-2026 via 5 truly
orthogonal LETF sleeves (mean correlation 0.02). Replicating that level
of cross-sleeve orthogonality with a fundamentally different toolkit
isn't achievable on this empirical work — I tested it.

What this strategy does deliver:
  - Higher CAGR than buy-and-hold or the unlevered MERIDIAN
  - Standalone (doesn't reuse Phoenix returns)
  - Genuinely different alpha source (stock momentum + LETF rotation)
  - Survivorship-bias-corrected disclosed CAGR

It DOES NOT beat Phoenix on Sharpe — Phoenix's structure is hard to
beat. It DOES outperform many leveraged-only strategies that exist in
the public literature (CTA-style, pure 3x rotation, etc.).

Strategy
========
Three sleeves, all using close[t-1] -> open[t] -> open[t+1]:

  S1 STOCK_TOP_2_W  : Top-2 stocks by 126d return, weekly Wed rebal.    40%
  S2 LETF_TOP_2_W   : Top-2 of 17 LETFs by 126d return, weekly rebal.   30%
  S3 STOCK_TOP_3_M  : Top-3 stocks by 252d return, monthly rebal.       30%

Universes:
  - 90 large-cap stocks (survivorship-biased; 3% haircut on stock portion)
  - 17 leveraged ETFs: TQQQ, UPRO, SOXL, TECL, QLD, SSO, FAS, ERX, EDC,
    YINN, DRN (3x equity); TMF, TYD, UBT (3x bonds); UGL, UCO (2x);
    NUGT (3x miners)

Risk overlays (de-risk only):
  - DD throttle, floor at -25% on 252d HWM
  - Vol-regime gate at 99th pct of 60d vol distribution

Performance (2011-2026)
=======================
  FULL  Sh=1.20  CAGR=33%  MDD=-30%  (raw before haircut)
  Haircut-adjusted CAGR: ~31% (3% × 70% stock portion = 2.1%)
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

IS_START = pd.Timestamp("2011-01-04")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")

TC_BPS = 5.0
DD_FLOOR = -0.25
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252
VOL_WIN = 60

SURVIVORSHIP_HAIRCUT_PCT = 3.0

LETF_UNIVERSE = ["TQQQ", "UPRO", "SOXL", "TECL", "QLD", "SSO", "FAS", "ERX",
                 "EDC", "YINN", "DRN", "TMF", "TYD", "UBT", "UGL", "UCO", "NUGT"]


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


def topk_sleeve(universe, opens, closes, top_k, lookback, freq, tc_bps):
    cl = closes.shift(1)
    momo = cl[universe].pct_change(lookback)
    eligible = momo > 0
    rk = momo.where(eligible).rank(axis=1, ascending=False, method="first")
    pick = (rk <= top_k).astype(float)
    n = pick.sum(axis=1).replace(0, np.nan)
    w = pick.div(n, axis=0).fillna(0.0)
    weights = pd.DataFrame(0.0, index=opens.index, columns=opens.columns)
    for col in universe:
        weights[col] = w[col]
    weights["BIL"] = (1 - weights[universe].sum(axis=1)).clip(lower=0)
    idx = opens.index
    if freq == "W":
        rebal_mask = pd.Series(idx, index=idx).dt.dayofweek == 2
        held = weights.copy(); held[~rebal_mask.values] = np.nan
        held = held.ffill().fillna(0.0)
    elif freq == "M":
        m = pd.Series(idx, index=idx).groupby(
            [idx.year, idx.month]).transform("first") == pd.Series(idx, index=idx)
        held = weights.copy(); held[~m.values] = np.nan
        held = held.ffill().fillna(0.0)
    o2o = opens.pct_change()
    held_lag = held.shift(1).fillna(0.0)
    ret = (held_lag * o2o.reindex(columns=held.columns)).sum(axis=1)
    turnover = (held - held.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = (turnover * tc_bps / 1e4).shift(1).fillna(0.0)
    return ret - cost


def apply_overlays(raw, dd_floor=DD_FLOOR, dd_win=DD_WIN,
                   vol_gate_pct=VOL_GATE_PCT, vol_gate_lb=VOL_GATE_LOOKBACK,
                   vol_win=VOL_WIN):
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = (cum / hwm - 1)
    dd_mult = (1.0 + dd / dd_floor).clip(lower=0.0, upper=1.0).shift(1).fillna(1.0)
    rv = raw.rolling(vol_win).std()
    rv_thr = rv.rolling(vol_gate_lb, min_periods=60).quantile(vol_gate_pct)
    vol_gate_ok = (rv <= rv_thr).shift(1).fillna(True).astype(float)
    vg_mult = vol_gate_ok + (1 - vol_gate_ok) * 0.5
    total_mult = (dd_mult * vg_mult).clip(upper=1.0)
    return raw * total_mult


def run_strategy():
    # Build panels
    opens_d, closes_d = {}, {}
    for t in STOCK_UNIVERSE:
        d = load_etf(t, folder="stocks")
        if d is not None:
            opens_d[t] = d["Open"]; closes_d[t] = d["Close"]
    for t in LETF_UNIVERSE + ["BIL"]:
        d = load_etf(t, folder="etfs")
        if d is not None:
            opens_d[t] = d["Open"]; closes_d[t] = d["Close"]
    opens = pd.DataFrame(opens_d); closes = pd.DataFrame(closes_d)
    idx = pd.bdate_range(IS_START, closes.index.max())
    opens = opens.reindex(idx).ffill(limit=3); closes = closes.reindex(idx).ffill(limit=3)

    # 3 sleeves
    s1 = topk_sleeve(STOCK_UNIVERSE, opens, closes, 2, 126, "W", tc_bps=3.0)
    s2 = topk_sleeve(LETF_UNIVERSE, opens, closes, 2, 126, "W", tc_bps=5.0)
    s3 = topk_sleeve(STOCK_UNIVERSE, opens, closes, 3, 252, "M", tc_bps=3.0)

    sleeves = pd.concat({"STK_2_W": s1, "LETF_2_W": s2, "STK_3_M": s3},
                        axis=1).fillna(0.0).loc[IS_START:]

    print("Per-sleeve metrics:")
    for col in sleeves.columns:
        m = metrics(sleeves[col].loc[IS_START:])
        print(f"  {col:10s}: Sh={m['sharpe']:.2f} CAGR={m['cagr']*100:.1f}% MDD={m['mdd']*100:.1f}% Sortino={m['sortino']:.2f}")

    print("\nCorrelations:")
    print(sleeves.corr().round(2).to_string())

    # 40/30/30 blend
    weights = pd.Series({"STK_2_W": 0.40, "LETF_2_W": 0.30, "STK_3_M": 0.30})
    raw = sleeves @ weights

    print("\nApplying overlays (de-risk only)...")
    net = apply_overlays(raw)

    m_full = metrics(net.loc[IS_START:], "FULL")
    m_is = metrics(net.loc[IS_START:IS_END], "IS")
    m_oos = metrics(net.loc[OOS_START:], "OOS")
    m_raw = metrics(raw.loc[IS_START:], "RAW")

    stock_w = weights["STK_2_W"] + weights["STK_3_M"]  # 70% stocks
    haircut = m_full["cagr"] - SURVIVORSHIP_HAIRCUT_PCT / 100.0 * stock_w

    print(f"\n{'='*70}")
    print(f"MERIDIAN-LEV — standalone leveraged-ETF strategy (no Phoenix)")
    print(f"{'='*70}")
    for label, m in [("RAW", m_raw), ("FULL", m_full), ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {label:5s}: Sh={m['sharpe']:.2f} CAGR={m['cagr']*100:.1f}% Vol={m['vol']*100:.1f}% "
              f"MDD={m['mdd']*100:.1f}% Sortino={m['sortino']:.2f} Calmar={m['calmar']:.2f}")
    print(f"\n  Haircut-adjusted FULL CAGR: {haircut*100:.1f}% "
          f"({SURVIVORSHIP_HAIRCUT_PCT}% × {stock_w*100:.0f}% stock = {SURVIVORSHIP_HAIRCUT_PCT*stock_w:.1f}% blended)")

    # Phoenix benchmark
    try:
        phx = pd.read_csv(RES / "phoenix_production_returns.csv", parse_dates=["Date"]).set_index("Date")
        phx_r = phx["net_ret"].dropna().loc[IS_START:]
        m_phx = metrics(phx_r, "PHOENIX")
        print(f"\n  PHOENIX benchmark (same window): Sh={m_phx['sharpe']:.2f} CAGR={m_phx['cagr']*100:.1f}% MDD={m_phx['mdd']*100:.1f}%")
    except FileNotFoundError:
        pass

    out = {
        "params": {"tc_bps": TC_BPS, "dd_floor": DD_FLOOR,
                    "vol_gate_pct": VOL_GATE_PCT, "vol_gate_lb": VOL_GATE_LOOKBACK,
                    "survivorship_haircut_pct": SURVIVORSHIP_HAIRCUT_PCT,
                    "stock_weight_total": float(stock_w),
                    "rule": "Standalone LETF strategy (no Phoenix). 40% stock top-2 weekly + "
                             "30% LETF top-2 weekly + 30% stock top-3 monthly.",
                    "letf_universe": LETF_UNIVERSE, "stock_universe_size": len(STOCK_UNIVERSE)},
        "weights": {k: float(v) for k, v in weights.items()},
        "full": m_full, "is": m_is, "oos": m_oos, "raw_full": m_raw,
        "cagr_haircut": float(haircut),
        "correlations": sleeves.corr().round(3).to_dict(),
    }
    with open(RES / "meridian_lev_metrics.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    state_df = pd.DataFrame({"raw": raw, "net": net})
    state_df.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_lev_returns.csv", index=False)
    sleeves.reset_index().rename(columns={"index": "Date"}).to_csv(
        RES / "meridian_lev_sleeves.csv", index=False)
    return out


if __name__ == "__main__":
    run_strategy()
