"""
POLARIS v2 — Adds Stock-Sharpe-Rank as a 5th Sleeve
====================================================

POLARIS v2 augments the original 4-sleeve POLARIS (VOLT_RP, DONCHIAN_BO,
VRP, BOND_DIP -- all built from leveraged ETFs) with a 5th sleeve that
uses INDIVIDUAL STOCKS:

    S5.  STK_SHARPE_RANK  --  weekly top-K by 126d realised Sharpe

This is genuinely novel relative to both POLARIS v1 (no stock-level
exposure) and Meridian (which uses 126d *return*-momentum on the same
stock universe; v2 uses 126d *Sharpe* ranking on a different rebalance
day).

----------------------------------------------------------------------
Survivorship-bias DISCLOSURE
----------------------------------------------------------------------
The 90-stock universe is the *current* S&P 500 large caps with full
data back to 2010, sourced from `data/stocks/`. Bankrupt/delisted
names (Lehman, WaMu, Wachovia etc.) are NOT in the dataset. This
introduces survivorship bias in the stock sleeve. Following Meridian,
we apply a conservative *3% CAGR haircut* in the headline numbers
proportional to stock weight (the stock sleeve typically receives
~18-20% of the IS inverse-vol weighting).

----------------------------------------------------------------------
S5.  STK_SHARPE_RANK
----------------------------------------------------------------------
Universe: 90 large-cap US stocks listed by 2010-01-01 (full IS coverage).
Signal:
    sharpe_126d[s, t] = mean_126d_returns / std_126d_returns   (lagged)
Construction:
    * Pick top-K=5 stocks by sharpe_126d each Friday (different rebal
      day from Meridian's Wednesday).
    * Equal-weight the K picks (gross 1.0).
    * Eligibility: sharpe > 0 (no shorting; if all negative, cash).
    * Same self-throttle (252d HWM, -20% floor) as other sleeves.
    * Stock signals are LAGGED shift(1); fills at next day's open;
      5 bps TC one-way on |dw|.

NEUTRINO/POLARIS-style rate-velocity gate is NOT applied to STK_SHARPE_RANK
because it would defeat the diversification purpose and the gate is
designed for leveraged-ETF protection.

----------------------------------------------------------------------
S1-S4 are unchanged from POLARIS v1
----------------------------------------------------------------------

----------------------------------------------------------------------
Headline (full-period 2010-03-11 -> 2026-05-06)
----------------------------------------------------------------------
POLARIS v1 (4 sleeves):  Sharpe 1.16  CAGR 23.0%  MDD -24.4%
POLARIS v2 (this, 5):    Sharpe 1.31  CAGR 26.4%  MDD -21.7%
                         OOS Sharpe 1.37 (vs 1.28 v1)
                         CAGR (with 3% survivorship haircut on stock sleeve):
                         ~26.0%  (haircut applied to stock-weighted CAGR)

Sharpe lift: +0.15. OOS Sharpe lift: +0.09. MDD: -3pp tighter.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data" / "etfs"
FRED_DIR = ROOT / "data" / "fred"
STK_DIR = ROOT / "data" / "stocks"
RESULTS = ROOT / "data" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

IS_START   = pd.Timestamp("2010-03-11")
IS_END     = pd.Timestamp("2018-12-31")
OOS_START  = pd.Timestamp("2019-01-01")

TC_BPS = 5.0
TC_RATE = TC_BPS / 1e4
DAYS = 252

SELF_DD_WIN   = 252
SELF_DD_FLOOR = -0.20

TARGET_VOL = 0.18
VT_LOOKBACK = 60
VT_MIN_PERIODS = 20
VT_LOWER, VT_UPPER = 0.5, 2.5
PORT_DD_WIN = 252
PORT_DD_FLOOR = -0.15

# Stock sleeve params
STK_K = 5                  # top-K stocks
STK_LOOKBACK = 126         # 6-month Sharpe lookback
STK_REBAL_DOW = 4          # Friday (different from Meridian's Wednesday)
STK_INCEPTION_CUTOFF = pd.Timestamp("2010-01-01")
SURVIVORSHIP_HAIRCUT = 0.03   # 3% CAGR haircut on stock sleeve, conservative


def _load_etf(t: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{t}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def _load_fred(name: str) -> pd.Series:
    df = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[name].astype(float)


def _load_stk(t: str) -> pd.DataFrame:
    df = pd.read_csv(STK_DIR / f"{t}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def build_panels(tickers: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for t in tickers:
        d = _load_etf(t)
        opens[t] = d["Open"]; closes[t] = d["Close"]
    O = pd.DataFrame(opens).sort_index()
    C = pd.DataFrame(closes).sort_index()
    idx = pd.bdate_range(O.index.min(), O.index.max())
    O = O.reindex(idx).ffill(limit=2)
    C = C.reindex(idx).ffill(limit=2)
    return O, C


def build_stock_panels(idx: pd.DatetimeIndex):
    """Build (open, close) panels for the eligible stock universe."""
    all_stk = sorted([f.replace(".csv", "") for f in os.listdir(STK_DIR) if f.endswith(".csv")])
    valid = []
    for s in all_stk:
        df = _load_stk(s)
        if df.index.min() <= STK_INCEPTION_CUTOFF:
            valid.append(s)
    opens = pd.DataFrame({s: _load_stk(s)["Open"].reindex(idx).ffill(limit=2) for s in valid})
    closes = pd.DataFrame({s: _load_stk(s)["Close"].reindex(idx).ffill(limit=2) for s in valid})
    return opens, closes, valid


def backtest_open_to_open(weights: pd.DataFrame, opens: pd.DataFrame,
                          tc_rate: float = TC_RATE) -> pd.DataFrame:
    common = [c for c in weights.columns if c in opens.columns]
    W = weights[common].fillna(0.0)
    O = opens[common].reindex(W.index)
    o2o_fwd = O.shift(-1) / O - 1.0
    gross = (W * o2o_fwd).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * tc_rate
    net = (gross - cost).fillna(0.0)
    return pd.DataFrame({"gross_ret": gross.fillna(0.0), "cost": cost,
                         "net_ret": net, "turnover": turnover})


def self_throttle(r: pd.Series, dd_win: int = SELF_DD_WIN,
                  dd_floor: float = SELF_DD_FLOOR) -> pd.Series:
    cum = (1 + r).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = cum / hwm - 1.0
    mult = (1.0 + dd / dd_floor).clip(0.0, 1.0).shift(1).fillna(1.0)
    return r * mult


def _rate_velocity_gate(idx) -> pd.Series:
    dgs10 = _load_fred("DGS10").reindex(idx).ffill()
    rv_yoy = (dgs10 - dgs10.shift(252)).shift(1)
    g = pd.Series(1.0, index=idx)
    g[rv_yoy > 1.0] = 0.5
    g[rv_yoy > 2.0] = 0.0
    return g


# --------------------------------------------------------------------------- #
#  S1-S4 unchanged from POLARIS v1
# --------------------------------------------------------------------------- #
S1_TVOL = {"QLD": 0.20, "TYD": 0.10, "UGL": 0.10}
S1_VOL_LB = 21
S1_GROSS_CAP = 1.5


def sleeve_volt_rp(opens, closes):
    cols = [c for c in S1_TVOL if c in opens.columns]
    idx = opens.index
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for t in cols:
        r = closes[t].pct_change()
        v = r.rolling(S1_VOL_LB).std() * np.sqrt(DAYS)
        W[t] = (S1_TVOL[t] / v).clip(0, S1_GROSS_CAP).shift(1).fillna(0.0)
    g = _rate_velocity_gate(idx)
    return W.mul(g, axis=0)


def sleeve_donchian_bo(opens, closes):
    idx = opens.index
    qc = closes["QLD"]
    hh = qc.shift(1).rolling(40).max()
    ll = qc.shift(1).rolling(20).min()
    long = pd.Series(0.0, index=idx)
    pos = 0
    for i in range(2, len(idx)):
        c_prev = qc.iloc[i - 1]
        if pos == 0 and pd.notna(hh.iloc[i - 1]) and c_prev >= hh.iloc[i - 1]:
            pos = 1
        elif pos == 1 and pd.notna(ll.iloc[i - 1]) and c_prev <= ll.iloc[i - 1]:
            pos = 0
        long.iloc[i] = float(pos)
    return pd.DataFrame({"QLD": long * _rate_velocity_gate(idx)}, index=idx)


def sleeve_vrp(opens, closes):
    idx = opens.index
    spy_close = _load_etf("SPY")["Close"].reindex(idx).ffill()
    spy_ret = spy_close.pct_change()
    rv21_ann = spy_ret.rolling(21).std() * np.sqrt(DAYS) * 100.0
    vix = _load_fred("VIXCLS").reindex(idx).ffill()
    vrp = (vix - rv21_ann).shift(1)
    rv = rv21_ann.shift(1)
    cols = [c for c in ["QLD", "UGL"] if c in opens.columns]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    if "QLD" in cols:
        W.loc[((vrp > 5.0) & (rv < 25.0)).fillna(False), "QLD"] = 1.0
    if "UGL" in cols:
        W.loc[(rv > 30.0).fillna(False), "UGL"] = 1.0
    return W


def sleeve_bond_dip(opens):
    idx = opens.index
    dgs10 = _load_fred("DGS10").reindex(idx).ffill()
    t10y2y = _load_fred("T10Y2Y").reindex(idx).ffill()
    bull = ((dgs10 < dgs10.rolling(60).mean())
            & (t10y2y > -0.5)).shift(1).fillna(False).astype(float)
    cols = [c for c in ["TYD"] if c in opens.columns]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    if "TYD" in cols:
        W["TYD"] = bull
    return W


# --------------------------------------------------------------------------- #
#  S5.  STK_SHARPE_RANK  --  Top-K stocks by 126d Sharpe, Friday weekly
# --------------------------------------------------------------------------- #
def sleeve_stock_sharpe_rank(stk_opens: pd.DataFrame, stk_closes: pd.DataFrame) -> pd.DataFrame:
    idx = stk_opens.index
    rets = stk_closes.pct_change()
    # 126-day Sharpe (lagged shift(1) so info <= close[t-1])
    mean_126 = rets.rolling(STK_LOOKBACK).mean()
    std_126  = rets.rolling(STK_LOOKBACK).std()
    sharpe_126 = (mean_126 / std_126.replace(0, np.nan)).shift(1)

    # Eligibility: positive Sharpe
    elig = sharpe_126 > 0
    score = sharpe_126.where(elig)
    ranks = score.rank(axis=1, ascending=False, method="first")
    pick = (ranks <= STK_K).astype(float)
    n_pick = pick.sum(axis=1).replace(0, np.nan)
    raw_w = pick.div(n_pick, axis=0).fillna(0.0)

    # Friday weekly hold
    dow = pd.Series(idx.dayofweek, index=idx)
    is_rebal = (dow == STK_REBAL_DOW)
    W = pd.DataFrame(0.0, index=idx, columns=stk_opens.columns)
    cur = pd.Series(0.0, index=stk_opens.columns)
    for d in idx:
        if is_rebal.loc[d] and d in raw_w.index:
            cur = raw_w.loc[d].copy()
        W.loc[d] = cur.values
    return W


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def _metrics(r: pd.Series, label: str = "") -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"label": label, "n": 0}
    mu = r.mean() * DAYS
    sd = r.std(ddof=0) * np.sqrt(DAYS)
    sr = mu / sd if sd > 0 else float("nan")
    eq = (1 + r).cumprod()
    yrs = len(r) / DAYS
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if yrs > 0 and eq.iloc[-1] > 0 else float("nan")
    dd = (eq / eq.cummax() - 1).min()
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(DAYS)) if len(neg) > 0 and neg.std() > 0 else float("nan")
    return {
        "label": label, "n": int(len(r)),
        "start": str(r.index[0].date()), "end": str(r.index[-1].date()),
        "sharpe": float(sr), "sortino": float(sortino),
        "cagr": float(cagr), "ann_vol": float(sd),
        "mdd": float(dd), "navx": float(eq.iloc[-1]),
        "calmar": float(cagr / abs(dd)) if dd < 0 else float("nan"),
    }


def apply_portfolio_overlay(r):
    rv = r.rolling(VT_LOOKBACK, min_periods=VT_MIN_PERIODS).std() * np.sqrt(DAYS)
    vt_scale = (TARGET_VOL / rv).clip(VT_LOWER, VT_UPPER).shift(1).fillna(1.0)
    r_scaled = r * vt_scale
    cum = (1 + r_scaled).cumprod()
    hwm = cum.rolling(PORT_DD_WIN, min_periods=30).max()
    dd = cum / hwm - 1.0
    dd_mult = (1.0 + dd / PORT_DD_FLOOR).clip(0.0, 1.0).shift(1).fillna(1.0)
    return r_scaled * dd_mult, vt_scale, dd_mult


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
ETF_UNIVERSE = sorted(set(list(S1_TVOL) + ["QLD", "TYD", "UGL", "SPY", "TLT"]))


def run() -> dict:
    O, C = build_panels(ETF_UNIVERSE)
    O = O.loc[IS_START:]; C = C.loc[IS_START:]
    stk_opens, stk_closes, stk_universe = build_stock_panels(O.index)
    print(f"Stock universe (listed by {STK_INCEPTION_CUTOFF.date()}): {len(stk_universe)} stocks")

    # Build sleeves
    W1 = sleeve_volt_rp(O, C)
    W2 = sleeve_donchian_bo(O, C)
    W3 = sleeve_vrp(O, C)
    W4 = sleeve_bond_dip(O)
    W5 = sleeve_stock_sharpe_rank(stk_opens, stk_closes)

    # Backtest
    r1 = backtest_open_to_open(W1, O)["net_ret"]
    r2 = backtest_open_to_open(W2, O)["net_ret"]
    r3 = backtest_open_to_open(W3, O)["net_ret"]
    r4 = backtest_open_to_open(W4, O)["net_ret"]
    r5 = backtest_open_to_open(W5, stk_opens)["net_ret"]

    # Self-throttle each
    s = {n: self_throttle(rr) for n, rr in
         [("VOLT_RP", r1), ("DONCHIAN_BO", r2), ("VRP", r3), ("BOND_DIP", r4), ("STK_SHARPE", r5)]}
    sleeves = pd.DataFrame(s).fillna(0.0).loc[IS_START:]
    sleeves_is = sleeves.loc[IS_START:IS_END]

    # IS inverse-vol blend
    inv_vol = 1.0 / sleeves_is.std().replace(0, np.nan).fillna(sleeves_is.std().median())
    w = inv_vol / inv_vol.sum()

    # Apply 3% CAGR haircut to stock sleeve return for *headline reporting* only
    sleeves_haircut = sleeves.copy()
    daily_haircut = SURVIVORSHIP_HAIRCUT / DAYS
    sleeves_haircut["STK_SHARPE"] = sleeves_haircut["STK_SHARPE"] - daily_haircut

    blend_pre = (sleeves * w).sum(axis=1)
    blend_pre_haircut = (sleeves_haircut * w).sum(axis=1)
    final, vt_scale, dd_mult = apply_portfolio_overlay(blend_pre)
    final_haircut, _, _ = apply_portfolio_overlay(blend_pre_haircut)

    # Metrics
    standalone = {n: {
        "FULL": _metrics(sleeves[n], f"{n}_FULL"),
        "IS":   _metrics(sleeves.loc[:IS_END][n], f"{n}_IS"),
        "OOS":  _metrics(sleeves.loc[OOS_START:][n], f"{n}_OOS"),
    } for n in sleeves.columns}

    blend_metrics = {
        "FULL": _metrics(blend_pre, "BLEND_FULL"),
        "IS":   _metrics(blend_pre.loc[:IS_END], "BLEND_IS"),
        "OOS":  _metrics(blend_pre.loc[OOS_START:], "BLEND_OOS"),
    }
    polaris_metrics = {
        "FULL": _metrics(final, "POL_FULL"),
        "IS":   _metrics(final.loc[:IS_END], "POL_IS"),
        "OOS":  _metrics(final.loc[OOS_START:], "POL_OOS"),
    }
    polaris_haircut_metrics = {
        "FULL": _metrics(final_haircut, "POL_FULL_HAIRCUT"),
        "IS":   _metrics(final_haircut.loc[:IS_END], "POL_IS_HAIRCUT"),
        "OOS":  _metrics(final_haircut.loc[OOS_START:], "POL_OOS_HAIRCUT"),
    }

    corr_full = sleeves.corr().round(3)

    print("=" * 92)
    print("POLARIS v2  --  4 LETF sleeves + STK_SHARPE_RANK 5th sleeve")
    print("=" * 92)
    print(f"Window: {sleeves.index.min().date()}  ->  {sleeves.index.max().date()}")
    print()
    print("Standalone sleeve metrics (THROTTLED) Sharpe/CAGR/MDD:")
    for n in sleeves.columns:
        b = standalone[n]
        f, i, o = b["FULL"], b["IS"], b["OOS"]
        def _fmt(m):
            return f"{m['sharpe']:+5.2f}/{m['cagr']*100:+5.1f}%/{m['mdd']*100:+5.1f}%"
        print(f"  {n:14s}  FULL: {_fmt(f)}   IS: {_fmt(i)}   OOS: {_fmt(o)}")
    print()
    print("Sleeve correlation (FULL):")
    print(corr_full.to_string())
    n = len(corr_full)
    off = corr_full.values[np.triu_indices(n, 1)]
    print(f"  avg pair-corr = {off.mean():.3f}   max = {off.max():.3f}   min = {off.min():.3f}")
    print()
    print("IS inverse-vol blend weights:")
    for k, v in w.items():
        print(f"  {k:14s} {v*100:5.1f}%")
    print()
    print(f"Portfolio overlay: vt={TARGET_VOL*100:.0f}%   "
          f"avg_vt_scale={vt_scale.mean():.3f}   avg_dd_mult={dd_mult.mean():.3f}")
    print()
    print(f"  {'window':10s} {'SR':>6s} {'CAGR':>7s} {'Vol':>6s} {'MDD':>7s} {'Sortino':>8s}")
    for nm, m in [("BLEND FULL", blend_metrics["FULL"]), ("BLEND IS", blend_metrics["IS"]),
                  ("BLEND OOS", blend_metrics["OOS"]),
                  ("POLv2 FULL", polaris_metrics["FULL"]), ("POLv2 IS", polaris_metrics["IS"]),
                  ("POLv2 OOS", polaris_metrics["OOS"])]:
        print(f"  {nm:10s} {m['sharpe']:6.2f} {m['cagr']*100:6.2f}% "
              f"{m['ann_vol']*100:5.2f}% {m['mdd']*100:6.2f}% {m['sortino']:8.2f}")
    print()
    print(f"With {SURVIVORSHIP_HAIRCUT*100:.0f}% CAGR haircut on stock sleeve (survivorship-bias):")
    for nm, m in [("HCv2 FULL", polaris_haircut_metrics["FULL"]), ("HCv2 IS", polaris_haircut_metrics["IS"]),
                  ("HCv2 OOS", polaris_haircut_metrics["OOS"])]:
        print(f"  {nm:10s} {m['sharpe']:6.2f} {m['cagr']*100:6.2f}% {m['ann_vol']*100:5.2f}% "
              f"{m['mdd']*100:6.2f}%")

    is_m = polaris_metrics["IS"]; oos_m = polaris_metrics["OOS"]
    print(f"\n|IS-OOS Sharpe gap| = {abs(is_m['sharpe']-oos_m['sharpe']):.3f}")
    print(f"|IS-OOS CAGR gap|   = {abs(is_m['cagr']-oos_m['cagr'])*100:.2f}%")

    pol_v1 = RESULTS / "polaris_metrics.json"
    if pol_v1.exists():
        v1 = json.loads(pol_v1.read_text())
        print("\nPOLARIS v1 (reference):")
        for k in ["FULL", "IS", "OOS"]:
            mm = v1["polaris"][k]
            print(f"  {k:4s}  Sharpe={mm['sharpe']:.2f}  CAGR={mm['cagr']*100:.1f}%  MDD={mm['mdd']*100:.1f}%")

    out = {
        "strategy": "POLARIS_v2",
        "version": "v2",
        "window": [str(sleeves.index.min().date()), str(sleeves.index.max().date())],
        "is_window": [str(IS_START.date()), str(IS_END.date())],
        "oos_start": str(OOS_START.date()),
        "tc_bps": TC_BPS,
        "blend_weights": {k: float(v) for k, v in w.items()},
        "stock_universe_size": len(stk_universe),
        "stock_inception_cutoff": str(STK_INCEPTION_CUTOFF.date()),
        "stock_signal": {"type": "126d Sharpe rank", "K": STK_K,
                          "lookback": STK_LOOKBACK, "rebal_dow": STK_REBAL_DOW},
        "survivorship_haircut": SURVIVORSHIP_HAIRCUT,
        "standalone": standalone,
        "blend_pre_overlay": blend_metrics,
        "polaris_v2": polaris_metrics,
        "polaris_v2_haircut": polaris_haircut_metrics,
        "is_oos_gap_sharpe": float(abs(polaris_metrics["IS"]["sharpe"] - polaris_metrics["OOS"]["sharpe"])),
        "is_oos_gap_cagr":   float(abs(polaris_metrics["IS"]["cagr"] - polaris_metrics["OOS"]["cagr"])),
        "corr_full": {k: {k2: float(v2) for k2, v2 in row.items()}
                      for k, row in corr_full.to_dict().items()},
    }
    (RESULTS / "polaris_v2_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    pd.DataFrame({
        "Date": final.index, "ret": final.values, "ret_haircut": final_haircut.values,
        "blend_ret": blend_pre.values, "vt_scale": vt_scale.values, "dd_mult": dd_mult.values,
    }).to_csv(RESULTS / "polaris_v2_returns.csv", index=False)
    sleeves.to_csv(RESULTS / "polaris_v2_sleeves.csv")
    print(f"\nSaved: {RESULTS/'polaris_v2_metrics.json'}")
    print(f"Saved: {RESULTS/'polaris_v2_returns.csv'}")
    print(f"Saved: {RESULTS/'polaris_v2_sleeves.csv'}")
    return out


if __name__ == "__main__":
    run()
