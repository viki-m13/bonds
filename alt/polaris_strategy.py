"""
POLARIS — Polar-Orthogonal Leveraged-ETF Adaptive Risk-premia Strategy
======================================================================

A standalone, self-contained 4-sleeve LETF ensemble designed to be
ORTHOGONAL, in both signal-type and information-content, to:

    * PHOENIX (VANGUARD, ORION, HELIOS, QUANTUM)
    * MERIDIAN (stock-momentum + ETF rotation family)

POLARIS deliberately AVOIDS every signal Phoenix uses:

    * No VIX-LEVEL macro gate, no HY-OAS macro gate.
    * No 200-day SMA trend filter.
    * No cross-sectional 12-month / 9-month price-momentum on LETFs.
    * No XGBoost / ML rank-IC model.
    * No top-K rotation by raw price-momentum.
    * No inverse-vol blend over price-momentum sleeves.

Each sleeve is driven by a fundamentally different information source.

----------------------------------------------------------------------
S1.  VOLT_RP   (vol-targeted risk-parity basket, rate-velocity gate)
----------------------------------------------------------------------
Per-asset volatility-targeting on QLD / TYD / UGL with target vols
intentionally low so the basket runs at modest gross. The signal is
pure RISK control -- there is no momentum, trend, RSI, or VIX/HY gate.
Defensive trigger uses the FED-cycle, not VIX/HY:

    rv_yoy = DGS10 - DGS10.shift(252)        (pp change over trailing year)
    multiplier  =  1.0   if rv_yoy <= 1.0
                   0.5   if 1.0 < rv_yoy <= 2.0
                   0.0   if rv_yoy > 2.0

This catches 2022 cleanly (DGS10 went 1.6% -> 4.2%, +260bp) using
yield-curve dynamics, NOT the VIX/HY composite Phoenix relies on.

----------------------------------------------------------------------
S2.  DONCHIAN_BO   (40/20 Donchian channel breakout on QLD)
----------------------------------------------------------------------
Long QLD when close[t-1] makes a new 40-day HIGH; remains long until
close[t-1] makes a new 20-day LOW. This is the classical Turtle-Traders
breakout -- a CHANNEL-RANGE signal, not a moving-average filter.
Phoenix uses 200-day SMA (rolling MEAN); Donchian uses rolling MAX/MIN.
Same graduated rate-velocity gate as S1 (zero-VIX defense).

----------------------------------------------------------------------
S3.  VRP_HARVEST   (vol-risk-premium SPREAD, not VIX level)
----------------------------------------------------------------------
Trades the SPREAD between implied (VIX) and realised (SPY 21d ann) vol:

    vrp = VIX - SPY_21d_realised*sqrt(252)*100      (in % units)
    rv  = SPY_21d_realised*sqrt(252)*100
    if vrp > 5  and rv < 25:    long QLD     (calm, harvest VRP)
    elif rv > 30:               long UGL     (panic hedge)
    else:                       cash

Daily check, daily reposition. Phoenix uses VIX *level*; POLARIS uses
the *spread*, with a *realised-vol* (not VIX-level) panic gate.

----------------------------------------------------------------------
S4.  BOND_DIP   (single-asset TYD on rate-direction signal)
----------------------------------------------------------------------
Long TYD when DGS10 < its 60-day average AND T10Y2Y > -0.5
(rates falling, curve not deeply inverted). Cash otherwise.
Bond-only sleeve; provides genuine asset-class diversification with
near-zero correlation to the equity-leaning sleeves.

----------------------------------------------------------------------
Per-sleeve self-throttle (every sleeve uses the same risk overlay)
----------------------------------------------------------------------
Each sleeve return stream is throttled by its OWN rolling 252d HWM
drawdown, floor -20%. Generic risk control, not a macro gate.

----------------------------------------------------------------------
Blend
----------------------------------------------------------------------
1. Inverse-volatility weights computed on IS-only (2010-03-11..2018-12-31)
   applied unchanged to OOS.
2. Portfolio-level vol-target overlay scales daily exposure by
        clip(target_vol / realised_60d_vol, 0.5, 2.5)
   target_vol = 18%  (matches Phoenix's full-sample vol of ~16% with
   slight headroom; chosen by inspection, not optimised).
3. Final overlay drawdown throttle: 252d HWM, -15% floor.

Execution
---------
* All signals use info <= close[t-1].
* Trades at open[t]; PnL uses open[t] -> open[t+1].
* TC = 5 bps one-way on |dw_i| at time of weight change.
* No Kelly sizing, no SMA trend filter.

HONEST DISCLOSURE
-----------------
POLARIS is a NOVEL strategy that uses signals fundamentally different
from Phoenix's. Standalone metrics on this dataset (2010-03-11 .. 2026):
its Sharpe is comparable in magnitude to a single Phoenix sleeve but
LOWER than the 4-sleeve Phoenix v2 ensemble. Its main values are:
    * IS-OOS robustness  -- tiny sharpe gap, OOS often beats IS
    * Diversifier vs Phoenix -- low sleeve-level correlation
    * Different signal family (rates / channel / VRP-spread / curve)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data" / "etfs"
FRED_DIR = ROOT / "data" / "fred"
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

# Portfolio overlay parameters
TARGET_VOL = 0.18
VT_LOOKBACK = 60
VT_MIN_PERIODS = 20
VT_LOWER, VT_UPPER = 0.5, 2.5
PORT_DD_WIN = 252
PORT_DD_FLOOR = -0.15


# --------------------------------------------------------------------------- #
#  Data loaders
# --------------------------------------------------------------------------- #
def _load_etf(t: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{t}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def _load_fred(name: str) -> pd.Series:
    df = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[name].astype(float)


def build_panels(tickers: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    opens, closes = {}, {}
    for t in tickers:
        d = _load_etf(t)
        opens[t] = d["Open"]
        closes[t] = d["Close"]
    O = pd.DataFrame(opens).sort_index()
    C = pd.DataFrame(closes).sort_index()
    idx = pd.bdate_range(O.index.min(), O.index.max())
    O = O.reindex(idx).ffill(limit=2)
    C = C.reindex(idx).ffill(limit=2)
    return O, C


# --------------------------------------------------------------------------- #
#  Backtester (open-to-open, daily, 5 bps TC)
# --------------------------------------------------------------------------- #
def backtest_open_to_open(weights: pd.DataFrame, opens: pd.DataFrame,
                          tc_rate: float = TC_RATE) -> pd.DataFrame:
    """weights[t] = target weights set at close[t-1]; trade at open[t]."""
    common_cols = [c for c in weights.columns if c in opens.columns]
    W = weights[common_cols].fillna(0.0)
    O = opens[common_cols].reindex(W.index)
    o2o_fwd = O.shift(-1) / O - 1.0
    gross = (W * o2o_fwd).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(0.0)
    cost = turnover * tc_rate
    net = (gross - cost).fillna(0.0)
    return pd.DataFrame({
        "gross_ret": gross.fillna(0.0),
        "cost": cost,
        "net_ret": net,
        "turnover": turnover,
    })


def self_throttle(r: pd.Series, dd_win: int = SELF_DD_WIN,
                  dd_floor: float = SELF_DD_FLOOR) -> pd.Series:
    cum = (1 + r).cumprod()
    hwm = cum.rolling(dd_win, min_periods=30).max()
    dd = cum / hwm - 1.0
    mult = (1.0 + dd / dd_floor).clip(0.0, 1.0).shift(1).fillna(1.0)
    return r * mult


# --------------------------------------------------------------------------- #
#  Common helper: graduated rate-velocity gate (FED-cycle, not VIX/HY)
#    rv_yoy <= 1.0      ->  1.0
#    1.0 < rv_yoy <= 2.0 -> 0.5
#    rv_yoy > 2.0       ->  0.0
# --------------------------------------------------------------------------- #
def _rate_velocity_gate(idx: pd.DatetimeIndex) -> pd.Series:
    dgs10 = _load_fred("DGS10").reindex(idx).ffill()
    rv_yoy = (dgs10 - dgs10.shift(252)).shift(1)
    g = pd.Series(1.0, index=idx)
    g[rv_yoy > 1.0] = 0.5
    g[rv_yoy > 2.0] = 0.0
    return g


# --------------------------------------------------------------------------- #
#  S1. VOLT_RP  --  vol-targeted RP (QLD/TYD/UGL) + graduated rate gate
# --------------------------------------------------------------------------- #
S1_TVOL = {"QLD": 0.20, "TYD": 0.10, "UGL": 0.10}
S1_VOL_LB = 21
S1_GROSS_CAP = 1.5


def sleeve_volt_rp(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in S1_TVOL if c in opens.columns]
    idx = opens.index
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    for t in cols:
        r = closes[t].pct_change()
        v = r.rolling(S1_VOL_LB).std() * np.sqrt(DAYS)
        w = (S1_TVOL[t] / v).clip(0, S1_GROSS_CAP).shift(1).fillna(0.0)
        W[t] = w
    gate = _rate_velocity_gate(idx)
    return W.mul(gate, axis=0)


# --------------------------------------------------------------------------- #
#  S2. DONCHIAN_BO  --  40/20 channel breakout on QLD + graduated rate gate
# --------------------------------------------------------------------------- #
S2_NH = 40    # entry: new N_H-day high
S2_NL = 20    # exit:  new N_L-day low


def sleeve_donchian_bo(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    idx = opens.index
    qc = closes["QLD"]
    hh = qc.shift(1).rolling(S2_NH).max()
    ll = qc.shift(1).rolling(S2_NL).min()

    long = pd.Series(0.0, index=idx)
    pos = 0
    for i in range(2, len(idx)):
        c_prev = qc.iloc[i - 1]
        if pos == 0 and pd.notna(hh.iloc[i - 1]) and c_prev >= hh.iloc[i - 1]:
            pos = 1
        elif pos == 1 and pd.notna(ll.iloc[i - 1]) and c_prev <= ll.iloc[i - 1]:
            pos = 0
        long.iloc[i] = float(pos)

    gate = _rate_velocity_gate(idx)
    return pd.DataFrame({"QLD": long * gate}, index=idx)


# --------------------------------------------------------------------------- #
#  S3. VRP_HARVEST  --  IV-RV spread (with realized-vol panic gate)
# --------------------------------------------------------------------------- #
VRP_LONG = "QLD"
VRP_SAFE = "UGL"
VRP_THR = 5.0
VRP_RV_CAP = 25.0
VRP_PANIC_RV = 30.0


def sleeve_vrp_harvest(opens: pd.DataFrame, closes: pd.DataFrame) -> pd.DataFrame:
    idx = opens.index
    if "SPY" in closes.columns:
        spy_close = closes["SPY"]
    else:
        spy_close = _load_etf("SPY")["Close"].reindex(idx).ffill()
    spy_ret = spy_close.pct_change()
    rv21_ann = spy_ret.rolling(21).std() * np.sqrt(DAYS) * 100.0

    vix = _load_fred("VIXCLS").reindex(idx).ffill()
    vrp = (vix - rv21_ann).shift(1)
    rv = rv21_ann.shift(1)

    cols = list({VRP_LONG, VRP_SAFE} & set(opens.columns))
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    if VRP_LONG in cols:
        W.loc[((vrp > VRP_THR) & (rv < VRP_RV_CAP)).fillna(False), VRP_LONG] = 1.0
    if VRP_SAFE in cols:
        W.loc[(rv > VRP_PANIC_RV).fillna(False), VRP_SAFE] = 1.0
    return W


# --------------------------------------------------------------------------- #
#  S4. BOND_DIP  --  single-asset TYD on rate-direction
# --------------------------------------------------------------------------- #
def sleeve_bond_dip(opens: pd.DataFrame) -> pd.DataFrame:
    idx = opens.index
    dgs10 = _load_fred("DGS10").reindex(idx).ffill()
    t10y2y = _load_fred("T10Y2Y").reindex(idx).ffill()

    bull = ((dgs10 < dgs10.rolling(60).mean()) & (t10y2y > -0.5)).shift(1).fillna(False)
    cols = [c for c in ["TYD"] if c in opens.columns]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    if "TYD" in cols:
        W["TYD"] = bull.astype(float)
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


# --------------------------------------------------------------------------- #
#  Build & run
# --------------------------------------------------------------------------- #
def build_full_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    uni = sorted(set(list(S1_TVOL) + ["QLD", "TYD", VRP_SAFE, "SPY"]))
    O, C = build_panels(uni)
    O = O.loc[IS_START:]
    C = C.loc[IS_START:]
    return O, C


def build_sleeve_returns() -> Tuple[Dict[str, pd.Series], Dict[str, pd.Series]]:
    O, C = build_full_universe()

    W1 = sleeve_volt_rp(O, C)
    W2 = sleeve_donchian_bo(O, C)
    W3 = sleeve_vrp_harvest(O, C)
    W4 = sleeve_bond_dip(O)

    bt1 = backtest_open_to_open(W1, O)
    bt2 = backtest_open_to_open(W2, O)
    bt3 = backtest_open_to_open(W3, O)
    bt4 = backtest_open_to_open(W4, O)

    sleeves_raw = {
        "VOLT_RP":     bt1["net_ret"],
        "DONCHIAN_BO": bt2["net_ret"],
        "VRP":         bt3["net_ret"],
        "BOND_DIP":    bt4["net_ret"],
    }
    sleeves_thr = {k: self_throttle(v) for k, v in sleeves_raw.items()}
    return sleeves_raw, sleeves_thr


def apply_portfolio_overlay(r: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Apply (1) vol-target scaling and (2) portfolio DD throttle."""
    rv = r.rolling(VT_LOOKBACK, min_periods=VT_MIN_PERIODS).std() * np.sqrt(DAYS)
    vt_scale = (TARGET_VOL / rv).clip(VT_LOWER, VT_UPPER).shift(1).fillna(1.0)

    r_scaled = r * vt_scale

    cum = (1 + r_scaled).cumprod()
    hwm = cum.rolling(PORT_DD_WIN, min_periods=30).max()
    dd = cum / hwm - 1.0
    dd_mult = (1.0 + dd / PORT_DD_FLOOR).clip(0.0, 1.0).shift(1).fillna(1.0)

    final = r_scaled * dd_mult
    return final, vt_scale, dd_mult


def main(verbose: bool = True) -> dict:
    sleeves_raw, sleeves_thr = build_sleeve_returns()
    raw_df = pd.DataFrame(sleeves_raw).fillna(0.0).loc[IS_START:]
    thr_df = pd.DataFrame(sleeves_thr).fillna(0.0).loc[IS_START:]

    raw_is = raw_df.loc[IS_START:IS_END]
    thr_is = thr_df.loc[IS_START:IS_END]

    standalone_thr = {n: {
        "FULL": _metrics(thr_df[n], f"{n}_FULL"),
        "IS":   _metrics(thr_is[n],  f"{n}_IS"),
        "OOS":  _metrics(thr_df[n].loc[OOS_START:], f"{n}_OOS"),
    } for n in thr_df.columns}

    standalone_raw = {n: {
        "FULL": _metrics(raw_df[n], f"{n}_FULL"),
        "IS":   _metrics(raw_is[n],  f"{n}_IS"),
        "OOS":  _metrics(raw_df[n].loc[OOS_START:], f"{n}_OOS"),
    } for n in raw_df.columns}

    corr_thr_full = thr_df.corr().round(3)
    corr_thr_is = thr_is.corr().round(3)

    # IS inverse-vol blend
    inv_vol = 1.0 / thr_is.std().replace(0, np.nan)
    inv_vol = inv_vol.fillna(inv_vol.median())
    w_blend = (inv_vol / inv_vol.sum())

    pol_blend = (thr_df * w_blend).sum(axis=1)
    pol_final, vt_scale, dd_mult = apply_portfolio_overlay(pol_blend)

    blend_metrics = {
        "FULL": _metrics(pol_blend, "BLEND_FULL"),
        "IS":   _metrics(pol_blend.loc[IS_START:IS_END], "BLEND_IS"),
        "OOS":  _metrics(pol_blend.loc[OOS_START:], "BLEND_OOS"),
    }
    polaris_metrics = {
        "FULL": _metrics(pol_final, "POL_FULL"),
        "IS":   _metrics(pol_final.loc[IS_START:IS_END], "POL_IS"),
        "OOS":  _metrics(pol_final.loc[OOS_START:], "POL_OOS"),
    }

    # Correlation vs Phoenix v2
    phoenix_corr = None
    ph_returns_path = RESULTS / "phoenix_v2_returns.csv"
    if ph_returns_path.exists():
        ph = pd.read_csv(ph_returns_path, parse_dates=["Date"]).set_index("Date")
        if "ret" in ph.columns:
            r_ph = ph["ret"].astype(float)
            joined = pd.concat({"POL": pol_final, "PHX": r_ph}, axis=1).dropna()
            phoenix_corr = float(joined["POL"].corr(joined["PHX"]))

    if verbose:
        print("=" * 92)
        print("POLARIS  --  novel 4-sleeve LETF ensemble (vol-target + DD throttle overlay)")
        print("=" * 92)
        print(f"Window: {raw_df.index.min().date()}  ->  {raw_df.index.max().date()}")
        print()
        print("Standalone sleeve metrics (THROTTLED) Sharpe/CAGR/MDD:")
        print(f"  {'sleeve':14s}  {'FULL':>22s}  {'IS':>22s}  {'OOS':>22s}")
        for n in thr_df.columns:
            f, i, o = standalone_thr[n]["FULL"], standalone_thr[n]["IS"], standalone_thr[n]["OOS"]
            def _fmt(m):
                return f"{m['sharpe']:+5.2f}/{m['cagr']*100:+5.1f}%/{m['mdd']*100:+5.1f}%"
            print(f"  {n:14s}  {_fmt(f):>22s}  {_fmt(i):>22s}  {_fmt(o):>22s}")
        print()
        print("Sleeve correlation (THROTTLED, FULL):")
        print(corr_thr_full.to_string())
        n = len(corr_thr_full)
        off = corr_thr_full.values[np.triu_indices(n, 1)]
        print(f"  avg pair-corr = {off.mean():.3f}   max = {off.max():.3f}   min = {off.min():.3f}")
        print()
        print("IS inverse-vol blend weights:")
        for k, v in w_blend.items():
            print(f"  {k:14s} {v*100:5.1f}%")
        print()
        print("Portfolio overlay:")
        print(f"  vol-target = {TARGET_VOL*100:.0f}%   lookback = {VT_LOOKBACK}d   "
              f"avg_scale = {vt_scale.mean():.3f}")
        print(f"  DD throttle: dd_win={PORT_DD_WIN}  dd_floor={PORT_DD_FLOOR*100:.0f}%   "
              f"avg_mult = {dd_mult.mean():.3f}")
        print()
        print(f"  {'window':10s} {'SR':>6s} {'CAGR':>7s} {'Vol':>6s} {'MDD':>7s} "
              f"{'Calmar':>7s} {'Sortino':>8s}")
        for nm, m in [("BLEND FULL", blend_metrics["FULL"]),
                      ("BLEND IS",   blend_metrics["IS"]),
                      ("BLEND OOS",  blend_metrics["OOS"]),
                      ("POL FULL",  polaris_metrics["FULL"]),
                      ("POL IS",    polaris_metrics["IS"]),
                      ("POL OOS",   polaris_metrics["OOS"])]:
            print(f"  {nm:10s} {m['sharpe']:6.2f} {m['cagr']*100:6.2f}% "
                  f"{m['ann_vol']*100:5.2f}% {m['mdd']*100:6.2f}% "
                  f"{m['calmar']:7.2f} {m['sortino']:8.2f}")
        print()
        is_m = polaris_metrics["IS"]; oos_m = polaris_metrics["OOS"]
        print(f"|IS-OOS Sharpe gap| = {abs(is_m['sharpe']-oos_m['sharpe']):.3f}")
        print(f"|IS-OOS CAGR gap|   = {abs(is_m['cagr']-oos_m['cagr'])*100:.2f}%")
        if phoenix_corr is not None:
            print(f"\nPOLARIS  vs  Phoenix v2  return correlation: {phoenix_corr:+.3f}")
        print()
        ph_path = RESULTS / "phoenix_v2_metrics.json"
        if ph_path.exists():
            ph = json.loads(ph_path.read_text())
            print("Phoenix v2  (reference):")
            for k in ["full", "is", "oos"]:
                m = ph["v2"][k]
                print(f"  {k.upper():4s} SR={m['sharpe']:.2f}  CAGR={m['cagr']*100:5.1f}% "
                      f"Vol={m['ann_vol']*100:.1f}%  MDD={m['mdd']*100:.1f}%")

    out = {
        "strategy": "POLARIS",
        "version": "v4-final",
        "window": [str(raw_df.index.min().date()), str(raw_df.index.max().date())],
        "is_window": [str(IS_START.date()), str(IS_END.date())],
        "oos_start": str(OOS_START.date()),
        "tc_bps": TC_BPS,
        "blend_weights": {k: float(v) for k, v in w_blend.items()},
        "self_throttle": {"dd_win": SELF_DD_WIN, "dd_floor": SELF_DD_FLOOR},
        "portfolio_overlay": {
            "target_vol": TARGET_VOL,
            "vt_lookback": VT_LOOKBACK,
            "vt_lower": VT_LOWER, "vt_upper": VT_UPPER,
            "dd_win": PORT_DD_WIN, "dd_floor": PORT_DD_FLOOR,
            "avg_vt_scale": float(vt_scale.mean()),
            "avg_dd_mult": float(dd_mult.mean()),
        },
        "standalone_raw": {k: v for k, v in standalone_raw.items()},
        "standalone_thr": {k: v for k, v in standalone_thr.items()},
        "corr_thr_full": {k: {k2: float(v2) for k2, v2 in row.items()}
                          for k, row in corr_thr_full.to_dict().items()},
        "corr_thr_is":   {k: {k2: float(v2) for k2, v2 in row.items()}
                          for k, row in corr_thr_is.to_dict().items()},
        "blend_pre_overlay": blend_metrics,
        "polaris": polaris_metrics,
        "is_oos_gap_sharpe": float(abs(polaris_metrics["IS"]["sharpe"]
                                       - polaris_metrics["OOS"]["sharpe"])),
        "is_oos_gap_cagr":   float(abs(polaris_metrics["IS"]["cagr"]
                                       - polaris_metrics["OOS"]["cagr"])),
        "phoenix_v2_corr": phoenix_corr,
    }
    (RESULTS / "polaris_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    pd.DataFrame({
        "Date": pol_final.index,
        "ret": pol_final.values,
        "blend_ret": pol_blend.values,
        "vt_scale": vt_scale.values,
        "dd_mult": dd_mult.values,
    }).to_csv(RESULTS / "polaris_returns.csv", index=False)
    pd.DataFrame(thr_df).to_csv(RESULTS / "polaris_sleeves.csv")
    pd.DataFrame(raw_df).to_csv(RESULTS / "polaris_sleeves_raw.csv")
    if verbose:
        print(f"\nSaved: {RESULTS/'polaris_metrics.json'}")
        print(f"Saved: {RESULTS/'polaris_returns.csv'}")
        print(f"Saved: {RESULTS/'polaris_sleeves.csv'}")
    return out


if __name__ == "__main__":
    main()
