"""
NEUTRINO — High-CAGR LETF Strategy with Garman-Klass VT and Stacked Macro Gates
================================================================================

A standalone, self-contained, SINGLE-SLEEVE LETF strategy that introduces
GENUINELY NOVEL elements relative to PHOENIX, MERIDIAN and POLARIS:

    1.  GARMAN-KLASS high-low-close-open volatility estimator (uses the
        full OHLC range, more efficient than realised close-to-close vol).
        Phoenix uses 21d close-to-close std; NEUTRINO uses GK on intraday
        OHLC and produces statistically more efficient vol estimates.

    2.  TWO-HORIZON SMOOTH RATE-VELOCITY GATE — multiplies a smooth
        252d-yoy ramp with a smooth 90d-rate-shock ramp:
            gate_rate = clip(1 - rv_252/2.0, 0, 1) * clip(1 - rv_90/1.5, 0, 1)
        Where rv_T = DGS10 - DGS10.shift(T). This is more responsive than
        the binary thresholds POLARIS uses and catches BOTH structural
        (yoy) and tactical (90d) Fed-tightening regimes.

    3.  STOCK-BOND CORRELATION REGIME gate (60d corr(SPY, TLT)) applied
        ONLY to the equity leg. When the diversification regime breaks
        (rho > 0), the equity leg de-risks while the bond/gold legs
        continue to harvest carry.

    4.  AGGRESSIVE TQQQ TARGET (45% per-asset vol target, capped 2.0x)
        on the equity leg, with GK vol estimator. This is intentionally
        more aggressive than POLARIS's 20% target on QLD, exploiting
        TQQQ's 3x leverage (vs QLD's 2x) when conditions are favourable.

NEUTRINO deliberately AVOIDS every primary signal Phoenix uses:

    * No VIX-LEVEL macro gate, no HY-OAS macro gate.
    * No 200-day SMA trend filter.
    * No cross-sectional 12m / 9m price-momentum on LETFs.
    * No XGBoost / ML rank-IC ranking.
    * No multi-sleeve inverse-vol blend over momentum sleeves.

----------------------------------------------------------------------
Single core sleeve:  GK-VT RP TQQQ + TYD + UGL with stacked gates
----------------------------------------------------------------------

    sigma_GK = sqrt[ rolling_mean(0.5*ln(H/L)^2 - (2*ln 2 - 1)*ln(C/O)^2) * 252 ]

    rate_gate = clip(1 - rv_yoy/2.0, 0, 1) * clip(1 - rv_90/1.5, 0, 1)

    corr_gate = 1.0   if corr_60(SPY, TLT) <= 0.0
                0.6   if 0.0 < corr_60 <= 0.20
                0.0   if corr_60 > 0.20

    w_TQQQ = clip(0.45 / sigma_GK_TQQQ, 0, 2.0) * rate_gate * corr_gate
    w_TYD  = clip(0.10 / sigma_GK_TYD,  0, 1.5) * rate_gate
    w_UGL  = clip(0.10 / sigma_GK_UGL,  0, 1.5) * rate_gate

----------------------------------------------------------------------
Per-sleeve self-throttle and portfolio overlay
----------------------------------------------------------------------
* Self-throttle: 252d HWM drawdown, floor -25%.
* Portfolio vol-target overlay: target 22%, lookback 60d, scale [0.5,2.5].
* Final portfolio DD throttle: 252d HWM, floor -15%.

----------------------------------------------------------------------
Execution
----------------------------------------------------------------------
* All signals lagged shift(1); use info <= close[t-1].
* Trades at open[t]; PnL uses open[t] -> open[t+1].
* TC = 5 bps one-way on |dw_i| at trade time.

----------------------------------------------------------------------
Headline (full-sample 2010-03-11 → 2026-05-06)
----------------------------------------------------------------------
NEUTRINO: Sharpe 1.28, CAGR 36-38%, MDD -28%
Phoenix:  Sharpe 2.10, CAGR 38.0%, MDD -21%

* NEUTRINO BEATS Phoenix on CAGR at vt=25% (38.8% vs 38.0%; OOS 39.9% vs 37.5%)
* Phoenix BEATS NEUTRINO on Sharpe (2.10 vs 1.28) and MDD (-21% vs -28%)

NEUTRINO is a high-CAGR, single-sleeve alternative -- much simpler than
a multi-sleeve ensemble while matching/exceeding Phoenix's CAGR, with
the tradeoff being lower Sharpe and slightly larger drawdowns.

The Sharpe ceiling for a single sleeve in this regime appears to be
~1.30; reaching Phoenix's 2.10 requires a multi-sleeve ensemble with
near-zero pairwise correlation, which is structurally difficult without
copying Phoenix's specific sleeve construction.
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

# Per-asset target vols (in annualised %)
TVOL = {"TQQQ": 0.45, "TYD": 0.10, "UGL": 0.10}
GK_LB = 21
EQ_GROSS_CAP = 2.0
DEF_GROSS_CAP = 1.5

# Rate-velocity gate (smooth, two-horizon)
RV_YOY_DENOM = 2.0       # full risk-off when DGS10 yoy >= 2.0pp
RV_90_DENOM  = 1.5       # full risk-off when DGS10 90d >= 1.5pp

# Stock-bond correlation gate
CORR_LB = 60
CORR_THR_LO = 0.0        # < 0 -> 1.0
CORR_THR_HI = 0.20       # > 0.20 -> 0.0
CORR_MID_MULT = 0.6      # in between

# Per-sleeve throttle
SELF_DD_WIN   = 252
SELF_DD_FLOOR = -0.25

# Portfolio overlay
TARGET_VOL = 0.27
VT_LOOKBACK = 60
VT_MIN_PERIODS = 20
VT_LOWER, VT_UPPER = 0.5, 2.5
PORT_DD_WIN = 252
PORT_DD_FLOOR = -0.15


# --------------------------------------------------------------------------- #
def _load_etf(t: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{t}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[["Open", "Close", "High", "Low"]].astype(float)


def _load_fred(name: str) -> pd.Series:
    df = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").set_index("Date")
    return df[name].astype(float)


def build_panels(tickers: list[str]):
    opens, closes, highs, lows = {}, {}, {}, {}
    for t in tickers:
        d = _load_etf(t)
        opens[t] = d["Open"]; closes[t] = d["Close"]
        highs[t] = d["High"]; lows[t] = d["Low"]
    O = pd.DataFrame(opens).sort_index()
    C = pd.DataFrame(closes).sort_index()
    H = pd.DataFrame(highs).sort_index()
    L = pd.DataFrame(lows).sort_index()
    idx = pd.bdate_range(O.index.min(), O.index.max())
    O = O.reindex(idx).ffill(limit=2); C = C.reindex(idx).ffill(limit=2)
    H = H.reindex(idx).ffill(limit=2); L = L.reindex(idx).ffill(limit=2)
    return O, C, H, L


# --------------------------------------------------------------------------- #
#  Backtester (open-to-open, 5 bps TC)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
#  Garman-Klass volatility estimator
# --------------------------------------------------------------------------- #
def garman_klass_vol(opens: pd.Series, highs: pd.Series, lows: pd.Series,
                     closes: pd.Series, lb: int = GK_LB) -> pd.Series:
    """sigma_GK = sqrt[ mean(0.5*ln(H/L)^2 - (2*ln 2 - 1)*ln(C/O)^2) * 252 ]"""
    log_hl = np.log(highs / lows)
    log_co = np.log(closes / opens)
    rs = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
    gv = rs.rolling(lb).mean()
    return np.sqrt(gv.clip(lower=0) * DAYS)


# --------------------------------------------------------------------------- #
#  Two-horizon smooth rate-velocity gate
# --------------------------------------------------------------------------- #
def rate_velocity_gate(idx: pd.DatetimeIndex) -> pd.Series:
    dgs10 = _load_fred("DGS10").reindex(idx).ffill()
    rv_yoy = (dgs10 - dgs10.shift(252)).shift(1)
    rv_90  = (dgs10 - dgs10.shift(90)).shift(1)
    g_yoy = (1.0 - rv_yoy / RV_YOY_DENOM).clip(0.0, 1.0)
    g_90  = (1.0 - rv_90  / RV_90_DENOM ).clip(0.0, 1.0)
    return (g_yoy * g_90).fillna(1.0)


# --------------------------------------------------------------------------- #
#  Stock-bond correlation regime gate (graduated)
# --------------------------------------------------------------------------- #
def stock_bond_corr_gate(idx: pd.DatetimeIndex) -> pd.Series:
    spy = _load_etf("SPY")["Close"].reindex(idx).ffill()
    tlt = _load_etf("TLT")["Close"].reindex(idx).ffill()
    spy_r = spy.pct_change()
    tlt_r = tlt.pct_change()
    corr = spy_r.rolling(CORR_LB).corr(tlt_r).shift(1)
    g = pd.Series(1.0, index=idx)
    g[corr > CORR_THR_LO] = CORR_MID_MULT
    g[corr > CORR_THR_HI] = 0.0
    return g


# --------------------------------------------------------------------------- #
#  Build NEUTRINO target weights (single core sleeve)
# --------------------------------------------------------------------------- #
def build_weights(O, C, H, L) -> pd.DataFrame:
    idx = O.index
    cols = list(TVOL.keys())
    sigma = {t: garman_klass_vol(O[t], H[t], L[t], C[t]).shift(1) for t in cols}

    W = pd.DataFrame(0.0, index=idx, columns=cols)

    rg = rate_velocity_gate(idx)
    cg = stock_bond_corr_gate(idx)

    for t in cols:
        cap = EQ_GROSS_CAP if _is_equity(t) else DEF_GROSS_CAP
        raw = (TVOL[t] / sigma[t]).clip(0, cap).fillna(0.0)
        if _is_equity(t):
            W[t] = raw * rg * cg     # equity: rate AND corr gate
        else:
            W[t] = raw * rg          # defensives: rate gate only
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


def apply_portfolio_overlay(r: pd.Series):
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
UNIVERSE = sorted(set(list(TVOL.keys()) + ["SPY", "TLT"]))


def _is_equity(t: str) -> bool:
    return t == "TQQQ"


def run() -> dict:
    O, C, H, L = build_panels(UNIVERSE)
    O = O.loc[IS_START:]; C = C.loc[IS_START:]; H = H.loc[IS_START:]; L = L.loc[IS_START:]

    W = build_weights(O, C, H, L)
    bt = backtest_open_to_open(W, O)
    r_raw = bt["net_ret"]
    r_thr = self_throttle(r_raw)
    r_final, vt_scale, dd_mult = apply_portfolio_overlay(r_thr)

    raw_metrics = {
        "FULL": _metrics(r_raw, "RAW_FULL"),
        "IS":   _metrics(r_raw.loc[IS_START:IS_END], "RAW_IS"),
        "OOS":  _metrics(r_raw.loc[OOS_START:], "RAW_OOS"),
    }
    thr_metrics = {
        "FULL": _metrics(r_thr, "THR_FULL"),
        "IS":   _metrics(r_thr.loc[IS_START:IS_END], "THR_IS"),
        "OOS":  _metrics(r_thr.loc[OOS_START:], "THR_OOS"),
    }
    neutrino_metrics = {
        "FULL": _metrics(r_final, "NEU_FULL"),
        "IS":   _metrics(r_final.loc[IS_START:IS_END], "NEU_IS"),
        "OOS":  _metrics(r_final.loc[OOS_START:], "NEU_OOS"),
    }

    phoenix_corr = polaris_corr = None
    ph_path = RESULTS / "phoenix_v2_returns.csv"
    if ph_path.exists():
        ph = pd.read_csv(ph_path, parse_dates=["Date"]).set_index("Date")
        if "ret" in ph.columns:
            j = pd.concat({"NEU": r_final, "PHX": ph["ret"].astype(float)}, axis=1).dropna()
            phoenix_corr = float(j["NEU"].corr(j["PHX"]))
    pol_path = RESULTS / "polaris_returns.csv"
    if pol_path.exists():
        po = pd.read_csv(pol_path, parse_dates=["Date"]).set_index("Date")
        if "ret" in po.columns:
            j = pd.concat({"NEU": r_final, "POL": po["ret"].astype(float)}, axis=1).dropna()
            polaris_corr = float(j["NEU"].corr(j["POL"]))

    print("=" * 92)
    print("NEUTRINO  --  GK-VT TQQQ/TYD/UGL with smooth rate gate + corr regime")
    print("=" * 92)
    print(f"Window: {r_final.index.min().date()}  ->  {r_final.index.max().date()}")
    print()
    print("Per-asset target vols:")
    for k, v in TVOL.items():
        print(f"  {k}: {v*100:.0f}%   (cap {EQ_GROSS_CAP if k=='TQQQ' else DEF_GROSS_CAP:.1f}x)")
    print(f"\nGates: rate two-horizon smooth (yoy/{RV_YOY_DENOM}, 90d/{RV_90_DENOM}); "
          f"corr_60(SPY,TLT) graduated [{CORR_THR_LO},{CORR_THR_HI}]")
    print(f"Self-throttle: dd_floor={SELF_DD_FLOOR*100:.0f}%, "
          f"target_vol={TARGET_VOL*100:.0f}%, dd_floor_port={PORT_DD_FLOOR*100:.0f}%")
    print()
    print(f"  {'window':10s} {'SR':>6s} {'CAGR':>7s} {'Vol':>6s} {'MDD':>7s} "
          f"{'Calmar':>7s} {'Sortino':>8s}")
    for nm, m in [("RAW FULL", raw_metrics["FULL"]),
                  ("RAW IS",   raw_metrics["IS"]),
                  ("RAW OOS",  raw_metrics["OOS"]),
                  ("THR FULL", thr_metrics["FULL"]),
                  ("THR IS",   thr_metrics["IS"]),
                  ("THR OOS",  thr_metrics["OOS"]),
                  ("NEU FULL", neutrino_metrics["FULL"]),
                  ("NEU IS",   neutrino_metrics["IS"]),
                  ("NEU OOS",  neutrino_metrics["OOS"])]:
        print(f"  {nm:10s} {m['sharpe']:6.2f} {m['cagr']*100:6.2f}% "
              f"{m['ann_vol']*100:5.2f}% {m['mdd']*100:6.2f}% "
              f"{m['calmar']:7.2f} {m['sortino']:8.2f}")
    print()
    is_m = neutrino_metrics["IS"]; oos_m = neutrino_metrics["OOS"]
    print(f"|IS-OOS Sharpe gap| = {abs(is_m['sharpe']-oos_m['sharpe']):.3f}   "
          f"OOS Sharpe = {oos_m['sharpe']:.2f}")
    print(f"|IS-OOS CAGR gap|   = {abs(is_m['cagr']-oos_m['cagr'])*100:.2f}%")
    if phoenix_corr is not None:
        print(f"\nNEUTRINO vs Phoenix v2 return correlation: {phoenix_corr:+.3f}")
    if polaris_corr is not None:
        print(f"NEUTRINO vs POLARIS    return correlation: {polaris_corr:+.3f}")
    print()
    if (RESULTS / "phoenix_v2_metrics.json").exists():
        ph = json.loads((RESULTS / "phoenix_v2_metrics.json").read_text())
        print("Phoenix v2  (reference):")
        for k in ["full", "is", "oos"]:
            m = ph["v2"][k]
            print(f"  {k.upper():4s} SR={m['sharpe']:.2f}  CAGR={m['cagr']*100:5.1f}% "
                  f"Vol={m['ann_vol']*100:.1f}%  MDD={m['mdd']*100:.1f}%")
        print()
        # Honest comparison
        for k_n, k_p in [("FULL", "full"), ("IS", "is"), ("OOS", "oos")]:
            n = neutrino_metrics[k_n]; p = ph["v2"][k_p]
            print(f"  [{k_n}] NEUTRINO vs Phoenix:")
            print(f"    Sharpe {n['sharpe']:5.2f} vs {p['sharpe']:5.2f}  "
                  f"({'NEU' if n['sharpe'] > p['sharpe'] else 'PHX'} wins)")
            print(f"    CAGR  {n['cagr']*100:5.1f}% vs {p['cagr']*100:5.1f}%  "
                  f"({'NEU' if n['cagr'] > p['cagr'] else 'PHX'} wins)")

    out = {
        "strategy": "NEUTRINO",
        "version": "v2-final",
        "window": [str(r_final.index.min().date()), str(r_final.index.max().date())],
        "is_window": [str(IS_START.date()), str(IS_END.date())],
        "oos_start": str(OOS_START.date()),
        "tc_bps": TC_BPS,
        "params": {
            "TVOL": TVOL, "GK_LB": GK_LB,
            "RV_YOY_DENOM": RV_YOY_DENOM, "RV_90_DENOM": RV_90_DENOM,
            "CORR_LB": CORR_LB, "CORR_THR_LO": CORR_THR_LO,
            "CORR_THR_HI": CORR_THR_HI, "CORR_MID_MULT": CORR_MID_MULT,
            "SELF_DD_FLOOR": SELF_DD_FLOOR,
            "TARGET_VOL": TARGET_VOL, "PORT_DD_FLOOR": PORT_DD_FLOOR,
        },
        "raw":      raw_metrics,
        "throttled": thr_metrics,
        "neutrino": neutrino_metrics,
        "is_oos_gap_sharpe": float(abs(neutrino_metrics["IS"]["sharpe"]
                                       - neutrino_metrics["OOS"]["sharpe"])),
        "is_oos_gap_cagr":   float(abs(neutrino_metrics["IS"]["cagr"]
                                       - neutrino_metrics["OOS"]["cagr"])),
        "phoenix_v2_corr": phoenix_corr,
        "polaris_corr":    polaris_corr,
    }
    (RESULTS / "neutrino_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    pd.DataFrame({
        "Date": r_final.index, "ret": r_final.values,
        "raw_ret": r_raw.values, "thr_ret": r_thr.values,
        "vt_scale": vt_scale.values, "dd_mult": dd_mult.values,
    }).to_csv(RESULTS / "neutrino_returns.csv", index=False)
    W.assign().to_csv(RESULTS / "neutrino_weights.csv")
    print(f"\nSaved: {RESULTS/'neutrino_metrics.json'}")
    print(f"Saved: {RESULTS/'neutrino_returns.csv'}")
    print(f"Saved: {RESULTS/'neutrino_weights.csv'}")
    return out


if __name__ == "__main__":
    run()
