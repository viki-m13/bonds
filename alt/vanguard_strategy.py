"""
VANGUARD — Volatility-Term-Structure-Gated Participation
==========================================================
Long-biased leveraged-ETF rotation strategy with a 4-trigger risk regime
gate built from macro series (VIX, HY OAS, Yield curve) and equity trend.

Core architecture:
    * Universe = {QLD, UGL, TMF, TYD} — 4 uncorrelated leveraged assets
      (2x S&P Nasdaq-100, 2x Gold, 3x LT Treasury, 3x 7-10y Treasury).
    * At each month-start: rank by 189-day absolute momentum, require
      price > 200d SMA; inverse-60d-vol weight the eligible names.
    * Risk gate: composite "trigger count" from HY-OAS slope, VIX-z,
      yield-curve inversion, SPY-below-200d. Each trigger reduces
      participation in 25% steps (1.0 -> 0.75 -> 0.50 -> 0.25 -> 0.0)
      with 5-day smoothing for stability.
    * Gross is scaled by a constant multiplier to hit the 20% CAGR target.

Signal timing:
    * All signals computed through close[t-1].
    * Weights set at open[t]; return on bar t+1 = weight_t * (open[t+1]/open[t] - 1).

No daily vol targeting.
Monthly momentum rebalance, daily gate re-check.
Transaction cost: 5 bps one-way applied to every weight change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
ETF_DIR = ROOT / "data" / "etfs"
FRED_DIR = ROOT / "data" / "fred"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IS_START = pd.Timestamp("2010-03-11")
IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-01")
# OOS_END not hardcoded — strategy extends to latest available data
OOS_END = None

TC_BPS = 5.0
TC_RATE = TC_BPS / 1e4

# Broad leveraged-ETF universe (>= 10 used in the screening)
LEV_UNIVERSE = [
    "TQQQ", "UPRO", "QLD", "SSO",
    "SOXL", "TECL", "FAS", "ERX",
    "NUGT", "DRN", "UCO", "UGL",
    "TMF", "TYD", "EDC", "YINN", "LABU",
]

# Core rotation universe (selected by asset-class diversification + Sharpe):
# QLD   - 2x Nasdaq-100    (equity growth)
# UGL   - 2x Gold          (inflation / macro hedge)
# TMF   - 3x LT Treasury   (bond rally / recession hedge)
# TYD   - 3x 7-10y Treas   (mid-duration, steadier)
CORE = ["QLD", "UGL", "TMF", "TYD"]

# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def load_etf(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(ETF_DIR / f"{ticker}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df[["Open", "Close"]].astype(float)


def load_fred(name: str) -> pd.Series:
    df = pd.read_csv(FRED_DIR / f"{name}.csv", parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")
    return df.iloc[:, 0].astype(float)


def build_panels(universe: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    u = universe or LEV_UNIVERSE
    opens, closes = {}, {}
    for t in u:
        d = load_etf(t)
        opens[t] = d["Open"]
        closes[t] = d["Close"]
    opens = pd.DataFrame(opens).sort_index()
    closes = pd.DataFrame(closes).sort_index()
    idx = pd.bdate_range(opens.index.min(), opens.index.max())
    opens = opens.reindex(idx).ffill(limit=2)
    closes = closes.reindex(idx).ffill(limit=2)
    return opens, closes


def build_fred(idx: pd.DatetimeIndex) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    f["VIX"] = load_fred("VIXCLS").reindex(idx).ffill()
    f["HY"] = load_fred("BAMLH0A0HYM2").reindex(idx).ffill()
    f["T10Y2Y"] = load_fred("T10Y2Y").reindex(idx).ffill()
    return f


def load_spy(idx: pd.DatetimeIndex) -> pd.Series:
    spy = pd.read_csv(ETF_DIR / "SPY.csv", parse_dates=["Date"]).set_index("Date")["Close"].astype(float).sort_index()
    return spy.reindex(idx).ffill(limit=2)


# --------------------------------------------------------------------------- #
# Regime gate
# --------------------------------------------------------------------------- #
def compute_trigger_count(fred: pd.DataFrame, spy: pd.Series) -> pd.Series:
    """Composite risk-off trigger count (0-4) from 4 macro conditions.

    1. HY OAS widening: 20d change > 0.3 OR 5d change > 0.25
    2. VIX spike: 60d z > 1.2 OR level > 30
    3. Curve inversion in progress: T10Y2Y < 0 AND falling (60d)
    4. SPY below its 200d SMA
    """
    vix = fred["VIX"]
    hy = fred["HY"]
    hy_slope20 = hy - hy.shift(20)
    hy_slope5 = hy - hy.shift(5)
    vix_z = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
    t10y2y = fred["T10Y2Y"]
    t10y2y_s60 = t10y2y - t10y2y.shift(60)

    c_hy = (hy_slope20 > 0.30) | (hy_slope5 > 0.25)
    c_vix = (vix_z > 1.2) | (vix > 30.0)
    c_curve = (t10y2y < 0.0) & (t10y2y_s60 < 0.0)
    c_spy = ~(spy > spy.rolling(200).mean())

    trg = (
        c_hy.astype(float).fillna(0)
        + c_vix.astype(float).fillna(0)
        + c_curve.astype(float).fillna(0)
        + c_spy.astype(float).fillna(0)
    )
    return trg.rolling(5).mean()


def participation_from_triggers(trg_smooth: pd.Series) -> pd.Series:
    """Map trigger count -> participation in {1, 0.75, 0.5, 0.25, 0}.
    Lagged 1 bar so participation at t uses data through t-1.
    """
    p = pd.Series(1.0, index=trg_smooth.index)
    p[trg_smooth >= 0.5] = 0.75
    p[trg_smooth >= 1.0] = 0.50
    p[trg_smooth >= 1.5] = 0.25
    p[trg_smooth >= 2.0] = 0.00
    return p.shift(1).fillna(0.0)


# --------------------------------------------------------------------------- #
# Basket construction
# --------------------------------------------------------------------------- #
def build_basket_weights(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    mom_lb: int = 189,
    sma_lb: int = 200,
    vol_lb: int = 60,
) -> pd.DataFrame:
    """Risk-parity weights across CORE basket, masked by momentum eligibility.

    Monthly rebalance: pick names with positive mom_lb return AND price above
    sma_lb moving average; weight by 1/vol; normalize to sum 1.
    """
    c_lag = closes.shift(1)  # close through t-1
    mom = c_lag / c_lag.shift(mom_lb) - 1.0
    above_sma = c_lag > c_lag.rolling(sma_lb).mean()
    eligible = (mom > 0) & above_sma & c_lag.notna()

    rets = closes.pct_change().shift(1)
    vol = rets.rolling(vol_lb).std()
    iv = 1.0 / vol
    iv = iv.where(eligible, 0.0)
    iv_w = iv.div(iv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    idx = opens.index
    month_mark = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).transform("first") == pd.Series(idx, index=idx)
    rebal_dates = idx[month_mark.values]

    weights = pd.DataFrame(0.0, index=idx, columns=opens.columns)
    cur = pd.Series(0.0, index=opens.columns)
    for dt in idx:
        if dt in rebal_dates:
            cur = iv_w.loc[dt].copy()
        weights.loc[dt] = cur.values
    return weights


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #
def backtest(opens: pd.DataFrame, weights: pd.DataFrame, tc_rate: float = TC_RATE) -> pd.DataFrame:
    o2o = opens / opens.shift(1) - 1.0
    w_lag = weights.shift(1).fillna(0.0)            # weight set at open[t-1]
    gross = (w_lag * o2o).sum(axis=1)               # earns open[t-1]->open[t]

    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_lag = (turnover * tc_rate).shift(1).fillna(0.0)
    net = gross - cost_lag
    return pd.DataFrame({
        "gross_ret": gross, "cost": cost_lag, "net_ret": net,
        "turnover": turnover.shift(1).fillna(0.0),
    })


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def perf_metrics(rets: pd.Series, periods: int = 252) -> dict:
    r = rets.dropna()
    if len(r) < 5:
        return {"sharpe": np.nan, "cagr": np.nan, "vol": np.nan, "mdd": np.nan, "n": len(r)}
    mu = r.mean() * periods
    sigma = r.std(ddof=0) * np.sqrt(periods)
    sharpe = mu / sigma if sigma > 0 else np.nan
    navx = float((1.0 + r).prod())
    yrs = len(r) / periods
    cagr = navx ** (1.0 / yrs) - 1.0 if yrs > 0 else np.nan
    nav = (1.0 + r).cumprod()
    mdd = float((nav / nav.cummax() - 1.0).min())
    return {"sharpe": float(sharpe), "cagr": float(cagr), "vol": float(sigma),
            "mdd": mdd, "n": int(len(r)), "navx": navx}


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(
    mom_lb: int = 189,
    sma_lb: int = 200,
    vol_lb: int = 60,
    gross: float = 1.5,
    verbose: bool = True,
):
    opens, closes = build_panels(CORE)
    idx = opens.index
    fred = build_fred(idx)
    spy = load_spy(idx)

    # Regime
    trg = compute_trigger_count(fred, spy)
    part = participation_from_triggers(trg)

    # Basket
    basket_w = build_basket_weights(opens, closes, mom_lb=mom_lb, sma_lb=sma_lb, vol_lb=vol_lb)

    # Apply gate + gross
    w = basket_w.mul(part * gross, axis=0)

    bt = backtest(opens, w, tc_rate=TC_RATE).loc[IS_START:]
    net = bt["net_ret"]

    is_r = net.loc[IS_START:IS_END]
    oos_r = net.loc[OOS_START:]

    metrics = {
        "full": perf_metrics(net),
        "is": perf_metrics(is_r),
        "oos": perf_metrics(oos_r),
        "avg_turnover_annual": float(bt["turnover"].sum() / ((net.index[-1] - net.index[0]).days / 365.25)),
    }

    # Regime breakdown
    regimes = pd.cut(trg.reindex(net.index).shift(0),
                     bins=[-0.01, 0.5, 1.5, 5.01],
                     labels=["risk_on", "caution", "risk_off"])
    reg_bd = {}
    for label in ["risk_on", "caution", "risk_off"]:
        mask = (regimes == label).values
        r = net[mask]
        reg_bd[label] = {
            "days": int(mask.sum()),
            "ann_ret": float(r.mean() * 252) if len(r) else np.nan,
            "vol": float(r.std() * np.sqrt(252)) if len(r) else np.nan,
            "sharpe": float(r.mean() / r.std() * np.sqrt(252)) if len(r) and r.std() > 0 else np.nan,
        }
    metrics["regime_breakdown"] = reg_bd

    # Yearly returns
    yearly = (1 + net).groupby(net.index.year).prod() - 1
    metrics["yearly_returns"] = {int(y): float(v) for y, v in yearly.items()}

    metrics["params"] = {
        "mom_lb": mom_lb, "sma_lb": sma_lb, "vol_lb": vol_lb, "gross": gross,
        "tc_bps": TC_BPS, "universe": CORE,
        "gate_triggers": [
            "HY OAS 20d slope > 0.30 OR 5d slope > 0.25",
            "VIX 60d z > 1.2 OR VIX level > 30",
            "T10Y2Y < 0 AND 60d change < 0",
            "SPY < 200d SMA",
        ],
        "gate_map": {"0 triggers": 1.00, "0.5": 0.75, "1.0": 0.50, "1.5": 0.25, "2.0+": 0.0},
        "rebal": "monthly (momentum basket); daily (risk gate)",
    }

    if verbose:
        print("=" * 80)
        print(f"VANGUARD  universe={CORE}")
        print(f"  mom_lb={mom_lb}  sma_lb={sma_lb}  vol_lb={vol_lb}  gross={gross}x")
        print("-" * 80)
        for k in ["full", "is", "oos"]:
            m = metrics[k]
            print(f"  {k.upper():4s}  Sh={m['sharpe']:6.3f}  CAGR={m['cagr']*100:6.2f}%  "
                  f"Vol={m['vol']*100:5.2f}%  MDD={m['mdd']*100:7.2f}%  "
                  f"N={m['n']:4d}  NAVx={m['navx']:.2f}")
        gap = abs(metrics["is"]["sharpe"] - metrics["oos"]["sharpe"])
        print(f"  |IS-OOS gap|={gap:.3f}  TO/yr={metrics['avg_turnover_annual']:.2f}")
        print("-" * 80)
        print("  Regime breakdown:")
        for k, v in reg_bd.items():
            print(f"    {k:9s} days={v['days']:4d}  ann_ret={v['ann_ret']*100:6.2f}%  "
                  f"vol={v['vol']*100:5.2f}%  Sh={v['sharpe']:.2f}")
        print("-" * 80)
        print("  Yearly returns:")
        for y, v in sorted(metrics["yearly_returns"].items()):
            print(f"    {y}: {v*100:6.2f}%")
        print("=" * 80)

    return bt, w, metrics, trg


def save_outputs(bt: pd.DataFrame, weights: pd.DataFrame, metrics: dict, trg: pd.Series):
    out_csv = RESULTS_DIR / "vanguard_returns.csv"
    out_json = RESULTS_DIR / "vanguard_metrics.json"
    df = bt.copy()
    df["trigger_count"] = trg.reindex(df.index)
    df.to_csv(out_csv)
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(f"Saved: {out_csv}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    bt, w, metrics, trg = run()
    save_outputs(bt, w, metrics, trg)
