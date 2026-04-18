"""
Proprietary V7 — "ZEPHYR-20Y" — validate the approach on longer history.

The V6 Sharpe 5 used JAAA/CLOI/JBBB which only have data from 2020-2022.
This V7 runs the same static-weight concept using long-history
equivalents: FLOT/BKLN/MINT/SHYG/ANGL/LQD/IEF.

We also build a "stitched" long-history CLO proxy: use FLOT+BKLN weighted
to match JAAA's vol profile pre-2020, then splice in real JAAA from 2020.

Goal: show the approach works in multiple rate regimes.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/user/bonds/data")
ETF = DATA / "etfs"
FRED = DATA / "fred"
RESULTS = Path("/home/user/bonds/alt/results")


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def metrics(r, name=""):
    r = r.loc[r.ne(0).idxmax():] if (r != 0).any() else r
    if r.std() == 0 or len(r) == 0:
        return {k: 0 for k in ["sharpe", "ann_return", "ann_vol", "max_dd", "sortino", "calmar", "total_return", "n_years"]}
    ar, av = r.mean() * 252, r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1); mdd = dd.min()
    neg = r[r < 0]
    sor = (r.mean() * 252) / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else float("inf")
    return {"sharpe": float(sr), "ann_return": float(ar), "ann_vol": float(av),
            "max_dd": float(mdd), "sortino": float(sor),
            "calmar": float(ar / abs(mdd)) if mdd < 0 else float("inf"),
            "total_return": float(cum.iloc[-1] - 1), "n_years": float(len(r) / 252),
            "start": str(r.index[0].date()), "end": str(r.index[-1].date())}


def backtest_static(weights_dict, rebalance_days=21, tc_bps=5.0,
                    start="2014-01-01", regime=None, cash_ticker="BIL"):
    tickers = list(weights_dict.keys())
    prices = pd.DataFrame({t: load_etf(t) for t in tickers}).dropna()
    prices = prices.loc[start:]
    rets = prices.pct_change().fillna(0)
    dates = rets.index
    target = pd.Series(weights_dict); target = target / target.sum()
    current = pd.Series(0.0, index=tickers)
    port = pd.Series(0.0, index=dates)
    last_idx = -rebalance_days
    bil = load_etf(cash_ticker).reindex(dates).ffill().pct_change().fillna(0) if cash_ticker else pd.Series(0.0, index=dates)
    for i, d in enumerate(dates):
        if i - last_idx >= rebalance_days:
            tc = (target - current).abs().sum() * (tc_bps / 1e4)
            port.iloc[i] -= tc
            current = target.copy()
            last_idx = i
        r = (rets.iloc[i] * current).sum()
        if regime is not None:
            g = float(regime.get(d, 1.0))
            r = g * r + (1 - g) * bil.iloc[i]
        port.iloc[i] += r
    return port


def regime_hy_smooth(dates, low=5.0, high=8.0):
    hy = load_fred("BAMLH0A0HYM2")
    if hy is None:
        return pd.Series(1.0, index=dates)
    h = hy.reindex(dates).ffill()
    g = ((high - h) / (high - low)).clip(0, 1)
    return g.shift(1).fillna(1.0)


def rate_trend_gate(dates, threshold=0.7):
    y = load_fred("DGS10")
    if y is None:
        return pd.Series(1.0, index=dates)
    yv = y.reindex(dates).ffill()
    chg = yv - yv.shift(63)
    g = (chg < threshold).astype(float)
    return g.shift(1).fillna(1.0)


def main():
    print("\n=== V7 ZEPHYR-20Y — LONG-HISTORY VALIDATION ===\n")

    # Portfolios using only long-history ETFs (all available >=2013)
    LONG_HISTORY = {
        "L1_mint_bkln":         {"MINT": 0.55, "BKLN": 0.20, "FLOT": 0.15, "SHYG": 0.10},
        "L2_equal_5":           {"MINT": 0.20, "BKLN": 0.20, "FLOT": 0.20, "SHYG": 0.20, "ANGL": 0.20},
        "L3_short_heavy":       {"MINT": 0.40, "FLOT": 0.20, "BKLN": 0.15, "SRLN": 0.10, "SHYG": 0.10, "GLD": 0.05},
        "L4_floating_mix":      {"BKLN": 0.25, "SRLN": 0.20, "FLOT": 0.25, "MINT": 0.15, "SHYG": 0.10, "GLD": 0.05},
        "L5_conservative":      {"MINT": 0.50, "FLOT": 0.25, "BKLN": 0.10, "SHYG": 0.10, "GLD": 0.05},
        "L6_with_short_hy":     {"MINT": 0.30, "FLOT": 0.20, "SHYG": 0.20, "BKLN": 0.15, "ANGL": 0.10, "GLD": 0.05},
        "L7_minimal":           {"MINT": 0.60, "BKLN": 0.30, "GLD": 0.10},
        "L8_mint_only":         {"MINT": 1.0},
        "L9_flot_only":         {"FLOT": 1.0},
        "L10_bkln_only":        {"BKLN": 1.0},
    }

    results = {}
    print("--- NO REGIME GATE ---")
    for name, w in LONG_HISTORY.items():
        r = backtest_static(w, start="2014-01-01")
        m = metrics(r, name)
        results[name] = {"weights": w, "metrics_nogate": m}
        print(f"{name:24s} SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  Vol={m['ann_vol']:.2%}  "
              f"MDD={m['max_dd']:.2%}  {m['n_years']:.1f}y  [{m['start']}..{m['end']}]")

    # With regime gate
    print("\n--- WITH HY GATE + RATE TREND GATE ---")
    bdates = load_etf("MINT").loc["2014-01-01":].index
    gate = regime_hy_smooth(bdates) * rate_trend_gate(bdates)
    for name, w in LONG_HISTORY.items():
        r = backtest_static(w, start="2014-01-01", regime=gate)
        m = metrics(r, name)
        results[name]["metrics_gate"] = m
        print(f"{name:24s} SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  Vol={m['ann_vol']:.2%}  "
              f"MDD={m['max_dd']:.2%}")

    # What if we include all available "modern" CLOs from 2022 forward?
    print("\n--- MODERN ENHANCED (2022+) with CLOs ---")
    MODERN = {
        "M1_clo_plus":  {"JPST": 0.25, "MINT": 0.15, "JAAA": 0.20, "CLOI": 0.10, "JBBB": 0.10, "BKLN": 0.10, "SHYG": 0.05, "GLD": 0.05},
        "M2_optimized": {"JPST": 0.20, "MINT": 0.15, "JAAA": 0.25, "CLOI": 0.15, "JBBB": 0.15, "BKLN": 0.05, "GLD": 0.05},
    }
    for name, w in MODERN.items():
        r = backtest_static(w, start="2022-06-01")
        m = metrics(r, name)
        results[name] = {"weights": w, "metrics_modern": m}
        print(f"{name:24s} SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  Vol={m['ann_vol']:.2%}  "
              f"MDD={m['max_dd']:.2%}  [{m['start']}..{m['end']}]")

    # Test year-by-year Sharpe on best long-history portfolio
    print("\n--- YEAR-BY-YEAR: L2_equal_5 ---")
    r = backtest_static(LONG_HISTORY["L2_equal_5"], start="2014-01-01")
    r = r.loc[r.ne(0).idxmax():]
    for y, g in r.groupby(r.index.year):
        ar = g.mean() * 252
        av = g.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        print(f"  {y}: SR={sr:.2f}  Ret={ar:.2%}  Vol={av:.2%}")

    with open(RESULTS / "proprietary_v7_zephyr20y.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
