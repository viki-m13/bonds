"""
Proprietary V6 — "ZEPHYR" Zero-Scaling Enhanced Persistent High-Yield Receiver.

Breakthrough thesis: Sharpe of a portfolio = (mean return) / (std dev).
You can reach Sharpe 3 two ways:
  (A) High excess return with moderate vol (equity-like strategies).
  (B) Modest positive return with ultra-low vol (credit-carry strategies).

Since we've ruled out vol scaling (which is how (A) strategies hit S=3),
the answer must be a (B) strategy. The asset class that intrinsically
delivers moderate return + ultra-low vol is: ultra-short-duration
corporate credit + AAA CLOs + floating-rate loans.

Design:
  - Core sleeve (60%): ultra-short credit (JPST, MINT, ICSH, NEAR) —
    earns overnight rate + 50-100bps spread, vol < 1%.
  - Carry sleeve (25%): AAA CLOs (JAAA, CLOI) + BBB CLOs (JBBB) —
    earns overnight + 100-250bps, vol 1-3%.
  - Floating sleeve (10%): leveraged loans (BKLN, SRLN, FLOT) —
    earns overnight + 300bps, vol 4-6%.
  - Crisis-alpha sleeve (5%): gold (GLD) — decorrelator for stress.

ALL weights are FROZEN. Monthly rebalance simply re-trues the weights
back to target. No vol scaling, no regime gating, no selection changes.

We also test a regime-gated variant where the carry and floating sleeves
rotate to BIL during crises (HY OAS > threshold).
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


def metrics(r):
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
            "total_return": float(cum.iloc[-1] - 1), "n_years": float(len(r) / 252)}


def backtest_static(weights_dict, rebalance_days=21, tc_bps=5.0,
                    start="2014-01-01", regime=None, cash_ticker="BIL"):
    """Run a backtest with static target weights. Monthly rebalance re-trues
    to target; no daily vol scaling. Optionally gate to cash via regime."""
    tickers = list(weights_dict.keys())
    prices = pd.DataFrame({t: load_etf(t) for t in tickers})
    prices = prices.dropna()
    prices = prices.loc[start:]
    rets = prices.pct_change().fillna(0)
    dates = rets.index

    target = pd.Series(weights_dict)
    target = target / target.sum()

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


def regime_hy_oas(dates, threshold=6.5):
    hy = load_fred("BAMLH0A0HYM2")
    if hy is None:
        return pd.Series(1.0, index=dates)
    h = hy.reindex(dates).ffill()
    g = (h < threshold).astype(float)
    return g.shift(1).fillna(1.0)


def regime_hy_oas_smooth(dates, low=5.0, high=8.0):
    """Smooth gate: full weight below `low`, zero weight above `high`, linear in between."""
    hy = load_fred("BAMLH0A0HYM2")
    if hy is None:
        return pd.Series(1.0, index=dates)
    h = hy.reindex(dates).ffill()
    g = ((high - h) / (high - low)).clip(0, 1)
    return g.shift(1).fillna(1.0)


def regime_rate_trend(dates, threshold=0.5):
    """Gate off duration when 10Y yield rose more than threshold (pp) in 3M."""
    y10 = load_fred("DGS10")
    if y10 is None:
        return pd.Series(1.0, index=dates)
    y = y10.reindex(dates).ffill()
    chg = y - y.shift(63)
    g = (chg < threshold).astype(float)
    return g.shift(1).fillna(1.0)


def regime_combined(dates):
    hy = regime_hy_oas_smooth(dates)
    rt = regime_rate_trend(dates)
    return (hy * rt).clip(0, 1)


def main():
    print("\n=== V6 ZEPHYR — Static Weight Portfolios ===\n")

    # Portfolio configurations
    portfolios = {
        "P1_ultra_short_heavy":   {"JPST": 0.30, "MINT": 0.25, "JAAA": 0.15, "CLOI": 0.10, "BKLN": 0.10, "JBBB": 0.05, "GLD": 0.05},
        "P2_clo_heavy":            {"JAAA": 0.35, "CLOI": 0.20, "JBBB": 0.15, "JPST": 0.15, "BKLN": 0.10, "GLD": 0.05},
        "P3_balanced":             {"JPST": 0.20, "MINT": 0.15, "JAAA": 0.15, "CLOI": 0.10, "BKLN": 0.10, "SRLN": 0.05, "JBBB": 0.10, "ANGL": 0.05, "GLD": 0.05, "TIP": 0.05},
        "P4_equal_8":              {"JPST": 0.125, "MINT": 0.125, "JAAA": 0.125, "CLOI": 0.125, "BKLN": 0.125, "SRLN": 0.125, "FLOT": 0.125, "JBBB": 0.125},
        "P5_with_hy":              {"JPST": 0.15, "MINT": 0.10, "JAAA": 0.15, "CLOI": 0.10, "BKLN": 0.10, "JBBB": 0.10, "ANGL": 0.10, "SHYG": 0.10, "HYG": 0.05, "GLD": 0.05},
        "P6_with_duration":        {"JPST": 0.20, "JAAA": 0.15, "BKLN": 0.10, "CLOI": 0.10, "IEF": 0.15, "SHY": 0.10, "GLD": 0.10, "TIP": 0.05, "JBBB": 0.05},
        "P7_clo_only":             {"JAAA": 0.50, "CLOI": 0.30, "JBBB": 0.20},
        "P8_minimal":              {"JPST": 0.40, "JAAA": 0.30, "JBBB": 0.15, "BKLN": 0.10, "GLD": 0.05},
        "P9_floating_heavy":       {"BKLN": 0.25, "SRLN": 0.20, "FLOT": 0.20, "JAAA": 0.15, "JPST": 0.10, "CLOI": 0.05, "JBBB": 0.05},
    }

    results = {}
    for name, w in portfolios.items():
        r = backtest_static(w)
        m = metrics(r)
        results[name] = {"weights": w, "metrics_nogate": m}
        print(f"{name:28s}  NO gate:   SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  Vol={m['ann_vol']:.2%}  MDD={m['max_dd']:.2%}")

    # With HY OAS gate
    print()
    temp_prices = load_etf("BIL").loc["2014-01-01":]
    dates = temp_prices.index
    reg = regime_hy_oas_smooth(dates)
    for name, w in portfolios.items():
        r = backtest_static(w, regime=reg)
        m = metrics(r)
        results[name]["metrics_gate_hy"] = m
        print(f"{name:28s}  HY gate:   SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  Vol={m['ann_vol']:.2%}  MDD={m['max_dd']:.2%}")

    # With combined gate (HY + rate trend)
    print()
    reg2 = regime_combined(dates)
    for name, w in portfolios.items():
        r = backtest_static(w, regime=reg2)
        m = metrics(r)
        results[name]["metrics_gate_combo"] = m
        print(f"{name:28s}  COMBO:     SR={m['sharpe']:.3f}  Ret={m['ann_return']:.2%}  Vol={m['ann_vol']:.2%}  MDD={m['max_dd']:.2%}")

    # Summary: find best
    best = {}
    for name, r in results.items():
        for variant in ["metrics_nogate", "metrics_gate_hy", "metrics_gate_combo"]:
            sr = r[variant]["sharpe"]
            best[f"{name}__{variant}"] = sr
    top = sorted(best.items(), key=lambda x: -x[1])[:10]
    print("\n=== TOP 10 ===")
    for k, v in top:
        print(f"  {k:50s} SR={v:.3f}")

    with open(RESULTS / "proprietary_v6_zephyr.json", "w") as f:
        json.dump(results, f, indent=2, default=str)


if __name__ == "__main__":
    main()
