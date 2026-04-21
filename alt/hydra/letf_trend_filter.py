"""Follow-up test: LETF strategies with a 200-day trend filter.

Idea: each LETF is held only if its *underlying* (SPY/QQQ/TLT/GLD/SOX-via-SMH)
is above its 200d SMA. Otherwise, allocation rotates to BIL (cash).

Still: no daily vol scaling, rebalance every N days.
The trend signal is checked at each rebalance (not intrabar).
"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats

TC_BPS = 15.0

# LETF -> underlying for trend detection
UNDERLYING = {
    "UPRO": "SPY", "SSO": "SPY",
    "TQQQ": "QQQ", "QLD": "QQQ",
    "TMF":  "TLT", "TYD":  "IEF",
    "UGL":  "GLD",
    "SOXL": "SMH",
    "TECL": "XLK",
    "FAS":  "XLF",
    "DRN":  "VNQ",
    "EDC":  "EEM",
    "ERX":  "XLE",
    "UCO":  "USO",
    "YINN": "FXI",
    "NUGT": "GDX",
}


def load_prices(tickers, start=None):
    frames = {}
    for t in tickers:
        s = load_etf(t)
        if s is None: continue
        frames[t] = s
    px = pd.DataFrame(frames).sort_index()
    if start is not None:
        px = px.loc[start:]
    return px


def trend_signals(underlyings, lookback=200, dates=None):
    """Return DataFrame[date x underlying] of bool: 1 if close > SMA(lookback)."""
    px = load_prices(underlyings)
    sma = px.rolling(lookback).mean()
    sig = (px > sma).astype(float).shift(1)    # T-1 close signal, act at T
    if dates is not None:
        sig = sig.reindex(dates).fillna(0.0)
    return sig


def run_trend_strategy(letf_weights, rebal_days=21, trend_lookback=200,
                       start="2011-01-01", tc_bps=TC_BPS):
    """letf_weights: dict ticker -> target weight (sum to 1).
    For each LETF, if underlying is above its SMA, hold full weight;
    otherwise redirect to BIL.
    """
    tickers = list(letf_weights.keys())
    underlyings = sorted(set(UNDERLYING[t] for t in tickers))

    # Load LETF + BIL returns
    all_tickers = tickers + ["BIL"]
    px = load_prices(all_tickers, start=start).dropna(how="any")
    rets = px.pct_change().fillna(0)

    # Trend signals on underlyings
    sig_under = trend_signals(underlyings, lookback=trend_lookback, dates=rets.index)

    idx = rets.index
    cols = rets.columns

    W = pd.DataFrame(0.0, index=idx, columns=cols)
    last_w = pd.Series(0.0, index=cols)
    prev_w = pd.Series(0.0, index=cols)
    tc_daily = pd.Series(0.0, index=idx)

    rebal_mask = pd.Series(False, index=idx)
    rebal_mask.iloc[::rebal_days] = True

    for i, d in enumerate(idx):
        if rebal_mask.iloc[i]:
            w = pd.Series(0.0, index=cols)
            cash = 0.0
            for t, wt in letf_weights.items():
                under = UNDERLYING[t]
                active = sig_under.iloc[i].get(under, 0.0)
                if active > 0.5:
                    w[t] = wt
                else:
                    cash += wt
            w["BIL"] = cash
            turnover = (w - prev_w).abs().sum()
            tc_daily.iloc[i] = turnover * (tc_bps / 1e4)
            prev_w = w
            last_w = w
        W.iloc[i] = last_w

    port_ret = (W * rets).sum(axis=1) - tc_daily
    return port_ret.dropna(), W


def run_summary(r, label):
    s = stats(r.dropna(), label)
    nav = (1 + r).cumprod()
    cagr = (nav.iloc[-1]) ** (252 / len(r)) - 1
    mdd = (nav / nav.cummax() - 1).min()
    cagr_mdd = cagr / abs(mdd) if mdd < 0 else float("inf")
    return {"label": label, "cagr": cagr * 100, "vol": s["vol"],
            "mdd": mdd * 100, "sharpe": s["sharpe"], "cagr_mdd": cagr_mdd,
            "navx": s["navx"]}


def main():
    recipes = {
        "100% TQQQ + trend":             {"TQQQ": 1.0},
        "100% UPRO + trend":             {"UPRO": 1.0},
        "100% SOXL + trend":             {"SOXL": 1.0},
        "100% TECL + trend":             {"TECL": 1.0},
        "HFEA 55/45 UPRO/TMF + trend":   {"UPRO": 0.55, "TMF": 0.45},
        "HFEA 60/40 UPRO/TMF + trend":   {"UPRO": 0.60, "TMF": 0.40},
        "HFEA-Tech 55/45 TQQQ/TMF + trend": {"TQQQ": 0.55, "TMF": 0.45},
        "HFEA-Tech 60/40 TQQQ/TMF + trend": {"TQQQ": 0.60, "TMF": 0.40},
        "5-sleeve 20/20/20/20/20 UPRO/TQQQ/SOXL/TMF/UGL + trend":
            {"UPRO": 0.20, "TQQQ": 0.20, "SOXL": 0.20, "TMF": 0.20, "UGL": 0.20},
        "3-way 40/40/20 UPRO/TMF/UGL + trend":
            {"UPRO": 0.40, "TMF": 0.40, "UGL": 0.20},
    }
    rows = []
    equities = {}
    for name, w in recipes.items():
        for nd in (3, 5, 10, 21):
            try:
                r, W = run_trend_strategy(w, rebal_days=nd)
            except Exception as e:
                print(f"  skip {name} @ {nd}d: {e}")
                continue
            if len(r) < 500: continue
            res = run_summary(r, f"{name} @ {nd}d")
            rows.append(res)
            equities[res["label"]] = r

    df = pd.DataFrame(rows).sort_values("cagr", ascending=False).reset_index(drop=True)
    print(f"{'Strategy @ Rebal':68s}  {'CAGR%':>6s}  {'Vol%':>6s}  {'MDD%':>7s}  "
          f"{'SR':>5s}  {'C/MDD':>6s}  {'NAVx':>7s}")
    print("-" * 115)
    for _, r in df.iterrows():
        print(f"{r['label']:68s}  {r['cagr']:>6.2f}  {r['vol']:>6.2f}  "
              f"{r['mdd']:>7.2f}  {r['sharpe']:>5.2f}  "
              f"{r['cagr_mdd']:>6.2f}  {r['navx']:>7.1f}")

    print("\nTop 8 by CAGR/|MDD|:")
    df2 = df.sort_values("cagr_mdd", ascending=False).head(8)
    for _, r in df2.iterrows():
        print(f"  {r['label']:68s}  CAGR={r['cagr']:>5.2f}%  "
              f"MDD={r['mdd']:>6.2f}%  C/MDD={r['cagr_mdd']:>5.2f}  "
              f"SR={r['sharpe']:>4.2f}")

    # Save best by CAGR/MDD
    best_ratio = df2.iloc[0]
    r_best = equities[best_ratio["label"]]
    out = pd.DataFrame({best_ratio["label"]: r_best})
    out.to_csv("/home/user/bonds/data/results/letf_trend_best.csv")
    print(f"\nWrote letf_trend_best.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
