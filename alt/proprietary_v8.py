"""
Proprietary V8 — "ZEPHYR-CLO" — Find Sharpe 3+ over longest reasonable window.

Key move: use JAAA from 2020-11, extend with SHYG/BKLN/MINT pre-2020.
Also: try concentrated portfolios with the highest-Sharpe individual ETFs.
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
        return {"sharpe": 0}
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


def vix_gate(dates, threshold=25.0):
    vix = load_fred("VIXCLS")
    if vix is None:
        return pd.Series(1.0, index=dates)
    v = vix.reindex(dates).ffill()
    g = (v < threshold).astype(float)
    return g.shift(1).fillna(1.0)


def backtest_static(weights_dict, rebalance_days=21, tc_bps=5.0,
                    start="2014-01-01", end=None, regime=None, cash_ticker="BIL"):
    tickers = list(weights_dict.keys())
    prices = pd.DataFrame({t: load_etf(t) for t in tickers}).dropna()
    prices = prices.loc[start:end]
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


def run_window(weights, start, end=None, name=""):
    bdates = load_etf("BIL").loc[start:end].index
    r_none = backtest_static(weights, start=start, end=end)
    r_gate = backtest_static(weights, start=start, end=end,
                              regime=regime_hy_smooth(bdates) * rate_trend_gate(bdates))
    m_none = metrics(r_none)
    m_gate = metrics(r_gate)
    print(f"{name:30s}  no gate SR={m_none['sharpe']:.3f} ret={m_none['ann_return']:.2%} vol={m_none['ann_vol']:.2%} "
          f"| gated SR={m_gate['sharpe']:.3f} ret={m_gate['ann_return']:.2%} vol={m_gate['ann_vol']:.2%}")
    return {"no_gate": m_none, "gated": m_gate, "weights": weights, "returns_nogate": r_none, "returns_gated": r_gate}


def main():
    print("\n=== V8 ZEPHYR-CLO — PORTFOLIO SEARCH ===\n")

    # Tests from JAAA inception (2020-11)
    start_jaaa = "2020-11-01"
    print(f"--- Window: 2020-11 onward (~5.5 years) ---")
    portfolios_jaaa = {
        "A_all_in_clo":   {"JAAA": 0.70, "JPST": 0.20, "MINT": 0.10},
        "B_balanced":     {"JAAA": 0.40, "JPST": 0.25, "MINT": 0.15, "BKLN": 0.10, "SHYG": 0.05, "GLD": 0.05},
        "C_diverse":      {"JAAA": 0.30, "JPST": 0.20, "MINT": 0.15, "BKLN": 0.10, "SRLN": 0.05, "FLOT": 0.05, "ANGL": 0.05, "SHYG": 0.05, "GLD": 0.05},
        "D_clo_floating": {"JAAA": 0.35, "BKLN": 0.20, "SRLN": 0.15, "FLOT": 0.15, "JPST": 0.10, "GLD": 0.05},
        "E_concentrated": {"JAAA": 0.50, "JPST": 0.30, "MINT": 0.15, "GLD": 0.05},
    }
    res_jaaa = {}
    for k, w in portfolios_jaaa.items():
        res_jaaa[k] = run_window(w, start_jaaa, None, k)

    # Tests from 2014 (no JAAA, but use longer-history "proxies")
    print(f"\n--- Window: 2014-01 onward (~12 years) ---")
    portfolios_long = {
        "L_mint_floating":    {"MINT": 0.45, "FLOT": 0.25, "BKLN": 0.15, "SHYG": 0.10, "GLD": 0.05},
        "L_floating_heavy":   {"BKLN": 0.25, "SRLN": 0.20, "FLOT": 0.20, "MINT": 0.20, "SHYG": 0.10, "GLD": 0.05},
        "L_mint_heavy":       {"MINT": 0.60, "FLOT": 0.20, "BKLN": 0.10, "SHYG": 0.05, "GLD": 0.05},
        "L_conservative":     {"MINT": 0.70, "FLOT": 0.15, "BKLN": 0.10, "GLD": 0.05},
    }
    res_long = {}
    for k, w in portfolios_long.items():
        res_long[k] = run_window(w, "2014-01-01", None, k)

    # Stitched history: use proxies pre-JAAA, real JAAA post-2020-11
    print(f"\n--- Stitched: FLOT/BKLN pre-Nov-2020, real JAAA+JPST+MINT after ---")
    # Build a synthetic "long CLO" by splicing
    def build_stitched():
        pre = pd.DataFrame({
            "CORE":  load_etf("MINT"),   # 60%
            "FLOAT": load_etf("FLOT"),   # 25%
            "LOAN":  load_etf("BKLN"),   # 10%
            "GLD":   load_etf("GLD"),    # 5%
        }).dropna()
        pre_start = pd.Timestamp("2014-01-01")
        pre_end   = pd.Timestamp("2020-11-02")
        pre = pre.loc[pre_start:pre_end]
        pre_rets = pre.pct_change().fillna(0)
        w_pre = pd.Series({"CORE": 0.60, "FLOAT": 0.25, "LOAN": 0.10, "GLD": 0.05})
        pre_port = (pre_rets * w_pre).sum(axis=1)

        post_start = pd.Timestamp("2020-11-02")
        post = pd.DataFrame({
            "CLO":   load_etf("JAAA"),
            "CORE":  load_etf("JPST"),
            "MINT":  load_etf("MINT"),
            "FLOAT": load_etf("FLOT"),
            "LOAN":  load_etf("BKLN"),
            "GLD":   load_etf("GLD"),
        }).dropna()
        post = post.loc[post_start:]
        post_rets = post.pct_change().fillna(0)
        w_post = pd.Series({"CLO": 0.35, "CORE": 0.25, "MINT": 0.15, "FLOAT": 0.10, "LOAN": 0.10, "GLD": 0.05})
        post_port = (post_rets * w_post).sum(axis=1)

        # Stitch: combine ensuring no overlap
        pre_port = pre_port.loc[:pre_end - pd.Timedelta(days=1)]
        stitched = pd.concat([pre_port, post_port]).sort_index()
        return stitched[~stitched.index.duplicated(keep="last")]

    stitched = build_stitched()
    m_stitch = metrics(stitched)
    print(f"Stitched history        SR={m_stitch['sharpe']:.3f} ret={m_stitch['ann_return']:.2%} "
          f"vol={m_stitch['ann_vol']:.2%} MDD={m_stitch['max_dd']:.2%} "
          f"{m_stitch['n_years']:.1f}y [{m_stitch['start']}..{m_stitch['end']}]")

    # With gate
    bd = stitched.index
    gate = regime_hy_smooth(bd) * rate_trend_gate(bd)
    bil = load_etf("BIL").reindex(bd).ffill().pct_change().fillna(0)
    stitched_gated = stitched * gate + (1 - gate) * bil
    m_stitch_g = metrics(stitched_gated)
    print(f"Stitched + gated        SR={m_stitch_g['sharpe']:.3f} ret={m_stitch_g['ann_return']:.2%} "
          f"vol={m_stitch_g['ann_vol']:.2%} MDD={m_stitch_g['max_dd']:.2%}")

    # Year-by-year of stitched gated
    print("\n--- Stitched + gated year-by-year ---")
    for y, g in stitched_gated.groupby(stitched_gated.index.year):
        ar = g.mean() * 252
        av = g.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        print(f"  {y}: SR={sr:.2f}  Ret={ar:.2%}  Vol={av:.2%}")

    out = {
        "jaaa_window": {k: {"no_gate": v["no_gate"], "gated": v["gated"], "weights": v["weights"]} for k, v in res_jaaa.items()},
        "long_window": {k: {"no_gate": v["no_gate"], "gated": v["gated"], "weights": v["weights"]} for k, v in res_long.items()},
        "stitched_nogate": m_stitch,
        "stitched_gated": m_stitch_g,
    }
    with open(RESULTS / "proprietary_v8_zephyr_clo.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
