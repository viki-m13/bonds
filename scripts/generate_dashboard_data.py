#!/usr/bin/env python3
"""
Generate dashboard_data.json and docs/index.html from strategy results.
Called by update_dashboard.py or GitHub Actions.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

DATA_DIR = Path("/home/user/bonds/data")
DOCS_DIR = Path("/home/user/bonds/docs")

# Allow override via env
if "REPO_ROOT" in __import__("os").environ:
    root = Path(__import__("os").environ["REPO_ROOT"])
    DATA_DIR = root / "data"
    DOCS_DIR = root / "docs"


def generate():
    # Load strategy returns
    strat = pd.read_csv(DATA_DIR / "results" / "dichs_returns.csv", parse_dates=[0])
    strat.columns = ["Date", "return"]
    strat = strat.set_index("Date").sort_index()

    # Load stream returns
    streams = pd.read_csv(DATA_DIR / "results" / "dichs_stream_returns.csv", parse_dates=[0])
    streams = streams.set_index(streams.columns[0]).sort_index()

    # Load AGG
    agg = pd.read_csv(DATA_DIR / "etfs" / "AGG.csv", parse_dates=["Date"]).set_index("Date").sort_index()
    agg_ret = agg["Close"].pct_change().rename("AGG")

    # Load SPY (try local first, then download)
    spy_path = DATA_DIR / "etfs" / "SPY.csv"
    if spy_path.exists():
        spy_df = pd.read_csv(spy_path, parse_dates=["Date"]).set_index("Date").sort_index()
        spy_ret = spy_df["Close"].pct_change().rename("SPY")
    else:
        try:
            import yfinance as yf
            spy_data = yf.download("SPY", start="2005-01-01", progress=False)
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.get_level_values(0)
            spy_ret = spy_data["Close"].pct_change().rename("SPY")
            spy_data.to_csv(spy_path)
        except Exception:
            spy_ret = pd.Series(dtype=float, name="SPY")

    # FRED
    fred = pd.read_csv(DATA_DIR / "fred" / "_combined_fred.csv", parse_dates=["Date"]).set_index("Date")
    fred = fred[~fred.index.duplicated(keep="first")].sort_index()
    for c in fred.columns:
        fred[c] = pd.to_numeric(fred[c], errors="coerce")
    fred = fred.ffill()

    # ETF prices for allocations
    etf_tickers = ["HYG", "JNK", "LQD", "VCIT", "VCSH", "IGIB", "EMB", "MUB",
                   "MBB", "TIP", "IEF", "IEI", "SHY", "TLT"]
    etf_prices = {}
    for t in etf_tickers:
        p = DATA_DIR / "etfs" / f"{t}.csv"
        if p.exists():
            df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            etf_prices[t] = df["Close"]
    etf_prices = pd.DataFrame(etf_prices)
    etf_ret = etf_prices.pct_change()

    # Current allocations
    pairs = [
        ("HYG", "IEF", "HY_mid"), ("HYG", "TLT", "HY_long"), ("HYG", "SHY", "HY_short"),
        ("JNK", "IEF", "JNK_mid"), ("LQD", "IEF", "IG_mid"),
        ("VCIT", "IEI", "MidCorp"), ("VCSH", "SHY", "ShortCorp"), ("IGIB", "IEI", "IG5yr"),
        ("EMB", "IEF", "EM_mid"), ("EMB", "TLT", "EM_long"),
        ("MUB", "SHY", "Muni_short"), ("MUB", "IEI", "Muni_mid"),
        ("MBB", "IEF", "MBS"), ("TIP", "IEF", "TIPS"),
    ]

    vix = fred.get("VIXCLS")
    latest_vix = float(vix.dropna().iloc[-1]) if vix is not None else 20.0
    vix_pctl = float(vix.rolling(252, min_periods=126).rank(pct=True).dropna().iloc[-1]) if vix is not None else 0.5
    stress_scale = max(0.5, min(1.3, 1.3 - 0.8 * vix_pctl))

    current_allocations = {}
    for long_e, hedge_e, name in pairs:
        if long_e in etf_ret.columns and hedge_e in etf_ret.columns:
            cov = etf_ret[long_e].rolling(252, min_periods=126).cov(etf_ret[hedge_e])
            var = etf_ret[hedge_e].rolling(252, min_periods=126).var()
            beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)
            latest_beta = float(beta.dropna().iloc[-1])
            hedged = etf_ret[long_e] - beta.shift(1) * etf_ret[hedge_e]
            rs = float(hedged.tail(63).mean() / hedged.tail(63).std() * np.sqrt(252)) if hedged.tail(63).std() > 0 else 0
            current_allocations[name] = {
                "long": long_e, "hedge": hedge_e,
                "beta": round(latest_beta, 3),
                "stress_scale": round(stress_scale, 3),
                "recent_63d_sharpe": round(rs, 2),
                "long_price": round(float(etf_prices[long_e].dropna().iloc[-1]), 2),
                "hedge_price": round(float(etf_prices[hedge_e].dropna().iloc[-1]), 2),
            }

    # Align
    combined = pd.DataFrame({"DICHS": strat["return"], "AGG": agg_ret, "SPY": spy_ret}).dropna()
    cum = (1 + combined).cumprod()

    # Weekly samples
    weekly_cum = cum.resample("W").last().dropna()
    equity_data = [{"date": dt.strftime("%Y-%m-%d"),
                    "DICHS": round(float(weekly_cum.loc[dt, "DICHS"]), 4),
                    "AGG": round(float(weekly_cum.loc[dt, "AGG"]), 4),
                    "SPY": round(float(weekly_cum.loc[dt, "SPY"]), 4)} for dt in weekly_cum.index]

    dd = {}
    for col in ["DICHS", "AGG", "SPY"]:
        c = cum[col]; dd[col] = (c - c.cummax()) / c.cummax()
    weekly_dd = pd.DataFrame(dd).resample("W").last().dropna()
    dd_data = [{"date": dt.strftime("%Y-%m-%d"),
                "DICHS": round(float(weekly_dd.loc[dt, "DICHS"]), 4),
                "AGG": round(float(weekly_dd.loc[dt, "AGG"]), 4),
                "SPY": round(float(weekly_dd.loc[dt, "SPY"]), 4)} for dt in weekly_dd.index]

    # Yearly
    yearly_rows = []
    for col in ["DICHS", "AGG", "SPY"]:
        for yr, g in combined[col].groupby(combined[col].index.year):
            if len(g) < 20:
                continue
            ar = g.mean() * 252; av = g.std() * np.sqrt(252)
            sr = ar / av if av > 0 else 0
            c = (1 + g).cumprod(); mdd = ((c - c.cummax()) / c.cummax()).min()
            wr = (g > 0).mean()
            yearly_rows.append({"strategy": col, "year": int(yr),
                "return": round(ar * 100, 2), "vol": round(av * 100, 2),
                "sharpe": round(sr, 3), "maxdd": round(mdd * 100, 2),
                "winrate": round(wr * 100, 1)})

    # Overall metrics
    def calc_metrics(r, name):
        r = r.dropna(); ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0; c = (1 + r).cumprod()
        mdd = ((c - c.cummax()) / c.cummax()).min()
        cal = ar / abs(mdd) if mdd != 0 else 0; wr = (r > 0).mean()
        ds = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else av
        sortino = ar / ds if ds > 0 else 0; total_ret = c.iloc[-1] - 1
        return {"name": name, "total_return": round(total_ret * 100, 2),
            "ann_return": round(ar * 100, 2), "ann_vol": round(av * 100, 2),
            "sharpe": round(sr, 3), "sortino": round(sortino, 3),
            "max_dd": round(mdd * 100, 2), "calmar": round(cal, 3),
            "win_rate": round(wr * 100, 1), "skew": round(r.skew(), 3),
            "kurt": round(r.kurtosis(), 3),
            "best_year": round(float(r.groupby(r.index.year).apply(lambda x: x.mean() * 252).max()) * 100, 2),
            "worst_year": round(float(r.groupby(r.index.year).apply(lambda x: x.mean() * 252).min()) * 100, 2),
            "n_years": round(len(r) / 252, 1)}

    overall_metrics = [
        calc_metrics(combined["DICHS"], "DICHS Strategy"),
        calc_metrics(combined["AGG"], "AGG (US Agg Bond)"),
        calc_metrics(combined["SPY"], "SPY (S&P 500)"),
    ]

    # Monthly
    monthly_ret = combined["DICHS"].resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_data = [{"year": dt.year, "month": dt.month,
                     "return": round(float(monthly_ret.loc[dt]) * 100, 2)} for dt in monthly_ret.index]

    # Rolling Sharpe
    rolling_sr = combined["DICHS"].rolling(252).mean() / combined["DICHS"].rolling(252).std() * np.sqrt(252)
    rolling_sr_w = rolling_sr.resample("W").last().dropna()
    rolling_data = [{"date": dt.strftime("%Y-%m-%d"),
                     "sharpe": round(float(rolling_sr_w.loc[dt]), 3)} for dt in rolling_sr_w.index]

    dashboard_data = {
        "equity_curve": equity_data, "drawdown": dd_data,
        "yearly": yearly_rows, "overall": overall_metrics,
        "monthly_returns": monthly_data, "rolling_sharpe": rolling_data,
        "allocations": current_allocations,
        "latest_vix": round(latest_vix, 2),
        "vix_percentile": round(vix_pctl * 100, 1),
        "stress_scale": round(stress_scale, 3),
        "last_updated": combined.index[-1].strftime("%Y-%m-%d"),
        "stream_names": list(streams.columns),
    }

    # Save JSON
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "dashboard_data.json", "w") as f:
        json.dump(dashboard_data, f)

    print(f"Dashboard data: {len(equity_data)} equity pts, {len(current_allocations)} allocations")
    return dashboard_data


if __name__ == "__main__":
    generate()
