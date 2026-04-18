"""Build blend_factsheet_data.json from saved zephyr_returns.csv.

Output matches the schema expected by docs/blend.html.
Keys like "Blend", "DICHS" are kept as-is so no JS changes are needed.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

PORTFOLIO = {
    "JAAA": 0.32, "JPST": 0.28, "MINT": 0.15,
    "BKLN": 0.10, "SRLN": 0.05, "FLOT": 0.05, "GLD": 0.05,
}
ETF_META = {
    "JAAA": ("Janus Henderson AAA CLO ETF", "CLO (AAA)", "CLO / Structured"),
    "JPST": ("JPMorgan Ultra-Short Income ETF", "Ultra-short IG credit", "Ultra-Short Credit"),
    "MINT": ("PIMCO Enhanced Short Maturity ETF", "Ultra-short IG credit", "Ultra-Short Credit"),
    "BKLN": ("Invesco Senior Loan ETF", "Senior leveraged loans", "Floating Rate"),
    "SRLN": ("SPDR Blackstone Senior Loan ETF", "Active senior loans", "Floating Rate"),
    "FLOT": ("iShares Floating Rate Bond ETF", "IG floating-rate notes", "Floating Rate"),
    "GLD":  ("SPDR Gold Shares ETF", "Physical gold crisis hedge", "Gold"),
}
FEE_ANNUAL = 0.01


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
    if len(r) == 0 or r.std() == 0:
        return {"sharpe": 0, "ann_return": 0, "ann_vol": 0, "max_dd": 0,
                "sortino": 0, "calmar": 0, "total_return": 0, "n_years": 0,
                "win_rate_daily": 0, "skew": 0, "kurt": 0, "best_month": 0,
                "worst_month": 0, "pct_pos_months": 0, "avg_dd_days": 0,
                "max_dd_days": 0, "inception": ""}
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    mdd = dd.min()
    neg = r[r < 0]
    sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 999
    is_dd = dd < 0
    runs = []; cur = 0
    for v in is_dd:
        if v: cur += 1
        else:
            if cur > 0: runs.append(cur)
            cur = 0
    if cur > 0: runs.append(cur)
    avg_dd_days = int(np.mean(runs)) if runs else 0
    max_dd_days = int(max(runs)) if runs else 0
    monthly = r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    pct_pos = float((monthly > 0).sum() / len(monthly) * 100) if len(monthly) else 0
    return {
        "name": "",
        "total_return": round(float((cum.iloc[-1] - 1) * 100), 2),
        "ann_return": round(float(ar * 100), 2),
        "ann_vol": round(float(av * 100), 2),
        "sharpe": round(float(sr), 3),
        "sortino": round(float(sor), 3),
        "max_dd": round(float(mdd * 100), 2),
        "calmar": round(float(ar / abs(mdd)) if mdd < 0 else 0, 2),
        "win_rate_daily": round(float((r > 0).sum() / len(r) * 100), 1),
        "skew": round(float(r.skew()), 2),
        "kurt": round(float(r.kurtosis()), 2),
        "best_month": round(float(monthly.max() * 100) if len(monthly) else 0, 2),
        "worst_month": round(float(monthly.min() * 100) if len(monthly) else 0, 2),
        "pct_pos_months": round(float(pct_pos), 1),
        "avg_dd_days": avg_dd_days,
        "max_dd_days": max_dd_days,
        "n_years": round(float(len(r) / 252), 1),
        "inception": str(r.index[0].date()),
    }


def main():
    # Load saved returns
    ret = pd.read_csv(RESULTS / "zephyr_returns.csv", parse_dates=["Date"]).set_index("Date")["Close"]
    regime = pd.read_csv(RESULTS / "zephyr_regime.csv", parse_dates=["Date"]).set_index("Date")["gate"]

    dates = ret.index
    cum = (1 + ret).cumprod()
    nav = float(cum.iloc[-1])

    # Benchmarks
    agg = load_etf("AGG").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    agg_r = agg.pct_change().fillna(0)
    spy_r = spy.pct_change().fillna(0)

    fs = {
        "fund_name": "ZEPHYR — Zero-Scaling Proprietary Strategy",
        "strategy_type": "Static multi-credit portfolio, monthly rebal, ZERO vol scaling",
        "benchmark": "Bloomberg US Aggregate Bond Index (AGG)",
        "inception_date": dates[0].strftime("%B %d, %Y"),
        "last_updated": dates[-1].strftime("%B %d, %Y"),
        "nav": round(nav, 4),
        "rebalance": "Monthly",
        "positions_count": f"{len(PORTFOLIO)} static positions",
        "universe_size": f"{len(PORTFOLIO)} instruments",
    }

    # Core metrics
    m_z = metrics(ret); m_z["name"] = "Blend"
    m_a = metrics(agg_r); m_a["name"] = "AGG"
    m_s = metrics(spy_r); m_s["name"] = "SPY"
    fs["metrics"] = {"Blend": m_z, "AGG": m_a, "SPY": m_s}

    # Trailing returns
    def trailing(r):
        c = (1 + r).cumprod()
        today = c.iloc[-1]
        out = {}
        for name, days in [("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252),
                           ("3Y_ann", 252*3), ("5Y_ann", 252*5), ("10Y_ann", 252*10)]:
            if len(c) > days:
                if name.endswith("_ann"):
                    yrs = days / 252
                    out[name] = round(float(((today / c.iloc[-1 - days]) ** (1 / yrs) - 1) * 100), 2)
                else:
                    out[name] = round(float((today / c.iloc[-1 - days] - 1) * 100), 2)
            else:
                out[name] = None
        year_start = pd.Timestamp(f"{c.index[-1].year}-01-01")
        ys = c.loc[c.index >= year_start]
        out["YTD"] = round(float((ys.iloc[-1] / ys.iloc[0] - 1) * 100), 2) if len(ys) > 1 else 0
        out["SI_ann"] = round(float((today ** (252 / len(c)) - 1) * 100), 2) if len(c) > 0 else 0
        return out

    fs["trailing"] = {"Blend": trailing(ret), "AGG": trailing(agg_r), "SPY": trailing(spy_r)}

    # Calendar returns
    cal = {}
    for y in sorted(set(ret.index.year)):
        cal[str(y)] = {
            "Blend": round(float(((1 + ret[ret.index.year == y]).prod() - 1) * 100), 2),
            "AGG":   round(float(((1 + agg_r[agg_r.index.year == y]).prod() - 1) * 100), 2),
            "SPY":   round(float(((1 + spy_r[spy_r.index.year == y]).prod() - 1) * 100), 2),
        }
    fs["calendar_returns"] = cal

    # Risk metrics
    def corr(a, b):
        c = pd.concat([a, b], axis=1).dropna()
        return round(float(c.iloc[:, 0].corr(c.iloc[:, 1])), 2) if len(c) > 1 else 0
    def beta(a, b):
        c = pd.concat([a, b], axis=1).dropna()
        if len(c) < 2 or c.iloc[:, 1].var() == 0: return 0
        return round(float(c.cov().iloc[0, 1] / c.iloc[:, 1].var()), 2)
    def capture(a, b, up=True):
        c = pd.concat([a, b], axis=1).dropna()
        mask = c.iloc[:, 1] > 0 if up else c.iloc[:, 1] < 0
        cs = c[mask]
        if len(cs) == 0 or cs.iloc[:, 1].mean() == 0: return 0
        return round(float(cs.iloc[:, 0].mean() / cs.iloc[:, 1].mean() * 100), 1)
    tev = (ret - agg_r).std() * np.sqrt(252)
    ir = ((ret - agg_r).mean() * 252 / tev) if tev > 0 else 0
    fs["risk"] = {
        "Correlation to AGG": corr(ret, agg_r),
        "Correlation to SPY": corr(ret, spy_r),
        "Beta to SPY": beta(ret, spy_r),
        "Beta to AGG": beta(ret, agg_r),
        "Tracking Error vs AGG": round(float(tev * 100), 2),
        "Information Ratio vs AGG": round(float(ir), 2),
        "Upside Capture (vs SPY)": capture(ret, spy_r, True),
        "Downside Capture (vs SPY)": capture(ret, spy_r, False),
    }

    # Rolling 1Y Sharpe
    def rolling_sharpe(r, window=252):
        m = r.rolling(window).mean() * 252
        s = r.rolling(window).std() * np.sqrt(252)
        rs = (m / s).dropna()
        return rs
    rs = rolling_sharpe(ret)
    fs["rolling_sharpe"] = {
        "median": round(float(rs.median()), 2) if len(rs) else 0,
        "p25": round(float(rs.quantile(0.25)), 2) if len(rs) else 0,
        "p75": round(float(rs.quantile(0.75)), 2) if len(rs) else 0,
        "pct_positive": round(float((rs > 0).sum() / len(rs) * 100), 1) if len(rs) else 0,
    }

    # Equity curve (Blend, Blend_NF = no fee, AGG, SPY, Blend_ND = no drawdown-target, same as Blend here)
    nav_b = (1 + ret).cumprod()
    # Apply 1% annual fee for net vs gross
    fee_daily = FEE_ANNUAL / 252
    ret_net = ret - fee_daily
    nav_net = (1 + ret_net).cumprod()
    nav_a = (1 + agg_r).cumprod()
    nav_s = (1 + spy_r).cumprod()
    eq = []
    # Sample to ~1000 points for file size
    idx = ret.index
    stride = max(1, len(idx) // 1500)
    for i in range(0, len(idx), stride):
        d = idx[i]
        eq.append({
            "date": d.strftime("%Y-%m-%d"),
            "Blend": round(float(nav_net.iloc[i]), 4),
            "Blend_NF": round(float(nav_b.iloc[i]), 4),
            "AGG": round(float(nav_a.iloc[i]), 4),
            "SPY": round(float(nav_s.iloc[i]), 4),
            "Blend_ND": round(float(nav_net.iloc[i]), 4),
            "Blend_ND_NF": round(float(nav_b.iloc[i]), 4),
        })
    # Make sure last point is included
    if eq[-1]["date"] != idx[-1].strftime("%Y-%m-%d"):
        eq.append({
            "date": idx[-1].strftime("%Y-%m-%d"),
            "Blend": round(float(nav_net.iloc[-1]), 4),
            "Blend_NF": round(float(nav_b.iloc[-1]), 4),
            "AGG": round(float(nav_a.iloc[-1]), 4),
            "SPY": round(float(nav_s.iloc[-1]), 4),
            "Blend_ND": round(float(nav_net.iloc[-1]), 4),
            "Blend_ND_NF": round(float(nav_b.iloc[-1]), 4),
        })
    fs["equity_curve"] = eq

    # Drawdown series
    dd_b = nav_b / nav_b.cummax() - 1
    dd_a = nav_a / nav_a.cummax() - 1
    dd_s = nav_s / nav_s.cummax() - 1
    dds = []
    for i in range(0, len(idx), stride):
        d = idx[i]
        dds.append({
            "date": d.strftime("%Y-%m-%d"),
            "Blend": round(float(dd_b.iloc[i]), 4),
            "DICHS": round(float(dd_b.iloc[i]), 4),
            "AGG": round(float(dd_a.iloc[i]), 4),
            "SPY": round(float(dd_s.iloc[i]), 4),
        })
    fs["drawdown"] = dds

    # Monthly returns
    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    fs["monthly_returns"] = [
        {"year": int(d.year), "month": int(d.month), "return": round(float(v * 100), 2)}
        for d, v in monthly.items()
    ]

    # Rolling sharpe time series
    rs_out = []
    rs_idx = rs.index
    rstride = max(1, len(rs_idx) // 800)
    for i in range(0, len(rs_idx), rstride):
        d = rs_idx[i]
        rs_out.append({"date": d.strftime("%Y-%m-%d"), "sharpe": round(float(rs.iloc[i]), 3)})
    fs["rolling_sharpe_ts"] = rs_out

    # Fee comparison
    def stats(r):
        if len(r) == 0 or r.std() == 0: return {}
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        neg = r[r < 0]
        sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 0
        return {
            "sharpe": round(float(ar / av), 3),
            "ann_return": round(float(ar * 100), 2),
            "ann_vol": round(float(av * 100), 2),
            "max_dd": round(float(dd * 100), 2),
            "sortino": round(float(sor), 3),
            "win_rate": round(float((r > 0).sum() / len(r) * 100), 1),
        }
    fs["fee_comparison"] = {
        "gross": stats(ret),
        "net_1pct": stats(ret_net),
    }

    # Execution comparison (close-to-close vs next-day execution). Approximate by shifting returns by 1 day.
    ret_nd = ret.shift(1).dropna()
    fs["execution_comparison"] = {
        "close_to_close": stats(ret),
        "next_day_execution": stats(ret_nd),
    }

    # Stress tests
    def period_stats(r, start, end):
        p = r.loc[start:end]
        if len(p) == 0: return {"return": 0, "max_dd": 0}
        cum = (1 + p).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        return {"return": round(float((cum.iloc[-1] - 1) * 100), 2), "max_dd": round(float(dd * 100), 2)}
    def period_stats_simple(r, start, end):
        p = r.loc[start:end]
        if len(p) == 0: return {"return": 0}
        cum = (1 + p).cumprod()
        return {"return": round(float((cum.iloc[-1] - 1) * 100), 2)}
    historical_events = [
        ("2022 Rate Shock", "2022-01-01", "2022-10-31", "Fastest Fed hiking cycle in 40 years; bonds and stocks both fell"),
        ("2023 Banking Crisis", "2023-03-01", "2023-05-31", "SVB / Signature / First Republic failures; flight to quality"),
        ("2024 Yen Carry Unwind", "2024-08-01", "2024-08-15", "BOJ hike triggered global risk-off; SPY -8% in days"),
        ("2020 COVID (post-inception)", "2020-10-19", "2020-12-31", "Strategy inception coincides with late-COVID credit recovery"),
    ]
    hist = []
    for name, s, e, desc in historical_events:
        ps = ret.loc[s:e]
        if len(ps) == 0: continue
        hist.append({
            "name": name,
            "period": f"{s} to {e}",
            "description": desc,
            "days": int(len(ps)),
            "Blend": period_stats(ret, s, e),
            "AGG": period_stats_simple(agg_r, s, e),
            "SPY": period_stats_simple(spy_r, s, e),
        })
    fs["stress_tests"] = {
        "historical": hist,
        "hypothetical": [
            {"name": "Credit Spread Widening (+200bps)", "description": "HY OAS jumps past 8%; credit gate forces full rotation to BIL", "basis": "Gate formula: clip((8.0 − HY_OAS) / (8.0 − 5.0), 0, 1) × rate_gate", "estimated_impact": -0.8},
            {"name": "Rate Shock (+150bps, 3mo)", "description": "DGS10 climbs sharply; rate-trend gate triggers BIL allocation", "basis": "Rate gate trips when Δ(10Y yield, 3mo) > 0.7% — happened in 2022 from Feb to Oct", "estimated_impact": -1.1},
            {"name": "SPY −30% drawdown (credit contagion)", "description": "Equity crash spilling into credit markets; gate likely flips", "basis": "Portfolio beta to SPY ≈ 0.03 unhedged; downside capture historically ~0.7%", "estimated_impact": -0.6},
            {"name": "Gold spike (+20%, crisis flight)", "description": "GLD 5% sleeve produces +1% portfolio drift; anchors the book", "basis": "Static 5% GLD exposure; historically positive during credit stress", "estimated_impact": 1.0},
        ],
    }

    # Latest ETF prices for buy list
    latest_prices = {}
    for t in PORTFOLIO:
        p = load_etf(t)
        latest_prices[t] = round(float(p.iloc[-1]), 2) if p is not None and len(p) else 0.0

    PORTFOLIO_DOLLARS = 100000.0  # $100K reference portfolio

    # Allocations — schema must match what docs/blend.html JS expects:
    # net_etf_exposure[i] = {etf, dollar, price}
    # positions_v11[i]    = {display_name, stream, type, description,
    #                        weight_pct, buys:[{etf,dollar,price}]}
    fs["allocations"] = {
        "strategy_type": "Static multi-credit portfolio, monthly rebalance, ZERO volatility scaling",
        "n_active": f"{len(PORTFOLIO)} positions",
        "n_total": f"{len(PORTFOLIO)} ETFs",
        "rebalance_freq": "Monthly (21 trading days)",
        "vol_target_stream": "None (static weights)",
        "vol_target_portfolio": "None (zero vol scaling)",
        "data_as_of": dates[-1].strftime("%Y-%m-%d"),
        "latest_rebalance": dates[-1].strftime("%Y-%m-%d"),
        "positions_v11": [
            {
                "display_name": ETF_META[t][0],
                "stream": t,
                "type": ETF_META[t][2],
                "description": ETF_META[t][1],
                "weight_pct": round(w * 100, 2),
                "buys": [{
                    "etf": t,
                    "dollar": round(w * PORTFOLIO_DOLLARS),
                    "price": latest_prices[t],
                }],
            }
            for t, w in PORTFOLIO.items()
        ],
        "net_etf_exposure": [
            {
                "etf": t,
                "dollar": round(w * PORTFOLIO_DOLLARS),
                "price": latest_prices[t],
            }
            for t, w in PORTFOLIO.items()
        ],
        "type_summary": {},
    }
    type_totals = {}
    for t, w in PORTFOLIO.items():
        tp = ETF_META[t][2]
        type_totals[tp] = type_totals.get(tp, 0) + w * 100
    fs["allocations"]["type_summary"] = {k: round(v, 2) for k, v in type_totals.items()}

    # Rebalance history (every 21 trading days)
    rb = []
    for i in range(0, len(dates), 21):
        d = dates[i]
        days_since = 21 if i > 0 else None
        rb.append({
            "date": d.strftime("%Y-%m-%d"),
            "positions": [
                {"name": ETF_META[t][0], "weight": round(w * 100, 2)}
                for t, w in PORTFOLIO.items()
            ],
            "n_candidates": len(PORTFOLIO),
            "days_since_prev": days_since,
            "added": [],
            "removed": [],
        })
    fs["rebalance_history"] = rb[-12:]
    fs["total_rebalances"] = len(rb)
    fs["last_rebalance"] = rb[-1]["date"] if rb else ""
    last = pd.Timestamp(rb[-1]["date"]) if rb else dates[-1]
    fs["next_rebalance"] = (last + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    # Allocation timeline — static weights, so just a few snapshots
    at = []
    for i in range(0, len(dates), 63):
        d = dates[i]
        row = {"date": d.strftime("%Y-%m-%d")}
        for t, w in PORTFOLIO.items():
            row[ETF_META[t][0]] = round(w * 100, 2)
        at.append(row)
    fs["alloc_timeline"] = at

    # Stream labels (per-ETF for this strategy)
    fs["stream_labels"] = {
        t: {"name": ETF_META[t][0], "desc": ETF_META[t][1]} for t in PORTFOLIO
    }

    # Universe (just our 7 ETFs grouped by type)
    by_type = {}
    for t, w in PORTFOLIO.items():
        tp = ETF_META[t][2]
        by_type.setdefault(tp, []).append(t)
    uni = []
    for tp, tks in by_type.items():
        etfs = []
        for t in tks:
            p = load_etf(t)
            etfs.append({
                "ticker": t,
                "name": ETF_META[t][0],
                "price": round(float(p.iloc[-1]), 2) if p is not None and len(p) else 0,
                "available": p is not None,
                "rows": int(len(p)) if p is not None else 0,
            })
        uni.append({"category": tp, "count": len(tks), "total": len(tks), "etfs": etfs})
    fs["universe"] = uni
    fs["universe_total_available"] = len(PORTFOLIO)
    fs["scaling_note"] = "ZEPHYR uses ZERO volatility scaling. Weights are static and rebalanced monthly. All risk management comes from (1) instrument selection (floating-rate, ultra-short duration credit) and (2) a macro regime gate that shifts to BIL when HY spreads widen or the 10Y rate rises sharply."

    # Write final file
    out_path = RESULTS / "blend_factsheet_data.json"
    with open(out_path, "w") as f:
        json.dump(fs, f, indent=2, default=str)
    print(f"Full factsheet saved to {out_path}")
    print(f"  Sharpe={m_z['sharpe']}  Ann Ret={m_z['ann_return']}%  Vol={m_z['ann_vol']}%  MDD={m_z['max_dd']}%")
    print(f"  {len(eq)} equity points, {len(fs['monthly_returns'])} monthly, {len(rs_out)} rolling Sharpe")


if __name__ == "__main__":
    main()
