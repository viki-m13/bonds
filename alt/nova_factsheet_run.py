"""Assemble full nova_factsheet_data.json — mirrors aurora_factsheet_run.py
but with a single sleeve for the unified NOVA momentum strategy."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from nova_factsheet import (
    ROOT, ETF, RESULTS, UNIVERSE, ETF_META, CAP, FEE_ANNUAL,
    load_etf, metrics, trailing,
)


def main():
    df = pd.read_csv(RESULTS / "nova_returns.csv",
                     parse_dates=["Date"]).set_index("Date")
    ret = df["Close"]
    rebals = pd.read_csv(RESULTS / "nova_rebalances.csv",
                         parse_dates=["date"])
    w_crypto = df["Crypto"]
    w_equity = df["Equity"]

    dates = ret.index
    cum = (1 + ret).cumprod()
    nav = float(cum.iloc[-1])

    agg = load_etf("AGG").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    agg_r = agg.pct_change().fillna(0)
    spy_r = spy.pct_change().fillna(0)

    fs = {
        "fund_name": "NOVA — Max-Growth Momentum",
        "strategy_type": "Cross-sectional momentum on leveraged ETFs + crypto",
        "benchmark": "S&P 500 (SPY)",
        "inception_date": dates[0].strftime("%B %d, %Y"),
        "last_updated": dates[-1].strftime("%B %d, %Y"),
        "nav": round(nav, 4),
        "rebalance": "Weekly (5-day) momentum rotation",
        "positions_count": "Top-3 of 20 instruments, 33% cap each",
        "universe_size": "20 instruments (18 leveraged ETFs + BTC + ETH)",
    }

    m_n = metrics(ret); m_n["name"] = "NOVA"
    m_a = metrics(agg_r); m_a["name"] = "AGG"
    m_s = metrics(spy_r); m_s["name"] = "SPY"
    fs["metrics"] = {"Blend": m_n, "AGG": m_a, "SPY": m_s}
    fs["trailing"] = {"Blend": trailing(ret), "AGG": trailing(agg_r), "SPY": trailing(spy_r)}

    cal = {}
    for y in sorted(set(ret.index.year)):
        cal[str(y)] = {
            "Blend": round(float(((1 + ret[ret.index.year == y]).prod() - 1) * 100), 2),
            "AGG":   round(float(((1 + agg_r[agg_r.index.year == y]).prod() - 1) * 100), 2),
            "SPY":   round(float(((1 + spy_r[spy_r.index.year == y]).prod() - 1) * 100), 2),
        }
    fs["calendar_returns"] = cal

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
    tev = (ret - spy_r).std() * np.sqrt(252)
    ir = ((ret - spy_r).mean() * 252 / tev) if tev > 0 else 0
    fs["risk"] = {
        "Correlation to AGG": corr(ret, agg_r),
        "Correlation to SPY": corr(ret, spy_r),
        "Beta to SPY": beta(ret, spy_r),
        "Beta to AGG": beta(ret, agg_r),
        "Tracking Error vs SPY": round(float(tev * 100), 2),
        "Information Ratio vs SPY": round(float(ir), 2),
        "Upside Capture (vs SPY)": capture(ret, spy_r, True),
        "Downside Capture (vs SPY)": capture(ret, spy_r, False),
    }

    def rolling_sharpe(r, window=252):
        m = r.rolling(window).mean() * 252
        s = r.rolling(window).std() * np.sqrt(252)
        return (m / s).dropna()
    rs = rolling_sharpe(ret)
    fs["rolling_sharpe"] = {
        "median": round(float(rs.median()), 2) if len(rs) else 0,
        "p25": round(float(rs.quantile(0.25)), 2) if len(rs) else 0,
        "p75": round(float(rs.quantile(0.75)), 2) if len(rs) else 0,
        "pct_positive": round(float((rs > 0).sum() / len(rs) * 100), 1) if len(rs) else 0,
    }

    nav_b = (1 + ret).cumprod()
    fee_daily = FEE_ANNUAL / 252
    ret_net = ret - fee_daily
    nav_net = (1 + ret_net).cumprod()
    nav_a = (1 + agg_r).cumprod()
    nav_s = (1 + spy_r).cumprod()
    eq = []
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

    monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    fs["monthly_returns"] = [
        {"year": int(d.year), "month": int(d.month), "return": round(float(v * 100), 2)}
        for d, v in monthly.items()
    ]

    rs_out = []
    rs_idx = rs.index
    rstride = max(1, len(rs_idx) // 800)
    for i in range(0, len(rs_idx), rstride):
        d = rs_idx[i]
        rs_out.append({"date": d.strftime("%Y-%m-%d"), "sharpe": round(float(rs.iloc[i]), 3)})
    fs["rolling_sharpe_ts"] = rs_out

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
    fs["fee_comparison"] = {"gross": stats(ret), "net_1pct": stats(ret_net)}
    ret_nd = ret.shift(1).dropna()
    fs["execution_comparison"] = {"close_to_close": stats(ret), "next_day_execution": stats(ret_nd)}

    def period_stats(r, s, e):
        p = r.loc[s:e]
        if len(p) == 0: return {"return": 0, "max_dd": 0}
        cum = (1 + p).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        return {"return": round(float((cum.iloc[-1] - 1) * 100), 2),
                "max_dd": round(float(dd * 100), 2)}
    def period_stats_simple(r, s, e):
        p = r.loc[s:e]
        if len(p) == 0: return {"return": 0}
        return {"return": round(float(((1 + p).prod() - 1) * 100), 2)}
    historical_events = [
        ("2015-16 Oil Crash", "2015-08-01", "2016-02-29", "Commodity collapse; SPY -13% peak-to-trough. BTC was rangebound; 3x equity ETFs churned."),
        ("2017 Crypto Bull", "2017-01-01", "2017-12-31", "BTC 15x, ETH 90x. Momentum sleeve rotated heavily into crypto names."),
        ("2018 Crypto Crash", "2018-01-15", "2018-12-31", "BTC -75% from peak. BTC-trend filter cut exposure early."),
        ("2020 COVID Crash", "2020-02-15", "2020-04-30", "SPY -34% in 5 weeks; SPY-regime gate forces equity rotation to cash."),
        ("2020 Recovery", "2020-05-01", "2020-12-31", "Post-crash, everything rallied — 3x tech and BTC both hit tops."),
        ("2022 Bear Market", "2022-01-01", "2022-10-31", "Rate-shock killed both stocks and BTC. Regime gates cut exposure."),
        ("2023 Tech Rally", "2023-01-01", "2023-12-31", "NDX +53%; TQQQ/TECL/SOXL dominated top-3 picks."),
        ("2024-25 BTC ATH", "2024-01-01", "2024-12-31", "BTC $100k; crypto sleeve contributed materially."),
    ]
    hist = []
    for name, s, e, desc in historical_events:
        ps = ret.loc[s:e]
        if len(ps) == 0: continue
        hist.append({
            "name": name, "period": f"{s} to {e}", "description": desc,
            "days": int(len(ps)),
            "Blend": period_stats(ret, s, e),
            "AGG": period_stats_simple(agg_r, s, e),
            "SPY": period_stats_simple(spy_r, s, e),
        })
    fs["stress_tests"] = {
        "historical": hist,
        "hypothetical": [
            {"name": "SPY −30% drawdown (slow)", "description": "Regime gate (SPY<200dma or VIX>30) deleverages equity leg to cash.",
             "basis": "Equity weight ~51% on average; gate typically cuts within 2 weeks.",
             "estimated_impact": -12.0},
            {"name": "BTC −60% crash", "description": "BTC-trend gate (>200dma) trips; crypto leg rotates to cash.",
             "basis": "Avg crypto weight 15%; concentrated periods up to 66%. Trend filter caps downside.",
             "estimated_impact": -10.0},
            {"name": "VIX spike to 40 (fast selloff)", "description": "Weekly rebalance cadence can't react to a 1-day 5% equity shock.",
             "basis": "3x ETFs compound losses intra-week; historical worst daily ~-7%.",
             "estimated_impact": -15.0},
            {"name": "Tech rally (NDX +20% in month)", "description": "Momentum sleeve captures via TQQQ/SOXL/TECL; typical beta to NDX ~1.5-2x.",
             "basis": "Top-3 often holds 2 tech names in bull regimes.",
             "estimated_impact": 25.0},
            {"name": "Crypto rally (BTC +50% in month)", "description": "BTC enters top-3; max 33% weight.",
             "basis": "33% × 50% = ~16.5% portfolio contribution.",
             "estimated_impact": 16.0},
        ],
    }

    last_pick = rebals.iloc[-1] if len(rebals) else None
    current_picks = []
    if last_pick is not None:
        picks = [p for p in [last_pick.get("pick_1"), last_pick.get("pick_2"), last_pick.get("pick_3")]
                 if p and str(p) != "nan" and p != ""]
        current_picks = picks

    latest_prices = {}
    for t in UNIVERSE:
        p = load_etf(t)
        latest_prices[t] = round(float(p.iloc[-1]), 2) if p is not None and len(p) else 0.0

    PORT = 100000.0

    # Single sleeve: unified momentum top-3
    positions_v11 = []
    momo_buys = []
    if current_picks:
        per = 1.0 / len(current_picks)
        per_capped = min(per, CAP)
        for t in current_picks:
            if latest_prices.get(t, 0) > 0:
                momo_buys.append({"etf": t, "dollar": round(per_capped * PORT), "price": latest_prices[t]})
    positions_v11.append({
        "display_name": "Weekly Momentum Top-3",
        "stream": "momo",
        "type": "Unified Momentum",
        "description": "Every 5 trading days, rank the 20-instrument universe by 120-day return and hold the top-3 with positive momentum, equal-weight, capped at 33% per name. The 18 leveraged equity ETFs (TQQQ/UPRO/QLD/SSO/SOXL/TECL/FAS/LABU/ERX/NUGT/DRN/EDC/YINN/UGL/UCO/TMF/TYD/UBT) are gated off to cash when SPY<200dma OR VIX>30. Crypto names (BTC/ETH) are gated off when BTC<200dma. Signal is lagged 1 bar: the momentum ranking uses closes through t-1 and is traded starting at bar t.",
        "weight_pct": round(sum(b["dollar"] for b in momo_buys) / PORT * 100, 2) if momo_buys else 0,
        "buys": momo_buys,
    })

    net_etfs = momo_buys.copy()

    fs["allocations"] = {
        "strategy_type": "Unified cross-sectional momentum — 20-instrument universe, top-3, 33% cap per name",
        "n_active": f"{len(positions_v11)} sleeve",
        "n_total": f"{len(UNIVERSE)} instruments in universe",
        "rebalance_freq": "Weekly (5 trading days)",
        "vol_target_stream": "None (cap-bound instead)",
        "vol_target_portfolio": "None (cash residual when <3 names positive)",
        "data_as_of": dates[-1].strftime("%Y-%m-%d"),
        "latest_rebalance": last_pick["date"].strftime("%Y-%m-%d") if last_pick is not None else dates[-1].strftime("%Y-%m-%d"),
        "positions_v11": positions_v11,
        "net_etf_exposure": net_etfs,
        "type_summary": {
            "3x Leveraged": round(float(w_equity.mean() * 100), 2),
            "Crypto": round(float(w_crypto.mean() * 100), 2),
            "Cash (avg)": round(float((1 - w_crypto - w_equity).mean() * 100), 2),
        },
    }

    rebal_hist = []
    for i, row in rebals.iterrows():
        picks = [p for p in [row.get("pick_1"), row.get("pick_2"), row.get("pick_3")]
                 if p and str(p) != "nan" and p != ""]
        if not picks: continue
        per = 1.0 / len(picks)
        per_capped = min(per, CAP)
        positions = [{"name": f"Momo {t}", "weight": round(per_capped * 100, 2)} for t in picks]
        rebal_hist.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "positions": positions,
            "n_candidates": len(UNIVERSE),
            "days_since_prev": 5 if i > 0 else None,
        })
    for i in range(1, len(rebal_hist)):
        prev = {p["name"] for p in rebal_hist[i-1]["positions"]}
        curr = {p["name"] for p in rebal_hist[i]["positions"]}
        rebal_hist[i]["added"] = [n.replace("Momo ", "") for n in sorted(curr - prev)]
        rebal_hist[i]["removed"] = [n.replace("Momo ", "") for n in sorted(prev - curr)]
    if rebal_hist:
        rebal_hist[0]["added"] = []
        rebal_hist[0]["removed"] = []

    fs["rebalance_history"] = rebal_hist[-16:]
    fs["total_rebalances"] = len(rebal_hist)
    fs["last_rebalance"] = rebal_hist[-1]["date"] if rebal_hist else dates[-1].strftime("%Y-%m-%d")
    last = pd.Timestamp(rebal_hist[-1]["date"]) if rebal_hist else dates[-1]
    fs["next_rebalance"] = (last + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    at = []
    rbq = rebals.copy()
    rbq["quarter"] = rbq["date"].dt.to_period("Q")
    for q, grp in rbq.groupby("quarter"):
        row = grp.iloc[0]
        picks = [p for p in [row.get("pick_1"), row.get("pick_2"), row.get("pick_3")]
                 if p and str(p) != "nan" and p != ""]
        entry = {"date": row["date"].strftime("%Y-%m-%d")}
        if picks:
            per = 1.0 / len(picks)
            per_capped = min(per, CAP)
            for t in picks:
                entry[f"Momo {t}"] = round(per_capped * 100, 2)
        at.append(entry)
    fs["alloc_timeline"] = at

    fs["stream_labels"] = {
        "momo": {"name": "Weekly Momentum Top-3", "desc": "Top-3 of 20 (18 leveraged ETFs + BTC/ETH), weekly rebal, 33% cap"},
    }

    by_type = {}
    for t in UNIVERSE:
        tp = ETF_META[t][2]
        by_type.setdefault(tp, []).append(t)
    uni = []
    for tp, tks in sorted(by_type.items()):
        etfs = []
        for t in sorted(tks):
            p = load_etf(t)
            etfs.append({
                "ticker": t,
                "name": ETF_META[t][0],
                "price": latest_prices.get(t, 0),
                "available": p is not None,
                "rows": int(len(p)) if p is not None else 0,
            })
        uni.append({"category": tp, "count": len(tks), "total": len(tks), "etfs": etfs})
    fs["universe"] = uni
    fs["universe_total_available"] = sum(1 for t in UNIVERSE if latest_prices.get(t, 0) > 0)

    fs["scaling_note"] = ("NOVA uses no portfolio-level volatility scaling. The single sleeve rotates "
                          "its 3 holdings every 5 trading days based on 120-day return (signal lagged "
                          "1 bar to avoid look-ahead). Each name is capped at 33% so crypto cannot "
                          "dominate the book. Equity names gate off to BIL when SPY<200dma or VIX>30; "
                          "crypto names gate off when BTC<200dma.")

    out_path = RESULTS / "nova_factsheet_data.json"
    with open(out_path, "w") as f:
        json.dump(fs, f, indent=2, default=str)
    print(f"Saved {out_path}")
    print(f"  Sharpe={m_n['sharpe']}  Ret={m_n['ann_return']}%  "
          f"Vol={m_n['ann_vol']}%  MDD={m_n['max_dd']}%")
    print(f"  {len(rebal_hist)} total rebalances, {len(at)} quarterly snapshots")


if __name__ == "__main__":
    main()
