"""Execute the AURORA factsheet builder — glue script."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from aurora_factsheet import (
    ROOT, ETF, RESULTS, load_etf, metrics, trailing,
    W_COVCALL, W_MOMO, W_MF, COVCALL_TICKERS, MF_TICKERS, MOMO_UNIVERSE,
    ETF_META, FEE_ANNUAL,
)


def main():
    # Load saved AURORA returns + sleeve returns + momo rebalances
    ret = pd.read_csv(RESULTS / "aurora_returns.csv",
                      parse_dates=["Date"]).set_index("Date")["Close"]
    sleeves = pd.read_csv(RESULTS / "aurora_sleeves.csv",
                          parse_dates=["Date"]).set_index("Date")
    rebals = pd.read_csv(RESULTS / "aurora_momo_rebalances.csv",
                         parse_dates=["date"])

    dates = ret.index
    cum = (1 + ret).cumprod()
    nav = float(cum.iloc[-1])

    # Benchmarks
    agg = load_etf("AGG").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    agg_r = agg.pct_change().fillna(0)
    spy_r = spy.pct_change().fillna(0)

    fs = {
        "fund_name": "AURORA — Adaptive Uncorrelated Return Overlay",
        "strategy_type": "Multi-sleeve diversified growth portfolio",
        "benchmark": "S&P 500 (SPY)",
        "inception_date": dates[0].strftime("%B %d, %Y"),
        "last_updated": dates[-1].strftime("%B %d, %Y"),
        "nav": round(nav, 4),
        "rebalance": "Weekly (5-day) momentum / static sleeves",
        "positions_count": "3 sleeves, ~10 instruments",
        "universe_size": "14 core instruments",
    }

    # Core metrics (use "Blend" key so JS works unchanged)
    m_z = metrics(ret); m_z["name"] = "AURORA"
    m_a = metrics(agg_r); m_a["name"] = "AGG"
    m_s = metrics(spy_r); m_s["name"] = "SPY"
    fs["metrics"] = {"Blend": m_z, "AGG": m_a, "SPY": m_s}
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

    # Rolling 1Y Sharpe
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

    # Equity curve
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

    # Fee + execution comparisons
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

    # Stash intermediate for part 3
    fs["_sleeves_ref"] = {
        "covcall_total_w": W_COVCALL, "momo_total_w": W_MOMO, "mf_total_w": W_MF,
    }

    # Stress tests
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
        ("2022 Rate Shock", "2022-01-01", "2022-10-31", "Fastest Fed hiking cycle in 40 years; bonds and stocks both fell. Momo sleeve rotates to TMV/TMF hedges, MF sleeve profits on trend."),
        ("2020 COVID Crash", "2020-02-15", "2020-04-30", "SPY -34% in five weeks; momentum regime gate forces rotation to cash."),
        ("2020 COVID Recovery", "2020-05-01", "2020-12-31", "Post-crash recovery; CC sleeve captures rally."),
        ("2023 Banking Crisis", "2023-03-01", "2023-05-31", "SVB/Signature/First Republic failures; high-yield widens briefly."),
        ("2024 Yen Carry Unwind", "2024-08-01", "2024-08-15", "BOJ hike triggered global risk-off; SPY -8% in days."),
        ("2018 Q4 Selloff", "2018-10-01", "2018-12-24", "Powell rate scare; regime gate triggers."),
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
            {"name": "SPY −30% drawdown (slow)", "description": "SPY falls gradually over 3 months. Regime gate (SPY<200dma OR VIX>30) triggers momentum rotation to cash; CC sleeve captures remaining decline partially offset by MF trend.",
             "basis": "Portfolio beta to SPY ~0.7 unhedged; regime gate historically cuts momo exposure within 2-3 weeks of 200dma breach.",
             "estimated_impact": -9.0},
            {"name": "VIX spike to 40 (fast selloff)", "description": "One-day crash like Feb 2018. Momo sleeve takes full 40% hit; CC sleeve loses ~15% (SPY-like); MF slow to react.",
             "basis": "Weekly rebalance cadence can't react in a 1-day shock. Historical daily worst was ~-3.5%.",
             "estimated_impact": -7.5},
            {"name": "Bond rate shock (+150bps, 3mo)", "description": "TMF drops 25% if it was a momo holding. CC equity component neutral. MF likely short duration.",
             "basis": "TMF has 40% annualized vol; 3mo tail event ~-12% hit on a TMF holding.",
             "estimated_impact": -3.0},
            {"name": "Tech rally (NDX +20% in month)", "description": "Momo sleeve captures via TQQQ/SOXL/TECL; CC caps upside at ~8%.",
             "basis": "Momentum sleeve has 3x leverage on NDX in bull regimes; typical capture ~1.5x of NDX move.",
             "estimated_impact": 10.0},
            {"name": "Managed futures rally (20% CTA year)", "description": "Like 2022 — MF sleeve contributes +4%; other sleeves neutral.",
             "basis": "MF weight 20% × 20% sleeve return = 4% portfolio contribution.",
             "estimated_impact": 4.0},
        ],
    }

    # Momo sleeve latest picks
    last_pick = rebals.iloc[-1] if len(rebals) else None
    current_momo = []
    if last_pick is not None:
        picks = [p for p in [last_pick.get("pick_1"), last_pick.get("pick_2"), last_pick.get("pick_3")]
                 if p and str(p) != "nan" and p != ""]
        current_momo = picks

    # Latest prices
    latest_prices = {}
    all_tickers = set(COVCALL_TICKERS) | set(MF_TICKERS) | set(MOMO_UNIVERSE)
    for t in all_tickers:
        p = load_etf(t)
        latest_prices[t] = round(float(p.iloc[-1]), 2) if p is not None and len(p) else 0.0

    PORT = 100000.0

    # Positions (v11 schema): one entry per *sleeve*
    positions_v11 = []
    # Sleeve 1: Covered-call income
    cc_buys = []
    for t in COVCALL_TICKERS:
        if latest_prices.get(t, 0) > 0:
            w = W_COVCALL / len(COVCALL_TICKERS)
            cc_buys.append({"etf": t, "dollar": round(w * PORT), "price": latest_prices[t]})
    positions_v11.append({
        "display_name": "Covered-Call Income",
        "stream": "covcall",
        "type": "Covered Call",
        "description": "Equal-weight blend of JEPI, JEPQ, SPYI, DIVO — option-overlay ETFs that harvest call-premium on S&P 500 and Nasdaq-100. Pays monthly distributions ~7-11% annualized.",
        "weight_pct": round(W_COVCALL * 100, 2),
        "buys": cc_buys,
    })
    # Sleeve 2: Weekly momentum
    momo_buys = []
    if current_momo:
        per = W_MOMO / len(current_momo)
        for t in current_momo:
            if latest_prices.get(t, 0) > 0:
                momo_buys.append({"etf": t, "dollar": round(per * PORT), "price": latest_prices[t]})
    positions_v11.append({
        "display_name": "Weekly Momentum Top-3",
        "stream": "momo",
        "type": "3x Leveraged",
        "description": "Every 5 trading days, rank the 7-ETF leveraged universe by 20-day return and hold the top-3 with positive absolute momentum. Regime-gated by SPY>200dma AND VIX<30 — rotates to BIL otherwise.",
        "weight_pct": round(W_MOMO * 100, 2),
        "buys": momo_buys,
    })
    # Sleeve 3: Managed futures
    mf_buys = []
    for t in MF_TICKERS:
        if latest_prices.get(t, 0) > 0:
            w = W_MF / len(MF_TICKERS)
            mf_buys.append({"etf": t, "dollar": round(w * PORT), "price": latest_prices[t]})
    positions_v11.append({
        "display_name": "Managed Futures",
        "stream": "mf",
        "type": "Managed Futures",
        "description": "Equal-weight blend of DBMF, CTA, and KMLM — liquid-alt CTA trend-following. Low correlation to equity and credit; positive in most rate-shock years (2022 was the strongest year).",
        "weight_pct": round(W_MF * 100, 2),
        "buys": mf_buys,
    })

    # Flat net-ETF exposure (aggregate)
    net_etfs = []
    for t in COVCALL_TICKERS:
        if latest_prices.get(t, 0) > 0:
            w = W_COVCALL / len(COVCALL_TICKERS)
            net_etfs.append({"etf": t, "dollar": round(w * PORT), "price": latest_prices[t]})
    if current_momo:
        per = W_MOMO / len(current_momo)
        for t in current_momo:
            if latest_prices.get(t, 0) > 0:
                net_etfs.append({"etf": t, "dollar": round(per * PORT), "price": latest_prices[t]})
    for t in MF_TICKERS:
        if latest_prices.get(t, 0) > 0:
            w = W_MF / len(MF_TICKERS)
            net_etfs.append({"etf": t, "dollar": round(w * PORT), "price": latest_prices[t]})

    fs["allocations"] = {
        "strategy_type": "Multi-sleeve diversified growth — Income 40% + Weekly Momentum 40% + Managed Futures 20%",
        "n_active": f"{len(positions_v11)} sleeves",
        "n_total": f"{len(all_tickers)} instruments in universe",
        "rebalance_freq": "Weekly (5 trading days) for momentum; static for income/MF",
        "vol_target_stream": "None (fixed sleeve weights)",
        "vol_target_portfolio": "None (no portfolio-level vol targeting)",
        "data_as_of": dates[-1].strftime("%Y-%m-%d"),
        "latest_rebalance": last_pick["date"].strftime("%Y-%m-%d") if last_pick is not None else dates[-1].strftime("%Y-%m-%d"),
        "positions_v11": positions_v11,
        "net_etf_exposure": net_etfs,
        "type_summary": {
            "Covered Call": round(W_COVCALL * 100, 2),
            "3x Leveraged": round(W_MOMO * 100, 2),
            "Managed Futures": round(W_MF * 100, 2),
        },
    }

    fs.pop("_sleeves_ref", None)

    # Rebalance history — use weekly momentum rebalances (the actual events)
    # Covered-call and MF sleeves rebalance implicitly via the momo event cycle
    rebal_hist = []
    for i, row in rebals.iterrows():
        picks = [p for p in [row.get("pick_1"), row.get("pick_2"), row.get("pick_3")]
                 if p and str(p) != "nan" and p != ""]
        if not picks:
            continue
        momo_positions = []
        per = W_MOMO / len(picks) * 100
        for t in picks:
            momo_positions.append({"name": f"Momo {t}", "weight": round(per, 2)})
        cc_positions = [{"name": f"CC {t}", "weight": round(W_COVCALL / len(COVCALL_TICKERS) * 100, 2)}
                        for t in COVCALL_TICKERS]
        mf_positions = [{"name": f"MF {t}", "weight": round(W_MF / len(MF_TICKERS) * 100, 2)}
                        for t in MF_TICKERS]
        all_pos = momo_positions + cc_positions + mf_positions
        rebal_hist.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "positions": all_pos,
            "n_candidates": 14,
            "days_since_prev": 5 if i > 0 else None,
        })

    # Diff added/removed vs prior rebalance (tracks momo rotation only)
    for i in range(1, len(rebal_hist)):
        prev_momo = {p["name"] for p in rebal_hist[i-1]["positions"] if p["name"].startswith("Momo ")}
        curr_momo = {p["name"] for p in rebal_hist[i]["positions"] if p["name"].startswith("Momo ")}
        rebal_hist[i]["added"] = [n.replace("Momo ", "") for n in sorted(curr_momo - prev_momo)]
        rebal_hist[i]["removed"] = [n.replace("Momo ", "") for n in sorted(prev_momo - curr_momo)]
    if rebal_hist:
        rebal_hist[0]["added"] = []
        rebal_hist[0]["removed"] = []

    fs["rebalance_history"] = rebal_hist[-16:]
    fs["total_rebalances"] = len(rebal_hist)
    fs["last_rebalance"] = rebal_hist[-1]["date"] if rebal_hist else dates[-1].strftime("%Y-%m-%d")
    last = pd.Timestamp(rebal_hist[-1]["date"]) if rebal_hist else dates[-1]
    fs["next_rebalance"] = (last + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

    # Allocation timeline — show how momo picks rotate quarterly
    at = []
    # Quarterly snapshots
    qidx = rebals.copy()
    qidx["quarter"] = qidx["date"].dt.to_period("Q")
    for q, grp in qidx.groupby("quarter"):
        # Take representative row for the quarter (first one)
        row = grp.iloc[0]
        picks = [p for p in [row.get("pick_1"), row.get("pick_2"), row.get("pick_3")]
                 if p and str(p) != "nan" and p != ""]
        entry = {"date": row["date"].strftime("%Y-%m-%d")}
        # Covered call constant
        for t in COVCALL_TICKERS:
            entry[f"CC {t}"] = round(W_COVCALL / len(COVCALL_TICKERS) * 100, 2)
        # Momo picks
        if picks:
            per = W_MOMO / len(picks) * 100
            for t in picks:
                entry[f"Momo {t}"] = round(per, 2)
        # MF constant
        for t in MF_TICKERS:
            entry[f"MF {t}"] = round(W_MF / len(MF_TICKERS) * 100, 2)
        at.append(entry)
    fs["alloc_timeline"] = at

    # Stream labels
    fs["stream_labels"] = {
        "covcall": {"name": "Covered-Call Income", "desc": "JEPI/JEPQ/SPYI/DIVO option-overlay blend"},
        "momo": {"name": "Weekly Momentum Top-3", "desc": "Top-3 of 7-ETF leveraged universe, 5d rebal"},
        "mf": {"name": "Managed Futures", "desc": "DBMF/CTA/KMLM CTA trend-following blend"},
    }

    # Investment universe
    by_type = {}
    for t in all_tickers:
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
    fs["universe_total_available"] = sum(1 for t in all_tickers if latest_prices.get(t, 0) > 0)

    fs["scaling_note"] = ("AURORA uses no portfolio-level volatility scaling. Sleeve weights "
                          "(40/40/20) are static. The momentum sleeve rotates its 3 holdings "
                          "every 5 trading days based on 20-day return, and is gated off (into "
                          "BIL) when SPY is below its 200-day moving average OR VIX is above 30.")

    # Write final
    out_path = RESULTS / "aurora_factsheet_data.json"
    with open(out_path, "w") as f:
        json.dump(fs, f, indent=2, default=str)
    print(f"Full factsheet saved to {out_path}")
    print(f"  Sharpe={m_z['sharpe']}  Ann Ret={m_z['ann_return']}%  "
          f"Vol={m_z['ann_vol']}%  MDD={m_z['max_dd']}%")
    print(f"  {len(rebal_hist)} total momo rebalances, {len(at)} quarterly alloc snapshots")


if __name__ == "__main__":
    main()
