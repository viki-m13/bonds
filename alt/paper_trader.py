"""PHOENIX paper trader — track a virtual portfolio that executes the canonical
live signal at each next-day market open, using the same yfinance Open prices
the backtest uses. By construction, paper NAV = backtest open-to-open NAV
within whatever frictions are configured.

Reads:
    data/results/live_positions.csv   — running log of target weights per signal date
    data/etfs/{ticker}.csv            — opens for each held ticker
    data/results/phoenix_production_returns.csv  — backtest reference for tracking error

Writes:
    data/results/paper_nav.csv        — one row per fill event: NAV, realized
                                        return, turnover, TC drag
    data/results/paper_fills.csv      — one row per traded ticker per event:
                                        weight delta, fill price, shares, notional
    data/results/paper_summary.json   — webapp-facing snapshot: current NAV,
                                        total return, fills, tracking error

Configuration (env vars, with sensible defaults):
    PAPER_INITIAL_NAV  initial cash, default 10000
    PAPER_TC_BPS       per-side transaction cost in bps applied to turnover,
                       default 5

Usage:
    python3 alt/paper_trader.py
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
ETF = ROOT / "data/etfs"
R = ROOT / "data/results"

INITIAL_NAV = float(os.environ.get("PAPER_INITIAL_NAV", "10000"))
TC_BPS = float(os.environ.get("PAPER_TC_BPS", "5"))
TRADE_THRESHOLD = 0.005  # match live_signal.py

CASH = "BIL"


def load_opens(tickers: list[str]) -> pd.DataFrame:
    """Load Open prices for each ticker, aligned on a union calendar."""
    series = {}
    for t in tickers:
        p = ETF / f"{t}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        series[t] = pd.to_numeric(df["Open"], errors="coerce")
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index()


def main():
    pos_file = R / "live_positions.csv"
    if not pos_file.exists():
        print("[paper_trader] live_positions.csv missing — nothing to do.")
        return

    pos = pd.read_csv(pos_file, parse_dates=["Date"]).set_index("Date").sort_index()
    pos = pos.drop(columns=["ret"], errors="ignore").fillna(0.0)
    if pos.empty:
        print("[paper_trader] live_positions.csv is empty.")
        return

    tickers = pos.columns.tolist()
    opens = load_opens(tickers)
    if opens.empty:
        print("[paper_trader] no Open price data available.")
        return

    # Build trade events. Each row in pos is signal_date d; the trade
    # executes at the next available market open after d.
    events: list[dict] = []
    for sig_d, w in pos.iterrows():
        future = opens.index[opens.index > sig_d]
        if len(future) == 0:
            continue  # signal hasn't been fillable yet
        trade_d = future[0]
        events.append({"signal_date": sig_d, "trade_date": trade_d, "weights": w})

    if not events:
        print("[paper_trader] no fillable trade events yet.")
        return

    nav = INITIAL_NAV
    prev_w = pd.Series(0.0, index=tickers)  # bootstrap: 100% cash, position = $0
    # Initial cash: model as a BIL position so the bootstrap fill is BIL → first signal
    if CASH in prev_w.index:
        prev_w[CASH] = 1.0

    nav_rows = []
    fill_rows = []

    for i, ev in enumerate(events):
        sig_d = ev["signal_date"]
        trade_d = ev["trade_date"]
        w_new = ev["weights"].astype(float)

        # ---- 1. Apply realized return for the prior holding period -------
        # Held prev_w from prev_trade_date to trade_d (open-to-open).
        if i > 0:
            prev_trade_d = events[i - 1]["trade_date"]
            opens_prev = opens.loc[prev_trade_d]
            opens_curr = opens.loc[trade_d]
            held = prev_w[prev_w.abs() > 1e-9]
            common = [t for t in held.index
                      if t in opens.columns
                      and not np.isnan(opens_prev.get(t, np.nan))
                      and not np.isnan(opens_curr.get(t, np.nan))]
            if common:
                rets = opens_curr[common] / opens_prev[common] - 1.0
                realized = float((held[common] * rets).sum())
            else:
                realized = 0.0
            nav *= (1.0 + realized)
        else:
            realized = 0.0  # first event: nothing held before trade

        # ---- 2. Apply transaction costs for this rebalance ---------------
        delta = (w_new - prev_w).abs()
        turnover = float(delta.sum())
        tc_drag = turnover * (TC_BPS / 1e4)
        nav_after_tc = nav * (1.0 - tc_drag)

        # ---- 3. Record fills ---------------------------------------------
        for t in tickers:
            d = float(w_new.get(t, 0.0)) - float(prev_w.get(t, 0.0))
            if abs(d) < TRADE_THRESHOLD:
                continue
            fill_price = float(opens.loc[trade_d, t]) if t in opens.columns else float("nan")
            if not np.isfinite(fill_price) or fill_price <= 0:
                continue
            notional = d * nav_after_tc
            shares = notional / fill_price
            fill_rows.append({
                "Date": trade_d.strftime("%Y-%m-%d"),
                "signal_date": sig_d.strftime("%Y-%m-%d"),
                "Ticker": t,
                "Side": "BUY" if d > 0 else "SELL",
                "Weight_Delta": round(d, 4),
                "Fill_Price": round(fill_price, 4),
                "Shares": round(shares, 4),
                "Notional_USD": round(notional, 2),
            })

        nav_rows.append({
            "Date": trade_d.strftime("%Y-%m-%d"),
            "signal_date": sig_d.strftime("%Y-%m-%d"),
            "NAV": round(nav_after_tc, 2),
            "realized_ret": round(realized, 6),
            "turnover": round(turnover, 4),
            "tc_drag": round(tc_drag, 6),
        })

        nav = nav_after_tc
        prev_w = w_new.copy()

    nav_df = pd.DataFrame(nav_rows)
    fill_df = pd.DataFrame(fill_rows)
    nav_df.to_csv(R / "paper_nav.csv", index=False)
    fill_df.to_csv(R / "paper_fills.csv", index=False)

    # ---- Tracking error vs backtest -------------------------------------
    # Compare each day's realized return to the backtest's net_ret for the
    # same trade_date.
    bt_file = R / "phoenix_production_returns.csv"
    tracking = {}
    if bt_file.exists() and len(nav_df) > 0:
        bt = pd.read_csv(bt_file, parse_dates=["Date"]).set_index("Date")["net_ret"]
        nav_df_dt = nav_df.copy()
        nav_df_dt["Date"] = pd.to_datetime(nav_df_dt["Date"])
        merged = nav_df_dt.set_index("Date").join(bt.rename("backtest_ret"), how="left")
        merged["diff"] = merged["realized_ret"] - merged["backtest_ret"]
        diffs = merged["diff"].dropna()
        if len(diffs) > 0:
            tracking = {
                "n_days": int(len(diffs)),
                "mean_diff_bps": round(float(diffs.mean()) * 1e4, 2),
                "stdev_diff_bps": round(float(diffs.std()) * 1e4, 2) if len(diffs) > 1 else 0.0,
                "max_abs_diff_bps": round(float(diffs.abs().max()) * 1e4, 2),
                "ann_tracking_error_bps": round(float(diffs.std()) * np.sqrt(252) * 1e4, 1)
                                          if len(diffs) > 1 else 0.0,
                "note": "diff = paper_realized_ret - backtest_net_ret per trade-open day. "
                        "Source of drift: TC bps applied here vs in backtest, plus any "
                        "timing-convention differences across sleeves.",
            }

    # ---- Webapp summary -------------------------------------------------
    if len(nav_df) > 0:
        latest = nav_df.iloc[-1]
        first = nav_df.iloc[0]
        n_days_held = (pd.Timestamp(latest["Date"]) - pd.Timestamp(first["Date"])).days
        years = max(n_days_held / 365.25, 1 / 252)  # avoid div0
        total_ret = (latest["NAV"] / INITIAL_NAV) - 1
        # Use realized_ret series for vol/sharpe/MDD
        r = pd.to_numeric(nav_df["realized_ret"], errors="coerce").dropna()
        if len(r) > 1 and r.std() > 0:
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
        else:
            ann_ret = ann_vol = sharpe = 0.0
        eq = (1 + r).cumprod()
        mdd = float((eq / eq.cummax() - 1).min()) if len(eq) > 0 else 0.0

        # Most recent fills (top 30 by date desc)
        recent_fills = (fill_df.sort_values("Date", ascending=False)
                               .head(30)
                               .to_dict(orient="records")) if len(fill_df) > 0 else []

        # Full NAV time series (all fields) for charting + table
        nav_series = []
        for _, row in nav_df.iterrows():
            nav_series.append({
                "d": row["Date"],
                "v": float(row["NAV"]),
                "ret": float(row["realized_ret"]),
                "turnover": float(row["turnover"]),
                "tc": float(row["tc_drag"]),
            })

        # Benchmark: SPY buy-and-hold with same initial_nav and same start date.
        # Uses Open prices (open-to-open, identical convention to paper).
        spy_series = []
        spy_csv = ETF / "SPY.csv"
        if spy_csv.exists() and len(nav_df) > 0:
            spy_df = pd.read_csv(spy_csv, parse_dates=["Date"]).set_index("Date").sort_index()
            spy_df = spy_df[~spy_df.index.duplicated(keep="first")]
            spy_open = pd.to_numeric(spy_df["Open"], errors="coerce")
            start_date = pd.Timestamp(first["Date"])
            spy_at_start = spy_open.reindex([start_date], method="ffill").iloc[0]
            if np.isfinite(spy_at_start) and spy_at_start > 0:
                for _, row in nav_df.iterrows():
                    d = pd.Timestamp(row["Date"])
                    spy_at_d = spy_open.reindex([d], method="ffill").iloc[0]
                    if np.isfinite(spy_at_d) and spy_at_d > 0:
                        spy_nav = INITIAL_NAV * (spy_at_d / spy_at_start)
                        spy_series.append({"d": row["Date"], "v": round(float(spy_nav), 2)})

        # SPY total return over the same period (for KPI comparison)
        spy_total_return = None
        if spy_series:
            spy_total_return = round((spy_series[-1]["v"] / INITIAL_NAV - 1) * 100, 2)

        summary = {
            "config": {
                "initial_nav": INITIAL_NAV,
                "tc_bps_per_side": TC_BPS,
                "trade_threshold_pct": TRADE_THRESHOLD * 100,
            },
            "as_of": str(latest["Date"]),
            "start_date": str(first["Date"]),
            "current_nav": float(latest["NAV"]),
            "total_return_pct": round(total_ret * 100, 2),
            "spy_total_return_pct": spy_total_return,
            "n_trade_days": int(len(nav_df)),
            "ann_return_pct": round(ann_ret * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "mdd_pct": round(mdd * 100, 2),
            "tracking_error_vs_backtest": tracking,
            "recent_fills": recent_fills,
            "nav_series": nav_series,
            "spy_series": spy_series,
        }
    else:
        summary = {"config": {"initial_nav": INITIAL_NAV, "tc_bps_per_side": TC_BPS},
                   "note": "No fillable events yet."}

    (R / "paper_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Pretty print ----------------------------------------------------
    print("=" * 74)
    print(f"PAPER TRADER — initial ${INITIAL_NAV:,.0f}, TC {TC_BPS} bps/side")
    print("=" * 74)
    if len(nav_df) > 0:
        print(f"Period:        {first['Date']} → {latest['Date']}")
        print(f"Current NAV:   ${latest['NAV']:,.2f}  ({total_ret*100:+.2f}% total)")
        print(f"Trade events:  {len(nav_df)}")
        print(f"Fills logged:  {len(fill_df)}")
        if tracking:
            print(f"Tracking error vs backtest:")
            print(f"  mean diff:    {tracking['mean_diff_bps']:+.2f} bps/day")
            print(f"  stdev diff:   {tracking['stdev_diff_bps']:.2f} bps/day")
            print(f"  ann tracking: {tracking['ann_tracking_error_bps']:.1f} bps")
        print()
        print("Last 5 NAV updates:")
        for _, row in nav_df.tail(5).iterrows():
            print(f"  {row['Date']}: NAV ${row['NAV']:,.2f}  "
                  f"return {row['realized_ret']*100:+.3f}%  "
                  f"turnover {row['turnover']*100:.1f}%")
    else:
        print("No trade events.")
    print(f"\nWrote: {R/'paper_nav.csv'}, {R/'paper_fills.csv'}, {R/'paper_summary.json'}")


if __name__ == "__main__":
    main()
