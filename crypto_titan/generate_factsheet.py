"""Generate factsheet JSON for the TITAN webapp page.

Reads strategy returns + meta JSON and produces a single titan_data.json
with everything titan.html needs (metrics, equity curve, sleeve metrics,
yearly stats, etc.).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
from util import OUT, metrics, regime_slice, DPY


def main():
    rets = pd.read_csv(OUT / "crypto_titan_returns.csv",
                        index_col=0, parse_dates=True).iloc[:, 0]
    rets = rets.fillna(0.0)
    weights = pd.read_csv(OUT / "crypto_titan_weights.csv",
                           index_col=0, parse_dates=True)
    meta = json.loads((OUT / "crypto_titan_meta.json").read_text())

    # Equity curve at base $10,000
    equity = (1 + rets).cumprod() * 10000
    eq_dates = [d.strftime("%Y-%m-%d") for d in equity.index]
    eq_values = [round(v, 2) for v in equity.values]

    # Drawdown curve
    dd = equity / equity.cummax() - 1
    dd_dates = eq_dates
    dd_values = [round(v, 4) for v in dd.values]

    # Rolling 12m Sharpe
    rolling_sharpe = (rets.rolling(252).mean() / rets.rolling(252).std()
                      * np.sqrt(DPY))
    rs_clean = rolling_sharpe.dropna()
    rs_dates = [d.strftime("%Y-%m-%d") for d in rs_clean.index]
    rs_values = [round(v, 3) for v in rs_clean.values]

    # ===== ALLOCATION DATA =====
    # Filter weights to non-trivial positions only
    pos_mask = weights.abs() > 1e-4
    n_pos = pos_mask.sum(axis=1)
    gross = weights.abs().sum(axis=1)

    # Current target (latest non-zero or zero if cash)
    last_w = weights.iloc[-1]
    current_positions = [
        {"coin": c, "weight": round(float(last_w[c]), 4)}
        for c in last_w.index if abs(last_w[c]) > 1e-4
    ]
    current_positions.sort(key=lambda x: -abs(x["weight"]))

    # Find LAST DATE strategy was actually in the market (for "most recent
    # allocation" display when current is cash)
    in_market_dates = weights.index[gross > 1e-4]
    last_active_date = in_market_dates[-1] if len(in_market_dates) else None
    last_active_positions = []
    if last_active_date is not None:
        row = weights.loc[last_active_date]
        last_active_positions = [
            {"coin": c, "weight": round(float(row[c]), 4)}
            for c in row.index if abs(row[c]) > 1e-4
        ]
        last_active_positions.sort(key=lambda x: -abs(x["weight"]))

    # Position-frequency table — % of all days each coin was held
    held_pct = pos_mask.mean(axis=0).sort_values(ascending=False)
    held_pct = held_pct[held_pct > 0]
    coin_freq = [
        {"coin": c, "days_held": int(pos_mask[c].sum()),
         "pct_of_days": round(float(held_pct[c]) * 100, 2),
         "avg_weight_when_held": round(float(weights[c][pos_mask[c]].mean()), 4)
         if pos_mask[c].any() else 0.0}
        for c in held_pct.index
    ]

    # Position-count distribution
    pos_count_dist = {
        "cash":    int((n_pos == 0).sum()),
        "one":     int((n_pos == 1).sum()),
        "two":     int((n_pos == 2).sum()),
        "three+":  int((n_pos >= 3).sum()),
    }

    # MONTHLY allocation history — sample weights at each month-end
    monthly_dates = weights.index[weights.index.day >= 25]
    # Group by year-month, take last day of each month
    monthly_w = weights.resample("ME").last().fillna(0.0)
    # Only the coins that were ever held
    held_coins = [c for c in held_pct.index]
    monthly_history = {
        "dates": [d.strftime("%Y-%m-%d") for d in monthly_w.index],
        "coins": held_coins,
        "values": {
            c: [round(float(v), 4) for v in monthly_w[c].values]
            for c in held_coins
        },
        "gross": [round(float(v), 4) for v in monthly_w.abs().sum(axis=1).values],
    }

    # WEEKLY exposure history (gross + net) for chart
    weekly_w = weights.resample("W-WED").last().fillna(0.0)
    exposure_history = {
        "dates": [d.strftime("%Y-%m-%d") for d in weekly_w.index],
        "gross": [round(float(v), 4) for v in weekly_w.abs().sum(axis=1).values],
        "btc":   [round(float(weekly_w[c].iloc[i]), 4)
                  for i, c in enumerate(["BTC"] * len(weekly_w))]
                 if "BTC" in weekly_w.columns else [0.0] * len(weekly_w),
        "eth":   [round(float(v), 4) for v in
                  (weekly_w["ETH"].values if "ETH" in weekly_w.columns
                   else [0.0] * len(weekly_w))],
    }
    # Fix BTC list (above was awkward)
    exposure_history["btc"] = ([round(float(v), 4) for v in
                                 (weekly_w["BTC"].values if "BTC" in weekly_w.columns
                                  else [0.0] * len(weekly_w))])

    # Recent positions table (last 12 weekly snapshots)
    recent_snapshots = []
    for date in weekly_w.index[-12:]:
        row = weekly_w.loc[date]
        positions = [
            {"coin": c, "weight": round(float(row[c]), 4)}
            for c in row.index if abs(row[c]) > 1e-4
        ]
        positions.sort(key=lambda x: -abs(x["weight"]))
        recent_snapshots.append({
            "date": date.strftime("%Y-%m-%d"),
            "gross": round(float(row.abs().sum()), 4),
            "n_positions": len(positions),
            "positions": positions,
        })

    # Yearly metrics
    yearly = {}
    for y in range(2014, 2027):
        ys = rets[rets.index.year == y]
        if len(ys) < 30:
            continue
        m = metrics(ys)
        yearly[str(y)] = {
            "sharpe": m["sharpe"],
            "cagr": m["cagr"],
            "vol": m["vol"],
            "mdd": m["mdd"],
            "calmar": m["calmar"],
            "n": m["n"],
        }

    data = {
        "name": "CRYPTO-TITAN",
        "tagline": "BTC + ETH Timing Strategy with 21-Model Conviction Ensemble",
        "description": (
            "BTC + ETH + cash timing strategy. Tradable universe: BTC and ETH "
            "only (max 2 simultaneous positions). Signal universe: 50 coins "
            "(35 survivors + 15 explicitly-dead) used for breadth signals, "
            "master gate, and survivorship-bias robustness. The 21 models "
            "decide weekly: hold BTC, hold ETH, hold both, or stay in cash. "
            "Conviction-scaled leverage (1× → 3.2×) executed via Hyperliquid "
            "perpetuals or OKX leveraged tokens. Alt-trading sleeves were "
            "tested empirically and dropped — daily-bar alt momentum picks "
            "parabolic alts right before they reverse."
        ),
        "as_of": str(rets.index[-1].date()),
        "metrics": meta.get("metrics", {}),
        "benchmarks": meta.get("benchmarks", {}),
        "sleeve_metrics": meta.get("sleeve_metrics", {}),
        "survivorship_bias": meta.get("survivorship_bias", {}),
        "yearly": yearly,
        "equity_curve": {"dates": eq_dates, "values": eq_values},
        "drawdown_curve": {"dates": dd_dates, "values": dd_values},
        "rolling_1y_sharpe": {"dates": rs_dates, "values": rs_values},
        # ALLOCATION DATA (new)
        "current_positions": current_positions,
        "last_active_date": str(last_active_date.date()) if last_active_date is not None else None,
        "last_active_positions": last_active_positions,
        "coin_frequency": coin_freq,
        "position_count_dist": pos_count_dist,
        "monthly_history": monthly_history,
        "exposure_history": exposure_history,
        "recent_snapshots": recent_snapshots,
        "config": {
            "target_vol": meta.get("target_vol", 0.18),
            "dd_floor": meta.get("dd_floor", -0.12),
            "tc_bps": meta.get("tc_bps", 20.0),
            "smooth_span": meta.get("smooth_span", 7),
            "n_sleeves": len(meta.get("sleeves", [])),
            "universe_size": len(meta.get("universe_full", [])),
            "n_survivors": len(meta.get("survivors", [])),
            "n_dead": len(meta.get("dead", [])),
        },
        "iterations": [
            {"v": "v1",   "ssr_full": 1.19, "ssr_oos": 0.54, "cagr": 12.0, "note": "4 sleeves, 20 coins"},
            {"v": "v2",   "ssr_full": 1.81, "ssr_oos": 1.21, "cagr": 25.9, "note": "15 sleeves, ensemble"},
            {"v": "v3",   "ssr_full": 1.63, "ssr_oos": 1.49, "cagr": 22.2, "note": "+ proprietary edges"},
            {"v": "v4",   "ssr_full": 1.68, "ssr_oos": 1.42, "cagr": 21.0, "note": "50-coin universe + daily vol scaling"},
            {"v": "v5",   "ssr_full": 1.63, "ssr_oos": 1.40, "cagr": 22.8, "note": "111-coin survivorship test"},
            {"v": "v6",   "ssr_full": 1.55, "ssr_oos": 1.01, "cagr": 19.5, "note": "shorts tested — failed empirically"},
            {"v": "v7",   "ssr_full": 1.46, "ssr_oos": 1.15, "cagr": 55.6, "note": "leverage 3× → CAGR target hit"},
            {"v": "v8",   "ssr_full": 1.47, "ssr_oos": 1.13, "cagr": 51.4, "note": "+ vol-of-vol regime gate"},
        ],
    }

    out_fp = Path(__file__).parent.parent / "docs" / "titan_data.json"
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(data, indent=2, default=str))
    print(f"Wrote {out_fp} ({out_fp.stat().st_size:,} bytes)")
    print(f"  Metrics keys: {list(data['metrics'].keys())[:8]}")
    print(f"  Equity points: {len(eq_dates)}")
    print(f"  Sleeves: {len(data['sleeve_metrics'])}")


if __name__ == "__main__":
    main()
