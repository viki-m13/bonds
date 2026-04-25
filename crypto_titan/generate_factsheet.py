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
        "tagline": "21-Sleeve Long-Only Crypto Ensemble with Conviction-Scaled Leverage",
        "description": (
            "21 weakly-correlated sleeves spanning trend, breakout, regime, "
            "calendar, volume, range, autocorrelation, and convex right-tail "
            "models. Conviction-scaled leverage (1× → 3.2×) executed via "
            "Hyperliquid perpetuals or OKX leveraged tokens. Weekly rebalance "
            "(Wednesdays). Survivorship-bias-robust universe of 50 coins "
            "(35 survivors + 15 dead/collapsed)."
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
