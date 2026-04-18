"""Patch aurora_factsheet_data.json and blend_factsheet_data.json with
proxy-extended equity curves + period-split metrics.

For each strategy we add:
  proxy_equity_curve: [{date,value,source}] — normalized so NAV=10000 at start
  proxy_metrics:      {full: {...}, live: {...}, proxy_only: {...}}
  proxy_inception:    date string of first 'live' row
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
RESULTS = ROOT / "data/results"


def metrics(r):
    if len(r) < 2 or r.std() == 0:
        return {"sharpe": 0, "ann_return": 0, "ann_vol": 0, "max_dd": 0,
                "sortino": 0, "n_years": round(len(r)/252, 1)}
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av
    cum = (1 + r).cumprod()
    mdd = (cum / cum.cummax() - 1).min()
    neg = r[r < 0]
    sor = ar / (neg.std() * np.sqrt(252)) if len(neg) and neg.std() > 0 else 999
    return {
        "sharpe": round(float(sr), 3),
        "ann_return": round(float(ar * 100), 2),
        "ann_vol": round(float(av * 100), 2),
        "max_dd": round(float(mdd * 100), 2),
        "sortino": round(float(sor), 3),
        "n_years": round(float(len(r) / 252), 1),
    }


def build_equity_curve(df):
    """Downsample to weekly, normalize to NAV=10000. Keep source label."""
    r = df["Close"]
    src = df["source"]
    cum = (1 + r).cumprod() * 10000
    # Sample weekly (Friday) to keep payload light
    idx = cum.resample("W-FRI").last().dropna().index
    cum_w = cum.reindex(idx).dropna()
    src_w = src.reindex(idx).ffill()
    return [{"date": d.strftime("%Y-%m-%d"),
             "value": round(float(cum_w.loc[d]), 2),
             "source": str(src_w.loc[d])}
            for d in cum_w.index]


def patch(returns_csv, factsheet_json, name):
    df = pd.read_csv(RESULTS / returns_csv, parse_dates=["Date"]).set_index("Date")
    r = df["Close"]
    src = df["source"]
    live_mask = (src == "live")
    live_start = df.index[live_mask].min() if live_mask.any() else None

    out = {
        "proxy_equity_curve": build_equity_curve(df),
        "proxy_inception": live_start.strftime("%Y-%m-%d") if live_start is not None else None,
        "proxy_metrics": {
            "full":       {**metrics(r), "period": f"{df.index[0].date()} — {df.index[-1].date()}"},
            "live":       {**metrics(r[live_mask]),
                           "period": f"{live_start.date()} — {df.index[-1].date()}" if live_start is not None else ""},
            "proxy_only": {**metrics(r[~live_mask]),
                           "period": f"{df.index[0].date()} — {(live_start - pd.Timedelta(days=1)).date()}" if live_start is not None else ""},
        },
    }

    fs_path = RESULTS / factsheet_json
    fs = json.loads(fs_path.read_text())
    fs.update(out)
    fs_path.write_text(json.dumps(fs, separators=(",", ":")))
    print(f"{name}: patched {fs_path.name}")
    print(f"  full:       SR {out['proxy_metrics']['full']['sharpe']}  "
          f"Ret {out['proxy_metrics']['full']['ann_return']}%  "
          f"MDD {out['proxy_metrics']['full']['max_dd']}%  "
          f"({out['proxy_metrics']['full']['n_years']}y)")
    print(f"  live:       SR {out['proxy_metrics']['live']['sharpe']}  "
          f"Ret {out['proxy_metrics']['live']['ann_return']}%  "
          f"MDD {out['proxy_metrics']['live']['max_dd']}%  "
          f"({out['proxy_metrics']['live']['n_years']}y)")
    print(f"  proxy only: SR {out['proxy_metrics']['proxy_only']['sharpe']}  "
          f"Ret {out['proxy_metrics']['proxy_only']['ann_return']}%  "
          f"MDD {out['proxy_metrics']['proxy_only']['max_dd']}%  "
          f"({out['proxy_metrics']['proxy_only']['n_years']}y)")
    print(f"  curve points: {len(out['proxy_equity_curve'])}")


if __name__ == "__main__":
    patch("zephyr_proxy_returns.csv", "blend_factsheet_data.json", "ZEPHYR")
    print()
    patch("aurora_proxy_returns.csv", "aurora_factsheet_data.json", "AURORA")
