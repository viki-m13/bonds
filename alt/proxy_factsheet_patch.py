"""Patch aurora_factsheet_data.json and blend_factsheet_data.json with
proxy-extended equity curves (strategy + SPY + AGG benchmarks) and
period-split metrics.

New fields added to each factsheet JSON:
  proxy_equity_curve: [{date,value,spy,agg,source,tier}] — weekly samples
  proxy_metrics:      {full, live, tier2, tier3, spy_full, agg_full}
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
    """Weekly sample, normalized so NAV=10000 at start. Carry SPY+AGG+tier."""
    r = df["Close"]
    r_spy = df["SPY"]
    r_agg = df["AGG"]
    src = df["source"]
    tier = df["tier"] if "tier" in df.columns else pd.Series(1, index=df.index)
    cum = (1 + r).cumprod() * 10000
    cum_spy = (1 + r_spy).cumprod() * 10000
    cum_agg = (1 + r_agg).cumprod() * 10000
    idx = cum.resample("W-FRI").last().dropna().index
    return [{
        "date": d.strftime("%Y-%m-%d"),
        "value": round(float(cum.reindex(idx).loc[d]), 2),
        "spy":   round(float(cum_spy.reindex(idx).loc[d]), 2),
        "agg":   round(float(cum_agg.reindex(idx).loc[d]), 2),
        "source": str(src.reindex(idx).ffill().loc[d]),
        "tier":   int(tier.reindex(idx).ffill().loc[d]),
    } for d in idx]


def patch(returns_csv, factsheet_json, name, tier_labels):
    df = pd.read_csv(RESULTS / returns_csv, parse_dates=["Date"]).set_index("Date")
    r = df["Close"]
    tier = df["tier"] if "tier" in df.columns else pd.Series(1, index=df.index)
    live_mask = (tier == 1)
    tier2_mask = (tier == 2)
    tier3_mask = (tier == 3)
    live_start = df.index[live_mask].min() if live_mask.any() else None

    def period(mask):
        if not mask.any():
            return {"period": "", **metrics(pd.Series([], dtype=float))}
        sub = df.index[mask]
        return {"period": f"{sub.min().date()} — {sub.max().date()}",
                **metrics(r[mask])}

    out = {
        "proxy_equity_curve": build_equity_curve(df),
        "proxy_inception": live_start.strftime("%Y-%m-%d") if live_start is not None else None,
        "proxy_metrics": {
            "full":  {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(r)},
            "live":  period(live_mask),
            "tier2": period(tier2_mask),
            "tier3": period(tier3_mask),
            "spy_full": {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(df["SPY"])},
            "agg_full": {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(df["AGG"])},
        },
        "proxy_tier_labels": tier_labels,
    }

    fs_path = RESULTS / factsheet_json
    fs = json.loads(fs_path.read_text())
    fs.update(out)
    fs_path.write_text(json.dumps(fs, separators=(",", ":")))
    print(f"{name}: patched {fs_path.name}  ({len(out['proxy_equity_curve'])} weekly pts)")
    for key in ["full", "live", "tier2", "tier3", "spy_full", "agg_full"]:
        m = out["proxy_metrics"][key]
        print(f"  {key:10s} {m.get('period',''):28s}  "
              f"SR {m.get('sharpe',0):>5}  Ret {m.get('ann_return',0):>6}%  "
              f"MDD {m.get('max_dd',0):>6}%  ({m.get('n_years',0)}y)")


if __name__ == "__main__":
    ZEPHYR_TIERS = {
        "1": "Live (all real ETFs)",
        "2": "Near-proxy (JAAA→FLOT+BKLN, JPST→MINT)",
        "3": "Early-proxy (credit sleeves→SHY)",
    }
    AURORA_TIERS = {
        "1": "Live (all real ETFs)",
        "2": "Mid-proxy (XYLD/QYLD/SCHD + synthetic trend)",
        "3": "Early-proxy (SPY-3%/yr + synthetic 3x underliers)",
    }
    patch("zephyr_proxy_returns.csv", "blend_factsheet_data.json", "ZEPHYR", ZEPHYR_TIERS)
    print()
    patch("aurora_proxy_returns.csv", "aurora_factsheet_data.json", "AURORA", AURORA_TIERS)
