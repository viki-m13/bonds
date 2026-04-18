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
    """Weekly sample, normalized so NAV=10000 at start. Carry SPY+AGG+tier.
    Friday holidays (e.g. Good Friday) yield NaN on resample; ffill to last
    available print so the series stays JSON-serializable and continuous."""
    r = df["Close"]
    r_spy = df["SPY"]
    r_agg = df["AGG"]
    src = df["source"]
    tier = df["tier"] if "tier" in df.columns else pd.Series(1, index=df.index)
    cum = ((1 + r).cumprod() * 10000).resample("W-FRI").last().ffill()
    cum_spy = ((1 + r_spy).cumprod() * 10000).resample("W-FRI").last().ffill()
    cum_agg = ((1 + r_agg).cumprod() * 10000).resample("W-FRI").last().ffill()
    src_w = src.resample("W-FRI").last().ffill()
    tier_w = tier.resample("W-FRI").last().ffill()
    idx = cum.index
    return [{
        "date": d.strftime("%Y-%m-%d"),
        "value": round(float(cum.loc[d]), 2),
        "spy":   round(float(cum_spy.loc[d]), 2),
        "agg":   round(float(cum_agg.loc[d]), 2),
        "source": str(src_w.loc[d]),
        "tier":   int(tier_w.loc[d]),
    } for d in idx]


def patch(returns_csv, factsheet_json, name, tier_labels):
    df = pd.read_csv(RESULTS / returns_csv, parse_dates=["Date"]).set_index("Date")
    r = df["Close"]
    tier = df["tier"] if "tier" in df.columns else pd.Series(1, index=df.index)
    live_mask = (tier == 1)
    live_start = df.index[live_mask].min() if live_mask.any() else None

    def period(mask):
        if not mask.any():
            return {"period": "", **metrics(pd.Series([], dtype=float))}
        sub = df.index[mask]
        return {"period": f"{sub.min().date()} — {sub.max().date()}",
                **metrics(r[mask])}

    pm = {
        "full": {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(r)},
        "live": period(live_mask),
        "spy_full": {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(df["SPY"])},
        "agg_full": {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(df["AGG"])},
    }
    # Dynamic tier buckets from labels dict (skip "1" which is `live`)
    for k in tier_labels:
        if k == "1": continue
        pm[f"tier{k}"] = period(tier == int(k))

    out = {
        "proxy_equity_curve": build_equity_curve(df),
        "proxy_inception": live_start.strftime("%Y-%m-%d") if live_start is not None else None,
        "proxy_metrics": pm,
        "proxy_tier_labels": tier_labels,
    }

    fs_path = RESULTS / factsheet_json
    fs = json.loads(fs_path.read_text())
    fs.update(out)
    fs_path.write_text(json.dumps(fs, separators=(",", ":")))
    print(f"{name}: patched {fs_path.name}  ({len(out['proxy_equity_curve'])} weekly pts)")
    order = ["full", "live"] + [f"tier{k}" for k in tier_labels if k != "1"] + ["spy_full", "agg_full"]
    for key in order:
        m = pm.get(key)
        if m is None: continue
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
    NOVA_TIERS = {
        "1": "Live (all real ETFs + BTC + ETH)",
        "2": "Crypto-partial (real ETFs + BTC, pre-ETH)",
        "3": "No-crypto (real ETFs only, pre-BTC)",
        "4": "Synth-leverage (synthetic daily×lev on underliers)",
    }
    patch("zephyr_proxy_returns.csv", "blend_factsheet_data.json", "ZEPHYR", ZEPHYR_TIERS)
    print()
    patch("aurora_proxy_returns.csv", "aurora_factsheet_data.json", "AURORA", AURORA_TIERS)
    print()
    patch("nova_proxy_returns.csv", "nova_factsheet_data.json", "NOVA", NOVA_TIERS)
