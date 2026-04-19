"""Patch nova_factsheet_data.json with METEOR proxy-extended equity curves
and period-split metrics (both the with-crypto and no-crypto variants).

Adds:
  proxy_equity_curve:          weekly [{date,value,spy,agg,source,tier}]
  proxy_nocrypto_equity_curve: weekly [{date,value,spy,agg,source,tier}]
  proxy_metrics:               {full, live, tier2, tier3, tier4, spy_full, agg_full}
  proxy_nocrypto_metrics:      {full, live, tier4, spy_full, agg_full}
  proxy_inception:             date of first 'live' row (with-crypto)
  proxy_nocrypto_inception:    date of first 'live' row (no-crypto)
  proxy_tier_labels:           {"1":..., "2":..., "3":..., "4":...}
  proxy_nocrypto_tier_labels:  {"1":..., "4":...}
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
    r = df["Close"]
    r_spy = df["SPY"]; r_agg = df["AGG"]; src = df["source"]
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


def compute(returns_csv, tier_labels):
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
    for k in tier_labels:
        if k == "1": continue
        pm[f"tier{k}"] = period(tier == int(k))

    curve = build_equity_curve(df)
    return curve, pm, live_start


def main():
    fs_path = RESULTS / "nova_factsheet_data.json"
    fs = json.loads(fs_path.read_text())

    TIERS = {
        "1": "Live (all real ETFs + BTC + ETH)",
        "2": "Crypto-partial (real ETFs + BTC, pre-ETH)",
        "3": "No-crypto (real ETFs only, pre-BTC)",
        "4": "Synth-leverage (synthetic daily×lev on underliers)",
    }
    NC_TIERS = {
        "1": "Live (all real ETFs, equity-only)",
        "4": "Synth-leverage (synthetic daily×lev on underliers)",
    }

    curve_c, pm_c, live_c = compute("nova_meteor_proxy_returns.csv", TIERS)
    curve_nc, pm_nc, live_nc = compute("nova_meteor_proxy_nocrypto_returns.csv", NC_TIERS)

    fs["proxy_equity_curve"] = curve_c
    fs["proxy_metrics"] = pm_c
    fs["proxy_inception"] = live_c.strftime("%Y-%m-%d") if live_c is not None else None
    fs["proxy_tier_labels"] = TIERS

    fs["proxy_nocrypto_equity_curve"] = curve_nc
    fs["proxy_nocrypto_metrics"] = pm_nc
    fs["proxy_nocrypto_inception"] = live_nc.strftime("%Y-%m-%d") if live_nc is not None else None
    fs["proxy_nocrypto_tier_labels"] = NC_TIERS

    fs_path.write_text(json.dumps(fs, separators=(",", ":")))

    print(f"patched {fs_path.name}")
    print(f"  with-crypto curve: {len(curve_c)} weekly pts, "
          f"inception {fs['proxy_inception']}")
    print(f"  no-crypto curve  : {len(curve_nc)} weekly pts, "
          f"inception {fs['proxy_nocrypto_inception']}")

    def pr(pm, order):
        for key in order:
            m = pm.get(key)
            if m is None: continue
            print(f"    {key:10s} {m.get('period',''):28s}  "
                  f"SR {m.get('sharpe',0):>6}  Ret {m.get('ann_return',0):>7}%  "
                  f"MDD {m.get('max_dd',0):>7}%  ({m.get('n_years',0)}y)")

    print("\n  WITH-CRYPTO tiers:")
    pr(pm_c, ["full","live","tier2","tier3","tier4","spy_full","agg_full"])
    print("\n  NO-CRYPTO tiers:")
    pr(pm_nc, ["full","live","tier4","spy_full","agg_full"])


if __name__ == "__main__":
    main()
