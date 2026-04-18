"""Add proxy_nocrypto_* fields to nova_factsheet_data.json.

Produces a companion equity curve (weekly) and metrics for the NOVA
strategy run with CRYPTO=[] so users can see how the strategy would
have performed on leveraged ETFs alone since 2005.

Fields added:
  proxy_nocrypto_equity_curve: [{date, value, tier}]
  proxy_nocrypto_metrics:      {full, live, tier4}
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


def main():
    df = pd.read_csv(RESULTS / "nova_proxy_nocrypto_returns.csv",
                     parse_dates=["Date"]).set_index("Date")
    r = df["Close"]
    tier = df["tier"]

    cum = ((1 + r).cumprod() * 10000).resample("W-FRI").last().ffill()
    tier_w = tier.resample("W-FRI").last().ffill()
    curve = [{
        "date": d.strftime("%Y-%m-%d"),
        "value": round(float(cum.loc[d]), 2),
        "tier": int(tier_w.loc[d]),
    } for d in cum.index]

    def period(mask):
        if not mask.any():
            return {"period": "", **metrics(pd.Series([], dtype=float))}
        sub = df.index[mask]
        return {"period": f"{sub.min().date()} — {sub.max().date()}",
                **metrics(r[mask])}

    m = {
        "full":  {"period": f"{df.index[0].date()} — {df.index[-1].date()}", **metrics(r)},
        "live":  period(tier == 1),
        "tier4": period(tier == 4),
    }

    fs_path = RESULTS / "nova_factsheet_data.json"
    fs = json.loads(fs_path.read_text())
    fs["proxy_nocrypto_equity_curve"] = curve
    fs["proxy_nocrypto_metrics"] = m
    fs_path.write_text(json.dumps(fs, separators=(",", ":")))

    print(f"NOVA no-crypto: patched {fs_path.name}  ({len(curve)} weekly pts)")
    for k in ["full", "live", "tier4"]:
        d = m[k]
        print(f"  {k:10s} {d.get('period',''):28s}  "
              f"SR {d.get('sharpe',0):>5}  Ret {d.get('ann_return',0):>6}%  "
              f"MDD {d.get('max_dd',0):>6}%  ({d.get('n_years',0)}y)")


if __name__ == "__main__":
    main()
