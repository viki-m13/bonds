"""De-lucked PHOENIX: rebuild the production blend using BAGGED sleeves.

The bagging study showed ORION's and HELIOS's canonical OOS Sharpes were partly
parameter luck (large negative IS->OOS gaps that shrink sharply under bagging).
This asks the scientific question: what is PHOENIX's OOS Sharpe when each
rules-based sleeve is replaced by its bag-averaged (overfit-robust) version?

Swaps VANGUARD/ORION/HELIOS for their bagged streams; keeps QUANTUM and CRYPTO
(QUANTUM is ML, separately known to be IS-overfit; CRYPTO unchanged). Recomputes
IS inverse-vol weights and applies the exact production overlay. Compares the
canonical-sleeve blend vs the bagged-sleeve blend, IS and OOS.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "data/results"
ETF = ROOT / "data/etfs"
BAG = ROOT / "phoenix5/bagging"
OUT = BAG

IS_END = "2018-12-31"
OOS = "2019-01-02"


def px(t):
    s = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    return s[~s.index.duplicated()].sort_index()


def metrics(r):
    r = r.dropna()
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    return {"sr": round(float(mu / sd), 3), "cagr": round(float(c.iloc[-1] ** (1 / yrs) - 1), 4),
            "vol": round(float(sd), 4), "mdd": round(float(mdd), 4)}


def overlay(raw, bil):
    rv = raw.rolling(60).std() * np.sqrt(252)
    vol_mult = (0.15 / rv).clip(0.25, 1.0).shift(1).fillna(1.0)
    scaled = raw * vol_mult
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd_mult = (1.0 + (cum / hwm - 1) / -0.10).clip(0, 1).shift(1).fillna(1.0)
    sv = scaled.rolling(60).std()
    thr = sv.rolling(252, min_periods=60).quantile(0.99)
    ok = (sv <= thr).shift(1).fillna(True).astype(float)
    gate = ok + (1 - ok) * 0.5
    total = (vol_mult * dd_mult * gate).clip(0, 1.0)
    tc = total.diff().abs().fillna(0) * (10 / 1e4)
    return raw * total - tc


def main():
    van_c = pd.read_csv(R / "vanguard_returns.csv", parse_dates=[0], index_col=0)["net_ret"]
    ori_c = pd.read_csv(R / "orion_returns.csv", parse_dates=["Date"]).set_index("Date")["orion"]
    hel_c = pd.read_csv(R / "helios_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    qua = pd.read_csv(R / "quantum_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    cry = pd.read_csv(R / "crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

    van_b = pd.read_csv(BAG / "vanguard_bagged_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    ori_b = pd.read_csv(BAG / "orion_bagged_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    hel_b = pd.read_csv(BAG / "helios_bagged_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

    bil = px("BIL").pct_change()

    def blend(van, ori, hel):
        df = pd.concat({"VANGUARD": van, "ORION": ori, "HELIOS": hel,
                        "QUANTUM": qua, "CRYPTO": cry}, axis=1, sort=True).fillna(0.0).loc["2010-03-11":]
        iv = 1.0 / df.loc[:IS_END].std()
        w = iv / iv.sum()
        return (df @ w), w

    raw_c, w_c = blend(van_c, ori_c, hel_c)
    raw_b, w_b = blend(van_b, ori_b, hel_b)
    bil_a = bil.reindex(raw_c.index).fillna(0)

    net_c = overlay(raw_c, bil_a)
    net_b = overlay(raw_b, bil_a)

    print("PHOENIX blend, canonical vs bagged sleeves (production overlay):")
    out = {}
    for name, r in [("canonical", net_c), ("bagged", net_b)]:
        i, o, f = metrics(r.loc[:IS_END]), metrics(r.loc[OOS:]), metrics(r)
        out[name] = {"is": i, "oos": o, "full": f,
                     "is_oos_gap": round(i["sr"] - o["sr"], 3)}
        print(f"  {name:10s} IS SR={i['sr']:.2f}  OOS SR={o['sr']:.2f}  "
              f"OOS CAGR={o['cagr']*100:.1f}%  OOS MDD={o['mdd']*100:.1f}%  "
              f"IS-OOS gap={i['sr']-o['sr']:+.2f}")
    print(f"\n  IS inv-vol weights canonical: {dict((k, round(v,3)) for k,v in w_c.items())}")
    print(f"  IS inv-vol weights bagged   : {dict((k, round(v,3)) for k,v in w_b.items())}")
    print(f"\n  Interpretation: the bagged-sleeve OOS Sharpe is the overfit-robust")
    print(f"  estimate of PHOENIX's true OOS performance. The gap to the canonical")
    print(f"  OOS number quantifies how much rests on lucky parameter picks.")

    (OUT / "bagged_blend_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"canonical": net_c, "bagged": net_b}).dropna().to_csv(OUT / "bagged_blend_returns.csv")
    print(f"\nSaved bagged_blend_metrics.json, bagged_blend_returns.csv")


if __name__ == "__main__":
    main()
