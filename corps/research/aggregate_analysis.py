"""Honest exploration of the FINRA aggregate corporate datasets (the data a
basic API credential can access) against corporate-bond ETFs.

Reproduces CORP_FINDINGS.md: customer-flow and breadth signals vs forward
LQD/HYG returns, and a breadth-timing backtest that does NOT beat buy-and-hold.

  python corps/scripts/download_aggregates.py   # needs FINRA creds (once)
  python corps/research/aggregate_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "data" / "aggregates"
ETF = ROOT / "data" / "etf"


def _etf(t: str) -> pd.Series:
    return pd.read_csv(ETF / f"{t}.csv.gz", parse_dates=["date"]).set_index(
        "date")["adjclose"]


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def flow_signal(grade: str) -> pd.Series:
    s = pd.read_csv(AGG / "corporateMarketSentiment.csv.gz",
                    parse_dates=["tradeReportDate"])
    s["totalVolume"] = pd.to_numeric(s["totalVolume"], errors="coerce")
    g = s[s["tradeType"] == grade]
    piv = g.pivot_table(index="tradeReportDate", columns="productCategory",
                        values="totalVolume", aggfunc="sum")
    cb, cs = piv.get("customer buy"), piv.get("customer sell")
    return ((cb - cs) / (cb + cs)).rename("imb")


def breadth_z(grade: str) -> pd.Series:
    b = pd.read_csv(AGG / "corporateMarketBreadth.csv.gz",
                    parse_dates=["tradeReportDate"])
    b = _num(b, ["fiftyTwoWeekHigh", "fiftyTwoWeekLow"])
    g = b[b["productCategory"] == grade].set_index("tradeReportDate").sort_index()
    hl = (g["fiftyTwoWeekHigh"] - g["fiftyTwoWeekLow"]) / (
        g["fiftyTwoWeekHigh"] + g["fiftyTwoWeekLow"] + 1)
    return ((hl - hl.rolling(120, min_periods=40).mean())
            / hl.rolling(120, min_periods=40).std()).shift(1).rename("z")


def corr_table():
    print("Signal → forward ETF return correlations:")
    for grade, t in [("investment grade", "LQD"), ("high yield", "HYG")]:
        px = _etf(t)
        imb, z = flow_signal(grade), breadth_z(grade)
        for name, sig in [("flow", imb), ("breadth-z", z)]:
            line = f"  {grade:17} {name:10}"
            for h in (10, 21, 42):
                fwd = px.reindex(sig.index).ffill().pct_change(h).shift(-h)
                d = pd.concat([sig.rename("s"), fwd.rename("f")], axis=1).dropna()
                line += f"  h{h}={d['s'].corr(d['f']):+.3f}"
            print(line)


def timing_backtest():
    print("\nBreadth-timing vs buy-and-hold (HY / HYG):")
    z = breadth_z("high yield")
    ret = _etf("HYG").pct_change()
    df = pd.concat([z, ret.rename("ret")], axis=1).dropna()

    def stats(r):
        return ((r.add(1).prod()) ** (252 / len(r)) - 1,
                r.mean() / r.std() * np.sqrt(252) if r.std() else 0,
                (r.add(1).cumprod() / r.add(1).cumprod().cummax() - 1).min())

    for thr in (0.5, 1.0, 1.5):
        expo = (df["z"] < thr).astype(float)
        a, s, d = stats(expo * df["ret"])
        print(f"  timed (thr={thr}): ann {a*100:+.1f}%  Sharpe {s:.2f}  "
              f"maxDD {d*100:.1f}%  exposure {expo.mean()*100:.0f}%")
    a, s, d = stats(df["ret"])
    print(f"  buy-and-hold HYG : ann {a*100:+.1f}%  Sharpe {s:.2f}  maxDD {d*100:.1f}%")
    print("  -> timing does not beat buy-and-hold on this 3-year sample.")


if __name__ == "__main__":
    corr_table()
    timing_backtest()
