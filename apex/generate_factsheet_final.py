"""APEX — generate factsheet JSON from the final strategy."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd
import util

OUT = Path("/home/user/bonds/data/apex")


def equity_curve_dict(r: pd.Series) -> dict:
    c = (1 + r.fillna(0)).cumprod()
    m = c.resample("ME").last().dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in m.index],
        "values": [round(float(v), 4) for v in m.values],
    }


def drawdown_dict(r: pd.Series) -> dict:
    c = (1 + r.fillna(0)).cumprod()
    hwm = c.cummax()
    dd = c / hwm - 1
    m = dd.resample("ME").last().dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in m.index],
        "values": [round(float(v), 4) for v in m.values],
    }


def rolling_sharpe_dict(r: pd.Series, win: int = 252) -> dict:
    mu = r.rolling(win).mean() * util.DPY
    sd = r.rolling(win).std() * np.sqrt(util.DPY)
    rs = (mu / sd.replace(0, np.nan)).dropna()
    m = rs.resample("ME").last().dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in m.index],
        "values": [round(float(v), 3) for v in m.values],
    }


def yearly_returns(r: pd.Series) -> list[dict]:
    r = r.dropna()
    out = []
    for y, grp in r.groupby(r.index.year):
        if len(grp) < 20:
            continue
        ret = (1 + grp).prod() - 1
        sd = grp.std() * np.sqrt(util.DPY)
        cum = (1 + grp).cumprod()
        mdd = (cum / cum.cummax() - 1).min()
        out.append({
            "year": int(y),
            "ret": round(float(ret), 4),
            "vol": round(float(sd), 4),
            "mdd": round(float(mdd), 4),
            "sharpe": round(float(grp.mean() / grp.std() * np.sqrt(util.DPY)) if grp.std() > 0 else 0, 3),
        })
    return out


def monthly_heatmap(r: pd.Series) -> dict:
    r = r.dropna()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    years = sorted(r.index.year.unique())
    result = {"years": [int(y) for y in years], "months": {}}
    for y in years:
        row = []
        for mm in range(1, 13):
            try:
                v = float(m.loc[(y, mm)])
                row.append(round(v, 4))
            except KeyError:
                row.append(None)
        result["months"][int(y)] = row
    return result


def main():
    op, cp = util.load_prices()
    net = pd.read_csv(OUT / "apex_final_returns.csv", parse_dates=["Date"]).set_index("Date")["apex_net_ret"]
    net = net.dropna()

    # Windows
    windows = [
        ("full", "1999-01-01", "2027-12-31"),
        ("is", "2005-01-01", "2018-12-31"),
        ("oos", "2019-01-02", "2027-12-31"),
        ("pre08", "2000-01-01", "2008-12-31"),
        ("gfc", "2007-01-01", "2009-12-31"),
        ("covid", "2020-01-01", "2020-12-31"),
        ("ratehike22", "2022-01-01", "2022-12-31"),
        ("recovery2324", "2023-01-01", "2024-12-31"),
    ]
    metrics = {}
    for lbl, s, e in windows:
        m = util.metrics(util.regime_slice(net, s, e))
        metrics[lbl] = m

    # Benchmarks
    spy = cp["SPY"].pct_change()
    bench = {"SPY": util.metrics(spy)}
    if "TLT" in cp.columns:
        b60 = 0.6 * spy + 0.4 * cp["TLT"].pct_change()
        bench["SPY_TLT_60_40"] = util.metrics(b60.dropna())
    # TQQQ buy-hold
    if "TQQQ" in cp.columns:
        bench["TQQQ_BH"] = util.metrics(cp["TQQQ"].pct_change().dropna())
    # 60/40 of UPRO/TMF rebalanced daily (HFEA-like)
    if "UPRO" in cp.columns and "TMF" in cp.columns:
        hfea = 0.6 * cp["UPRO"].pct_change() + 0.4 * cp["TMF"].pct_change()
        bench["HFEA_UPRO_TMF"] = util.metrics(hfea.dropna())

    # Load sleeve returns from the ml+v3 stack
    sleeves = {}
    for name, csv in [
        ("ML5", "ml5_returns.csv"),
        ("TREND", None),
        ("RPAR_CF", None),
        ("PAA", None),
        ("ORION", None),
        ("HELIOS", None),
    ]:
        if csv:
            fp = OUT / csv
            if fp.exists():
                df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date")
                s = df.iloc[:, 0]
                sleeves[name] = util.metrics(s)

    # Rolling 3y sharpe
    rs = net.rolling(756).mean() / net.rolling(756).std() * np.sqrt(util.DPY)
    rolling3y = rs.dropna()
    rolling_dict = {
        "dates": [d.strftime("%Y-%m-%d") for d in rolling3y.resample("ME").last().dropna().index],
        "values": [round(float(v), 3) for v in rolling3y.resample("ME").last().dropna().values],
    }

    # Current weights (last 30d)
    w_final = pd.read_csv(OUT / "apex_final_weights.csv", parse_dates=["Date"]).set_index("Date")
    w_recent = w_final.iloc[-30:].mean()
    w_recent = w_recent[w_recent > 0.001].sort_values(ascending=False)
    current_alloc = [{"ticker": k, "weight": round(float(v), 4)} for k, v in w_recent.items()]

    data = {
        "name": "APEX",
        "tagline": "Multi-Sleeve Leveraged-ETF Ensemble with ML Alpha Engine",
        "description": "Six-sleeve blend (50% XGBoost ML + 50% rule-based), daily vol scaling, no portfolio margin, pre-2008 stress-tested.",
        "as_of": net.index.max().strftime("%Y-%m-%d"),
        "metrics": metrics,
        "benchmarks": bench,
        "sleeve_metrics": sleeves,
        "equity_curve": equity_curve_dict(net),
        "drawdown": drawdown_dict(net),
        "rolling_3y_sharpe": rolling_dict,
        "rolling_1y_sharpe": rolling_sharpe_dict(net, 252),
        "yearly_returns": yearly_returns(net),
        "monthly_heatmap": monthly_heatmap(net),
        "current_allocation": current_alloc,
        "params": json.loads((OUT / "apex_final_meta.json").read_text()),
        "universe": ["UPRO","TQQQ","TECL","SOXL","FAS","EDC","YINN","TMF","UBT","TYD","UGL","UCO","DRN",
                     "SSO","QLD","ERX","SPY","QQQ","TLT","GLD","BIL","SHY"],
    }
    fp = OUT / "apex_factsheet_final.json"
    fp.write_text(json.dumps(data, indent=2, default=str))
    print(f"Saved {fp} ({fp.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
