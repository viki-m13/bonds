"""APEX — generate factsheet JSON for the webpage."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd

import util
from apex_production import run_apex, BLEND_WEIGHTS

OUT = Path("/home/user/bonds/data/apex")
DOCS = Path("/home/user/bonds/docs")


def equity_curve(r: pd.Series) -> dict:
    c = (1 + r.fillna(0)).cumprod()
    # Monthly end values to keep payload small
    monthly = c.resample("ME").last().dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in monthly.index],
        "values": [round(float(v), 4) for v in monthly.values],
    }


def drawdown_curve(r: pd.Series) -> dict:
    c = (1 + r.fillna(0)).cumprod()
    hwm = c.cummax()
    dd = c / hwm - 1
    monthly = dd.resample("ME").last().dropna()
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in monthly.index],
        "values": [round(float(v), 4) for v in monthly.values],
    }


def yearly_returns(r: pd.Series) -> list[dict]:
    # Year-by-year return (geometric)
    r = r.dropna()
    out = []
    for y, grp in r.groupby(r.index.year):
        ret = (1 + grp).prod() - 1
        sd = grp.std() * np.sqrt(util.DPY)
        sr = ((1 + grp).prod() ** (252 / len(grp)) - 1) / sd if sd > 0 else 0
        out.append({
            "year": int(y),
            "ret": round(float(ret), 4),
            "vol": round(float(sd), 4),
            "sharpe": round(float(sr), 3),
            "n": int(len(grp)),
        })
    return out


def monthly_heatmap(r: pd.Series) -> dict:
    r = r.dropna()
    m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    years = sorted(r.index.year.unique())
    result = {"years": years, "months": {}}
    for y in years:
        result["months"][y] = []
        for mm in range(1, 13):
            try:
                v = float(m.loc[(y, mm)])
                result["months"][y].append(round(v, 4))
            except KeyError:
                result["months"][y].append(None)
    return result


def main():
    op, cp = util.load_prices()
    print("Running APEX production...")
    r, state, P, sleeve_rets = run_apex(cp)

    # Load existing metrics
    with open(OUT / "apex_production_metrics.json") as f:
        prod = json.load(f)

    # Load stress tests
    try:
        with open(OUT / "stress_summary.json") as f:
            stress = json.load(f)
    except FileNotFoundError:
        stress = {}

    # Regime table
    regs = pd.read_csv(OUT / "stress_regimes.csv")

    # Walk-forward
    wf = pd.read_csv(OUT / "stress_walkforward.csv")

    # Sleeve metrics (full sample)
    R = pd.DataFrame(sleeve_rets)
    sleeve_metrics = {}
    for n in R.columns:
        m = util.metrics(R[n])
        sleeve_metrics[n] = {
            "sharpe": m["sharpe"], "cagr": m["cagr"],
            "vol": m["vol"], "mdd": m["mdd"],
        }

    # Correlations
    corr = R.corr().round(3).to_dict()

    # Windows
    metrics = {}
    for lbl, (s, e) in [("full", ("1999-01-01", "2027-12-31")),
                        ("is", ("2005-01-01", "2018-12-31")),
                        ("oos", ("2019-01-02", "2027-12-31")),
                        ("pre08", ("2000-01-01", "2008-12-31")),
                        ("gfc", ("2007-01-01", "2009-12-31")),
                        ("covid", ("2020-01-01", "2020-12-31")),
                        ("ratehike22", ("2022-01-01", "2022-12-31")),
                        ("recovery2324", ("2023-01-01", "2024-12-31"))]:
        metrics[lbl] = util.metrics(util.regime_slice(r, s, e))

    # Benchmark: 60/40 SPY/TLT
    spy_r = cp["SPY"].pct_change()
    tlt_r = cp["TLT"].pct_change() if "TLT" in cp.columns else None
    bench_rets = {}
    bench_rets["SPY"] = util.metrics(spy_r)
    if tlt_r is not None:
        b60 = 0.6 * spy_r + 0.4 * tlt_r
        bench_rets["SPY_TLT_60_40"] = util.metrics(b60.dropna())

    # Equity curves
    eq_apex = equity_curve(r)
    dd_apex = drawdown_curve(r)
    eq_spy = equity_curve(spy_r.reindex(r.index).fillna(0))

    # Yearly
    yrs = yearly_returns(r)
    heatmap = monthly_heatmap(r)

    # Current weights (last 30d average)
    w_current = P.iloc[-30:].mean()
    w_current = w_current[w_current > 0.001].sort_values(ascending=False)
    current_allocation = [{"ticker": k, "weight": round(float(v), 4)} for k, v in w_current.items()]

    data = {
        "name": "APEX",
        "tagline": "Six-Sleeve Leveraged ETF Ensemble | Daily Vol Scaling | No Margin",
        "as_of": r.index.max().strftime("%Y-%m-%d"),
        "params": prod["params"],
        "metrics": metrics,
        "sleeve_metrics": sleeve_metrics,
        "correlations": corr,
        "benchmarks": bench_rets,
        "regimes": regs.to_dict(orient="records"),
        "walkforward": wf.to_dict(orient="records"),
        "stress_summary": stress,
        "equity_curve_apex": eq_apex,
        "equity_curve_spy": eq_spy,
        "drawdown_apex": dd_apex,
        "yearly_returns": yrs,
        "monthly_heatmap": heatmap,
        "current_allocation": current_allocation,
    }
    out_fp = OUT / "apex_factsheet_data.json"
    out_fp.write_text(json.dumps(data, indent=2, default=str))
    print(f"Saved {out_fp} ({out_fp.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
