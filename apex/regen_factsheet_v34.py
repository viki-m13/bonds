"""Regenerate apex factsheet from v34 (max-5-positions) returns + weights.

Adds Phoenix-style regime periods and uses the new capped allocation."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import json
import numpy as np
import pandas as pd
import util

OUT = Path("/home/user/bonds/data/apex")


def equity_curve_dict(r: pd.Series, start: str = None) -> dict:
    if start:
        r = r.loc[start:]
    c = (1 + r.fillna(0)).cumprod()
    m = c.resample("ME").last().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in m.index],
            "values": [round(float(v), 4) for v in m.values]}


def overlay_series(weights_fp: Path) -> dict:
    """Daily gross LETF exposure = sum of row weights. Phoenix-style overlay chart."""
    if not weights_fp.exists():
        return {"dates": [], "values": []}
    W = pd.read_csv(weights_fp, parse_dates=["Date"]).set_index("Date")
    gross = W.sum(axis=1).clip(lower=0)
    # Weekly resample to keep file tidy
    gw = gross.resample("W-FRI").last().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in gw.index],
            "values": [round(float(v), 4) for v in gw.values]}


def drawdown_dict(r: pd.Series) -> dict:
    c = (1 + r.fillna(0)).cumprod()
    hwm = c.cummax()
    dd = c / hwm - 1
    m = dd.resample("ME").last().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in m.index],
            "values": [round(float(v), 4) for v in m.values]}


def rolling_sharpe_dict(r: pd.Series, win: int = 756) -> dict:
    mu = r.rolling(win).mean() * util.DPY
    sd = r.rolling(win).std() * np.sqrt(util.DPY)
    rs = (mu / sd.replace(0, np.nan)).dropna()
    m = rs.resample("ME").last().dropna()
    return {"dates": [d.strftime("%Y-%m-%d") for d in m.index],
            "values": [round(float(v), 3) for v in m.values]}


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
        sharpe = (grp.mean() / grp.std() * np.sqrt(util.DPY)) if grp.std() > 0 else 0
        out.append({"year": int(y), "ret": round(float(ret), 4),
                    "vol": round(float(sd), 4), "mdd": round(float(mdd), 4),
                    "sharpe": round(float(sharpe), 3)})
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
    net = pd.read_csv(OUT / "apex_v34_returns.csv", parse_dates=["Date"]).set_index("Date")["apex_v34_ret"]
    net = net.dropna()

    # Phoenix-style periods (ALL of Phoenix's named windows + APEX-only deep history)
    metrics = {}
    for lbl, s, e in [
        ("full", "1999-01-01", "2027-12-31"),
        ("phx_window", "2010-01-01", "2027-12-31"),  # Phoenix native window
        ("is", "2005-01-01", "2018-12-31"),
        ("oos", "2019-01-02", "2027-12-31"),
        ("pre08", "2000-01-01", "2008-12-31"),
        ("gfc", "2007-01-01", "2009-12-31"),
        ("gfc_strict", "2008-01-01", "2009-06-30"),  # Phoenix's stress window
        ("y2008", "2008-01-01", "2008-12-31"),       # 2008 calendar
        ("covid", "2020-01-01", "2020-12-31"),
        ("ratehike22", "2022-01-01", "2022-12-31"),
        ("recovery2324", "2023-01-01", "2024-12-31"),
        ("y2025plus", "2025-01-01", "2027-12-31"),
    ]:
        metrics[lbl] = util.metrics(util.regime_slice(net, s, e))

    # Benchmarks
    spy = cp["SPY"].pct_change()
    bench = {"SPY": util.metrics(spy)}
    if "TLT" in cp.columns:
        bench["SPY_TLT_60_40"] = util.metrics((0.6 * spy + 0.4 * cp["TLT"].pct_change()).dropna())
    if "TQQQ" in cp.columns:
        bench["TQQQ_BH"] = util.metrics(cp["TQQQ"].pct_change().dropna())
    if "UPRO" in cp.columns and "TMF" in cp.columns:
        hfea = 0.6 * cp["UPRO"].pct_change() + 0.4 * cp["TMF"].pct_change()
        bench["HFEA_UPRO_TMF"] = util.metrics(hfea.dropna())

    # Current allocation from v34 weights — already capped at top-5
    current_alloc = []
    pos_stats = {}
    try:
        w_v34 = pd.read_csv(OUT / "apex_v34_weights.csv", parse_dates=["Date"]).set_index("Date")
        # Most recent date with any position
        w_recent = w_v34.iloc[-30:].mean()
        w_recent = w_recent[w_recent > 0.001].sort_values(ascending=False).head(5)
        current_alloc = [{"ticker": k, "weight": round(float(v), 4)} for k, v in w_recent.items()]
        # Position-count stats
        pc = (w_v34 > 0.001).sum(axis=1)
        pos_stats = {
            "mean": round(float(pc.mean()), 2),
            "median": int(pc.median()),
            "max": int(pc.max()),
            "p95": int(pc.quantile(0.95)),
        }
    except Exception as e:
        print(f"WARN: weights load: {e}")

    # Sleeve metrics — use the prior file if exists (sleeve definitions unchanged)
    sleeves = {}
    try:
        R = pd.read_csv(OUT / "apex_final_sleeve_returns.csv",
                        parse_dates=["Date"]).set_index("Date")
        for n in R.columns:
            sleeves[n] = util.metrics(R[n])
    except Exception:
        pass

    data = {
        "name": "APEX",
        "tagline": "Six-Sleeve LETF Ensemble with Phoenix-Style Overlays — capped at MAX 5 positions",
        "description": "Six uncorrelated sleeves (PX_HELIOS + HMM + DIVERGENCE + ACCEL_MOM + SKEW_MOM + HURST) blended with dynamic crypto allocation and capped at 5 LETF positions per day, mirroring Phoenix's concentration discipline.",
        "as_of": net.index.max().strftime("%Y-%m-%d"),
        "metrics": metrics,
        "benchmarks": bench,
        "sleeve_metrics": sleeves,
        "equity_curve": equity_curve_dict(net),
        "equity_curve_phx": equity_curve_dict(net, start="2010-01-01"),
        "drawdown": drawdown_dict(net),
        "rolling_3y_sharpe": rolling_sharpe_dict(net, 756),
        "rolling_1y_sharpe": rolling_sharpe_dict(net, 252),
        "overlay_exposure": overlay_series(OUT / "apex_v34_weights.csv"),
        "yearly_returns": yearly_returns(net),
        "monthly_heatmap": monthly_heatmap(net),
        "current_allocation": current_alloc,
        "position_stats": pos_stats,
        "params": json.loads((OUT / "apex_v34_meta.json").read_text()),
        "universe": ["UPRO", "TQQQ", "TECL", "SOXL", "FAS", "EDC", "YINN", "TMF", "UBT", "TYD",
                     "UGL", "UCO", "DRN", "SSO", "QLD", "ERX", "SPY", "QQQ", "TLT", "GLD",
                     "BIL", "SHY"],
    }
    fp = OUT / "apex_v34_factsheet.json"
    fp.write_text(json.dumps(data, default=str))
    print(f"Saved {fp} ({fp.stat().st_size / 1024:.1f} KB)")
    print(f"\nKey metrics (max-5-positions):")
    for k, m in metrics.items():
        if m and m.get("n", 0) > 0:
            print(f"  {k:18s} SR={m['sharpe']:>5.2f}  CAGR={m['cagr']*100:>6.1f}%  "
                  f"MDD={m['mdd']*100:>6.1f}%  n={m['n']}")
    print(f"\nPosition stats: {pos_stats}")
    print(f"Current allocation: {current_alloc}")


if __name__ == "__main__":
    main()
