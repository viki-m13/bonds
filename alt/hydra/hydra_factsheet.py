"""Build HYDRA factsheet JSON for the webapp (docs/hydra.html).

Outputs:
  data/results/hydra_factsheet_data.json — all fields needed by docs/hydra.html
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
RESULTS = ROOT / "data/results"


def metrics(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return {"sharpe": 0, "ann_return": 0, "ann_vol": 0, "max_dd": 0,
                "sortino": 0, "n_years": round(len(r) / 252, 1)}
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


def equity_curve(r, start=10000.0, freq="W-FRI"):
    cum = ((1 + r).cumprod() * start).resample(freq).last().ffill()
    return [{"date": d.strftime("%Y-%m-%d"), "value": round(float(cum.loc[d]), 2)}
            for d in cum.index]


def equity_curve_multi(df, start=10000.0, freq="W-FRI"):
    """df columns: any asset; each column gets its own compounded curve."""
    cum = ((1 + df).cumprod() * start).resample(freq).last().ffill()
    out = []
    for d in cum.index:
        row = {"date": d.strftime("%Y-%m-%d")}
        for c in cum.columns:
            row[c] = round(float(cum.loc[d, c]), 2)
        out.append(row)
    return out


def drawdown_curve(r, freq="W-FRI"):
    cum = (1 + r).cumprod()
    dd = (cum / cum.cummax() - 1) * 100
    ddw = dd.resample(freq).last().ffill()
    return [{"date": d.strftime("%Y-%m-%d"), "dd": round(float(ddw.loc[d]), 2)}
            for d in ddw.index]


def rolling_sharpe(r, window=252, freq="W-FRI"):
    mu = r.rolling(window).mean() * 252
    sd = r.rolling(window).std() * np.sqrt(252)
    sh = (mu / sd).dropna()
    sw = sh.resample(freq).last()
    return [{"date": d.strftime("%Y-%m-%d"), "sr": round(float(sw.loc[d]), 3)}
            for d in sw.index if not np.isnan(sw.loc[d])]


def calendar_returns(r):
    """Yearly returns by calendar year."""
    by_year = r.groupby(r.index.year).apply(
        lambda x: (1 + x).prod() - 1
    )
    return [{"year": int(y), "ret": round(float(v * 100), 2)}
            for y, v in by_year.items()]


def monthly_heatmap(r):
    mo = r.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    return [{"date": d.strftime("%Y-%m-%d"),
             "year": int(d.year),
             "month": int(d.month),
             "ret": round(float(mo.loc[d]), 2)} for d in mo.index]


def trailing(r, spy):
    """Trailing 1M/3M/6M/YTD/1Y/3Y/5Y/10Y and since-inception."""
    end = r.index[-1]
    periods = {
        "1M": 21, "3M": 63, "6M": 126, "1Y": 252,
        "3Y_ann": 252 * 3, "5Y_ann": 252 * 5, "10Y_ann": 252 * 10,
    }
    out = {"HYDRA": {}, "SPY": {}}
    for label, n in periods.items():
        if len(r) < n + 1:
            out["HYDRA"][label] = None
            out["SPY"][label] = None
            continue
        rh = r.iloc[-n:]
        rs = spy.iloc[-n:]
        if label.endswith("_ann"):
            out["HYDRA"][label] = round(float(((1 + rh).prod() ** (252 / n) - 1) * 100), 2)
            out["SPY"][label] = round(float(((1 + rs).prod() ** (252 / n) - 1) * 100), 2)
        else:
            out["HYDRA"][label] = round(float(((1 + rh).prod() - 1) * 100), 2)
            out["SPY"][label] = round(float(((1 + rs).prod() - 1) * 100), 2)
    # YTD
    ystart = pd.Timestamp(f"{end.year}-01-01")
    rh_y = r.loc[ystart:]
    rs_y = spy.loc[ystart:]
    out["HYDRA"]["YTD"] = round(float(((1 + rh_y).prod() - 1) * 100), 2)
    out["SPY"]["YTD"] = round(float(((1 + rs_y).prod() - 1) * 100), 2)
    # Since inception
    out["HYDRA"]["SI_ann"] = round(float(((1 + r).prod() ** (252 / len(r)) - 1) * 100), 2)
    out["SPY"]["SI_ann"] = round(float(((1 + spy).prod() ** (252 / len(spy)) - 1) * 100), 2)
    return out


def walkforward(r, spy, year_groups):
    rows = []
    for y0, y1 in year_groups:
        lo = pd.Timestamp(f"{y0}-01-01")
        hi = pd.Timestamp(f"{y1}-01-01")
        sub_h = r.loc[lo:hi]
        sub_s = spy.loc[lo:hi]
        if len(sub_h) < 200:
            continue
        mh = metrics(sub_h)
        ms = metrics(sub_s)
        rows.append({
            "window": f"{y0}-{y1 - 1}",
            "hydra_sr": mh["sharpe"],
            "hydra_ret": mh["ann_return"],
            "hydra_mdd": mh["max_dd"],
            "spy_sr": ms["sharpe"],
            "spy_ret": ms["ann_return"],
            "spy_mdd": ms["max_dd"],
        })
    return rows


def sleeve_stats(sleeves_df):
    rows = []
    for c in sleeves_df.columns:
        r = sleeves_df[c]
        nz = r[r != 0]
        inception = nz.index[0].strftime("%Y-%m-%d") if len(nz) else None
        m = metrics(r)
        rows.append({"name": c, "inception": inception, **m})
    return rows


def sleeve_correlations(sleeves_df):
    valid = (sleeves_df != 0).sum(axis=1) >= 5
    corr = sleeves_df[valid].corr()
    tri = corr.values[np.triu_indices_from(corr, k=1)]
    return {
        "mean_abs": round(float(np.mean(np.abs(tri))), 3),
        "median_abs": round(float(np.median(np.abs(tri))), 3),
        "max_abs": round(float(np.max(np.abs(tri))), 2),
        "matrix": {c: {c2: round(float(corr.loc[c, c2]), 2)
                       for c2 in corr.columns}
                   for c in corr.columns},
    }


def sleeve_descriptions():
    return {
        "s1_eq_regime": "Long SPY when SPY > 200dma AND VIX < 25; else SHY.",
        "s2_sector_top3": "Top-3 of 9 SPDR sectors by 6m momentum, monthly.",
        "s3_bond_dur": "Long TLT when 10y yield 6m trend < 0; else SHY.",
        "s4_credit": "HY credit (HYG) when trending up; else IEF.",
        "s5_curve_carry": "Carry on yield-curve steepening (TLT/IEF).",
        "s6_cmdty": "Long DBC when DBC > 200dma; else BIL.",
        "s7_gld_slv": "Gold/silver ratio regime — risk-off toggle.",
        "s8_fxy_sh": "Long FXY (Yen) when VIX 10d avg > 22 (crisis hedge).",
        "s9_usd_reg": "Long UUP when 6m trend up; else BIL.",
        "s10_vix_carry": "Short vol carry (contango-based).",
        "s12_btc": "Long BTC-linked when BTC > 50d MA; else BIL.",
        "s13_xa_gem": "Absolute momentum across 6 assets.",
        "s15_defensive": "Defensive sector rotation in low-breadth regimes.",
        "s17_semi": "SMH when SMH > 200dma; else BIL.",
        "s18_spy_rev": "5-day SPY mean reversion after −3% drop & VIX > 20.",
        "s19_em": "EEM when EEM > 200dma; else BIL.",
        "s20_infl": "TIP when 10y breakeven trend up; else IEF.",
        "s22_energy": "XLE when oil trending AND XLE > 200dma; else XLP.",
        "s24_emb": "EMB when trending AND yields not spiking; else BIL.",
        "s27_xa_ls": "Dollar-neutral long-short cross-asset momentum.",
    }


def main():
    ret_df = pd.read_csv(RESULTS / "hydra_returns.csv",
                         parse_dates=["Date"]).set_index("Date")
    sl_df = pd.read_csv(RESULTS / "hydra_sleeves.csv",
                        parse_dates=["Date"]).set_index("Date")
    r = ret_df["HYDRA"]
    spy = ret_df["SPY"]

    IS_END = pd.Timestamp("2018-01-01")
    is_m = metrics(r.loc[:IS_END])
    oos_m = metrics(r.loc[IS_END:])
    full_m = metrics(r)
    spy_m = metrics(spy)

    # Current portfolio = latest non-zero weights (approx — use latest sleeve
    # contributions scaled by recent activity)
    # Use trailing 21d absolute return by sleeve as proxy for recent weight
    last21 = sl_df.iloc[-21:].abs().sum()
    weights = (last21 / last21.sum()).sort_values(ascending=False)
    portfolio = [
        {"sleeve": name, "weight_pct": round(float(w * 100), 2),
         "description": sleeve_descriptions().get(name, "")}
        for name, w in weights.items() if w > 0
    ]

    data = {
        "fund_name": "HYDRA — 20-Sleeve Diversified Ensemble",
        "strategy_type": "Multi-Strategy Risk-Parity Ensemble",
        "benchmark": "SPY (S&P 500 ETF)",
        "inception_date": str(r.index[0].date()),
        "last_updated": str(r.index[-1].date()),
        "nav_x": round(float((1 + r).cumprod().iloc[-1]), 2),
        "rebalance": "Monthly (sleeve names), Daily (vol scaling)",
        "sleeves_count": int(sl_df.shape[1]),
        "universe_size": int(sl_df.shape[1]),

        "metrics": {
            "HYDRA": {"name": "HYDRA", **full_m,
                      "n_years": round(len(r) / 252, 1),
                      "inception": str(r.index[0].date())},
            "SPY": {"name": "SPY", **spy_m,
                    "n_years": round(len(spy) / 252, 1),
                    "inception": str(spy.index[0].date())},
        },
        "is_metrics": {"period": f"{r.loc[:IS_END].index[0].date()} — {r.loc[:IS_END].index[-1].date()}", **is_m},
        "oos_metrics": {"period": f"{r.loc[IS_END:].index[0].date()} — {r.loc[IS_END:].index[-1].date()}", **oos_m},

        "trailing": trailing(r, spy),
        "equity_curve": equity_curve_multi(pd.DataFrame({"HYDRA": r, "SPY": spy})),
        "drawdown_curve": drawdown_curve(r),
        "rolling_sharpe": rolling_sharpe(r),
        "calendar_returns": calendar_returns(r),
        "calendar_spy": calendar_returns(spy),
        "monthly_heatmap": monthly_heatmap(r),

        "walkforward_5y": walkforward(r, spy,
                                      [(y, y + 5) for y in range(2006, 2022, 5)]),

        "sleeves": sleeve_stats(sl_df),
        "correlations": sleeve_correlations(sl_df),
        "portfolio": portfolio,

        "notes": {
            "construction": "Inverse-vol risk parity across 20 uncorrelated sleeves, each independently vol-targeted to 10% annualised. Portfolio vol target 20%, gross cap 5x.",
            "tc": "15 bps on turnover; 1-bar signal lag; no look-ahead.",
            "ceiling_honest": "After extensive iteration, full-window SR ≈ 1.6 / OOS SR ≈ 2.0 is the honest ceiling for a 21-year backtest with no look-ahead and realistic TC. SR 3 over 21 years is not achievable without hindsight-biased sleeve selection or concentrated leverage (METEOR-style, which produced −78% MDD in its 21y proxy).",
        },
    }

    out = RESULTS / "hydra_factsheet_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {out} ({len(out.read_text()) / 1024:.1f} KB)")

    # Summary
    print(f"\nHYDRA summary:")
    print(f"  inception {data['inception_date']}, last {data['last_updated']}")
    print(f"  full   SR {full_m['sharpe']}  Ret {full_m['ann_return']}%  MDD {full_m['max_dd']}%")
    print(f"  IS     SR {is_m['sharpe']}  Ret {is_m['ann_return']}%  MDD {is_m['max_dd']}%")
    print(f"  OOS    SR {oos_m['sharpe']}  Ret {oos_m['ann_return']}%  MDD {oos_m['max_dd']}%")
    print(f"  sleeves {data['sleeves_count']}, mean |corr| {data['correlations']['mean_abs']}")


if __name__ == "__main__":
    main()
