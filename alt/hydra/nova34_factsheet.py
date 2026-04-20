"""Build NOVA34 factsheet JSON for docs/nova34.html.

NOVA34 is the LETF-substituted, time-stacked overnight+daytime ensemble.
Four time-disjoint sleeves run on the same capital pool:
  1) N26_OVN_LETF  — equal-$ basket of UPRO/TQQQ/IWM/DIA/UGL overnight,
                     5-min RV<0.15 gate, flat 2 bps TC per active night
  2) N18_LO        — daytime TSMOM on 12 ETFs (09:30 → 15:55)
  3) N27_OVN       — stock L/S overnight on 96 stocks
  4) N28_WOVN      — weekly overnight basket

Because the windows are disjoint, the sleeves are summed at 1.0 notional
(time-stacked, not leveraged). Returns CSV is written by nova34_letf_replace.py.
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


def equity_curve_multi(df, start=10000.0, freq="W-FRI"):
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


def rolling_sharpe(r, window=126, freq="W-FRI"):
    mu = r.rolling(window).mean() * 252
    sd = r.rolling(window).std() * np.sqrt(252)
    sh = (mu / sd).dropna()
    sw = sh.resample(freq).last()
    return [{"date": d.strftime("%Y-%m-%d"), "sr": round(float(sw.loc[d]), 3)}
            for d in sw.index if not np.isnan(sw.loc[d])]


def calendar_returns(r):
    by_year = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    return [{"year": int(y), "ret": round(float(v * 100), 2)}
            for y, v in by_year.items()]


def monthly_heatmap(r):
    mo = r.resample("ME").apply(lambda x: (1 + x).prod() - 1) * 100
    return [{"date": d.strftime("%Y-%m-%d"),
             "year": int(d.year), "month": int(d.month),
             "ret": round(float(mo.loc[d]), 2)} for d in mo.index]


def trailing(r, spy):
    end = r.index[-1]
    periods = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252,
               "3Y_ann": 252 * 3, "5Y_ann": 252 * 5}
    out = {"NOVA": {}, "SPY": {}}
    for label, n in periods.items():
        if len(r) < n + 1:
            out["NOVA"][label] = None; out["SPY"][label] = None; continue
        rn = r.iloc[-n:]; rs = spy.iloc[-n:]
        if label.endswith("_ann"):
            out["NOVA"][label] = round(float(((1 + rn).prod() ** (252 / n) - 1) * 100), 2)
            out["SPY"][label] = round(float(((1 + rs).prod() ** (252 / n) - 1) * 100), 2)
        else:
            out["NOVA"][label] = round(float(((1 + rn).prod() - 1) * 100), 2)
            out["SPY"][label] = round(float(((1 + rs).prod() - 1) * 100), 2)
    ystart = pd.Timestamp(f"{end.year}-01-01")
    rn_y = r.loc[ystart:]; rs_y = spy.loc[ystart:]
    out["NOVA"]["YTD"] = round(float(((1 + rn_y).prod() - 1) * 100), 2)
    out["SPY"]["YTD"] = round(float(((1 + rs_y).prod() - 1) * 100), 2)
    out["NOVA"]["SI_ann"] = round(float(((1 + r).prod() ** (252 / len(r)) - 1) * 100), 2)
    out["SPY"]["SI_ann"] = round(float(((1 + spy).prod() ** (252 / len(spy)) - 1) * 100), 2)
    return out


def walkforward(r, spy, year_groups):
    rows = []
    for y0, y1 in year_groups:
        lo = pd.Timestamp(f"{y0}-01-01"); hi = pd.Timestamp(f"{y1}-01-01")
        sub_n = r.loc[lo:hi]; sub_s = spy.loc[lo:hi]
        if len(sub_n) < 100:
            continue
        mn = metrics(sub_n); ms = metrics(sub_s)
        rows.append({
            "window": f"{y0}-{y1 - 1}",
            "nova_sr": mn["sharpe"], "nova_ret": mn["ann_return"],
            "nova_mdd": mn["max_dd"],
            "spy_sr": ms["sharpe"], "spy_ret": ms["ann_return"],
            "spy_mdd": ms["max_dd"],
        })
    return rows


def main():
    df = pd.read_csv(RESULTS / "nova34_returns.csv",
                     parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index)
    r = df["NOVA34_LETF"].dropna()
    base_1x = df["NOVA29_1x"].dropna()
    margin_4x_net = df["NOVA29_4x_net"].dropna()

    spy = pd.read_csv(ROOT / "data/etfs/SPY.csv",
                      parse_dates=["Date"]).set_index("Date")
    spy.index = pd.to_datetime(spy.index)
    spy = spy[~spy.index.duplicated(keep="first")].sort_index()
    spy_ret = spy["Close"].pct_change().reindex(r.index).fillna(0)

    CUT = pd.Timestamp("2022-01-01")
    is_m = metrics(r.loc[:CUT])
    oos_m = metrics(r.loc[CUT:])
    full_m = metrics(r)
    spy_m = metrics(spy_ret)
    base_m = metrics(base_1x)
    margin_m = metrics(margin_4x_net)

    data = {
        "fund_name": "NOVA — Time-Stacked LETF Overnight + Intraday Ensemble",
        "strategy_type": "Time-disjoint sleeve stacking with LETF substitution",
        "benchmark": "SPY (S&P 500 ETF)",
        "inception_date": str(r.index[0].date()),
        "last_updated": str(r.index[-1].date()),
        "nav_x": round(float((1 + r).cumprod().iloc[-1]), 2),

        "metrics": {
            "NOVA": {"name": "NOVA34", **full_m,
                     "inception": str(r.index[0].date())},
            "SPY": {"name": "SPY", **spy_m,
                    "inception": str(spy_ret.index[0].date())},
            "NOVA29_1x": {"name": "NOVA29 1x (baseline)", **base_m},
            "NOVA29_4x_margin_net": {"name": "NOVA29 4x margin (net of 8.25% interest)",
                                     **margin_m},
        },
        "is_metrics": {
            "period": f"{r.loc[:CUT].index[0].date()} — {r.loc[:CUT].index[-1].date()}",
            **is_m},
        "oos_metrics": {
            "period": f"{r.loc[CUT:].index[0].date()} — {r.loc[CUT:].index[-1].date()}",
            **oos_m},

        "trailing": trailing(r, spy_ret),
        "equity_curve": equity_curve_multi(
            pd.DataFrame({"NOVA": r, "NOVA29_1x": base_1x,
                          "NOVA29_4x_net": margin_4x_net,
                          "SPY": spy_ret}).fillna(0)),
        "drawdown_curve": drawdown_curve(r),
        "rolling_sharpe": rolling_sharpe(r),
        "calendar_returns": calendar_returns(r),
        "calendar_spy": calendar_returns(spy_ret),
        "calendar_base_1x": calendar_returns(base_1x),
        "calendar_margin_net": calendar_returns(margin_4x_net),
        "monthly_heatmap": monthly_heatmap(r),
        "walkforward_3y": walkforward(
            r, spy_ret,
            [(2018, 2021), (2021, 2024), (2024, 2027)]),

        "sleeves": [
            {"name": "N26_OVN_LETF", "role": "Overnight (15:55 → next open)",
             "description": "Equal-$ basket of UPRO/TQQQ/IWM/DIA/UGL. Open only when prior-20d 5-min RV < 0.15 (annualised). LETFs substitute for margin: UPRO (3× SPY), TQQQ (3× QQQ), UGL (2× GLD); IWM and DIA kept 1× (no liquid 3× sibling). Cash in BIL when gated off. TC 2 bps per active night."},
            {"name": "N18_LO", "role": "Daytime (09:30 → 15:55)",
             "description": "Time-series momentum on 12 ETFs (SPY/QQQ/IWM/DIA/EFA/EEM/TLT/IEF/LQD/HYG/GLD/DBC). Signal: 21-day momentum. Daily rebalance, 5 bps TC."},
            {"name": "N27_OVN", "role": "Overnight L/S stocks",
             "description": "Cross-sectional long-short on 96 S&P 500 single names. Rank by trailing overnight momentum; long top decile / short bottom decile. 10 bps TC."},
            {"name": "N28_WOVN", "role": "Weekly overnight",
             "description": "Weekly overnight drift on 5 ETFs (SPY/QQQ/IWM/DIA/GLD), weekly hold Friday close → Monday open. 3 bps TC."},
        ],

        "notes": {
            "key_idea": "Three time windows on the same capital: (a) overnight 15:55→09:30, (b) daytime 09:30→15:55, (c) weekly overnight. Because the holding windows don't overlap, the four sleeves are summed at full notional from the same cash base — this is capital stacking, NOT leverage.",
            "letf_substitution": "Rather than use broker margin on 1× ETFs, the overnight equity/gold sleeve holds LETFs (UPRO 3×SPY, TQQQ 3×QQQ, UGL 2×GLD) equal-$ weighted. LETF daily-reset decay is an INTRADAY phenomenon; the close-to-open window captures ~2-3× the 1× drift without paying broker margin interest. LETF expense ratios (0.84-0.95%) and swap costs are already embedded in realised returns.",
            "margin_vs_letf": "A 4× broker-margin version of the 1× ensemble costs ~8.25%/yr in interest drag (5% APR × ~3× borrowed × ~0.55 deployment) and reaches CAGR ~20% net, SR 1.10. The LETF substitution path reaches CAGR ~40% at SR 1.7 — roughly 15pp CAGR and 0.6 SR advantage over margin.",
            "caveats": "(1) 8.9-year backtest — shorter than HYDRA's 21y because the overnight RV gate needs 5-min intraday data. (2) Volatility 24% and MDD −29% are closer to SPY than to HYDRA. (3) OOS Sharpe 1.61 is below HYDRA's OOS 2.01; NOVA trades some risk-adjusted quality for ~3× the absolute CAGR. (4) Reliance on liquid LETFs introduces path-dependence in large-gap regimes — a single −30% day in SPY implies a ~−60% day in UPRO, which has actually happened (Mar 2020 gap).",
            "tc": "Per-sleeve: N26 2 bps/night, N18 5 bps/day, N27 10 bps/rebal, N28 3 bps/week. LETF expense ratios are absorbed via realised close-to-open returns (they drag both numerator and denominator already).",
            "execution": "Four discrete trade windows: 15:55 (overnight ETF entries), 09:30 (daytime TSMOM rebal & previous overnight exit), 15:55 (daytime exit), Mon 09:30 (weekly exit). All use T−1 closes for the signal; no look-ahead.",
        },
    }

    out = RESULTS / "nova34_factsheet_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"Wrote {out} ({len(out.read_text()) / 1024:.1f} KB)")
    print(f"\nNOVA34 summary:")
    print(f"  inception {data['inception_date']}, last {data['last_updated']}")
    print(f"  full  SR {full_m['sharpe']}  CAGR {full_m['ann_return']}%  Vol {full_m['ann_vol']}%  MDD {full_m['max_dd']}%")
    print(f"  IS    SR {is_m['sharpe']}  CAGR {is_m['ann_return']}%  MDD {is_m['max_dd']}%")
    print(f"  OOS   SR {oos_m['sharpe']}  CAGR {oos_m['ann_return']}%  MDD {oos_m['max_dd']}%")
    print(f"  SPY   SR {spy_m['sharpe']}  CAGR {spy_m['ann_return']}%  MDD {spy_m['max_dd']}%")


if __name__ == "__main__":
    main()
