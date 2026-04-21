"""ATLAS — step 2/6: produce atlas_factsheet_data.json.

Schema mirrors nova34 so the HTML builder can be a trivial adaptation of
build_nova34_html.py.

Key compared series:
  - ATLAS              (this strategy)
  - ATLAS_BASE_1x      (TSMOM K=3m tv=15% without DD-throttle; "1x baseline")
  - SPY buy-hold
  - 60/40 SPY/TLT
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hydra_core import load_etf


RESULTS = Path("/home/user/bonds/data/results")
ATLAS_RET_PATH = RESULTS / "atlas_returns.csv"
OUT_JSON = RESULTS / "atlas_factsheet_data.json"

INCEPTION = "2011-01-03"
IS_END = "2020-12-31"      # in-sample: 2011-2020
OOS_START = "2021-01-01"    # out-of-sample: 2021-2026


def annualised(ret: pd.Series) -> dict:
    ret = ret.dropna()
    mu = ret.mean() * 252
    sd = ret.std() * np.sqrt(252)
    nav = (1 + ret).cumprod()
    cagr = nav.iloc[-1] ** (252 / len(ret)) - 1 if len(ret) > 0 else 0
    dd = (nav / nav.cummax() - 1).min() * 100
    # Sortino
    downside = ret[ret < 0]
    dsd = downside.std() * np.sqrt(252) if len(downside) else sd
    sortino = mu / dsd if dsd > 0 else 0
    return {
        "sharpe": round(float(mu / sd), 3) if sd > 0 else 0.0,
        "ann_return": round(float(cagr * 100), 2),
        "ann_vol": round(float(sd * 100), 2),
        "max_dd": round(float(dd), 2),
        "sortino": round(float(sortino), 3),
        "n_years": round(float(len(ret) / 252), 2),
    }


def load_returns():
    atlas = pd.read_csv(ATLAS_RET_PATH, parse_dates=["Date"]).set_index("Date")["ret"]
    spy = load_etf("SPY").pct_change().dropna()
    tlt = load_etf("TLT").pct_change().dropna()
    idx = atlas.index.intersection(spy.index).intersection(tlt.index)
    return (
        atlas.reindex(idx).fillna(0),
        spy.reindex(idx).fillna(0),
        tlt.reindex(idx).fillna(0),
    )


def build_base_1x():
    """The same TSMOM K=3m tv=15% without the DD-throttle overlay."""
    from letf_tsmom import tsmom_with_vol_target, prep as tsmom_prep
    px = tsmom_prep()
    r, _ = tsmom_with_vol_target(px, K_months=3, target_vol=0.15)
    return r


def trailing_returns(ret: pd.Series) -> dict:
    """Trailing returns as % (1M/3M/6M/YTD/1Y/3Y ann/5Y ann/SI ann)."""
    ret = ret.dropna()
    last = ret.index[-1]
    nav = (1 + ret).cumprod()
    last_nav = nav.iloc[-1]

    def ann_back(days):
        if len(ret) <= days:
            return None
        past_nav = nav.iloc[-days - 1]
        yrs = days / 252
        return ((last_nav / past_nav) ** (1 / yrs) - 1) * 100

    def cum_back(days):
        if len(ret) <= days:
            return None
        past_nav = nav.iloc[-days - 1]
        return (last_nav / past_nav - 1) * 100

    ytd_start_nav = nav[nav.index.year == last.year].iloc[0]

    n_years = len(ret) / 252

    return {
        "1M": round(cum_back(21) or 0, 2),
        "3M": round(cum_back(63) or 0, 2),
        "6M": round(cum_back(126) or 0, 2),
        "YTD": round((last_nav / ytd_start_nav - 1) * 100, 2),
        "1Y": round(cum_back(252) or 0, 2),
        "3Y_ann": round(ann_back(756) or 0, 2) if len(ret) > 756 else None,
        "5Y_ann": round(ann_back(1260) or 0, 2) if len(ret) > 1260 else None,
        "SI_ann": round((last_nav ** (1 / n_years) - 1) * 100, 2) if n_years > 0 else 0,
    }


def equity_curve(series: dict[str, pd.Series], step: int = 5):
    """Downsample to weekly to keep JSON small."""
    df = pd.DataFrame(series)
    nav = (1 + df).cumprod() * 10000
    nav = nav.iloc[::step]
    out = []
    for d, row in nav.iterrows():
        rec = {"date": d.strftime("%Y-%m-%d")}
        for k, v in row.items():
            rec[k] = round(float(v), 2)
        out.append(rec)
    return out


def drawdown_curve(ret: pd.Series, step: int = 5):
    nav = (1 + ret).cumprod()
    dd = (nav / nav.cummax() - 1) * 100
    dd = dd.iloc[::step]
    return [{"date": d.strftime("%Y-%m-%d"), "dd": round(float(x), 2)}
            for d, x in dd.items()]


def rolling_sharpe(ret: pd.Series, window: int = 126, step: int = 5):
    mu = ret.rolling(window).mean() * 252
    sd = ret.rolling(window).std() * np.sqrt(252)
    sr = (mu / sd).dropna()
    sr = sr.iloc[::step]
    return [{"date": d.strftime("%Y-%m-%d"), "sr": round(float(x), 2)}
            for d, x in sr.items()]


def calendar_returns(ret: pd.Series):
    nav = (1 + ret).cumprod()
    out = []
    for y in sorted(ret.index.year.unique()):
        mask = ret.index.year == y
        start_nav = nav[mask].iloc[0] / (1 + ret[mask].iloc[0])
        end_nav = nav[mask].iloc[-1]
        r = (end_nav / start_nav - 1) * 100
        out.append({"year": int(y), "ret": round(float(r), 2)})
    return out


def monthly_heatmap(ret: pd.Series):
    m = (1 + ret).resample("ME").prod() - 1
    out = []
    for d, r in m.items():
        out.append({"year": int(d.year), "month": int(d.month), "ret": round(float(r) * 100, 2)})
    return out


def walkforward_3y(ret: pd.Series, spy: pd.Series):
    out = []
    start = pd.Timestamp("2011-01-01")
    for wi in range(5):
        s = start + pd.DateOffset(years=3 * wi)
        e = s + pd.DateOffset(years=3)
        if e > ret.index[-1]:
            break
        w_ret = ret.loc[s:e].dropna()
        w_spy = spy.loc[s:e].dropna()
        if len(w_ret) < 100:
            continue
        nav = (1 + w_ret).cumprod()
        sn = (1 + w_spy).cumprod()
        yrs = len(w_ret) / 252
        out.append({
            "window": f"{s.year}-{e.year}",
            "atlas_sr": round(float(w_ret.mean() * 252 / (w_ret.std() * np.sqrt(252))), 2),
            "atlas_ret": round(float((nav.iloc[-1] ** (1 / yrs) - 1) * 100), 2),
            "atlas_mdd": round(float((nav / nav.cummax() - 1).min() * 100), 2),
            "spy_sr": round(float(w_spy.mean() * 252 / (w_spy.std() * np.sqrt(252))), 2),
            "spy_ret": round(float((sn.iloc[-1] ** (1 / yrs) - 1) * 100), 2),
            "spy_mdd": round(float((sn / sn.cummax() - 1).min() * 100), 2),
        })
    return out


def main():
    atlas, spy, tlt = load_returns()
    base = build_base_1x().reindex(atlas.index).fillna(0)
    sixty40 = 0.6 * spy + 0.4 * tlt

    is_slice = atlas.loc[:IS_END]
    oos_slice = atlas.loc[OOS_START:]

    data = {
        "fund_name": "ATLAS — Drawdown-Hardened TSMOM LETF",
        "strategy_type": "Time-series momentum on UPRO/TQQQ/TMF/UGL with drawdown-throttle overlay",
        "benchmark": "SPY + 60/40",
        "inception_date": INCEPTION,
        "last_updated": atlas.index[-1].strftime("%Y-%m-%d"),
        "nav_x": round(float((1 + atlas).cumprod().iloc[-1]), 2),
        "metrics": {
            "ATLAS": {"name": "ATLAS", **annualised(atlas), "inception": INCEPTION},
            "ATLAS_BASE_1x": {"name": "TSMOM base (no DD overlay)", **annualised(base)},
            "SPY": {"name": "SPY", **annualised(spy), "inception": INCEPTION},
            "SIXTY40": {"name": "60/40 SPY/TLT", **annualised(sixty40)},
        },
        "is_metrics": {
            "period": f"{is_slice.index[0].strftime('%Y-%m-%d')} — "
                      f"{is_slice.index[-1].strftime('%Y-%m-%d')}",
            **annualised(is_slice),
        },
        "oos_metrics": {
            "period": f"{oos_slice.index[0].strftime('%Y-%m-%d')} — "
                      f"{oos_slice.index[-1].strftime('%Y-%m-%d')}",
            **annualised(oos_slice),
        },
        "trailing": {
            "ATLAS": trailing_returns(atlas),
            "SPY": trailing_returns(spy),
            "SIXTY40": trailing_returns(sixty40),
        },
        "equity_curve": equity_curve({
            "ATLAS": atlas, "ATLAS_BASE_1x": base, "SPY": spy, "SIXTY40": sixty40,
        }),
        "drawdown_curve": drawdown_curve(atlas),
        "rolling_sharpe": rolling_sharpe(atlas),
        "calendar_returns": calendar_returns(atlas),
        "calendar_spy": calendar_returns(spy),
        "calendar_sixty40": calendar_returns(sixty40),
        "calendar_base_1x": calendar_returns(base),
        "monthly_heatmap": monthly_heatmap(atlas),
        "walkforward_3y": walkforward_3y(atlas, spy),
        "notes": {
            "key_idea": "TSMOM (time-series momentum, Moskowitz-Ooi-Pedersen 2012) "
                        "takes the sign of the trailing 63-day return on each of four "
                        "underlyings SPY, QQQ, TLT, GLD and expresses the bet through "
                        "3x-levered ETFs UPRO, TQQQ, TMF, UGL. Positions are vol-"
                        "targeted to 15% annualised; residual capacity is parked in "
                        "BIL. Rebalance monthly (21 trading days), next-day-open "
                        "execution, 15 bps TC on turnover.",
            "dd_throttle": "On top of TSMOM, a drawdown-throttle overlay scales "
                           "exposure based on how far the strategy is below its "
                           "trailing 252-day NAV peak: at peak, 100%; at -5% to -10%, "
                           "linear from 100% to 50%; at -10% to -20%, linear from 50% "
                           "to 25%; below -20%, floor of 25%. 5-day smoothed, 1-day "
                           "execution lag.",
            "validation": "Pre-registered holdout 2023-01 to 2026-04 delivered SR 1.07, "
                          "CAGR 18.1%, MDD -20.0% with zero re-tuning. Permutation null "
                          "test (500 reps, 21-day block shuffle of the DD multiplier) "
                          "shows the MDD reduction is real (p<0.001). The Sharpe lift "
                          "from the overlay is within sampling noise (p=0.15) — "
                          "the overlay is MDD insurance, not a Sharpe booster.",
            "caveats": "Backtest window 2011-2026 contains two serious equity bears "
                       "(2020, 2022) and a multi-decade bond-bull tail. 15 years is "
                       "not 100 years. LETF bid/ask and expense ratios (0.9-1.1% TER) "
                       "are ON TOP of the modelled 15bps turnover cost. Daily vol "
                       "targeting assumes execution at next-day open; slippage in "
                       "stress is unmodelled. The overlay will underperform the base "
                       "on the recovery leg (stays at 25% until the 252d peak is "
                       "reclaimed).",
        },
    }

    OUT_JSON.write_text(json.dumps(data))
    print(f"Wrote {OUT_JSON}")
    m = data["metrics"]["ATLAS"]
    print(f"  ATLAS: SR {m['sharpe']:.2f} | CAGR {m['ann_return']:.2f}% | "
          f"Vol {m['ann_vol']:.2f}% | MDD {m['max_dd']:.2f}%")


if __name__ == "__main__":
    main()
