"""Generate SUMMIT factsheet payloads from a panel + the committed strategy.

Writes three JSON files consumed by docs/summit.html:
  docs/summit_data.json    -- equity curves (value & money-multiple) for several
                              start horizons (ITD / 10y / 7y / 5y / 3y / 1y),
                              monthly-sampled, for SUMMIT, QQQ-DCA, SPY-DCA and
                              cumulative contributions.
  docs/summit_returns.json -- returns table: for each horizon, the final money
                              multiple and money-weighted IRR of SUMMIT vs
                              QQQ-DCA and SPY-DCA, plus per-calendar-year DCA
                              value growth.
  docs/summit_signal.json  -- current regime + next biweekly picks.

This is the single source of truth for the live page; the daily cron calls
build_factsheet() after refreshing prices.
"""
import json
import os

import numpy as np
import pandas as pd

import data as data_mod
import engine
import strategy_dca

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs")

HORIZONS = [("ITD", None), ("10y", 10), ("7y", 7), ("5y", 5), ("3y", 3),
            ("1y", 1)]
ANCHOR = "2006-01-03"


def _irr(value: pd.Series, invested: pd.Series, freq_guess=26) -> float:
    from scipy.optimize import brentq
    flows = invested.diff().fillna(invested.iloc[0])
    flows = flows[flows > 0]
    if len(flows) == 0 or value.iloc[-1] <= 0:
        return float("nan")
    t_end = value.index[-1]
    yrs = np.array([(t_end - d).days / 365.25 for d in flows.index])
    amts = flows.values
    fv = float(value.iloc[-1])

    def f(r):
        return (amts * (1 + r) ** yrs).sum() - fv
    try:
        return brentq(f, -0.95, 5.0)
    except ValueError:
        return float("nan")


def _run_all(P, k=2, cost_bps=5.0):
    """Run SUMMIT + QQQ + SPY DCA once over the full anchor->end window.
    Returns dict of daily Series aligned on the strategy index."""
    s = strategy_dca.build_scores(P)
    end = P["close"].index[-1]
    res = engine.run_dca(P["open"], P["close"], s, P["member"], k=k,
                         every=10, start=ANCHOR, end=end, cost_bps=cost_bps)
    qqq = data_mod.load_benchmark("QQQ")
    spy = data_mod.load_benchmark("SPY")
    bq = engine.run_benchmark_dca(qqq.loc[:end], start=ANCHOR, end=end,
                                  cost_bps=cost_bps)
    bs = engine.run_benchmark_dca(spy.loc[:end], start=ANCHOR, end=end,
                                  cost_bps=cost_bps)
    idx = res.value.index
    return {
        "summit": res.value, "invested": res.invested,
        "qqq": bq.value.reindex(idx).ffill(),
        "spy": bs.value.reindex(idx).ffill(),
        "qqq_inv": bq.invested.reindex(idx).ffill(),
        "spy_inv": bs.invested.reindex(idx).ffill(),
    }


def _curve_for_horizon(P, years, k=2, cost_bps=5.0):
    """A *fresh* DCA started `years` ago (None = since anchor). Returns the
    monthly-sampled value/multiple curves for each series."""
    end = P["close"].index[-1]
    start = ANCHOR if years is None else \
        (end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    s = strategy_dca.build_scores(P)
    res = engine.run_dca(P["open"], P["close"], s, P["member"], k=k,
                         every=10, start=start, end=end, cost_bps=cost_bps)
    qqq = data_mod.load_benchmark("QQQ")
    spy = data_mod.load_benchmark("SPY")
    bq = engine.run_benchmark_dca(qqq.loc[:end], start=start, end=end,
                                  cost_bps=cost_bps)
    bs = engine.run_benchmark_dca(spy.loc[:end], start=start, end=end,
                                  cost_bps=cost_bps)
    idx = res.value.index
    inv = res.invested
    out = {
        "summit": res.value, "qqq": bq.value.reindex(idx).ffill(),
        "spy": bs.value.reindex(idx).ffill(), "invested": inv,
    }
    metrics = {
        "summit_mult": float(res.value.iloc[-1] / inv.iloc[-1]),
        "summit_irr": _irr(res.value, inv),
        "qqq_mult": float(bq.value.iloc[-1] / bq.invested.iloc[-1]),
        "qqq_irr": _irr(bq.value, bq.invested),
        "spy_mult": float(bs.value.iloc[-1] / bs.invested.iloc[-1]),
        "spy_irr": _irr(bs.value, bs.invested),
        "start": str(pd.Timestamp(start).date()), "end": str(end.date()),
    }
    return out, metrics


def build_factsheet(P=None, k=2, cost_bps=5.0, write=True):
    if P is None:
        P = data_mod.build_panel()

    # ---- curves + returns table per horizon ----
    curves = {}
    returns_rows = []
    for label, yrs in HORIZONS:
        out, metrics = _curve_for_horizon(P, yrs, k, cost_bps)
        m = out["summit"].resample("ME").last().dropna()
        def samp(series):
            return [round(float(v), 1) for v in
                    series.resample("ME").last().reindex(m.index).ffill().values]
        curves[label] = {
            "dates": [d.strftime("%Y-%m") for d in m.index],
            "summit": samp(out["summit"]), "qqq": samp(out["qqq"]),
            "spy": samp(out["spy"]), "invested": samp(out["invested"]),
        }
        returns_rows.append({"horizon": label, **metrics})

    # ---- calendar-year value growth of the ITD portfolio ----
    full = _run_all(P, k, cost_bps)
    yr_rows = []
    val, inv = full["summit"], full["invested"]
    for yr, grp in val.groupby(val.index.year):
        q = full["qqq"].loc[grp.index]
        sp = full["spy"].loc[grp.index]
        # within-year value growth net of that year's contributions
        def yr_ret(v):
            contrib = inv.loc[grp.index].iloc[-1] - inv.loc[grp.index].iloc[0]
            v0 = v.iloc[0]
            return float((v.iloc[-1] - contrib) / v0 - 1) if v0 > 0 else float("nan")
        yr_rows.append({"year": int(yr),
                        "summit": yr_ret(grp), "qqq": yr_ret(q), "spy": yr_ret(sp)})

    # ---- current signal (computed from the supplied panel) ----
    scores = strategy_dca.build_scores(P)
    member, close = P["member"], P["close"]
    enough = close.notna().rolling(252).count() >= 252
    row = (scores.iloc[-1].where(member.iloc[-1]).where(enough.iloc[-1])
           .dropna().sort_values(ascending=False))
    regime = ("RISK-OFF (rebound sleeve)"
              if bool(strategy_dca.risk_off(P).iloc[-1])
              else "RISK-ON (momentum sleeve)")
    signal = {
        "as_of": str(close.index[-1].date()),
        "regime": regime,
        "picks": list(row.index[:k]),
    }

    headline = {
        "itd_mult": returns_rows[0]["summit_mult"],
        "itd_irr": returns_rows[0]["summit_irr"],
        "qqq_mult": returns_rows[0]["qqq_mult"],
        "spy_mult": returns_rows[0]["spy_mult"],
        "as_of": signal["as_of"],
    }

    data_json = {"horizons": [h[0] for h in HORIZONS], "curves": curves,
                 "headline": headline}
    returns_json = {"table": returns_rows, "years": yr_rows,
                    "as_of": signal["as_of"]}

    if write:
        os.makedirs(DOCS, exist_ok=True)
        _dump(data_json, os.path.join(DOCS, "summit_data.json"))
        _dump(returns_json, os.path.join(DOCS, "summit_returns.json"))
        _dump(signal, os.path.join(DOCS, "summit_signal.json"))
    return data_json, returns_json, signal


def _sanitize(o):
    """Replace NaN/Inf with None so the output is strict, browser-parseable
    JSON (json.dump otherwise emits bare NaN tokens that JSON.parse rejects)."""
    if isinstance(o, float):
        return None if (np.isnan(o) or np.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    return o


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(_sanitize(obj), f, allow_nan=False)


if __name__ == "__main__":
    d, r, s = build_factsheet()
    print("signal:", s)
    print("ITD: SUMMIT %.2fx  QQQ %.2fx  SPY %.2fx" % (
        r["table"][0]["summit_mult"], r["table"][0]["qqq_mult"],
        r["table"][0]["spy_mult"]))
    for row in r["table"]:
        print(f"  {row['horizon']:>4}: SUMMIT {row['summit_mult']:.2f}x "
              f"({row['summit_irr']*100:.1f}%)  QQQ {row['qqq_mult']:.2f}x  "
              f"SPY {row['spy_mult']:.2f}x")
