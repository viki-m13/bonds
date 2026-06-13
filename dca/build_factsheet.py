"""Generate factsheet JSON payloads for a DCA strategy (SUMMIT or ROTATOR).

For a strategy config (scores builder, optional sell builder, k, cost) this
writes three JSON files consumed by the matching docs/<prefix>.html page:
  docs/<prefix>_data.json    -- equity curves (value & money-multiple) for
                                several start horizons.
  docs/<prefix>_returns.json -- returns-by-horizon table + calendar-year
                                value growth.
  docs/<prefix>_signal.json  -- current regime + next picks.

Both strategies run in the *same* harness on the *same* point-in-time S&P 500
panel, so their factsheets are directly comparable.
"""
import json
import os

import numpy as np
import pandas as pd

import data as data_mod
import engine

DOCS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "docs")
HORIZONS = [("ITD", None), ("10y", 10), ("7y", 7), ("5y", 5), ("3y", 3),
            ("1y", 1)]
ANCHOR = "2006-01-03"


def _irr(value: pd.Series, invested: pd.Series) -> float:
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


def _bench(end):
    qqq = data_mod.load_benchmark("QQQ")
    spy = data_mod.load_benchmark("SPY")
    return qqq.loc[:end], spy.loc[:end]


def _curve_for_horizon(P, years, cfg):
    end = P["close"].index[-1]
    start = ANCHOR if years is None else \
        (end - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    s = cfg["scores"](P)
    sell = cfg["sell"](P) if cfg.get("sell") else None
    res = engine.run_dca(P["open"], P["close"], s, P["member"], k=cfg["k"],
                         every=10, start=start, end=end,
                         cost_bps=cfg["cost_bps"], sell=sell)
    qqq, spy = _bench(end)
    bq = engine.run_benchmark_dca(qqq, start=start, end=end,
                                  cost_bps=cfg["cost_bps"])
    bs = engine.run_benchmark_dca(spy, start=start, end=end,
                                  cost_bps=cfg["cost_bps"])
    idx = res.value.index
    inv = res.invested
    out = {"summit": res.value, "qqq": bq.value.reindex(idx).ffill(),
           "spy": bs.value.reindex(idx).ffill(), "invested": inv}
    metrics = {
        "strat_mult": float(res.value.iloc[-1] / inv.iloc[-1]),
        "strat_irr": _irr(res.value, inv),
        "qqq_mult": float(bq.value.iloc[-1] / bq.invested.iloc[-1]),
        "qqq_irr": _irr(bq.value, bq.invested),
        "spy_mult": float(bs.value.iloc[-1] / bs.invested.iloc[-1]),
        "spy_irr": _irr(bs.value, bs.invested),
        "start": str(pd.Timestamp(start).date()), "end": str(end.date()),
    }
    return out, metrics


def _calendar_years(P, cfg):
    end = P["close"].index[-1]
    s = cfg["scores"](P)
    sell = cfg["sell"](P) if cfg.get("sell") else None
    res = engine.run_dca(P["open"], P["close"], s, P["member"], k=cfg["k"],
                         every=10, start=ANCHOR, end=end,
                         cost_bps=cfg["cost_bps"], sell=sell)
    qqq, spy = _bench(end)
    bq = engine.run_benchmark_dca(qqq, start=ANCHOR, end=end,
                                  cost_bps=cfg["cost_bps"])
    bs = engine.run_benchmark_dca(spy, start=ANCHOR, end=end,
                                  cost_bps=cfg["cost_bps"])
    val, inv = res.value, res.invested
    q = bq.value.reindex(val.index).ffill()
    sp = bs.value.reindex(val.index).ffill()
    rows = []
    for yr, grp in val.groupby(val.index.year):
        contrib = inv.loc[grp.index].iloc[-1] - inv.loc[grp.index].iloc[0]

        def yr_ret(v):
            v0 = v.iloc[0]
            return float((v.iloc[-1] - contrib) / v0 - 1) if v0 > 0 else float("nan")
        rows.append({"year": int(yr), "summit": yr_ret(grp),
                     "qqq": yr_ret(q.loc[grp.index]),
                     "spy": yr_ret(sp.loc[grp.index])})
    return rows


def build(cfg, P=None, write=True):
    if P is None:
        P = data_mod.build_panel()
    curves, returns_rows = {}, []
    for label, yrs in HORIZONS:
        out, metrics = _curve_for_horizon(P, yrs, cfg)
        m = out["summit"].resample("ME").last().dropna()

        def samp(series):
            return [round(float(v), 1) for v in
                    series.resample("ME").last().reindex(m.index).ffill().values]
        curves[label] = {"dates": [d.strftime("%Y-%m") for d in m.index],
                         "summit": samp(out["summit"]), "qqq": samp(out["qqq"]),
                         "spy": samp(out["spy"]), "invested": samp(out["invested"])}
        returns_rows.append({"horizon": label, **metrics})

    yr_rows = _calendar_years(P, cfg)

    # current signal
    s = cfg["scores"](P)
    member, close = P["member"], P["close"]
    enough = close.notna().rolling(252).count() >= 252
    row = (s.iloc[-1].where(member.iloc[-1]).where(enough.iloc[-1])
           .dropna().sort_values(ascending=False))
    signal = {"as_of": str(close.index[-1].date()),
              "regime": cfg["regime"](P), "picks": list(row.index[:cfg["k"]]),
              "next_label": cfg.get("next_label", "next buy")}

    headline = {"itd_mult": returns_rows[0]["strat_mult"],
                "itd_irr": returns_rows[0]["strat_irr"],
                "qqq_mult": returns_rows[0]["qqq_mult"],
                "spy_mult": returns_rows[0]["spy_mult"], "as_of": signal["as_of"]}
    data_json = {"horizons": [h[0] for h in HORIZONS], "curves": curves,
                 "headline": headline}
    returns_json = {"table": returns_rows, "years": yr_rows,
                    "as_of": signal["as_of"]}
    if write:
        pre = cfg["prefix"]
        _dump(data_json, os.path.join(DOCS, f"{pre}_data.json"))
        _dump(returns_json, os.path.join(DOCS, f"{pre}_returns.json"))
        _dump(signal, os.path.join(DOCS, f"{pre}_signal.json"))
    return data_json, returns_json, signal


def _sanitize(o):
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


# ---- strategy configs ----

def summit_cfg():
    import strategy_dca
    return {"prefix": "summit", "k": 2, "cost_bps": 5,
            "scores": strategy_dca.build_scores, "sell": None,
            "next_label": "next buy",
            "regime": lambda P: ("RISK-OFF (rebound sleeve)"
                                 if bool(strategy_dca.risk_off(P).iloc[-1])
                                 else "RISK-ON (momentum sleeve)")}


def rotator_cfg():
    import strategy_rotator as R
    return {"prefix": "rotator", "k": 3, "cost_bps": 10,
            "scores": R.build_scores, "sell": R.build_sell,
            "next_label": "holding",
            "regime": lambda P: ("CASH (SPY below 210dma)"
                                 if bool(R._spy_bear(P["close"].index).iloc[-1])
                                 else "INVESTED (top-3 leaders)")}


# back-compat shim for update_summit.py
def build_factsheet(P=None, write=True):
    return build(summit_cfg(), P=P, write=write)


if __name__ == "__main__":
    for cfg in (summit_cfg(), rotator_cfg()):
        d, r, s = build(cfg)
        print(f"[{cfg['prefix']}] {s['regime']} picks={s['picks']}")
        for row in r["table"]:
            print(f"  {row['horizon']:>4}: {row['strat_mult']:.2f}x "
                  f"({row['strat_irr']*100:.0f}%) vs QQQ {row['qqq_mult']:.2f}x "
                  f"SPY {row['spy_mult']:.2f}x")
