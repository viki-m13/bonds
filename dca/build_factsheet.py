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
    return rows, res


def _extras(P, cfg, full_res):
    """Daily-fresh validation metrics for the focused (SUMMIT) page:
    win-rate by horizon, regime windows, the cadence robustness table, and the
    current portfolio holdings. Uses the fast grid engine on the supplied
    (stitched) panel so the numbers track live data."""
    import fast
    import protocol
    fd = fast.FastData(P["open"], P["close"], P["member"])
    protocol._cache.clear()
    protocol._cache["panels"] = P
    protocol._cache["fd"] = fd
    protocol._cache["qqq"] = data_mod.load_benchmark("QQQ")
    protocol._cache["spy"] = data_mod.load_benchmark("SPY")
    scores = cfg["scores"](P)
    sell = cfg["sell"](P) if cfg.get("sell") else None
    card = protocol.evaluate_signal(scores, "live", k=cfg["k"], every=10,
                                    cost_bps=cfg["cost_bps"], sell=sell,
                                    save=False, quiet=True)
    cadence = {}
    for cname, ev in (("daily", 1), ("weekly", 5), ("biweekly", 10),
                      ("monthly", 21)):
        c = protocol.evaluate_signal(scores, "live", k=cfg["k"], every=ev,
                                     cost_bps=cfg["cost_bps"], sell=sell,
                                     save=False, quiet=True)
        cadence[cname] = {"win_qqq": c["win_qqq"], "win_spy": c["win_spy"],
                          "med_vs_qqq": c["med_vs_qqq"],
                          "worst_vs_qqq": c["worst_vs_qqq"],
                          "full_mult": c["full_mult"]}
    tot = sum(v for v in full_res.holdings.values() if v == v and v > 0)
    hl = sorted(((t, v) for t, v in full_res.holdings.items()
                 if v == v and v > 0), key=lambda x: -x[1])
    holdings = [{"ticker": t, "pct": v / tot} for t, v in hl]

    # optional concentration-trim variants (no-cap default + annual caps)
    wins, bench = protocol._bench_grid(10, 0, 1000.0, cfg["cost_bps"])
    Snp = scores.reindex(index=fd.index, columns=fd.columns).to_numpy(float)

    def variant(vid, label, cap, period):
        vq, vs, full = [], [], None
        for wname, s, e in wins:
            if wname in protocol.REGIMES:
                continue
            _, vals, inv = fast.run_fast(fd, Snp, k=cfg["k"], every=10,
                                         start=s, end=e, cost_bps=cfg["cost_bps"],
                                         trim_cap=cap, trim_period=period)
            if inv[0] <= 0:
                continue
            m = vals[0] / inv[0]
            vq.append(m / bench["qqq"][wname] - 1)
            vs.append(m / bench["spy"][wname] - 1)
            if wname.endswith("_end") and full is None:
                full = m
        vq, vs = np.array(vq), np.array(vs)
        _, _, _, hold = fast.run_fast(fd, Snp, k=cfg["k"], every=10,
                                      start=ANCHOR, cost_bps=cfg["cost_bps"],
                                      trim_cap=cap, trim_period=period,
                                      return_holdings=True)
        t2 = sum(x for x in hold.values() if x == x and x > 0)
        hl2 = sorted(((t, x / t2) for t, x in hold.items()
                      if x == x and x > 0), key=lambda z: -z[1])
        return {"id": vid, "label": label,
                "holdings": [{"ticker": t, "pct": w} for t, w in hl2],
                "n": len(hl2), "top_weight": hl2[0][1] if hl2 else None,
                "win_qqq": float((vq > 0).mean()),
                "win_spy": float((vs > 0).mean()),
                "med_vs_qqq": float(np.median(vq)),
                "worst_vs_qqq": float(vq.min()), "full_mult": float(full)}

    trim = [variant("none", "No cap (default)", None, None),
            variant("a33", "Cap 33% a year", 0.33, "annual"),
            variant("a25", "Cap 25% a year", 0.25, "annual")]
    return {"winrate": card["by_horizon"], "regimes": card["regimes"],
            "cadence": cadence, "holdings": holdings,
            "n_positions": len(holdings), "trim": trim}


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

    yr_rows, full_res = _calendar_years(P, cfg)

    # current signal
    s = cfg["scores"](P)
    member, close = P["member"], P["close"]
    enough = close.notna().rolling(252).count() >= 252
    row = (s.iloc[-1].where(member.iloc[-1]).where(enough.iloc[-1])
           .dropna().sort_values(ascending=False))
    signal = {"as_of": str(close.index[-1].date()),
              "regime": cfg["regime"](P), "picks": list(row.index[:cfg["k"]]),
              "next_label": cfg.get("next_label", "next buy")}

    extras = _extras(P, cfg, full_res) if cfg.get("extras") else None
    if extras:
        signal["holdings"] = extras["holdings"]
        signal["n_positions"] = extras["n_positions"]

    headline = {"itd_mult": returns_rows[0]["strat_mult"],
                "itd_irr": returns_rows[0]["strat_irr"],
                "qqq_mult": returns_rows[0]["qqq_mult"],
                "spy_mult": returns_rows[0]["spy_mult"], "as_of": signal["as_of"]}
    data_json = {"horizons": [h[0] for h in HORIZONS], "curves": curves,
                 "headline": headline}
    returns_json = {"table": returns_rows, "years": yr_rows,
                    "as_of": signal["as_of"]}
    if extras:
        returns_json["winrate"] = extras["winrate"]
        returns_json["regimes"] = extras["regimes"]
        returns_json["cadence"] = extras["cadence"]
        returns_json["trim"] = extras["trim"]
    if write:
        pre = cfg["prefix"]
        _dump(data_json, os.path.join(DOCS, f"{pre}_data.json"))
        _dump(returns_json, os.path.join(DOCS, f"{pre}_returns.json"))
        _dump(signal, os.path.join(DOCS, f"{pre}_signal.json"))
    return data_json, returns_json, signal


def _sanitize(o):
    if isinstance(o, np.generic):
        o = o.item()
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
    return {"prefix": "summit", "k": 2, "cost_bps": 5, "extras": True,
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
