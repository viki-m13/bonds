"""Standard evaluation protocol for DCA stock-selection signals.

A *signal* is a DataFrame aligned to the panel (dates x tickers) where the
value at row date d may use information only through the CLOSE of d.
Execution happens at the next day's open (enforced by the engine).

`evaluate_signal` runs the full window grid and returns/saves a scorecard:
  * quarterly start dates 2006-01 .. (end - 3y), horizons 3y/5y/10y/to-end
  * named regime windows (GFC, COVID, 2022 bear, sideways 2015-16, ...)
  * win-rate / median / worst excess final-multiple vs QQQ and SPY DCA
  * comparison vs random-pick control (survivorship-bias control)

Anything that beats both benchmarks in ~every window here graduates to the
slow reference engine + leakage audit.
"""
import json
import os

import numpy as np
import pandas as pd

import data as data_mod
import fast

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "research", "scorecards")
os.makedirs(RESULTS_DIR, exist_ok=True)

REGIMES = {
    "GFC_2007_2009": ("2007-10-01", "2009-12-31"),
    "recovery_2009_2012": ("2009-03-01", "2012-12-31"),
    "bull_2013_2017": ("2013-01-01", "2017-12-31"),
    "sideways_2015_2016": ("2015-01-01", "2016-12-31"),
    "vol_2018": ("2018-01-01", "2019-06-30"),
    "covid_2020": ("2020-01-01", "2021-12-31"),
    "bear_2022": ("2022-01-01", "2023-06-30"),
    "ai_bull_2023_2026": ("2023-01-01", None),
}

_cache: dict = {}


def get_shared():
    if "fd" not in _cache:
        P = data_mod.build_panel()
        _cache["panels"] = P
        _cache["fd"] = fast.FastData(P["open"], P["close"], P["member"])
        _cache["qqq"] = data_mod.load_benchmark("QQQ")
        _cache["spy"] = data_mod.load_benchmark("SPY")
    return _cache


def grid_windows(index, min_years=3):
    end = index[-1]
    starts = pd.date_range("2006-01-01",
                           end - pd.DateOffset(years=min_years), freq="QS")
    wins = []
    for s in starts:
        for h in (3, 5, 10, None):
            e = end if h is None else s + pd.DateOffset(years=h)
            if e > end:
                continue
            wins.append((f"{s.date()}_{h or 'end'}", s, e))
    for name, (s, e) in REGIMES.items():
        wins.append((name, pd.Timestamp(s),
                     end if e is None else pd.Timestamp(e)))
    return wins


def _bench_grid(every, offset, contribution, cost_bps):
    key = ("bench", every, offset, contribution, cost_bps)
    if key in _cache:
        return _cache[key]
    sh = get_shared()
    idx = sh["fd"].index
    wins = grid_windows(idx)
    out = {}
    for bname in ("qqq", "spy"):
        b = sh[bname].loc[:idx[-1]]
        res = {}
        for name, s, e in wins:
            (v, inv), = fast.bench_fast(b, every=every, offset=offset,
                                        start=s, end=e,
                                        contribution=contribution,
                                        cost_bps=cost_bps, eval_dates=[e])
            res[name] = v / inv if inv else np.nan
        out[bname] = res
    _cache[key] = (wins, out)
    return wins, out


def evaluate_signal(scores: pd.DataFrame, name: str, k: int = 3,
                    every: int = 10, offset: int = 0,
                    cost_bps: float = 5.0, sell: pd.DataFrame | None = None,
                    contribution: float = 1000.0, save: bool = True,
                    quiet: bool = False) -> dict:
    sh = get_shared()
    fd = sh["fd"]
    S = scores.reindex(index=fd.index, columns=fd.columns).to_numpy(float)
    sellm = (sell.reindex(index=fd.index, columns=fd.columns)
             .fillna(False).to_numpy(bool) if sell is not None else None)
    wins, bench = _bench_grid(every, offset, contribution, cost_bps)
    rows = []
    for wname, s, e in wins:
        try:
            _, vals, inv = fast.run_fast(fd, S, k=k, every=every,
                                         offset=offset, start=s, end=e,
                                         contribution=contribution,
                                         cost_bps=cost_bps, sell=sellm)
        except (ValueError, IndexError):
            continue
        if inv[0] <= 0:
            continue
        mult = vals[0] / inv[0]
        rows.append({"window": wname, "start": str(s.date()),
                     "mult": mult,
                     "qqq": bench["qqq"][wname], "spy": bench["spy"][wname],
                     "vs_qqq": mult / bench["qqq"][wname] - 1,
                     "vs_spy": mult / bench["spy"][wname] - 1})
    df = pd.DataFrame(rows)
    grid = df[~df["window"].isin(REGIMES)]
    reg = df[df["window"].isin(REGIMES)]
    card = {
        "name": name, "k": k, "every": every, "offset": offset,
        "cost_bps": cost_bps, "n_windows": len(grid),
        "win_qqq": float((grid["vs_qqq"] > 0).mean()),
        "win_spy": float((grid["vs_spy"] > 0).mean()),
        "med_vs_qqq": float(grid["vs_qqq"].median()),
        "med_vs_spy": float(grid["vs_spy"].median()),
        "worst_vs_qqq": float(grid["vs_qqq"].min()),
        "worst_vs_spy": float(grid["vs_spy"].min()),
        "p10_vs_qqq": float(grid["vs_qqq"].quantile(0.10)),
        "full_mult": float(grid.loc[grid["window"].str.endswith("_end"),
                                    "mult"].iloc[0]) if len(grid) else None,
        "regimes": {r["window"]: {"mult": r["mult"], "vs_qqq": r["vs_qqq"],
                                  "vs_spy": r["vs_spy"]}
                    for _, r in reg.iterrows()},
    }
    if not quiet:
        print(f"[{name}] k={k} win_qqq={card['win_qqq']:.0%} "
              f"win_spy={card['win_spy']:.0%} "
              f"med_vs_qqq={card['med_vs_qqq']:+.1%} "
              f"worst_vs_qqq={card['worst_vs_qqq']:+.1%}")
    if save:
        safe = name.replace("/", "_").replace(" ", "_")
        with open(os.path.join(RESULTS_DIR, f"{safe}.json"), "w") as f:
            json.dump({"card": card,
                       "windows": df.to_dict(orient="records")},
                      f, indent=1, default=str)
    return card


def random_control(k=3, every=10, offset=0, cost_bps=5.0, n_draws=30,
                   seed=11) -> pd.DataFrame:
    """Random-pick DCA over the same eligible universe, full grid."""
    sh = get_shared()
    fd = sh["fd"]
    rng = np.random.default_rng(seed)
    wins, bench = _bench_grid(every, offset, 1000.0, cost_bps)
    rows = []
    for d in range(n_draws):
        S = rng.random((len(fd.index), len(fd.columns)))
        for wname, s, e in wins:
            _, vals, inv = fast.run_fast(fd, S, k=k, every=every,
                                         offset=offset, start=s, end=e,
                                         cost_bps=cost_bps)
            if inv[0] > 0:
                rows.append({"draw": d, "window": wname,
                             "mult": vals[0] / inv[0],
                             "qqq": bench["qqq"][wname],
                             "spy": bench["spy"][wname]})
    return pd.DataFrame(rows)
