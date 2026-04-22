"""PULSAR — Calendar / Event-Anomaly LETF Strategy.

Thesis: leveraged equities earn disproportionate returns on a handful of
structural calendar windows (turn-of-month, FOMC drift, monthly-opex week,
quarter-end flow, Santa rally, seasonal Nov-Apr bias). Naive TSMOM at a
single cadence plateaus near Sharpe 0.6-1.0 because it holds through noise
weeks. PULSAR only holds LETFs on days where at least one structural
calendar filter is ON — otherwise BIL.

Design:
  * Universe: 14 LETFs (TQQQ UPRO SSO QLD SOXL TECL FAS ERX DRN EDC YINN
    UCO UGL TMF) + BIL (cash).
  * Every REBAL_DAYS (FIXED uniform cadence), at close[t-1]:
        1. Read the union of active calendar filters over the next
           REBAL_DAYS holding window. A filter is ACTIVE if the rule
           flags at least one of those days.
        2. If NO filter fires --> hold BIL for the window.
        3. Else rank 14 LETFs by short-term momentum (MOM_LB bars), pick
           top-K, equal weight; hold for REBAL_DAYS.
  * Fills at open[t], return = open[t] -> open[t+REBAL_DAYS]. No look-ahead.
  * 10 bps/side transaction cost on turnover each rebalance.
  * No daily vol scaling.

Grid search on IS, pick best Sharpe (CAGR>=20%), eval OOS once.
"""
from __future__ import annotations
import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
RESULTS = ROOT / "data/results"
RESULTS.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["TQQQ", "UPRO", "SSO", "QLD", "SOXL", "TECL", "FAS",
            "ERX", "DRN", "EDC", "YINN", "UCO", "UGL", "TMF"]
CASH = "BIL"

IS_START = "2010-03-11"
IS_END   = "2018-12-31"
OOS_START = "2019-01-02"
OOS_END   = "2026-04-02"

TC_BPS = 10.0  # per side

# ------------------------------------------------------------------
# FOMC schedule (announcement dates, 8/yr). Hardcoded 2010-2026.
# Source: federalreserve.gov calendar. Dates verified.
FOMC_DATES = [
    # 2010
    "2010-01-27","2010-03-16","2010-04-28","2010-06-23","2010-08-10",
    "2010-09-21","2010-11-03","2010-12-14",
    # 2011
    "2011-01-26","2011-03-15","2011-04-27","2011-06-22","2011-08-09",
    "2011-09-21","2011-11-02","2011-12-13",
    # 2012
    "2012-01-25","2012-03-13","2012-04-25","2012-06-20","2012-07-31",
    "2012-09-13","2012-10-24","2012-12-12",
    # 2013
    "2013-01-30","2013-03-20","2013-05-01","2013-06-19","2013-07-31",
    "2013-09-18","2013-10-30","2013-12-18",
    # 2014
    "2014-01-29","2014-03-19","2014-04-30","2014-06-18","2014-07-30",
    "2014-09-17","2014-10-29","2014-12-17",
    # 2015
    "2015-01-28","2015-03-18","2015-04-29","2015-06-17","2015-07-29",
    "2015-09-17","2015-10-28","2015-12-16",
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-10",
    # 2026
    "2026-01-28","2026-03-18",
]
FOMC_DATES = pd.to_datetime(FOMC_DATES)


# ------------------------------------------------------------------
# Data loading
def load_px(tkr):
    p = ETF / f"{tkr}.csv"
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def build_panel():
    closes, opens = {}, {}
    for t in UNIVERSE + [CASH]:
        df = load_px(t)
        closes[t] = df["Close"]
        opens[t]  = df["Open"]
    C = pd.DataFrame(closes)
    O = pd.DataFrame(opens)
    # Align on union of dates; forward-fill gaps (but not before inception)
    idx = C.index.union(O.index).sort_values()
    C = C.reindex(idx)
    O = O.reindex(idx)
    # Drop dates where BIL is missing (pre-BIL inception -> no cash fallback)
    C = C[C[CASH].notna()]
    O = O.loc[C.index]
    return C, O


# ------------------------------------------------------------------
# Calendar filters. Each returns a boolean Series indexed by trading dates.
# TRUE on trigger days (days to BE IN the long basket).

def filter_tom(idx, n_end=2, n_start=3):
    """Last n_end trading days of month + first n_start of next month."""
    flags = pd.Series(False, index=idx)
    by_month = pd.Series(idx.month * 100 + idx.year * 10000 + idx.day, index=idx)
    # rank within month: asc and desc
    month_key = idx.to_period("M")
    mp = pd.Series(month_key, index=idx)
    # end-of-month: last n_end days per month
    for _, group in pd.Series(idx, index=idx).groupby(mp):
        days = group.index
        flags.loc[days[-n_end:]] = True
        flags.loc[days[:n_start]] = True
    return flags


def filter_fomc(idx, pre=1, post=2):
    """Long from close[-pre] before FOMC to close[+post] after.
    I.e. flag days in window [-pre+1 .. +post] relative to each FOMC date."""
    flags = pd.Series(False, index=idx)
    idx_arr = pd.DatetimeIndex(idx)
    for fd in FOMC_DATES:
        # position of fd (or next trading day if fd not a trading day)
        pos = idx_arr.searchsorted(fd)
        if pos >= len(idx_arr): continue
        # window: [pos - (pre-1) .. pos + post]
        lo = max(0, pos - (pre - 1))
        hi = min(len(idx_arr) - 1, pos + post)
        flags.iloc[lo:hi + 1] = True
    return flags


def filter_opex(idx):
    """Monday-Friday of the week containing the 3rd Friday of the month."""
    flags = pd.Series(False, index=idx)
    idx_arr = pd.DatetimeIndex(idx)
    # find 3rd Friday of each (year, month)
    ym = sorted({(d.year, d.month) for d in idx_arr})
    for y, m in ym:
        first = pd.Timestamp(year=y, month=m, day=1)
        # Friday = weekday 4
        off = (4 - first.weekday()) % 7
        third_fri = first + pd.Timedelta(days=off + 14)
        week_start = third_fri - pd.Timedelta(days=third_fri.weekday())
        week_end = week_start + pd.Timedelta(days=4)
        mask = (idx_arr >= week_start) & (idx_arr <= week_end)
        flags.iloc[np.where(mask)[0]] = True
    return flags


def filter_qend(idx, n=3):
    """Last n trading days of Mar/Jun/Sep/Dec."""
    flags = pd.Series(False, index=idx)
    s = pd.Series(idx, index=idx)
    mp = idx.to_period("M")
    for period, group in s.groupby(mp):
        if period.month in (3, 6, 9, 12):
            days = group.index
            flags.loc[days[-n:]] = True
    return flags


def filter_santa(idx):
    """Dec 23 through Jan 3 (calendar)."""
    flags = pd.Series(False, index=idx)
    idx_arr = pd.DatetimeIndex(idx)
    in_dec = (idx_arr.month == 12) & (idx_arr.day >= 23)
    in_jan = (idx_arr.month == 1) & (idx_arr.day <= 3)
    flags.iloc[np.where(in_dec | in_jan)[0]] = True
    return flags


def filter_onseason(idx):
    """Nov-Apr = ON. May-Oct = OFF."""
    idx_arr = pd.DatetimeIndex(idx)
    flags = pd.Series(np.isin(idx_arr.month, [11, 12, 1, 2, 3, 4]), index=idx)
    return flags


def build_filter(idx, filters_on):
    """Union of selected filters."""
    out = pd.Series(False, index=idx)
    for f in filters_on:
        if f == "TOM":    out |= filter_tom(idx)
        elif f == "FOMC": out |= filter_fomc(idx)
        elif f == "OPEX": out |= filter_opex(idx)
        elif f == "QEND": out |= filter_qend(idx)
        elif f == "SNTA": out |= filter_santa(idx)
        elif f == "ONSN": out |= filter_onseason(idx)
    return out


# ------------------------------------------------------------------
# Backtest (vectorized)
def backtest(C, O, flag_series, rebal_days, mom_lb, top_k,
             mom_cached=None, daily_ret_cached=None, assets_cached=None):
    """Vectorized. Returns a daily return series from open-to-open, fixed cadence."""
    assets = assets_cached if assets_cached is not None else UNIVERSE + [CASH]
    n_ast = len(assets)

    # daily open-to-open returns per asset, shape (T, n_ast)
    if daily_ret_cached is None:
        O_sub = O[assets]
        dr = (O_sub.shift(-1) / O_sub - 1.0).values
    else:
        dr = daily_ret_cached
    dr = np.nan_to_num(dr, nan=0.0)

    dates = C.index
    T = len(dates)

    # momentum at close[t] (usable at t+1)
    if mom_cached is None or mom_cached.get("lb") != mom_lb:
        lc = np.log(C[UNIVERSE].replace(0, np.nan).values)
        mom_vals = lc - np.roll(lc, mom_lb, axis=0)
        mom_vals[:mom_lb] = np.nan
    else:
        mom_vals = mom_cached["v"]

    # Build flag bool array
    flags = flag_series.reindex(dates).fillna(False).values.astype(bool)

    # Precompute asset index lookups
    uni_idx = np.array([assets.index(t) for t in UNIVERSE])
    cash_idx = assets.index(CASH)

    # rebalance days: 0, rebal_days, 2*rebal_days, ...
    rebal_is = np.arange(0, T, rebal_days)

    # For each rebalance day i, any flag in window [i, i+rebal_days-1]?
    # Use a rolling-forward 'any'. Build cumulative to do it fast.
    cflags = np.concatenate([[0], np.cumsum(flags.astype(int))])
    # any in [i, j] = cflags[j+1] - cflags[i] > 0
    window_any = np.zeros(len(rebal_is), dtype=bool)
    for k, i in enumerate(rebal_is):
        j = min(T - 1, i + rebal_days - 1)
        window_any[k] = (cflags[j + 1] - cflags[i]) > 0

    # Weights matrix: (T, n_ast). Start all zero; set on rebalance i, carry forward.
    W = np.zeros((T, n_ast))

    prev_w = np.zeros(n_ast)
    port_ret = np.zeros(T)

    for k, i in enumerate(rebal_is):
        # Decision at close[i-1]
        new_w = np.zeros(n_ast)
        if i == 0:
            new_w[cash_idx] = 1.0  # start in cash until first real rebalance
        elif not window_any[k]:
            new_w[cash_idx] = 1.0
        else:
            m_row = mom_vals[i - 1]  # momentum at close[i-1]
            # valid: has momentum value AND next-day (i) open is valid
            next_open = O[UNIVERSE].iloc[i].values if i < T else np.full(len(UNIVERSE), np.nan)
            valid = ~np.isnan(m_row) & ~np.isnan(next_open)
            if valid.sum() < top_k:
                new_w[cash_idx] = 1.0
            else:
                m_masked = np.where(valid, m_row, -np.inf)
                top_idx = np.argpartition(-m_masked, top_k)[:top_k]
                if np.mean(m_row[top_idx]) <= 0:
                    new_w[cash_idx] = 1.0
                else:
                    w = 1.0 / top_k
                    for ti in top_idx:
                        new_w[uni_idx[ti]] = w

        # TC on turnover
        turnover = np.abs(new_w - prev_w).sum()
        tc = turnover * (TC_BPS / 10000.0)
        port_ret[i] -= tc
        prev_w = new_w

        # Fill weights from i to next rebalance - 1
        end_i = rebal_is[k + 1] if k + 1 < len(rebal_is) else T
        W[i:end_i] = new_w

    # daily portfolio return = sum_j W[t,j] * dr[t,j]
    daily_port = (W * dr).sum(axis=1)
    port_ret += daily_port

    return pd.Series(port_ret, index=dates)


# ------------------------------------------------------------------
def metrics(ret, rf_daily=0.0):
    r = ret.dropna()
    if len(r) < 10:
        return dict(sharpe=np.nan, cagr=np.nan, vol=np.nan, mdd=np.nan, n=len(r))
    eq = (1 + r).cumprod()
    years = len(r) / 252.0
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() - rf_daily) / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    mdd = dd.min()
    return dict(sharpe=float(sharpe), cagr=float(cagr), vol=float(vol),
                mdd=float(mdd), n=int(len(r)))


# ------------------------------------------------------------------
def main():
    print("Loading data...")
    C, O = build_panel()
    print(f"  panel shape: {C.shape}, date range: {C.index[0]} .. {C.index[-1]}")

    full = C.index
    is_mask = (full >= IS_START) & (full <= IS_END)
    oos_mask = (full >= OOS_START) & (full <= OOS_END)
    is_idx = full[is_mask]
    oos_idx = full[oos_mask]
    print(f"  IS rows: {is_mask.sum()}, OOS rows: {oos_mask.sum()}")

    # Grid
    cadences = [3, 5, 10, 21]
    mom_lbs  = [3, 5, 10]
    top_ks   = [3, 4, 5]
    filter_sets = []
    base = ["TOM", "FOMC", "OPEX", "QEND", "SNTA", "ONSN"]
    # singletons
    for f in base: filter_sets.append((f,))
    # all pairs
    for combo in itertools.combinations(base, 2): filter_sets.append(combo)
    # select triples (meaningful)
    for combo in itertools.combinations(base, 3): filter_sets.append(combo)
    # all six
    filter_sets.append(tuple(base))

    results = []
    print(f"Grid size: {len(cadences)*len(mom_lbs)*len(top_ks)*len(filter_sets)}")

    # Precompute daily returns and per-filter flags once
    assets = UNIVERSE + [CASH]
    O_sub = O[assets]
    dr_cached = (O_sub.shift(-1) / O_sub - 1.0).values

    # Precompute each filter once
    filter_flags = {}
    for fs in filter_sets:
        filter_flags[fs] = build_filter(full, fs)

    # Precompute momentum per lb
    mom_cache = {}
    for lb in mom_lbs:
        lc = np.log(C[UNIVERSE].replace(0, np.nan).values)
        mv = lc - np.roll(lc, lb, axis=0)
        mv[:lb] = np.nan
        mom_cache[lb] = {"lb": lb, "v": mv}

    done = 0
    for cad in cadences:
        for mom_lb in mom_lbs:
            for top_k in top_ks:
                for fs in filter_sets:
                    ret = backtest(C, O, filter_flags[fs], cad, mom_lb, top_k,
                                   mom_cached=mom_cache[mom_lb],
                                   daily_ret_cached=dr_cached,
                                   assets_cached=assets)
                    is_ret = ret.loc[is_idx]
                    m_is = metrics(is_ret)
                    results.append(dict(cadence=cad, mom_lb=mom_lb, top_k=top_k,
                                        filters=",".join(fs),
                                        **{f"is_{k}": v for k, v in m_is.items()}))
                    done += 1
        print(f"  cadence={cad} done ({done} configs)")

    res_df = pd.DataFrame(results)
    # Require CAGR>=20% IS, pick max Sharpe
    feasible = res_df[(res_df["is_cagr"] >= 0.20)]
    if feasible.empty:
        print("No config met IS CAGR>=20%; relaxing to max Sharpe.")
        feasible = res_df
    feasible = feasible.sort_values("is_sharpe", ascending=False)
    print("\nTop 10 configs (IS):")
    print(feasible.head(10).to_string(index=False))

    winner = feasible.iloc[0]
    print(f"\nWINNER: {dict(winner)}")

    # Re-run winner on full period, split IS/OOS
    fs_win = tuple(winner["filters"].split(","))
    flags_full = build_filter(full, fs_win)
    ret_full = backtest(C, O, flags_full, int(winner["cadence"]),
                        int(winner["mom_lb"]), int(winner["top_k"]))

    m_is = metrics(ret_full.loc[is_idx])
    m_oos = metrics(ret_full.loc[oos_idx])
    m_full = metrics(ret_full.loc[(full >= IS_START) & (full <= OOS_END)])

    print("\n=== PULSAR ===")
    print(f"Params: cadence={int(winner['cadence'])} mom_lb={int(winner['mom_lb'])} "
          f"top_k={int(winner['top_k'])} filters={fs_win}")
    print(f"IS   Sharpe={m_is['sharpe']:.3f} CAGR={m_is['cagr']*100:.2f}% "
          f"Vol={m_is['vol']*100:.2f}% MDD={m_is['mdd']*100:.2f}%")
    print(f"OOS  Sharpe={m_oos['sharpe']:.3f} CAGR={m_oos['cagr']*100:.2f}% "
          f"Vol={m_oos['vol']*100:.2f}% MDD={m_oos['mdd']*100:.2f}%")
    print(f"FULL Sharpe={m_full['sharpe']:.3f} CAGR={m_full['cagr']*100:.2f}% "
          f"Vol={m_full['vol']*100:.2f}% MDD={m_full['mdd']*100:.2f}%")

    out = dict(
        strategy="PULSAR",
        params=dict(cadence=int(winner["cadence"]),
                    mom_lb=int(winner["mom_lb"]),
                    top_k=int(winner["top_k"]),
                    filters=list(fs_win),
                    universe=UNIVERSE,
                    cash=CASH,
                    tc_bps=TC_BPS),
        is_window=[IS_START, IS_END],
        oos_window=[OOS_START, OOS_END],
        IS=m_is, OOS=m_oos, FULL=m_full,
    )
    (RESULTS / "pulsar_metrics.json").write_text(json.dumps(out, indent=2, default=float))
    ret_full.to_frame("pulsar").to_csv(RESULTS / "pulsar_returns.csv")

    # Save grid too
    res_df.to_csv(RESULTS / "pulsar_grid.csv", index=False)
    print("\nSaved: pulsar_metrics.json, pulsar_returns.csv, pulsar_grid.csv")


if __name__ == "__main__":
    main()
