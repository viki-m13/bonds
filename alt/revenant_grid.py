"""Quick grid search on REVENANT params — tune ON IS ONLY."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import itertools

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

UNIVERSE = ["TQQQ", "UPRO", "QLD", "SSO", "SOXL", "TECL", "FAS", "ERX",
            "DRN", "EDC", "YINN", "UCO", "UGL", "NUGT", "TMF", "UBT", "TYD"]
IS_END, OOS_START = "2018-12-31", "2019-01-02"


def load_etf_close_open(t):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


def rsi(series, length):
    d = series.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ma_up = up.rolling(length).mean(); ma_dn = dn.rolling(length).mean()
    rs = ma_up / ma_dn.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def backtest(close, opn, bil_r, regime_ok, rsi_lag, oversold, n_max, max_hold, rsi_high_thr, tc_bps):
    dates = opn.index
    # per-day available universe
    pos_enter_ix = {}
    current_w = pd.Series(0.0, index=list(UNIVERSE) + ["BIL"])
    port_ret = np.zeros(len(dates))
    tov_arr = np.zeros(len(dates))
    n_arr = np.zeros(len(dates), dtype=int)
    opn_ret = opn[UNIVERSE].pct_change().fillna(0).values
    # rsi values array
    rsi_vals = rsi_lag[UNIVERSE].values
    oversold_vals = oversold[UNIVERSE].values.astype(bool)
    regime_arr = regime_ok.values.astype(bool)
    bil_vals = bil_r.values

    U = UNIVERSE
    for i in range(len(dates)):
        # earn return using yesterday's weights
        r = 0.0
        w_arr = current_w.values
        for k in range(len(U)):
            if w_arr[k] != 0:
                rk = opn_ret[i, k]
                if rk == rk:  # not nan
                    r += w_arr[k] * rk
        r += current_w.iloc[-1] * (bil_vals[i] if bil_vals[i] == bil_vals[i] else 0.0)
        port_ret[i] = r

        # decide new weights
        new_w = current_w.copy()
        # exits
        for k, t in enumerate(U):
            if new_w[t] > 0:
                held = i - pos_enter_ix[t]
                rv = rsi_vals[i, k]
                if held >= max_hold or (rv == rv and rv > rsi_high_thr):
                    new_w[t] = 0.0
                    pos_enter_ix.pop(t, None)

        # entries only if regime OK
        if regime_arr[i]:
            active = [t for t in U if new_w[t] > 0]
            slots = n_max - len(active)
            if slots > 0:
                cands = []
                for k, t in enumerate(U):
                    if new_w[t] > 0: continue
                    if oversold_vals[i, k]:
                        rv = rsi_vals[i, k]
                        if rv == rv:
                            cands.append((rv, t))
                cands.sort()
                for _, t in cands[:slots]:
                    new_w[t] = 1e-9  # placeholder
                    pos_enter_ix[t] = i

        # size equal weight
        active = [t for t in U if new_w[t] > 0]
        if active:
            w_each = 1.0 / len(active)
            new_w[:] = 0.0
            for t in active:
                new_w[t] = w_each
            new_w["BIL"] = 0.0
        else:
            new_w[:] = 0.0
            new_w["BIL"] = 1.0
            pos_enter_ix.clear()

        # turnover / TC
        tov = (new_w - current_w).abs().sum()
        tov_arr[i] = tov
        if i + 1 < len(dates):
            port_ret[i + 1] -= tov * (tc_bps / 1e4)
        n_arr[i] = sum(1 for t in U if new_w[t] > 0)
        current_w = new_w

    return pd.Series(port_ret, index=dates), pd.Series(tov_arr, index=dates), pd.Series(n_arr, index=dates)


def metrics(r):
    r = r.dropna()
    if len(r) == 0: return (0, 0, 0, 0)
    mu = r.mean() * 252; sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1
    return sr, cagr, sd, dd


def main():
    # Load data
    close = {}; opn = {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf_close_open(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) &
                  (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()

    bil_r = opn["BIL"].pct_change().fillna(0)

    results = []
    for rsi_len, rsi_low, max_hold, rsi_high, n_max, sma_len, hy_slope_thr, vix_max, tc_bps, trend_name in itertools.product(
        [2, 3],                    # rsi_len
        [5, 10, 15],               # rsi_low
        [2, 3, 5, 10],             # max_hold
        [60, 70, 80],              # rsi_high (exit)
        [2, 3, 4, 6],              # n_max
        [100, 200],                # sma_len
        [0.3, 1.0],                # hy_slope_thr
        [30, 40],                  # vix_max
        [10.0],                    # tc_bps
        [False, True],             # require per-name trend > 50dma
    ):
        rsi_df = pd.DataFrame({t: rsi(close[t], rsi_len) for t in UNIVERSE})
        rsi_lag = rsi_df.shift(1)

        hy_slope = hy - hy.shift(20)
        regime_ok = (spy_ok & (hy_slope < hy_slope_thr) & (vix < vix_max)).shift(1).fillna(False)
        if sma_len != 200:
            spy_ma2 = spy.rolling(sma_len).mean()
            spy_ok2 = (spy > spy_ma2) & (spy_ma2.diff(20) > 0)
            regime_ok = (spy_ok2 & (hy_slope < hy_slope_thr) & (vix < vix_max)).shift(1).fillna(False)

        if trend_name:
            sma50 = pd.DataFrame({t: close[t].rolling(50).mean() for t in UNIVERSE})
            trend_ok = (close[UNIVERSE] > sma50).shift(1).fillna(False)
            oversold = (rsi_lag < rsi_low) & trend_ok
        else:
            oversold = rsi_lag < rsi_low

        port_ret, tov, nh = backtest(close, opn, bil_r, regime_ok, rsi_lag, oversold,
                                     n_max, max_hold, rsi_high, tc_bps)
        sr_is, cagr_is, _, _ = metrics(port_ret.loc[:IS_END])
        sr_oos, cagr_oos, _, _ = metrics(port_ret.loc[OOS_START:])
        sr_full, cagr_full, vol_full, mdd_full = metrics(port_ret)
        turn_ann = float(tov.sum() / max(1, len(port_ret)) * 252)
        gap = abs(sr_is - sr_oos)
        results.append({
            "rsi_len": rsi_len, "rsi_low": rsi_low, "max_hold": max_hold,
            "rsi_high": rsi_high, "n_max": n_max, "sma_len": sma_len,
            "hy_slope_thr": hy_slope_thr, "vix_max": vix_max,
            "trend_name": trend_name,
            "sr_is": round(sr_is, 3), "sr_oos": round(sr_oos, 3),
            "sr_full": round(sr_full, 3), "cagr_full": round(cagr_full, 3),
            "mdd_full": round(mdd_full, 3), "vol_full": round(vol_full, 3),
            "gap": round(gap, 3), "turn": round(turn_ann, 1),
        })

    res = pd.DataFrame(results).sort_values("sr_is", ascending=False)
    res.to_csv(RESULTS / "revenant_grid.csv", index=False)
    print(res.head(30).to_string())
    print("\nTop IS Sharpe with small gap:")
    filt = res[(res.sr_is > 1.0) & (res.sr_oos > 1.0) & (res.gap < 0.5)]
    print(filt.head(30).to_string())


if __name__ == "__main__":
    main()
