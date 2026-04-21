"""Robustness check 6 — regime stress (worst sub-period, drawdown detail).

Pick 6 contenders. For each: report the single worst 1-yr/3-yr CAGR, the
worst 1-yr/3-yr MDD, peak-to-trough duration for the two biggest drawdowns,
and Calmar ratio (CAGR/|MDD|) computed on the FULL window.

This answers: "can you hold this for 1 yr? 3 yrs? what does a bad year
look like?"
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")


def invvol_fn(tickers, lookback):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        inv = 1 / r.std().replace(0, np.nan).fillna(0)
        w = inv / inv.sum()
        out = pd.Series(0.0, index=hist.columns)
        out.loc[w.index] = w
        return out
    return fn


def mom_fn(tickers, lookback, top_n):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        cum = (1 + r).prod() - 1
        picks = cum.sort_values(ascending=False).head(top_n).index.tolist()
        out = pd.Series(0.0, index=hist.columns)
        if picks:
            out.loc[picks] = 1.0 / len(picks)
        return out
    return fn


def rolling_worst(r, window_days):
    """Return (worst_window_cagr, worst_window_end_date, worst_window_mdd)."""
    nav = (1 + r).cumprod()
    # Rolling log-return ratio
    logr = np.log(1 + r)
    roll = logr.rolling(window_days).sum()
    if roll.dropna().empty:
        return np.nan, None, np.nan
    worst_i = roll.idxmin()
    worst_ret = np.expm1(roll.loc[worst_i])
    cagr = (1 + worst_ret) ** (252 / window_days) - 1
    # worst dd inside any window ending at that date
    # (approx: window MDD for that slice)
    end_iloc = r.index.get_loc(worst_i)
    start_iloc = max(0, end_iloc - window_days + 1)
    sub = r.iloc[start_iloc:end_iloc + 1]
    sub_nav = (1 + sub).cumprod()
    dd = (sub_nav / sub_nav.cummax() - 1).min()
    return float(cagr * 100), worst_i.strftime("%Y-%m-%d"), float(dd * 100)


def biggest_drawdowns(r, n=2):
    """Return list of (peak, trough, recovery, mdd%, peak_to_trough_days,
    peak_to_recovery_days) for the n biggest drawdowns."""
    nav = (1 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1
    events = []
    # Identify drawdown episodes: contiguous runs where dd < 0
    in_dd = False
    start_i = None
    local_min_i = None
    local_min_v = 0
    episodes = []
    for i, (d, v) in enumerate(dd.items()):
        if v < 0:
            if not in_dd:
                in_dd = True
                start_i = d
                local_min_v = v
                local_min_i = d
            elif v < local_min_v:
                local_min_v = v
                local_min_i = d
        else:
            if in_dd:
                episodes.append((start_i, local_min_i, d, local_min_v))
                in_dd = False
        if v == 0 and in_dd:
            pass
    if in_dd:
        episodes.append((start_i, local_min_i, dd.index[-1], local_min_v))

    episodes.sort(key=lambda e: e[3])
    out = []
    for (peak_d, trough_d, recov_d, mdd) in episodes[:n]:
        ptt = (trough_d - peak_d).days
        ptr = (recov_d - peak_d).days
        out.append({
            "peak": peak_d.strftime("%Y-%m-%d"),
            "trough": trough_d.strftime("%Y-%m-%d"),
            "recovery": recov_d.strftime("%Y-%m-%d") if recov_d != dd.index[-1]
                        else "NOT RECOVERED",
            "mdd_pct": mdd * 100,
            "peak_to_trough_days": ptt,
            "peak_to_recovery_days": ptr if recov_d != dd.index[-1] else -1,
        })
    return out


def summary(r, label):
    nav = (1 + r).cumprod()
    cagr = (nav.iloc[-1] ** (252 / len(r)) - 1) * 100 if nav.iloc[-1] > 0 else -100
    vol = r.std() * np.sqrt(252) * 100
    mdd = ((nav / nav.cummax() - 1).min()) * 100
    sr = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0
    calmar = cagr / abs(mdd) if mdd < 0 else np.inf

    w1y_cagr, w1y_end, w1y_dd = rolling_worst(r, 252)
    w3y_cagr, w3y_end, w3y_dd = rolling_worst(r, 756)
    dds = biggest_drawdowns(r, n=2)

    print(f"\n{label}")
    print(f"  Full: CAGR={cagr:5.1f}%  Vol={vol:4.1f}%  MDD={mdd:6.1f}%  "
          f"SR={sr:.2f}  Calmar={calmar:.2f}")
    print(f"  Worst 1y (ending {w1y_end}): CAGR={w1y_cagr:6.1f}%  "
          f"max-DD-in-window={w1y_dd:6.1f}%")
    print(f"  Worst 3y (ending {w3y_end}): CAGR={w3y_cagr:6.1f}%  "
          f"max-DD-in-window={w3y_dd:6.1f}%")
    for k, dd in enumerate(dds, 1):
        print(f"  DD#{k}: peak {dd['peak']} -> trough {dd['trough']} "
              f"({dd['mdd_pct']:5.1f}%)  "
              f"peak→trough {dd['peak_to_trough_days']}d  "
              f"→recovery {dd['peak_to_recovery_days']}d "
              f"({dd['recovery']})")


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]
    ew_all17 = {t: 1/len(LETF_LONG_2011) for t in LETF_LONG_2011}

    strategies = [
        ("100% TQQQ (static)", w_fixed({"TQQQ": 1.0})),
        ("HFEA-Tech 50/50 TQQQ/TMF", w_fixed({"TQQQ":0.5,"TMF":0.5})),
        ("EW5 UPRO/TQQQ/SOXL/TMF/UGL",
         w_fixed({"UPRO":0.2,"TQQQ":0.2,"SOXL":0.2,"TMF":0.2,"UGL":0.2})),
        ("EW-all17 (naive baseline)", w_fixed(ew_all17)),
        ("invvol clean4 lb=21", invvol_fn(clean4, 21)),
        ("invvol core6 lb=63", invvol_fn(core6, 63)),
        ("mom core6 lb=126 top4", mom_fn(core6, 126, 4)),
    ]
    for name, fn in strategies:
        r, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
        summary(r, name)

    spy = load_etf("SPY").pct_change().fillna(0).loc[rets.index[0]:rets.index[-1]]
    summary(spy, "SPY BH (benchmark)")


if __name__ == "__main__":
    main()
