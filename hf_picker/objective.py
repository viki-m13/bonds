"""The underwater-avoidance objective and its evaluation.

User goal, verbatim: "buying a stock that after it's purchased is not below
the purchase price often or at all."

We make that measurable. A *purchase* is a buy of one name at the OPEN of the
trading day after a signal date (strict next-open execution; a signal may use
data only through its own close). For a purchase with entry price P0 and the
subsequent close path C[1..H] over a holding horizon of H trading days we
define, per purchase:

  underwater_frac : mean( C < P0 )         fraction of held days below entry
  ever_underwater : any( C < P0 )          did it ever dip below entry
  max_dip         : min( C / P0 - 1 )      worst close drawdown below entry (<=0)
  time_to_recover : first day C >= P0 stays... (reported as days underwater)
  end_ret         : C[H] / P0 - 1          P&L at the horizon

The headline objective is a LOW mean `underwater_frac` and a LOW
`ever_underwater` rate, at acceptable `end_ret`. A perfect picker would buy
names that never close below P0 (underwater_frac = 0).

Evaluation is purchase-weighted (every buy counts once), which is the honest
unit for "after I buy, does it go below what I paid".
"""
import numpy as np
import pandas as pd

from data import load_panel, eligibility

TRADING_DAYS_MONTH = 21


def signal_positions(index: pd.DatetimeIndex, every: int = TRADING_DAYS_MONTH,
                     offset: int = 0, start=None, end=None) -> np.ndarray:
    idx = index
    keep = np.ones(len(idx), bool)
    if start is not None:
        keep &= idx >= pd.Timestamp(start)
    if end is not None:
        keep &= idx <= pd.Timestamp(end)
    pos = np.where(keep)[0][offset::every]
    return pos[pos + 1 < len(idx)]          # need a next open to execute


class Arrays:
    """Numpy views shared across all arms (built once)."""

    def __init__(self, min_history: int = 252):
        p = load_panel()
        self.index = p["close"].index
        self.columns = p["close"].columns
        self.open = p["open"].to_numpy(float)
        self.close = p["close"].to_numpy(float)
        self.volume = p["volume"].to_numpy(float)
        self.elig = eligibility(min_history).to_numpy(bool)


def select(arr: Arrays, scores: np.ndarray, k: int, sig_pos: np.ndarray):
    """Top-k eligible picks per signal date. Returns parallel arrays
    (exec_pos, ticker) for every purchase, plus the per-date pick lists."""
    exec_pos, tick = [], []
    per_date = []
    for p in sig_pos:
        row = scores[p].copy()
        mask = arr.elig[p] & ~np.isnan(arr.open[p + 1])
        row[~mask] = np.nan
        ok = np.where(~np.isnan(row))[0]
        if len(ok) == 0:
            per_date.append((p, []))
            continue
        kk = min(k, len(ok))
        top = ok[np.argsort(-row[ok])[:kk]]
        per_date.append((p, list(top)))
        for t in top:
            exec_pos.append(p + 1)
            tick.append(t)
    return np.array(exec_pos, int), np.array(tick, int), per_date


def underwater_metrics(arr: Arrays, exec_pos: np.ndarray, tick: np.ndarray,
                       horizon: int) -> pd.DataFrame:
    """Per-purchase underwater table over `horizon` trading days post-entry.

    Uses close prices for the path and the executed open as the entry price
    (what you actually paid). Purchases whose horizon runs past the data end
    are evaluated on the truncated path (still valid, just shorter)."""
    n = len(exec_pos)
    rows = {"exec_pos": exec_pos, "ticker": tick}
    entry = arr.open[exec_pos, tick]
    uw_frac = np.full(n, np.nan)
    ever = np.zeros(n, bool)
    max_dip = np.full(n, np.nan)
    days_uw = np.full(n, np.nan)
    end_ret = np.full(n, np.nan)
    hzn_used = np.zeros(n, int)
    T = arr.close.shape[0]
    for i in range(n):
        e = exec_pos[i]
        t = tick[i]
        hi = min(e + horizon, T - 1)
        path = arr.close[e + 1:hi + 1, t]          # strictly after entry day
        path = path[~np.isnan(path)]
        if len(path) == 0:
            continue
        p0 = entry[i]
        below = path < p0
        uw_frac[i] = below.mean()
        ever[i] = bool(below.any())
        days_uw[i] = int(below.sum())
        max_dip[i] = path.min() / p0 - 1.0
        end_ret[i] = path[-1] / p0 - 1.0
        hzn_used[i] = len(path)
    rows.update(uw_frac=uw_frac, ever_underwater=ever, max_dip=max_dip,
                days_underwater=days_uw, end_ret=end_ret, horizon_used=hzn_used,
                entry=entry)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    """Reduce a per-purchase table to headline objective stats."""
    d = df.dropna(subset=["uw_frac"])
    if len(d) == 0:
        return {}
    return {
        "n_buys": int(len(d)),
        "underwater_frac": float(d["uw_frac"].mean()),       # << headline
        "ever_underwater": float(d["ever_underwater"].mean()),
        "never_underwater": float((~d["ever_underwater"]).mean()),
        "mean_max_dip": float(d["max_dip"].mean()),
        "p10_max_dip": float(d["max_dip"].quantile(0.10)),   # tail dip
        "mean_days_uw": float(d["days_underwater"].mean()),
        "hit_rate_end": float((d["end_ret"] > 0).mean()),
        "mean_end_ret": float(d["end_ret"].mean()),
        "median_end_ret": float(d["end_ret"].median()),
    }


def evaluate_arm(arr: Arrays, scores: np.ndarray, k: int, horizon: int,
                 every: int = TRADING_DAYS_MONTH, start=None, end=None) -> dict:
    sp = signal_positions(arr.index, every, 0, start, end)
    ep, tk, _ = select(arr, scores, k, sp)
    if len(ep) == 0:
        return {}
    tbl = underwater_metrics(arr, ep, tk, horizon)
    return summarize(tbl)
