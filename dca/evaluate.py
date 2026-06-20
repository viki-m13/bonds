"""Evaluation harness: strategy DCA vs benchmark DCA across EVERY timeframe.

Core idea: for a grid of start dates (every quarter) and horizons, run both the
strategy and the benchmark with identical cadence/contributions, and compare
final value multiples (value / invested). Outperformance must hold across the
grid, not just in aggregate.

Also provides the survivorship control: random-pick DCA from the same eligible
universe. A strategy whose edge comes from universe survivorship (rather than
skill) will not beat this control.
"""
import numpy as np
import pandas as pd

from engine import run_dca, run_benchmark_dca, schedule_dates


def window_grid(index, first="2006-01-01", last=None, step_months=3,
                min_years=3):
    """All (start, end-of-data) windows from quarterly start dates, plus
    fixed-horizon windows (3y, 5y) from each start."""
    last = pd.Timestamp(last) if last else index[-1]
    starts = pd.date_range(first, last - pd.DateOffset(years=min_years),
                           freq=f"{step_months}MS")
    return [pd.Timestamp(s) for s in starts]


def run_grid(open_px, close_px, scores, member, bench_dfs: dict,
             starts, end=None, horizons=(3, 5, None), k=3, every=10,
             cost_bps=5.0, sell=None, contribution=1000.0):
    """Returns long DataFrame: one row per (start, horizon, benchmark)."""
    end = pd.Timestamp(end) if end else close_px.index[-1]
    rows = []
    for s in starts:
        for h in horizons:
            e = min(end, s + pd.DateOffset(years=h)) if h else end
            if (e - s).days < 700:
                continue
            try:
                res = run_dca(open_px, close_px, scores, member, k=k,
                              every=every, start=s, end=e, cost_bps=cost_bps,
                              sell=sell, contribution=contribution)
            except ValueError:
                continue
            row = {"start": s, "horizon": h or 99, "end": e,
                   "strat_mult": res.final_multiple, "strat_irr": res.irr()}
            for name, bdf in bench_dfs.items():
                b = run_benchmark_dca(bdf.loc[:e], every=every, start=s, end=e,
                                      cost_bps=cost_bps,
                                      contribution=contribution)
                row[f"{name.lower()}_mult"] = b.final_multiple
                row[f"{name.lower()}_irr"] = b.irr()
            rows.append(row)
    df = pd.DataFrame(rows)
    for name in bench_dfs:
        n = name.lower()
        df[f"vs_{n}"] = df["strat_mult"] / df[f"{n}_mult"] - 1
    return df


def random_control(open_px, close_px, member, starts, bench_dfs, end=None,
                   k=3, every=10, cost_bps=5.0, n_draws=20, seed=7,
                   min_history=252):
    """Random-pick DCA from same eligible universe: survivorship control.
    Returns DataFrame of mean random multiple per start (full horizon)."""
    rng = np.random.default_rng(seed)
    enough = close_px.notna().rolling(min_history).count() >= min_history
    end = pd.Timestamp(end) if end else close_px.index[-1]
    rows = []
    for s in starts:
        mults = []
        for d in range(n_draws):
            noise = pd.DataFrame(
                rng.random((len(close_px.index), len(close_px.columns))),
                index=close_px.index, columns=close_px.columns)
            noise = noise.where(close_px.notna() & enough)
            res = run_dca(open_px, close_px, noise, member, k=k, every=every,
                          start=s, end=end, cost_bps=cost_bps)
            mults.append(res.final_multiple)
        row = {"start": s, "rand_mean": np.mean(mults),
               "rand_p90": np.percentile(mults, 90),
               "rand_p10": np.percentile(mults, 10)}
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(grid: pd.DataFrame, benchmarks=("qqq", "spy")) -> str:
    lines = []
    for b in benchmarks:
        col = f"vs_{b}"
        win = (grid[col] > 0).mean()
        lines.append(
            f"vs {b.upper()}: win-rate {win:.0%} over {len(grid)} windows | "
            f"median excess {grid[col].median():+.1%} | "
            f"worst {grid[col].min():+.1%} | best {grid[col].max():+.1%}")
    return "\n".join(lines)
