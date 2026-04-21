"""Robustness check 2 — strict IS / OOS split.

Protocol:
  1. Fit period:  2011-01-01 .. 2019-01-01  (~8 years)
  2. Test period: 2019-01-01 .. 2026-04-02  (~7.3 years)

For each family (static / invvol / invvol-scaled / momentum) we:
  a. Enumerate a full grid of configurations (same grid as the published
     sweep — no sneaky filtering).
  b. Pick the config with the highest IS Sharpe.
  c. Freeze it. Report its OOS CAGR, vol, MDD, Sharpe.
  d. ALSO report the naive benchmark: equal-weight over the full 17 long
     LETF universe, and 60/40 SPY/AGG (from underlying proxies).

If the IS winner doesn't beat its family median OOS, it was overfit.
If its OOS Sharpe is ≤ half its IS Sharpe, same.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise,
                         w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")
IS_END   = "2019-01-01"
OOS_END  = "2026-12-31"


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


def invvol_scaled_fn(tickers, lookback, target_vol):
    def fn(d, hist):
        if len(hist) < lookback + 5: return None
        r = hist.iloc[-lookback:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0: return None
        sig = r.std() * np.sqrt(252)
        if (sig <= 0).all(): return None
        inv = 1 / sig.replace(0, np.nan).fillna(0)
        S = inv.sum()
        if S <= 0: return None
        w = inv / S
        c = 1.0 / S
        n = r.shape[1]
        naive = c * np.sqrt(n)
        k = min(target_vol / naive, 5.0) if naive > 0 else 1.0
        w = w * k
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


# --- config grids (identical to prior sweeps, modulo cadence = 21d) ---

def static_grid():
    from letf_sweep_static import build_recipes
    return {name: ("static", w_fixed(w)) for name, w in build_recipes().items()}


def invvol_grid():
    grid = {}
    baskets = {
        "core6":  ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
        "big8":   ["UPRO","TQQQ","SOXL","TECL","TMF","UGL","FAS","EDC"],
        "all17":  LETF_LONG_2011,
        "mix5":   ["UPRO","TQQQ","TMF","UGL","UCO"],
        "clean4": ["UPRO","TQQQ","TMF","UGL"],
    }
    for bn, ts in baskets.items():
        for lb in (21, 63, 126, 252):
            grid[f"invvol {bn} lb={lb}"] = ("invvol",
                                            invvol_fn(ts, lb))
            for tv in (0.25, 0.40, 0.60, 0.80):
                grid[f"invvol-s {bn} lb={lb} tv={int(tv*100)}%"] = \
                    ("invvol-scaled", invvol_scaled_fn(ts, lb, tv))
    return grid


def mom_grid():
    grid = {}
    baskets = {
        "core6": ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
        "big8":  ["UPRO","TQQQ","SOXL","TECL","TMF","UGL","FAS","EDC"],
        "all17": LETF_LONG_2011,
    }
    for bn, ts in baskets.items():
        for lb in (21, 63, 126, 252):
            for n in (1,2,3,4,5):
                if n > len(ts): continue
                grid[f"mom {bn} lb={lb} top{n}"] = ("mom", mom_fn(ts, lb, n))
    return grid


def run_all(rets, grid):
    """Run every strategy and return DataFrame of metrics."""
    rows = []
    for name, (fam, fn) in grid.items():
        r, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
        s = summarise(r, name)
        s["family"] = fam
        rows.append(s)
    return pd.DataFrame(rows)


def main():
    # 2011-2026 LETF-only window
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    is_rets = rets.loc[:IS_END]
    oos_rets = rets.loc[IS_END:OOS_END]

    print(f"IS  : {is_rets.index[0].date()} .. {is_rets.index[-1].date()} "
          f"({len(is_rets)} days)")
    print(f"OOS : {oos_rets.index[0].date()} .. {oos_rets.index[-1].date()} "
          f"({len(oos_rets)} days)")

    grids = {**static_grid(), **invvol_grid(), **mom_grid()}
    print(f"Evaluating {len(grids)} strategies on IS...")

    is_df = run_all(is_rets, grids)
    is_df = is_df.sort_values("sharpe", ascending=False).reset_index(drop=True)
    is_df.rename(columns={k: f"is_{k}" for k in
                          ["cagr","vol","mdd","sharpe","cagr_mdd","navx","n"]},
                  inplace=True)

    # Evaluate all on OOS
    print(f"Evaluating {len(grids)} strategies on OOS...")
    oos_df = run_all(oos_rets, grids)
    oos_df.rename(columns={k: f"oos_{k}" for k in
                           ["cagr","vol","mdd","sharpe","cagr_mdd","navx","n"]},
                  inplace=True)

    merged = is_df.merge(oos_df[["label"] + [c for c in oos_df.columns if c.startswith("oos_")]
                                 + ["family"]],
                         on=["label","family"])
    merged.to_csv(OUT / "letf_is_oos_all.csv", index=False)

    # Per-family: best by IS Sharpe, then show OOS
    print("\n=== IS-selected winners — OOS performance ===\n")
    for fam in sorted(merged["family"].unique()):
        sub = merged[merged.family == fam].sort_values("is_sharpe", ascending=False)
        best = sub.iloc[0]
        med_oos_sr = sub["oos_sharpe"].median()
        top5_oos = sub.head(5)["oos_sharpe"].mean()
        print(f"FAMILY: {fam}  ({len(sub)} configs)")
        print(f"  IS-best config: {best['label']}")
        print(f"    IS : CAGR={best['is_cagr']:>6.2f}%  SR={best['is_sharpe']:>4.2f}  "
              f"MDD={best['is_mdd']:>7.2f}%")
        print(f"    OOS: CAGR={best['oos_cagr']:>6.2f}%  SR={best['oos_sharpe']:>4.2f}  "
              f"MDD={best['oos_mdd']:>7.2f}%  "
              f"(degradation: ΔSR={best['oos_sharpe']-best['is_sharpe']:+.2f})")
        print(f"  Family medians: IS SR={sub['is_sharpe'].median():.2f}  "
              f"OOS SR={med_oos_sr:.2f}")
        print(f"  Top-5 IS avg OOS SR={top5_oos:.2f}")
        print()

    # Benchmark: all17 EW
    ew_w = {t: 1.0/len(LETF_LONG_2011) for t in LETF_LONG_2011}
    r_is, _ = run_backtest(is_rets, w_fixed(ew_w), rebal_days=21, exec_lag=1)
    r_oos, _ = run_backtest(oos_rets, w_fixed(ew_w), rebal_days=21, exec_lag=1)
    sb_is = summarise(r_is, "EW-all17")
    sb_oos = summarise(r_oos, "EW-all17")
    print(f"BENCH EW-all17  IS  CAGR={sb_is['cagr']:>5.2f}%  SR={sb_is['sharpe']:.2f}  "
          f"MDD={sb_is['mdd']:>6.2f}%")
    print(f"BENCH EW-all17  OOS CAGR={sb_oos['cagr']:>5.2f}%  SR={sb_oos['sharpe']:.2f}  "
          f"MDD={sb_oos['mdd']:>6.2f}%")

    # SPY buy-hold benchmark
    spy = load_etf("SPY")
    spy_r = spy.pct_change().fillna(0)
    r_is_spy = spy_r.loc[is_rets.index[0]:is_rets.index[-1]]
    r_oos_spy = spy_r.loc[oos_rets.index[0]:oos_rets.index[-1]]
    sb_is = summarise(r_is_spy, "SPY")
    sb_oos = summarise(r_oos_spy, "SPY")
    print(f"BENCH SPY BH    IS  CAGR={sb_is['cagr']:>5.2f}%  SR={sb_is['sharpe']:.2f}  "
          f"MDD={sb_is['mdd']:>6.2f}%")
    print(f"BENCH SPY BH    OOS CAGR={sb_oos['cagr']:>5.2f}%  SR={sb_oos['sharpe']:.2f}  "
          f"MDD={sb_oos['mdd']:>6.2f}%")


if __name__ == "__main__":
    main()
