"""Standalone LETF strategy search — highest CAGR with no daily vol scaling.

Universe: leveraged ETFs only (UPRO 3x SPY, TQQQ 3x QQQ, TMF 3x TLT, UGL 2x GLD,
SOXL 3x SOX, TECL 3x Tech, FAS 3x Financials, SSO 2x SPY, QLD 2x QQQ, etc.)

Rules of the game:
  * No intraday/daily vol targeting (the whole point).
  * Rebalance every N days, N ∈ {3, 5, 10, 21}.
  * Static target weights, or simple momentum/inv-vol-at-rebalance selection.
  * 15 bps on turnover, 1-day signal lag.

We report CAGR, vol, MDD, Sharpe, CAGR/MDD ratio over the full common window
for every strategy × rebalance-period combo, and the best configurations.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats

TC_BPS = 15.0

LETFS = {
    "UPRO": 3.0, "TQQQ": 3.0, "TMF": 3.0, "UGL": 2.0, "SOXL": 3.0,
    "TECL": 3.0, "SSO": 2.0, "QLD": 2.0, "FAS": 3.0, "EDC": 3.0,
    "DRN": 3.0, "ERX": 2.0, "TYD": 3.0, "UCO": 2.0, "YINN": 3.0,
    "NUGT": 2.0,
}


def load_prices(tickers, start=None):
    frames = {}
    for t in tickers:
        s = load_etf(t)
        if s is None:
            continue
        frames[t] = s
    px = pd.DataFrame(frames).sort_index()
    if start is not None:
        px = px.loc[start:]
    return px


def periodic_rebal_returns(rets, weights_fn, rebal_days, tc_bps=TC_BPS):
    """Run a rebalancing strategy given:
       rets         : DataFrame of daily % returns
       weights_fn   : function(date, history_rets) -> pd.Series of target weights
                      (indexed by ticker). History is rets.loc[:date-1] so no lookahead.
                      Returns None to keep prior weights.
       rebal_days   : integer N days between rebalances
    """
    idx = rets.index
    cols = rets.columns
    w = pd.Series(0.0, index=cols)
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    rebal_mask = pd.Series(False, index=idx)
    rebal_mask.iloc[::rebal_days] = True

    last_w = w.copy()
    tc_daily = pd.Series(0.0, index=idx)
    prev_w = pd.Series(0.0, index=cols)

    for i, d in enumerate(idx):
        if rebal_mask.iloc[i]:
            hist = rets.iloc[:i]  # all data strictly before d
            new_w = weights_fn(d, hist)
            if new_w is not None:
                new_w = new_w.reindex(cols).fillna(0.0)
                # turnover cost charged on d (executes on this bar)
                turnover = (new_w - prev_w).abs().sum()
                tc_daily.iloc[i] = turnover * (tc_bps / 1e4)
                prev_w = new_w
                last_w = new_w
        W.iloc[i] = last_w

    # drift: holding fixed weights still rebalances implicitly daily in this model;
    # that's a constant-mix approximation. Close enough for the exploration given
    # the small gap between rebalances.
    port_ret = (W * rets).sum(axis=1) - tc_daily
    return port_ret, W


def fixed_weights(target):
    t = pd.Series(target, dtype=float)
    s = t.sum()
    if s > 0:
        t = t / s if abs(s - 1.0) < 0.001 else t  # don't normalise unless intended ~1
    def fn(d, hist):
        return t
    return fn


def raw_fixed_weights(target):
    """Use weights as given (may not sum to 1)."""
    t = pd.Series(target, dtype=float)
    def fn(d, hist):
        return t
    return fn


def momentum_topn(lookback_days, top_n):
    def fn(d, hist):
        if len(hist) < lookback_days + 5:
            return None
        r = hist.iloc[-lookback_days:]
        # compound return
        cum = (1 + r).prod() - 1
        cum = cum.dropna().sort_values(ascending=False)
        picks = cum.head(top_n).index.tolist()
        w = pd.Series(0.0, index=hist.columns)
        if picks:
            w.loc[picks] = 1.0 / len(picks)
        return w
    return fn


def inv_vol_weights(lookback_days, tickers):
    def fn(d, hist):
        if len(hist) < lookback_days + 5:
            return None
        r = hist.iloc[-lookback_days:][tickers].dropna(axis=1, how="any")
        if r.shape[1] == 0:
            return None
        vol = r.std()
        if (vol <= 0).all():
            return None
        inv = (1 / vol.replace(0, np.nan)).fillna(0)
        w = pd.Series(0.0, index=hist.columns)
        w.loc[inv.index] = inv / inv.sum()
        return w
    return fn


def run_summary(r, label):
    s = stats(r.dropna(), label)
    nav = (1 + r).cumprod()
    cagr = (nav.iloc[-1]) ** (252 / len(r)) - 1
    mdd = (nav / nav.cummax() - 1).min()
    cagr_mdd = cagr / abs(mdd) if mdd < 0 else float("inf")
    return {
        "label": label,
        "n": s["n"],
        "cagr": cagr * 100,
        "vol": s["vol"],
        "mdd": mdd * 100,
        "sharpe": s["sharpe"],
        "cagr_mdd": cagr_mdd,
        "navx": s["navx"],
    }


def main():
    # Common window starts when every core candidate is live.
    # SOXL/TQQQ live from early 2010; use 2011-01-01 for safety.
    start = "2011-01-01"
    tickers = list(LETFS.keys())
    px = load_prices(tickers, start=start)
    # Keep only columns that have data throughout the window
    px = px.dropna(how="any")
    rets = px.pct_change().dropna(how="all").fillna(0)
    print(f"Universe: {list(px.columns)}")
    print(f"Window:   {rets.index[0].date()} .. {rets.index[-1].date()}  "
          f"({len(rets)} days)")
    print()

    strategies = []

    # ---- Static weight recipes ----
    recipes = {
        "100% UPRO":           {"UPRO": 1.0},
        "100% TQQQ":           {"TQQQ": 1.0},
        "100% SOXL":           {"SOXL": 1.0},
        "100% TECL":           {"TECL": 1.0},
        "HFEA 55/45 UPRO/TMF": {"UPRO": 0.55, "TMF": 0.45},
        "HFEA 60/40 UPRO/TMF": {"UPRO": 0.60, "TMF": 0.40},
        "HFEA 65/35 UPRO/TMF": {"UPRO": 0.65, "TMF": 0.35},
        "HFEA 70/30 UPRO/TMF": {"UPRO": 0.70, "TMF": 0.30},
        "HFEA-Tech 55/45 TQQQ/TMF": {"TQQQ": 0.55, "TMF": 0.45},
        "HFEA-Tech 60/40 TQQQ/TMF": {"TQQQ": 0.60, "TMF": 0.40},
        "HFEA-Tech 65/35 TQQQ/TMF": {"TQQQ": 0.65, "TMF": 0.35},
        "TMF/UPRO 50/50":      {"UPRO": 0.50, "TMF": 0.50},
        "Risk-parity-ish UPRO/TMF/UGL 40/40/20": {"UPRO": 0.40, "TMF": 0.40, "UGL": 0.20},
        "UPRO/TMF/UGL 50/30/20": {"UPRO": 0.50, "TMF": 0.30, "UGL": 0.20},
        "UPRO/TQQQ/TMF 30/30/40": {"UPRO": 0.30, "TQQQ": 0.30, "TMF": 0.40},
        "UPRO/TQQQ/TMF 25/35/40": {"UPRO": 0.25, "TQQQ": 0.35, "TMF": 0.40},
        "5-sleeve 20/20/20/20/20 UPRO/TQQQ/SOXL/TMF/UGL":
            {"UPRO": 0.20, "TQQQ": 0.20, "SOXL": 0.20, "TMF": 0.20, "UGL": 0.20},
        "UPRO/TQQQ/SOXL/TECL/TMF 15/15/15/15/40":
            {"UPRO": 0.15, "TQQQ": 0.15, "SOXL": 0.15, "TECL": 0.15, "TMF": 0.40},
        "Equal-weight all LETFs":
            {t: 1.0 / len(px.columns) for t in px.columns},
    }

    for label, w in recipes.items():
        strategies.append((f"[static] {label}", fixed_weights(w)))

    # ---- Momentum top-N ----
    for lb in (63, 126):
        for k in (1, 2, 3):
            strategies.append((f"[mom{lb}d-top{k}]", momentum_topn(lb, k)))

    # ---- Inverse-vol across a core LETF basket ----
    core = ["UPRO", "TQQQ", "SOXL", "TECL", "TMF", "UGL"]
    for lb in (63, 126):
        strategies.append((f"[invvol{lb}d core6]", inv_vol_weights(lb, core)))

    # ---- Run everything at multiple rebalance cadences ----
    rebal_set = [3, 5, 10, 21]
    rows = []
    equities = {}
    for sname, wfn in strategies:
        for nd in rebal_set:
            try:
                r, W = periodic_rebal_returns(rets, wfn, nd)
            except Exception as e:
                continue
            r = r.dropna()
            if len(r) < 200:
                continue
            res = run_summary(r, f"{sname} @ {nd}d")
            rows.append(res)
            equities[res["label"]] = r

    df = pd.DataFrame(rows)
    df = df.sort_values("cagr", ascending=False).reset_index(drop=True)

    print(f"{'Strategy @ Rebal':62s}  {'CAGR%':>7s}  {'Vol%':>6s}  {'MDD%':>8s}  "
          f"{'SR':>5s}  {'C/MDD':>6s}  {'NAVx':>8s}")
    print("-" * 115)
    for _, r in df.iterrows():
        print(f"{r['label']:62s}  {r['cagr']:>7.2f}  {r['vol']:>6.2f}  "
              f"{r['mdd']:>8.2f}  {r['sharpe']:>5.2f}  "
              f"{r['cagr_mdd']:>6.2f}  {r['navx']:>8.1f}")

    # Sorted by risk-adjusted CAGR/MDD (much more meaningful for LETFs)
    print("\nTop 15 by CAGR / |MDD|:")
    df2 = df.sort_values("cagr_mdd", ascending=False).head(15)
    for _, r in df2.iterrows():
        print(f"  {r['label']:62s}  CAGR={r['cagr']:>6.2f}%  "
              f"MDD={r['mdd']:>7.2f}%  C/MDD={r['cagr_mdd']:>5.2f}  "
              f"SR={r['sharpe']:>4.2f}")

    # Top 10 by Sharpe
    print("\nTop 10 by Sharpe:")
    df3 = df.sort_values("sharpe", ascending=False).head(10)
    for _, r in df3.iterrows():
        print(f"  {r['label']:62s}  SR={r['sharpe']:>4.2f}  "
              f"CAGR={r['cagr']:>6.2f}%  MDD={r['mdd']:>7.2f}%")

    # Save best-by-CAGR daily returns
    best = df.iloc[0]
    out_path = Path("/home/user/bonds/data/results/letf_best_returns.csv")
    eq_out = pd.DataFrame({best["label"]: equities[best["label"]]})
    eq_out.to_csv(out_path)
    print(f"\nWrote {out_path}  ({len(eq_out)} rows)  best by CAGR: {best['label']}")

    # Save summary
    df.to_csv(Path("/home/user/bonds/data/results/letf_strategy_summary.csv"), index=False)


if __name__ == "__main__":
    main()
