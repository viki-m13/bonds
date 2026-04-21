"""Robustness check 3 — block-bootstrap CI on headline strategies.

For each contender we resample the daily return series with replacement in
blocks of ~21 days (1 month), compute CAGR/Sharpe/MDD on each resample, and
report the 2.5 / 50 / 97.5 percentiles.

If the 95% CI is wide (e.g. Sharpe 0.2 .. 1.5) we have no business claiming
one headline number. If the lower bound crosses zero (Sharpe) or crosses SPY
the strategy is no better than noise.
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
N_BOOT = 500
BLOCK = 21
RNG = np.random.default_rng(20260421)


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


def block_bootstrap_returns(r, n_boot=N_BOOT, block=BLOCK):
    """Moving-block bootstrap. Return matrix [n_boot x T_target]."""
    r = r.values
    n = len(r)
    target_len = n
    out = np.empty((n_boot, target_len))
    n_blocks = int(np.ceil(target_len / block))
    for b in range(n_boot):
        starts = RNG.integers(0, n - block + 1, size=n_blocks)
        samp = np.concatenate([r[s:s+block] for s in starts])[:target_len]
        out[b] = samp
    return out


def metrics_from_returns(r):
    """Quick CAGR/Sharpe/MDD on a 1D array of daily returns."""
    if len(r) < 20:
        return (0., 0., 0.)
    nav = np.cumprod(1 + r)
    cagr = nav[-1] ** (252/len(r)) - 1 if nav[-1] > 0 else -1
    vol = np.std(r, ddof=1) * np.sqrt(252)
    sr = (np.mean(r) * 252) / vol if vol > 0 else 0
    running_max = np.maximum.accumulate(nav)
    dd = nav / running_max - 1
    mdd = dd.min()
    return (cagr * 100, sr, mdd * 100)


def ci_row(label, r):
    """Compute point + 2.5/50/97.5 percentile for CAGR/SR/MDD."""
    pt = metrics_from_returns(r.values)
    boot = block_bootstrap_returns(r)
    cagrs = np.empty(N_BOOT); srs = np.empty(N_BOOT); mdds = np.empty(N_BOOT)
    for i in range(N_BOOT):
        c, s, m = metrics_from_returns(boot[i])
        cagrs[i] = c; srs[i] = s; mdds[i] = m
    return {
        "label": label, "n": len(r),
        "cagr": pt[0], "sr": pt[1], "mdd": pt[2],
        "cagr_lo": np.percentile(cagrs, 2.5), "cagr_md": np.percentile(cagrs, 50),
        "cagr_hi": np.percentile(cagrs, 97.5),
        "sr_lo": np.percentile(srs, 2.5), "sr_md": np.percentile(srs, 50),
        "sr_hi": np.percentile(srs, 97.5),
        "mdd_lo": np.percentile(mdds, 2.5), "mdd_md": np.percentile(mdds, 50),
        "mdd_hi": np.percentile(mdds, 97.5),
        "sr_p_neg": float((srs <= 0).mean()),
    }


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]
    ew_all17 = {t: 1/len(LETF_LONG_2011) for t in LETF_LONG_2011}

    strategies = [
        ("100% TQQQ (static)", w_fixed({"TQQQ": 1.0})),
        ("HFEA 55/45 UPRO/TMF", w_fixed({"UPRO":0.55,"TMF":0.45})),
        ("HFEA-Tech 50/50 TQQQ/TMF", w_fixed({"TQQQ":0.5,"TMF":0.5})),
        ("EW5 UPRO/TQQQ/SOXL/TMF/UGL",
         w_fixed({"UPRO":0.2,"TQQQ":0.2,"SOXL":0.2,"TMF":0.2,"UGL":0.2})),
        ("EW-all17 (naive)", w_fixed(ew_all17)),
        ("invvol clean4 lb=21", invvol_fn(clean4, 21)),
        ("invvol core6 lb=63", invvol_fn(core6, 63)),
        ("mom core6 lb=126 top4", mom_fn(core6, 126, 4)),
    ]

    rows = []
    for name, fn in strategies:
        r, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
        rows.append(ci_row(name, r))
        print(f"  done {name}")

    # SPY BH bootstrap
    spy = load_etf("SPY").pct_change().fillna(0)
    spy = spy.loc[rets.index[0]:rets.index[-1]]
    rows.append(ci_row("SPY BH", spy))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_bootstrap.csv", index=False)

    print(f"\nN_BOOT={N_BOOT}, block={BLOCK}d, full 2011-2026 window")
    print("="*100)
    print(f"{'Strategy':<32s} {'CAGR [95% CI]':>24s} {'Sharpe [95% CI]':>22s}"
          f" {'MDD [95% CI]':>24s} {'P(SR≤0)':>8s}")
    print("-"*100)
    for r in rows:
        print(f"{r['label']:<32s} "
              f"{r['cagr']:>5.1f}% [{r['cagr_lo']:>5.1f}, {r['cagr_hi']:>5.1f}] "
              f"{r['sr']:>4.2f} [{r['sr_lo']:>4.2f}, {r['sr_hi']:>4.2f}] "
              f"{r['mdd']:>6.1f}% [{r['mdd_lo']:>6.1f}, {r['mdd_hi']:>6.1f}] "
              f"{r['sr_p_neg']:>8.1%}")


if __name__ == "__main__":
    main()
