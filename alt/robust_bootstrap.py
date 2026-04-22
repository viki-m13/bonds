"""Robustness Test 5: Block-bootstrap null test.

Tests whether the observed Sharpe ratio is statistically distinguishable from
luck. Uses circular block bootstrap (block length = 21 days ~ 1 month) to
preserve autocorrelation structure.

Two tests:
  (A) Sharpe CI: resample 10,000 times, compute Sharpe each time, build
      percentile confidence interval for the POINT estimate.

  (B) p-value vs null: same resampling on a demeaned version (zero expected
      return), count fraction of bootstrap samples with Sharpe >= observed.
      This is the probability of seeing this Sharpe under pure luck.

Run on both 4-sleeve (PHOENIX v2) and 5-sleeve (+ crypto) blends.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

R = Path("data/results")

IS_END = "2018-12-31"
OOS_START = "2019-01-02"
BLOCK_LEN = 21
N_BOOTSTRAP = 10000
RNG_SEED = 42


def sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) == 0 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(252))


def circ_block_bootstrap(returns: np.ndarray, block_len: int, rng) -> np.ndarray:
    """Sample a path of same length using circular overlapping blocks."""
    n = len(returns)
    n_blocks = int(np.ceil(n / block_len))
    # Random starting indices for each block
    starts = rng.integers(0, n, size=n_blocks)
    out = np.empty(n_blocks * block_len, dtype=returns.dtype)
    for b, s in enumerate(starts):
        # Circular slice
        idx = (s + np.arange(block_len)) % n
        out[b*block_len:(b+1)*block_len] = returns[idx]
    return out[:n]


def bootstrap_ci(returns, n_iter, block_len, rng):
    """Confidence interval for Sharpe via block-bootstrap."""
    rs = np.empty(n_iter)
    for i in range(n_iter):
        sample = circ_block_bootstrap(returns, block_len, rng)
        rs[i] = sharpe(sample)
    return rs


def bootstrap_pvalue(returns, observed_sr, n_iter, block_len, rng):
    """p-value: P(Sharpe >= observed | null)."""
    # Demean returns to create zero-expected-return null
    zero_mean = returns - returns.mean()
    rs = np.empty(n_iter)
    for i in range(n_iter):
        sample = circ_block_bootstrap(zero_mean, block_len, rng)
        rs[i] = sharpe(sample)
    return float((rs >= observed_sr).mean()), rs


def main():
    # Load 4-sleeve and 5-sleeve returns
    v2 = pd.read_csv(R/"phoenix_v2_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]
    v2c = pd.read_csv(R/"phoenix_v2_crypto_returns.csv", parse_dates=["Date"]).set_index("Date")["ret"]

    results = {}
    for name, r in [("phoenix_v2_4sleeve", v2), ("phoenix_v2_5sleeve_crypto", v2c)]:
        print(f"\n=== {name} ===")
        rng = np.random.default_rng(RNG_SEED)
        for win_name, w_slice in [("FULL", slice(None)),
                                   ("IS", slice(None, IS_END)),
                                   ("OOS", slice(OOS_START, None))]:
            ret = r.loc[w_slice].dropna().values
            if len(ret) < 100: continue
            obs = sharpe(ret)
            # CI
            rs_ci = bootstrap_ci(ret, N_BOOTSTRAP, BLOCK_LEN, rng)
            ci5 = float(np.percentile(rs_ci, 5))
            ci95 = float(np.percentile(rs_ci, 95))
            ci50 = float(np.percentile(rs_ci, 50))
            # p-value
            pval, rs_null = bootstrap_pvalue(ret, obs, N_BOOTSTRAP, BLOCK_LEN, rng)
            se = float(rs_ci.std())
            print(f"  {win_name:6s}  obs SR={obs:5.2f}  "
                  f"95% CI [{ci5:5.2f}, {ci95:5.2f}]  "
                  f"SE={se:4.2f}  p(SR>=obs | null)={pval:.4f}")
            results[f"{name}__{win_name}"] = {
                "observed_sharpe": obs,
                "ci_5pct":  ci5,
                "ci_50pct": ci50,
                "ci_95pct": ci95,
                "se": se,
                "p_value_vs_null": pval,
                "n_obs": int(len(ret)),
            }

    Path(R/"robustness_bootstrap.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved robustness_bootstrap.json ({N_BOOTSTRAP} bootstrap iterations each)")


if __name__ == "__main__":
    main()
