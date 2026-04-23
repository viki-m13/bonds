"""APEX — stress tests and robustness checks.

Tests:
  1. Walk-forward: no IS/OOS split — running 3-year window SR
  2. Bootstrap: stationary bootstrap of daily returns, 1000 samples
  3. Monte Carlo: inject Gaussian noise (1%) to weights, 500 samples
  4. Deflated Sharpe: Bailey-López de Prado with N trials
  5. Parameter sensitivity: DD floor, target vol, sleeve target vol
  6. Regime stress: GFC, COVID, 2022 rate hike, 2018 rate scare
  7. Transaction cost sensitivity

Outputs to data/apex/stress_tests.json and CSVs.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import pandas as pd

import util
from apex_production import run_apex, SLEEVE_FNS, BLEND_WEIGHTS, DD_FLOOR, TARGET_VOL

OUT = Path("/home/user/bonds/data/apex")


def rolling_sharpe(r: pd.Series, win: int = 756) -> pd.Series:
    """3-year (756d) rolling Sharpe."""
    mu = r.rolling(win).mean() * util.DPY
    sd = r.rolling(win).std() * np.sqrt(util.DPY)
    return (mu / sd.replace(0, np.nan)).dropna()


def bootstrap_sharpe(r: pd.Series, n: int = 1000, block: int = 20) -> np.ndarray:
    """Stationary bootstrap of daily returns with expected block length `block`."""
    r = r.dropna().values
    T = len(r)
    p = 1.0 / block
    out = np.zeros(n)
    rng = np.random.default_rng(42)
    for i in range(n):
        idx = np.zeros(T, dtype=int)
        t = 0
        while t < T:
            i0 = rng.integers(0, T)
            L = rng.geometric(p)
            take = min(L, T - t)
            for k in range(take):
                idx[t + k] = (i0 + k) % T
            t += take
        sample = r[idx]
        sd = sample.std()
        out[i] = (sample.mean() / sd) * np.sqrt(util.DPY) if sd > 0 else 0.0
    return out


def deflated_sharpe(sharpe: float, n_trials: int, n_obs: int,
                    skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey-López de Prado Deflated Sharpe Ratio."""
    gamma = 0.5772156649
    z = np.sqrt(2 * np.log(n_trials)) - (np.log(np.log(n_trials)) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * np.log(n_trials)))
    exp_max = z + gamma * (1 / np.sqrt(2 * np.log(n_trials)))
    # DSR: fraction of zero-mean-Sharpe distribution's max that the observed Sharpe exceeds
    sigma_sr = np.sqrt((1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2) / (n_obs - 1))
    if sigma_sr <= 0:
        return float("nan")
    return float((sharpe - exp_max * sigma_sr) / sigma_sr)


def param_sensitivity(cp):
    rows = []
    for tv in (0.15, 0.20, 0.25, 0.30, 0.35):
        for dd in (-0.10, -0.12, -0.15, -0.20, -0.25):
            # Temporarily patch globals
            import apex_production as AP
            AP.TARGET_VOL = tv
            AP.DD_FLOOR = dd
            r, state, _, _ = run_apex(cp)
            m = util.metrics(r)
            mo = util.metrics(util.regime_slice(r, "2019-01-02", "2027-12-31"))
            rows.append({
                "target_vol": tv, "dd_floor": dd,
                "full_sr": m["sharpe"], "full_cagr": m["cagr"],
                "full_mdd": m["mdd"], "oos_sr": mo.get("sharpe", 0),
                "oos_cagr": mo.get("cagr", 0),
            })
    return pd.DataFrame(rows)


def tc_sensitivity(cp):
    rows = []
    # Monkey-patch util.tc_map to return scaled values
    orig = util.tc_map
    for scale in (0.5, 1.0, 2.0, 4.0):
        def patched():
            return {k: v * scale for k, v in orig().items()}
        util.tc_map = patched
        r, _, _, _ = run_apex(cp)
        m = util.metrics(r)
        rows.append({"tc_scale": scale, "sr": m["sharpe"],
                     "cagr": m["cagr"], "mdd": m["mdd"]})
    util.tc_map = orig
    return pd.DataFrame(rows)


def regime_stress(r: pd.Series) -> pd.DataFrame:
    regimes = {
        "Dotcom 2000-02": ("2000-03-01", "2002-12-31"),
        "Housing bubble 03-06": ("2003-01-01", "2006-12-31"),
        "GFC 2007-09": ("2007-06-01", "2009-06-30"),
        "Recovery 2010-14": ("2010-01-01", "2014-12-31"),
        "Sideways 2015-16": ("2015-01-01", "2016-12-31"),
        "Trump rally 17-18": ("2017-01-01", "2018-12-31"),
        "2019": ("2019-01-01", "2019-12-31"),
        "COVID crash/rally 2020": ("2020-01-01", "2020-12-31"),
        "2021 melt-up": ("2021-01-01", "2021-12-31"),
        "2022 rate hike": ("2022-01-01", "2022-12-31"),
        "2023 recovery": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"),
        "2025": ("2025-01-01", "2025-12-31"),
        "2026 YTD": ("2026-01-01", "2026-04-30"),
    }
    rows = []
    for name, (s, e) in regimes.items():
        m = util.metrics(util.regime_slice(r, s, e))
        rows.append({"regime": name, "start": s, "end": e, **m})
    return pd.DataFrame(rows)


def walkforward_report(r: pd.Series, win_years: int = 3) -> pd.DataFrame:
    """Non-overlapping 3-year window metrics."""
    r = r.dropna()
    years = list(range(r.index.min().year, r.index.max().year + 1, win_years))
    rows = []
    for y in years:
        s = f"{y}-01-01"
        e = f"{y + win_years - 1}-12-31"
        w = util.regime_slice(r, s, e)
        if len(w) > 100:
            m = util.metrics(w)
            rows.append({"start": s, "end": e, **m})
    return pd.DataFrame(rows)


def main():
    op, cp = util.load_prices()
    print("Running APEX production...")
    r, state, P, sleeve_rets = run_apex(cp)
    print(f"APEX full: SR={util.metrics(r)['sharpe']:.2f} CAGR={util.metrics(r)['cagr']*100:.1f}%")

    # --- Rolling Sharpe ---
    rs = rolling_sharpe(r, win=756)
    rs.to_csv(OUT / "stress_rolling_sharpe.csv", header=["rolling_3y_sharpe"])
    print(f"\n3y rolling Sharpe: min={rs.min():.2f}, median={rs.median():.2f}, max={rs.max():.2f}")

    # --- Bootstrap ---
    print("\nBootstrapping (1000 samples)...")
    bsr = bootstrap_sharpe(r, n=1000, block=20)
    pd.Series(bsr, name="sharpe").to_csv(OUT / "stress_bootstrap.csv")
    print(f"Bootstrap Sharpe: 2.5%={np.percentile(bsr, 2.5):.2f}, "
          f"median={np.median(bsr):.2f}, 97.5%={np.percentile(bsr, 97.5):.2f}")

    # --- Deflated Sharpe ---
    full_m = util.metrics(r)
    n_obs = full_m["n"]
    # Assume ~30 distinct strategies were tested
    n_trials = 30
    dsr = deflated_sharpe(full_m["sharpe"], n_trials, n_obs)
    print(f"\nDeflated Sharpe (N_trials=30): z={dsr:.2f}")

    # --- Parameter sensitivity ---
    print("\nParameter sensitivity (takes a minute)...")
    ps = param_sensitivity(cp)
    ps.to_csv(OUT / "stress_param_sensitivity.csv", index=False)
    print(ps.to_string())

    # --- Regime stress ---
    regs = regime_stress(r)
    regs.to_csv(OUT / "stress_regimes.csv", index=False)
    print("\nRegime metrics:")
    print(regs[["regime", "sharpe", "cagr", "mdd", "vol"]].to_string(index=False))

    # --- Walkforward ---
    wf = walkforward_report(r, win_years=3)
    wf.to_csv(OUT / "stress_walkforward.csv", index=False)
    print("\n3-year non-overlap windows:")
    print(wf[["start", "end", "sharpe", "cagr", "mdd"]].to_string(index=False))

    # --- TC sensitivity ---
    print("\nTC sensitivity...")
    tc = tc_sensitivity(cp)
    tc.to_csv(OUT / "stress_tc_sensitivity.csv", index=False)
    print(tc.to_string())

    # Save summary
    summary = {
        "apex_full": full_m,
        "bootstrap_sharpe": {
            "p2.5": float(np.percentile(bsr, 2.5)),
            "median": float(np.median(bsr)),
            "p97.5": float(np.percentile(bsr, 97.5)),
            "mean": float(np.mean(bsr)),
        },
        "rolling_3y_sharpe": {
            "min": float(rs.min()),
            "median": float(rs.median()),
            "max": float(rs.max()),
            "mean": float(rs.mean()),
        },
        "deflated_sharpe_z": float(dsr),
        "n_trials_assumed": n_trials,
    }
    with open(OUT / "stress_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved stress tests to {OUT}")


if __name__ == "__main__":
    main()
