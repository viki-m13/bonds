"""Robustness check 4 — deflated Sharpe ratio.

Bailey & López de Prado (2014).  Given N=323 configs tested on the same
window, the *expected* maximum Sharpe of pure-noise strategies with the
same return distribution is much higher than zero.  The deflated Sharpe
asks: given the observed max-SR and the other SRs, what's the probability
that the maximum is due to luck rather than skill?

Inputs:
  - the SR distribution across all tested configs (from letf_is_oos_all.csv)
  - the return distribution (skewness, kurtosis) of the single top strategy

Formula:
  E[max SR] ≈ σ_SR * [(1 - γ) Φ^-1(1 - 1/N) + γ Φ^-1(1 - 1/(N e))]
  where γ = Euler-Mascheroni ≈ 0.5772
  DSR = Φ( (SR_0 - E[max SR]) sqrt(T-1) /
          sqrt(1 - skew · SR_0 + (kurt-1)/4 · SR_0^2) )

We compute DSR for each family's top-1 (by IS Sharpe) on the FULL sample.
If DSR < 0.95, the headline is statistically indistinguishable from luck.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scs

from letf_engine import (common_window_returns, run_backtest, w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011


OUT = Path("/home/user/bonds/data/results")
GAMMA = 0.5772156649  # Euler-Mascheroni


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


def daily_sharpe(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0: return 0, 0, 0, 0, 0
    mu = r.mean(); sd = r.std()
    sk = scs.skew(r.values)
    ku = scs.kurtosis(r.values, fisher=False)  # Pearson (normal=3)
    sr_daily = mu / sd
    sr_ann = sr_daily * np.sqrt(252)
    return sr_daily, sr_ann, sk, ku, len(r)


def deflated_sharpe(sr_daily, sr_distribution, T, skew, kurt):
    """Returns DSR (probability that true SR > 0 given the max came from N trials)."""
    N = len(sr_distribution)
    sr_daily_ref = sr_distribution / np.sqrt(252)  # convert ann->daily
    sigma_sr = float(np.std(sr_daily_ref, ddof=1))
    if sigma_sr <= 0 or T <= 1:
        return np.nan
    # Expected max sharpe (daily) under H0 of no-skill + same vol as observed set
    z1 = scs.norm.ppf(1 - 1/N)
    z2 = scs.norm.ppf(1 - 1/(N * np.e))
    e_max = sigma_sr * ((1 - GAMMA) * z1 + GAMMA * z2)
    # Denominator adjusts for non-normality (higher kurt = wider uncertainty)
    denom = np.sqrt((1 - skew * sr_daily + (kurt - 1) / 4 * sr_daily**2) / (T - 1))
    z = (sr_daily - e_max) / denom
    return float(scs.norm.cdf(z)), float(e_max * np.sqrt(252)), sigma_sr


def main():
    # Re-run the winners so we have their daily return series
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    core6 = ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"]
    clean4 = ["UPRO","TQQQ","TMF","UGL"]
    ew_all17 = {t: 1/len(LETF_LONG_2011) for t in LETF_LONG_2011}

    winners = [
        ("static", "HFEA-Tech 50/50 TQQQ/TMF", w_fixed({"TQQQ":0.5,"TMF":0.5})),
        ("invvol", "invvol clean4 lb=21", invvol_fn(clean4, 21)),
        ("invvol-scaled", "invvol-s core6 lb=126 tv=60%",
         invvol_scaled_fn(core6, 126, 0.60)),
        ("mom", "mom core6 lb=126 top5", mom_fn(core6, 126, 5)),
        ("baseline", "EW-all17", w_fixed(ew_all17)),
    ]

    # Load the full IS/OOS merged sheet (we need the SR distribution)
    all_df = pd.read_csv(OUT / "letf_is_oos_all.csv")
    # use full-sample SR (re-run on all data) — easiest approx: use is_sharpe
    # as the per-family distribution.

    print(f"N configs in distribution = {len(all_df)}")
    print(f"SR (IS) quantiles:   min={all_df.is_sharpe.min():.2f}  "
          f"25%={all_df.is_sharpe.quantile(.25):.2f}  "
          f"med={all_df.is_sharpe.median():.2f}  "
          f"75%={all_df.is_sharpe.quantile(.75):.2f}  "
          f"max={all_df.is_sharpe.max():.2f}")
    sr_dist = all_df.is_sharpe.values

    rows = []
    for fam, name, fn in winners:
        r, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
        sr_d, sr_a, sk, ku, T = daily_sharpe(r)
        dsr, e_max, sigma_sr = deflated_sharpe(sr_d, sr_dist, T, sk, ku)
        rows.append({
            "family": fam, "label": name, "T": T,
            "sr_ann": sr_a, "skew": sk, "kurt": ku,
            "E[max SR] ann": e_max,
            "sigma_SR (ann)": sigma_sr * np.sqrt(252),
            "DSR": dsr,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_deflated_sharpe.csv", index=False)

    print("\n=== Deflated Sharpe ratio (full 2011-2026) ===")
    print("SR_ref distribution = 323 strategies tested on IS period.")
    print("DSR = P(true SR > 0 given selection bias & return non-normality).\n")
    print(f"{'Family':<15s} {'Strategy':<30s} {'SR':>5s} {'Skew':>6s} "
          f"{'Kurt':>6s} {'E[max SR]':>10s} {'DSR':>6s}")
    print("-"*90)
    for r in rows:
        print(f"{r['family']:<15s} {r['label']:<30s} "
              f"{r['sr_ann']:>5.2f} {r['skew']:>6.2f} {r['kurt']:>6.2f} "
              f"{r['E[max SR] ann']:>10.2f} {r['DSR']:>6.2%}")

    print("\nDSR > 95% is conventional threshold for 'real' skill.")
    print("DSR < 95% means we cannot reject the hypothesis that the SR is luck"
          " (given we tested 323 configs).")


if __name__ == "__main__":
    main()
