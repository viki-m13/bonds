"""Priority 1 — effective-N correction to deflated Sharpe.

The previous letf_deflated_sharpe.py used N=323 for the Bailey & Lopez de Prado
expected-max-SR-under-null. That assumes the 323 strategies are INDEPENDENT.
They are emphatically not: most invvol configs differ only in one lookback
parameter, most statics share underlying tickers. Their realised returns are
heavily correlated.

The standard fix: compute effective N via
  N_eff = (Σ λ_i)^2 / Σ λ_i^2        (participation-ratio / PCA)
where λ_i are eigenvalues of the daily-return correlation matrix across
strategies. This is the Roll-1988 formulation; same as 1 / Σ w_i^2 on
normalised eigenvalue weights.

We then recompute E[max SR | N_eff] and the DSR. If N_eff is 5-15 (not 323),
the headline strategies may actually pass the 95% threshold.

Also compute DSR within each family separately for a tighter correction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as scs

from letf_engine import (common_window_returns, run_backtest, w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011
from letf_sweep_static import build_recipes

OUT = Path("/home/user/bonds/data/results")
GAMMA = 0.5772156649


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


def build_all_grids():
    grids = {}
    for name, w in build_recipes().items():
        grids[name] = ("static", w_fixed(w))
    baskets = {
        "core6":  ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
        "big8":   ["UPRO","TQQQ","SOXL","TECL","TMF","UGL","FAS","EDC"],
        "all17":  LETF_LONG_2011,
        "mix5":   ["UPRO","TQQQ","TMF","UGL","UCO"],
        "clean4": ["UPRO","TQQQ","TMF","UGL"],
    }
    for bn, ts in baskets.items():
        for lb in (21, 63, 126, 252):
            grids[f"invvol {bn} lb={lb}"] = ("invvol", invvol_fn(ts, lb))
            for tv in (0.25, 0.40, 0.60, 0.80):
                grids[f"invvol-s {bn} lb={lb} tv={int(tv*100)}%"] = \
                    ("invvol-scaled", invvol_scaled_fn(ts, lb, tv))
    mbask = {
        "core6": ["UPRO","TQQQ","SOXL","TECL","TMF","UGL"],
        "big8":  ["UPRO","TQQQ","SOXL","TECL","TMF","UGL","FAS","EDC"],
        "all17": LETF_LONG_2011,
    }
    for bn, ts in mbask.items():
        for lb in (21, 63, 126, 252):
            for n in (1,2,3,4,5):
                if n > len(ts): continue
                grids[f"mom {bn} lb={lb} top{n}"] = ("mom", mom_fn(ts, lb, n))
    return grids


def effective_n(ret_matrix):
    """Participation ratio from eigenvalues of correlation matrix."""
    # ret_matrix: T x N of daily returns per strategy
    corr = np.corrcoef(ret_matrix, rowvar=False)
    eigvals = np.linalg.eigvalsh(corr)
    eigvals = eigvals[eigvals > 1e-12]
    neff = (eigvals.sum() ** 2) / (eigvals ** 2).sum()
    return float(neff), eigvals


def e_max_sr(sigma_sr, N):
    """E[max SR under null] given N independent trials with equal sigma_SR."""
    z1 = scs.norm.ppf(1 - 1/N)
    z2 = scs.norm.ppf(1 - 1/(N * np.e))
    return sigma_sr * ((1 - GAMMA) * z1 + GAMMA * z2)


def dsr_one(sr_daily, T, skew, kurt, e_max_daily):
    denom = np.sqrt((1 - skew * sr_daily + (kurt - 1)/4 * sr_daily**2) / (T - 1))
    z = (sr_daily - e_max_daily) / denom
    return float(scs.norm.cdf(z))


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)
    grids = build_all_grids()
    print(f"Running {len(grids)} strategies on full 2011-2026 for return matrix...")

    ret_mat = np.empty((len(rets), len(grids)))
    labels = []
    families = []
    sr_ann = np.empty(len(grids))
    skews = np.empty(len(grids))
    kurts = np.empty(len(grids))
    for j, (name, (fam, fn)) in enumerate(grids.items()):
        r, _ = run_backtest(rets, fn, rebal_days=21, exec_lag=1)
        r = r.reindex(rets.index).fillna(0)
        ret_mat[:, j] = r.values
        labels.append(name); families.append(fam)
        mu = r.mean(); sd = r.std()
        sr_ann[j] = (mu * 252) / (sd * np.sqrt(252)) if sd > 0 else 0
        skews[j] = scs.skew(r.values)
        kurts[j] = scs.kurtosis(r.values, fisher=False)
        if (j+1) % 50 == 0:
            print(f"  ...{j+1}/{len(grids)}")

    T = len(rets)
    # Overall effective N
    neff_all, evs = effective_n(ret_mat)
    print(f"\nRaw N = {len(grids)}")
    print(f"Effective N (global) = {neff_all:.1f}")
    print(f"Top eigenvalues (first 5): {np.sort(evs)[::-1][:5]}")
    print(f"  -> top eigenvalue explains "
          f"{np.sort(evs)[::-1][0]/evs.sum()*100:.1f}% of variance")

    # sigma_SR on ANN basis, then convert to daily for formulas
    sigma_sr_ann = np.std(sr_ann, ddof=1)
    sigma_sr_d = sigma_sr_ann / np.sqrt(252)

    # Headline winners
    winners = {
        "static": "HFEA-Tech 50/50 TQQQ/TMF",
        "invvol": "invvol clean4 lb=21",
        "invvol-scaled": "invvol-s core6 lb=126 tv=60%",
        "mom": "mom core6 lb=126 top5",
    }

    print(f"\nσ(SR) ann = {sigma_sr_ann:.3f}  (σ(SR) daily = {sigma_sr_d:.4f})")
    print(f"\n{'='*95}")
    print(f"{'Strategy':<32s} {'SR':>5s} {'N_raw':>7s} {'N_eff':>6s} {'DSR_raw':>8s} {'DSR_eff':>8s} {'DSR_fam':>8s}")
    print("-"*95)

    # Per-family effective N
    fam_to_neff = {}
    fam_to_sigma = {}
    fam_to_maxsr = {}
    for fam in sorted(set(families)):
        idx = [i for i, f in enumerate(families) if f == fam]
        mat = ret_mat[:, idx]
        neff_f, _ = effective_n(mat)
        fam_to_neff[fam] = neff_f
        sr_fam = sr_ann[idx]
        fam_to_sigma[fam] = np.std(sr_fam, ddof=1) / np.sqrt(252)
        fam_to_maxsr[fam] = max(sr_fam)
        print(f"  family={fam:<15s}  N={len(idx)}  N_eff={neff_f:.1f}  "
              f"σ(SR) daily={fam_to_sigma[fam]:.4f}  max SR ann={fam_to_maxsr[fam]:.2f}")

    print()
    # Recompute DSR
    rows = []
    for j, name in enumerate(labels):
        if name not in winners.values():
            continue
        fam = families[j]
        sr_d = sr_ann[j] / np.sqrt(252)
        # raw-N DSR
        e_max_raw = e_max_sr(sigma_sr_d, len(grids))
        dsr_raw = dsr_one(sr_d, T, skews[j], kurts[j], e_max_raw)
        # effective-N DSR (global)
        e_max_eff = e_max_sr(sigma_sr_d, max(neff_all, 2))
        dsr_eff = dsr_one(sr_d, T, skews[j], kurts[j], e_max_eff)
        # family-only DSR
        e_max_fam = e_max_sr(fam_to_sigma[fam], max(fam_to_neff[fam], 2))
        dsr_fam = dsr_one(sr_d, T, skews[j], kurts[j], e_max_fam)
        rows.append({
            "family": fam, "label": name, "sr": sr_ann[j],
            "dsr_raw_N323": dsr_raw,
            "dsr_eff_N": dsr_eff,
            "dsr_family_only": dsr_fam,
            "N_eff_global": neff_all, "N_eff_family": fam_to_neff[fam],
            "skew": skews[j], "kurt": kurts[j],
        })
        print(f"{name:<32s} {sr_ann[j]:>5.2f} {len(grids):>7d} {neff_all:>6.1f} "
              f"{dsr_raw:>7.1%} {dsr_eff:>7.1%} {dsr_fam:>7.1%}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_dsr_effn.csv", index=False)

    print(f"\nInterpretation:")
    print(f"  DSR_raw     uses N={len(grids)} (over-correction, assumes independence)")
    print(f"  DSR_eff     uses global N_eff={neff_all:.1f} (PCA-based)")
    print(f"  DSR_fam     uses family-specific N_eff (tightest correction for the family's winner)")
    print(f"  > 95% = statistically real after multiple-testing correction")


if __name__ == "__main__":
    main()
