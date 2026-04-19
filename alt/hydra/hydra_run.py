"""HYDRA run v3 — cleaner vol handling, walk-forward, sleeve curation."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats
from hydra_sleeves_v3 import SLEEVES


PORT_VOL = 0.20
PORT_LEV_CAP = 5.0
MV_WINDOW = 252            # cov lookback for min-var
MV_SHRINK = 0.20           # diagonal shrinkage strength
MV_MAX_W = 0.25            # cap any single sleeve at 25% weight


def risk_parity_ensemble(sleeves_df, target_vol=PORT_VOL, window=63,
                         lev_cap=PORT_LEV_CAP):
    """Inverse-vol risk parity — baseline ensemble."""
    vols = sleeves_df.rolling(window).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = 1 / vols
    inv = inv.where(vols.notna(), 0)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raw = (w * sleeves_df).sum(axis=1)
    pv = raw.rolling(window).std().shift(1) * np.sqrt(252)
    scale = (target_vol / pv).clip(upper=lev_cap).fillna(0)
    return raw * scale, w, scale


def min_var_weights(cov, shrink=MV_SHRINK, w_max=MV_MAX_W):
    """Closed-form min-variance with shrinkage to identity and weight cap.
    cov: (n,n) sample cov. Returns w summing to 1, 0 ≤ w ≤ w_max.
    Iteratively binds weights that exceed w_max."""
    n = cov.shape[0]
    diag = np.diag(np.diag(cov))
    Cs = (1 - shrink) * cov + shrink * diag
    try:
        inv = np.linalg.pinv(Cs + 1e-6 * np.eye(n))
    except np.linalg.LinAlgError:
        return np.ones(n) / n
    w = inv @ np.ones(n)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        return np.ones(n) / n
    w = w / w.sum()
    # Enforce weight cap iteratively
    for _ in range(10):
        over = w > w_max
        if not over.any():
            break
        excess = (w[over] - w_max).sum()
        w[over] = w_max
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] = w[under] + excess * w[under] / w[under].sum()
        else:
            break
    return w


def min_var_ensemble(sleeves_df, target_vol=PORT_VOL, cov_window=MV_WINDOW,
                     vol_window=63, lev_cap=PORT_LEV_CAP, rebal_d=21):
    """Rolling min-variance ensemble with monthly rebal.
    - Compute 252d cov matrix
    - Solve for constrained min-var weights
    - Scale to target_vol at portfolio level"""
    n = sleeves_df.shape[1]
    cols = sleeves_df.columns
    dates = sleeves_df.index
    weights = pd.DataFrame(0.0, index=dates, columns=cols)
    last_w = np.zeros(n)
    for i in range(cov_window, len(dates)):
        if (i - cov_window) % rebal_d != 0:
            weights.iloc[i] = last_w
            continue
        window = sleeves_df.iloc[i - cov_window:i]
        # Only include sleeves with non-zero activity in window
        active = (window.abs().sum(axis=0) > 1e-8).values
        if active.sum() < 2:
            weights.iloc[i] = last_w
            continue
        sub = window.iloc[:, active]
        cov = sub.cov().values * 252
        w_sub = min_var_weights(cov)
        w_full = np.zeros(n)
        w_full[active] = w_sub
        last_w = w_full
        weights.iloc[i] = w_full
    weights = weights.shift(1).fillna(0)  # trade at next bar
    raw = (weights * sleeves_df).sum(axis=1)
    pv = raw.rolling(vol_window).std().shift(1) * np.sqrt(252)
    scale = (target_vol / pv).clip(upper=lev_cap).fillna(0)
    return raw * scale, weights, scale


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe window: {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")
    print(f"Building {len(SLEEVES)} sleeves...\n")

    out = {}
    for fn in SLEEVES:
        r = fn(dates)
        out[r.name] = r
        s = stats(r, r.name)
        nz = r[r != 0]
        start = nz.index[0].date() if len(nz) else None
        print(f"  {s['label']:18s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>6.2f}%  NAVx={s['navx']:>6.1f}  "
              f"live≥{start}")

    df = pd.DataFrame(out).reindex(dates).fillna(0)

    # Correlation only on rows where >= 5 sleeves are non-zero
    valid = (df != 0).sum(axis=1) >= 5
    corr = df[valid].corr()
    tri = corr.values[np.triu_indices_from(corr, k=1)]
    print(f"\nMean |pairwise corr| = {np.mean(np.abs(tri)):.3f}   "
          f"Max = {np.max(np.abs(tri)):.2f}   "
          f"Median = {np.median(np.abs(tri)):.3f}")

    port, w, scale = risk_parity_ensemble(df)
    port = port.fillna(0)
    nonzero = (df != 0).any(axis=1)
    port = port[nonzero]

    s = stats(port, "HYDRA")
    print(f"\n=== HYDRA (risk parity → vol-target {PORT_VOL:.0%}, cap {PORT_LEV_CAP}x) ===")
    print(f"  {s['label']:18s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>6.2f}%  NAVx={s['navx']:>6.1f}")

    IS_END = pd.Timestamp("2018-01-01")
    ir = stats(port.loc[:IS_END], "IS")
    or_ = stats(port.loc[IS_END:], "OOS")
    print(f"  IS  {port.loc[:IS_END].index[0].date()}..{port.loc[:IS_END].index[-1].date()}: "
          f"SR={ir['sharpe']} Ret={ir['ret']}% MDD={ir['mdd']}%")
    print(f"  OOS {port.loc[IS_END:].index[0].date()}..{port.loc[IS_END:].index[-1].date()}: "
          f"SR={or_['sharpe']} Ret={or_['ret']}% MDD={or_['mdd']}%")

    # Annual
    print("\nAnnual:")
    by_year = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print(by_year.to_string())

    # SPY benchmark on same window
    spy_r = load_etf("SPY").reindex(port.index).pct_change().fillna(0)
    sM = stats(spy_r, "SPY")
    print(f"\n  {sM['label']:18s} SR={sM['sharpe']:>5.2f}  Ret={sM['ret']:>6.2f}%  "
          f"Vol={sM['vol']:>5.2f}%  MDD={sM['mdd']:>6.2f}%  NAVx={sM['navx']:>6.1f}")

    final = pd.DataFrame({"HYDRA": port, "SPY": spy_r})
    final.to_csv(Path("/home/user/bonds/data/results/hydra_returns.csv"))
    df.to_csv(Path("/home/user/bonds/data/results/hydra_sleeves.csv"))


if __name__ == "__main__":
    main()
