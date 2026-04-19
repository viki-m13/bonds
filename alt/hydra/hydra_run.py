"""HYDRA v4 — production-grade ensemble. Inverse-vol risk parity + crisis
hedges in the sleeve roster. No walk-forward filter (was tested, hurt)
and no regime overlay (was tested, neutral). The lift comes from better
sleeve design: long/short commodity, JPY safe-haven."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats
from hydra_sleeves_v3 import SLEEVES


PORT_VOL = 0.20
PORT_LEV_CAP = 5.0


def risk_parity_ensemble(sleeves_df, target_vol=PORT_VOL, window=63,
                         lev_cap=PORT_LEV_CAP):
    """Inverse-vol risk parity → portfolio vol target."""
    vols = sleeves_df.rolling(window).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = 1 / vols
    inv = inv.where(vols.notna(), 0)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    raw = (w * sleeves_df).sum(axis=1)
    pv = raw.rolling(window).std().shift(1) * np.sqrt(252)
    scale = (target_vol / pv).clip(upper=lev_cap).fillna(0)
    return raw * scale, w, scale


def print_stats(r, label, prefix=""):
    s = stats(r.dropna(), label)
    print(f"{prefix}{s['label']:18s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")


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

    print(f"\n=== HYDRA (risk parity → vol-target {PORT_VOL:.0%}, cap {PORT_LEV_CAP}x) ===")
    print_stats(port, "HYDRA", "  ")

    IS_END = pd.Timestamp("2018-01-01")
    ir = stats(port.loc[:IS_END], "IS")
    or_ = stats(port.loc[IS_END:], "OOS")
    print(f"  IS  {port.loc[:IS_END].index[0].date()}..{port.loc[:IS_END].index[-1].date()}: "
          f"SR={ir['sharpe']} Ret={ir['ret']}% MDD={ir['mdd']}% NAVx={ir['navx']}")
    print(f"  OOS {port.loc[IS_END:].index[0].date()}..{port.loc[IS_END:].index[-1].date()}: "
          f"SR={or_['sharpe']} Ret={or_['ret']}% MDD={or_['mdd']}% NAVx={or_['navx']}")

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

    spy_r = load_etf("SPY").reindex(port.index).pct_change().fillna(0)
    print()
    print_stats(spy_r, "SPY", "  ")

    # Walk-forward validation: rolling 5-year windows
    print("\nWalk-forward 5y windows (rolling, non-overlapping):")
    year_groups = [(y, y + 5) for y in range(2006, 2022, 5)]
    for y0, y1 in year_groups:
        lo = pd.Timestamp(f"{y0}-01-01")
        hi = pd.Timestamp(f"{y1}-01-01")
        sub = port.loc[lo:hi]
        sub_spy = spy_r.loc[lo:hi]
        if len(sub) < 200:
            continue
        sh = stats(sub, "HYDRA")
        sp = stats(sub_spy, "SPY")
        print(f"  {y0}-{y1-1}  HYDRA SR={sh['sharpe']:>5.2f} Ret={sh['ret']:>6.2f}% MDD={sh['mdd']:>7.2f}%  "
              f"|  SPY SR={sp['sharpe']:>5.2f} Ret={sp['ret']:>6.2f}% MDD={sp['mdd']:>7.2f}%")

    # Monthly return distribution
    monthly = port.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    neg_months = (monthly < 0).sum()
    pct_pos = 100 * (monthly > 0).mean()
    worst_month = monthly.min() * 100
    best_month = monthly.max() * 100
    print(f"\nMonthly: {pct_pos:.0f}% positive months, {neg_months}/{len(monthly)} negative")
    print(f"  Worst month: {worst_month:.2f}%   Best month: {best_month:.2f}%")

    final = pd.DataFrame({"HYDRA": port, "SPY": spy_r})
    final.to_csv(Path("/home/user/bonds/data/results/hydra_returns.csv"))
    df.to_csv(Path("/home/user/bonds/data/results/hydra_sleeves.csv"))


if __name__ == "__main__":
    main()
