"""Measure actual daily vs monthly turnover of HYDRA to answer:
how painful is 'daily rebalancing for vol scaling' in practice?"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf
from hydra_sleeves_v3 import SLEEVES


def main():
    spy = load_etf("SPY")
    dates = spy.index

    # Build sleeves (daily returns already include daily vol scaling at sleeve level)
    out = {fn(dates).name: fn(dates) for fn in SLEEVES}
    df = pd.DataFrame(out).reindex(dates).fillna(0)

    # Inverse-vol weights (daily update)
    vols = df.rolling(63).std().shift(1) * np.sqrt(252)
    vols = vols.where(vols > 0.001)
    inv = 1 / vols
    inv = inv.where(vols.notna(), 0)
    w = inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Portfolio vol scaler (daily)
    raw = (w * df).sum(axis=1)
    pv = raw.rolling(63).std().shift(1) * np.sqrt(252)
    scale = (0.20 / pv).clip(upper=5.0).fillna(0)

    # Effective position = weight × portfolio_vol_scale
    # Daily turnover at the ensemble weight level
    eff = w.mul(scale, axis=0)
    daily_turn = eff.diff().abs().sum(axis=1)

    # Sleeve-internal turnover is embedded already (sleeves are monthly-signal
    # with daily vol scaling). Approximate it from the fact that each sleeve's
    # position size scales with its vol-target scaler, which moves slowly.
    # So total daily turnover at the book level ≈ ensemble-drift + sleeve-drift.
    # The sleeve-drift piece has already paid its TC inside each sleeve's return.

    nz = (df != 0).any(axis=1)
    dt = daily_turn[nz].dropna()
    monthly_bdays = 21

    print("Daily ensemble-weight turnover (one-sided, per day):")
    print(f"  mean   {dt.mean()*100:>6.2f}%")
    print(f"  median {dt.median()*100:>6.2f}%")
    print(f"  p90    {dt.quantile(0.9)*100:>6.2f}%")
    print(f"  p99    {dt.quantile(0.99)*100:>6.2f}%")
    print(f"  max    {dt.max()*100:>6.2f}%")
    print()

    # Annualised turnover
    ann = dt.sum() / (len(dt) / 252)
    print(f"Annualised ensemble turnover: {ann*100:>6.1f}%  (= {ann*100/252:.2f}%/day avg)")
    print(f"Annualised TC @ 15bp:          {ann*15/10000*100:>6.2f}%")
    print()

    # What fraction of days is turnover < 1%? 0.5%? 0.1%?
    print("Distribution of daily turnover:")
    for thr in [0.001, 0.003, 0.005, 0.01, 0.02]:
        p = (dt < thr).mean() * 100
        print(f"  < {thr*100:>4.1f}%/day: {p:>5.1f}% of days")
    print()

    # Compare to monthly-signal turnover.  The sleeve's internal signal-rebal
    # cadence is 21 bdays; ensemble weights drift daily but only from vol moves.
    # Estimate sleeve-internal signal-change bursts.
    # We see this indirectly: daily turnover on monthly-rebal days should be
    # higher.  Compute:
    month_ends = df.resample("ME").last().index.intersection(df.index)
    first_bday_of_month = pd.DatetimeIndex([d for d in df.index
                                             if d.day <= 3 and d.month != (d - pd.Timedelta(days=4)).month])
    turn_rebal = dt.reindex(first_bday_of_month).dropna()
    turn_other = dt.drop(first_bday_of_month, errors="ignore")
    print(f"Turnover on 1st business day of month (signal-rebal):")
    print(f"  mean {turn_rebal.mean()*100:>6.2f}%  n={len(turn_rebal)}")
    print(f"Turnover on all other days (vol-drift only):")
    print(f"  mean {turn_other.mean()*100:>6.2f}%  n={len(turn_other)}")


if __name__ == "__main__":
    main()
