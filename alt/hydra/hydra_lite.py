"""HYDRA-Lite — simplified variant, no portfolio vol targeting.
Equal-weight across 20 sleeves, monthly rebalance, static leverage
(2.43x) chosen once to hit ~10% annualised vol over the full history.
Same 20 sleeves as shipped HYDRA, so sleeves retain their internal
10%-vol targeting as part of their rule; but the ensemble layer is
stripped of all dynamic weighting and dynamic vol scaling.

This is the simplest practical version: rebalance book once a month,
static leverage, done."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats
from hydra_sleeves_v3 import SLEEVES


LITE_TARGET_VOL = 0.10
LITE_LEVERAGE = None     # solved from data; reported below
LITE_REBAL_DAYS = 21


def build_lite(dates, rebal_days=LITE_REBAL_DAYS, target_vol=LITE_TARGET_VOL):
    out = {fn(dates).name: fn(dates) for fn in SLEEVES}
    df = pd.DataFrame(out).reindex(dates).fillna(0)

    # Equal-weight across live sleeves (live = ever had a non-zero return)
    live = (df != 0).cummax().astype(float)
    w_daily = live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    # Freeze weights every rebal_days — "rebalance every N days"
    mask = pd.Series(False, index=w_daily.index)
    mask.iloc[::rebal_days] = True
    w = w_daily.where(mask, np.nan).ffill().fillna(0)

    raw = (w * df).sum(axis=1)
    nz = (df != 0).any(axis=1)
    raw = raw[nz].dropna()

    # Static leverage chosen ONCE to hit target vol full-sample
    native_vol = raw.std() * np.sqrt(252)
    lev = target_vol / native_vol
    return raw * lev, lev, df, w


def main():
    spy = load_etf("SPY")
    dates = spy.index

    print(f"Universe: {dates[0].date()} .. {dates[-1].date()}")
    print(f"HYDRA-Lite config: equal-weight, monthly ({LITE_REBAL_DAYS}-day) rebal, "
          f"static leverage to hit {LITE_TARGET_VOL:.0%} ann vol.\n")

    lite, lev, sleeves_df, weights = build_lite(dates)

    print(f"Static leverage used: {lev:.2f}x")
    print()

    def row(r, label):
        s = stats(r.dropna(), label)
        IS = pd.Timestamp("2018-01-01")
        si = stats(r.loc[:IS], "IS")
        so = stats(r.loc[IS:], "OOS")
        print(f"  {label:22s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}  "
              f"| IS SR={si['sharpe']:>5.2f}  OOS SR={so['sharpe']:>5.2f}")

    row(lite, "HYDRA-Lite")
    spy_r = load_etf("SPY").reindex(lite.index).pct_change().fillna(0)
    row(spy_r, "SPY")

    # Load shipped HYDRA for comparison
    shipped_path = Path("/home/user/bonds/data/results/hydra_returns.csv")
    shipped = pd.read_csv(shipped_path, parse_dates=["Date"]).set_index("Date")["HYDRA"]
    shipped = shipped.reindex(lite.index).dropna()
    row(shipped, "HYDRA (shipped)")

    # Annual stats
    print("\nHYDRA-Lite annual:")
    by_year = lite.groupby(lite.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print(by_year.to_string())

    # 5y walk-forward
    print("\nWalk-forward 5y windows:")
    year_groups = [(y, y + 5) for y in range(2006, 2022, 5)]
    for y0, y1 in year_groups:
        lo = pd.Timestamp(f"{y0}-01-01")
        hi = pd.Timestamp(f"{y1}-01-01")
        sub_l = lite.loc[lo:hi]
        sub_s = shipped.loc[lo:hi]
        sub_spy = spy_r.loc[lo:hi]
        if len(sub_l) < 200:
            continue
        sl = stats(sub_l, "Lite")
        ss = stats(sub_s, "HYDRA")
        sp = stats(sub_spy, "SPY")
        print(f"  {y0}-{y1-1}  Lite SR={sl['sharpe']:>5.2f} Ret={sl['ret']:>6.2f}%  "
              f"|  HYDRA SR={ss['sharpe']:>5.2f} Ret={ss['ret']:>6.2f}%  "
              f"|  SPY SR={sp['sharpe']:>5.2f} Ret={sp['ret']:>6.2f}%")

    # Save daily returns
    out = pd.DataFrame({"HYDRA_Lite": lite, "HYDRA": shipped, "SPY": spy_r}).dropna()
    out_path = Path("/home/user/bonds/data/results/hydra_lite_returns.csv")
    out.to_csv(out_path)
    print(f"\nWrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
