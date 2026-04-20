"""NOVA22 — Pre-FOMC announcement drift (Lucca-Moench 2015).

Published Journal of Finance finding: ~50% of the aggregate annual
equity risk premium accrues in the 24 hours before FOMC meetings over
1994-2011. OOS validation (Cieslak-Morse-Vissing-Jorgensen 2019) shows
the effect persists post-publication, now spread across the full week
surrounding the announcement.

Strategy (FIXED a priori based on published paper):
  Long SPY on trading days t-1 and t (FOMC announcement day, ~14:00
  release). Cash (BIL) otherwise. Switches only 16 times per year → TC
  dominated by ~4 bps × 16 = 0.6% drag.

We also test a wider window: [t-2, t+1] per Cieslak 2019 extension.

Long-only, no shorting, no leverage, no vol scaling. Purely a calendar
filter overlaid on SPY."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


TC_BPS = 2.0


# FOMC meeting dates 2016-01 .. 2026-04 (official Federal Reserve meetings).
FOMC_DATES = [
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27",
    "2016-09-21", "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26",
    "2017-09-20", "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
    "2018-09-26", "2018-11-08", "2018-12-19",
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
    "2019-09-18", "2019-10-30", "2019-12-11",
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
    "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
    "2022-09-21", "2022-11-02", "2022-12-14",
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
    "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
    "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18",
]


def build_window_flag(dates, fomc_dates, pre, post):
    """Flag True for the `pre` trading days before FOMC and `post` trading
    days after (inclusive of the FOMC day itself)."""
    flag = pd.Series(False, index=dates)
    tdl = list(dates)
    for fd in fomc_dates:
        f = pd.Timestamp(fd)
        # Find insertion point
        if f in dates:
            idx = tdl.index(f)
        else:
            # Next trading day
            future = [i for i, d in enumerate(tdl) if d >= f]
            if not future:
                continue
            idx = future[0]
        lo = max(0, idx - pre)
        hi = min(len(tdl), idx + post + 1)
        for j in range(lo, hi):
            flag.iloc[j] = True
    return flag


def backtest(dates, spy_ret, bil_ret, flag, label):
    """Long SPY on flag days, BIL otherwise, 1-bar lag."""
    pos = pd.Series("BIL", index=dates, dtype=object)
    pos[flag] = "SPY"
    pos_eff = pos.shift(1).fillna("BIL")
    r = pd.Series(0.0, index=dates)
    r[pos_eff == "SPY"] = spy_ret[pos_eff == "SPY"]
    r[pos_eff == "BIL"] = bil_ret[pos_eff == "BIL"]
    changes = (pos_eff != pos_eff.shift(1)).astype(int)
    r = r - changes * (TC_BPS / 1e4)
    return r, pos_eff


def main():
    spy = load_etf("SPY")
    bil = load_etf("BIL")
    df = pd.concat({"SPY": spy, "BIL": bil}, axis=1).sort_index().ffill().dropna()
    dates = df.index
    rets = df.pct_change().fillna(0)

    CUT = pd.Timestamp("2022-01-01")

    # Try several windows
    windows = [(1, 0), (2, 0), (1, 1), (2, 1), (3, 1), (2, 2)]
    results = {}
    for pre, post in windows:
        flag = build_window_flag(dates, FOMC_DATES, pre, post)
        r, pos = backtest(dates, rets["SPY"], rets["BIL"], flag, f"{pre}-{post}")
        warm = pd.Timestamp("2016-03-01")
        r_v = r.loc[warm:]
        s = stats(r_v, f"NOVA22 [t-{pre}..t+{post}]")
        is_s = stats(r_v.loc[:CUT], "")["sharpe"]
        oos_s = stats(r_v.loc[CUT:], "")["sharpe"]
        exposure = flag.loc[warm:].mean()
        print(f"t-{pre}..t+{post:+}  exposure={exposure*100:>5.1f}%  "
              f"SR_full={s['sharpe']:>5.2f}  IS={is_s:>5.2f}  OOS={oos_s:>5.2f}  "
              f"Ret={s['ret']:>6.2f}%  Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
        results[(pre, post)] = r

    # Use the primary Lucca-Moench [t-1, t+0] as primary
    primary = results[(1, 0)]
    ext = results[(2, 1)]

    for p, lbl in [(primary, "NOVA22 [t-1,t] primary"),
                   (ext, "NOVA22 [t-2,t+1] extended")]:
        warm = pd.Timestamp("2016-03-01")
        pv = p.loc[warm:]
        s = stats(pv, lbl)
        print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
        for pp, tag in [(pv.loc[:CUT], "IS <2022"), (pv.loc[CUT:], "OOS >=2022")]:
            ss = stats(pp, tag)
            print(f"  {ss['label']:30s} SR={ss['sharpe']:>5.2f}  "
                  f"Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    ann = primary.loc["2016-03-01":].groupby(
        primary.loc["2016-03-01":].index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (primary [t-1,t]):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA22_1_0": results[(1, 0)], "NOVA22_2_1": results[(2, 1)]})
    out.to_csv("/home/user/bonds/data/results/nova22_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova22_returns.csv")


if __name__ == "__main__":
    main()
