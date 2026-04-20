"""NOVA20 — Seasonality composite (turn-of-month + Halloween).

Two well-documented calendar anomalies, combined with no tuning:

A) TURN-OF-MONTH (Ogden 1990, Lakonishok-Smidt 1988):
   Long SPY during the last 4 trading days and first 3 trading days of
   each month. BIL otherwise. Historically produces SR 0.8-1.0 with very
   low vol (~6%) because it's in the market only 7/21 days/month.
   Explanation: month-end institutional rebalancing creates buying
   pressure into month-start.

B) HALLOWEEN / SELL-IN-MAY (Bouman-Jacobsen 2002):
   Long SPY Nov-Apr, long TLT May-Oct. Ensures always in some asset.
   Documented OOS SR 0.6-0.9 over multiple decades and 40+ countries.

Combined:
   Take two independent sleeves. ERC risk-parity weighting (pre-2022
   vol). Very low TC — only 2-4 rebalances per month.

Both sleeves are long-only discrete allocation, NOT vol scaling."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


TC_BPS = 5.0   # 2.5 bps × 2 legs


def turn_of_month_flag(dates):
    """Flag: True if the date is in the last 4 or first 3 trading days of month."""
    s = pd.Series(0, index=dates)
    # Day rank within month
    for ym, grp in pd.Series(dates).groupby([dates.year, dates.month]):
        first_days = grp.iloc[:3]
        last_days = grp.iloc[-4:]
        s.loc[first_days] = 1
        s.loc[last_days] = 1
    return s.astype(bool)


def main():
    spy = load_etf("SPY")
    tlt = load_etf("TLT")
    bil = load_etf("BIL")
    df = pd.concat({"SPY": spy, "TLT": tlt, "BIL": bil}, axis=1).sort_index().ffill()
    df = df.dropna()
    dates = df.index
    rets = df.pct_change().fillna(0)

    # A — Turn of month
    tom = turn_of_month_flag(dates)
    pos_a = pd.Series("BIL", index=dates, dtype=object)
    pos_a[tom] = "SPY"
    pos_a_eff = pos_a.shift(1).fillna("BIL")
    r_a = pd.Series(0.0, index=dates)
    r_a[pos_a_eff == "SPY"] = rets["SPY"][pos_a_eff == "SPY"]
    r_a[pos_a_eff == "BIL"] = rets["BIL"][pos_a_eff == "BIL"]
    changes = (pos_a_eff != pos_a_eff.shift(1)).astype(int)
    r_a = r_a - changes * (TC_BPS / 1e4)

    # B — Halloween
    pos_b = pd.Series("TLT", index=dates, dtype=object)
    in_equity = dates.month.isin([11, 12, 1, 2, 3, 4])
    pos_b[in_equity] = "SPY"
    pos_b_eff = pos_b.shift(1).fillna("TLT")
    r_b = pd.Series(0.0, index=dates)
    r_b[pos_b_eff == "SPY"] = rets["SPY"][pos_b_eff == "SPY"]
    r_b[pos_b_eff == "TLT"] = rets["TLT"][pos_b_eff == "TLT"]
    changes_b = (pos_b_eff != pos_b_eff.shift(1)).astype(int)
    r_b = r_b - changes_b * (TC_BPS / 1e4)

    warm = pd.Timestamp("2016-02-01")
    ra_v, rb_v = r_a.loc[warm:], r_b.loc[warm:]

    for r, lbl in [(ra_v, "A: TurnOfMonth"), (rb_v, "B: Halloween")]:
        s = stats(r, lbl)
        print(f"{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")

    CUT = pd.Timestamp("2022-01-01")
    for r, lbl in [(ra_v, "A TurnOfMonth"), (rb_v, "B Halloween")]:
        print(f"\n{lbl}:")
        for p, tag in [(r.loc[:CUT], "IS <2022"), (r.loc[CUT:], "OOS >=2022")]:
            ss = stats(p, tag)
            print(f"  {ss['label']:26s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
                  f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # ERC ensemble (pre-2022 vol)
    va = ra_v.loc[:CUT].std() * np.sqrt(252)
    vb = rb_v.loc[:CUT].std() * np.sqrt(252)
    wa = (1 / va) / (1 / va + 1 / vb)
    wb = 1 - wa
    print(f"\nERC weights (pre-2022): A={wa:.3f}  B={wb:.3f}")
    port = wa * ra_v + wb * rb_v
    s = stats(port, "NOVA20 seasonality ERC")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Annual ensemble
    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (ensemble):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA20": port, "TOM": r_a, "Halloween": r_b})
    out.to_csv("/home/user/bonds/data/results/nova20_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova20_returns.csv")


if __name__ == "__main__":
    main()
