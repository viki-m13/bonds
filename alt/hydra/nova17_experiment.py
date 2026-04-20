"""NOVA17 — RV-forecast regime-rotation between levered equity and levered bonds.

Building on NOVA16 finding: Chronos on RV delivers a coherent OOS signal
(SR 0.63) but allocating to UPRO in high-vol regimes blows up (2018 -41%,
2022 -30%). The vol signal is predictive but the position set was wrong:
when vol is high, equities tend to drop AND bonds (TLT/TMF) tend to rally
(flight to safety). So the levered equity ↔ levered bond rotation should
produce much better OOS SR than levered-equity ↔ cash rotation.

Strategy (FIXED a priori):
  Inputs: 21-day forward RV forecast from Chronos (reuse NOVA16 file).

  Allocation (on month-start, one-bar lag, TC 15 bps per regime change):
    RV_forecast < 0.12           → UPRO (3x SPY)
    0.12 ≤ RV_forecast < 0.20    → 50/50 SPY / TLT (balanced)
    RV_forecast ≥ 0.20           → TMF (3x TLT — flight to safety)

  Rationale:
    - Low-vol → quiet bull regime → UPRO drift is very positive
    - Mid-vol → mixed regime → balanced gets carry from both sides
    - High-vol → stress regime → equities down, long duration rallies
      (2008, 2020 COVID, 2022-Q1 pre-rate-hike). TMF offers 3x bond
      rally capture during these tail events.

Caveats we accept:
  - 2022 was an EXCEPTION to the equity-bond negative correlation (both fell
    due to inflation/rate-hike cycle). This is an OOS adversarial year that
    will hurt the TMF sleeve. We report it honestly.

No scaling, no fitting: vol buckets and tickers are specified up-front based
on the Moreira-Muir (2017) and Asness-Moskowitz (2012) literature on vol-
managed bond-equity rotation."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


TC_BPS = 15.0


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def bucket(rv):
    if pd.isna(rv):
        return "BIL"
    if rv < 0.12:
        return "UPRO"
    if rv < 0.20:
        return "MIX"
    return "TMF"


def main():
    fc = pd.read_csv("/home/user/bonds/data/results/nova16_returns.csv",
                     parse_dates=[0], index_col=0)
    rv = fc["rv_forecast"].dropna()

    spy = load_etf("SPY")
    dates = spy.index
    assets = {}
    for t in ["UPRO", "SPY", "TLT", "TMF", "BIL"]:
        p = load_etf(t)
        if p is None:
            print(f"MISSING {t}")
            assets[t] = pd.Series(0.0, index=dates)
        else:
            assets[t] = p.reindex(dates).ffill().pct_change().fillna(0)

    # Propagate monthly rebalance: position held until next month-start pred
    position = pd.Series("BIL", index=dates, dtype=object)
    last = "BIL"
    fc_dates = set(rv.index)
    for d in dates:
        if d in fc_dates:
            last = bucket(rv.loc[d])
        position.loc[d] = last
    position_eff = position.shift(1).fillna("BIL")

    r = pd.Series(0.0, index=dates)
    mask = position_eff == "UPRO"
    r.loc[mask] = assets["UPRO"].loc[mask]
    mask = position_eff == "TMF"
    r.loc[mask] = assets["TMF"].loc[mask]
    mix = position_eff == "MIX"
    r.loc[mix] = 0.5 * assets["SPY"].loc[mix] + 0.5 * assets["TLT"].loc[mix]
    bil = position_eff == "BIL"
    r.loc[bil] = assets["BIL"].loc[bil]

    changes = (position_eff != position_eff.shift(1)).astype(int)
    r = r - changes * (TC_BPS / 1e4) * 2

    first_pred = rv.index.min()
    r_v = r.loc[first_pred:]
    pos_v = position_eff.loc[first_pred:]

    print(f"NOVA17 — RV→(UPRO|MIX|TMF) rotation, first pred {first_pred.date()}")
    print("Position distribution:")
    print(pos_v.value_counts())

    s = stats(r_v, "NOVA17 RV rotation UPRO/MIX/TMF")
    print(f"\n{s['label']:40s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(r_v.loc[:CUT], "IS <2022"), (r_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    ann = r_v.groupby(r_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA17": r, "position": position_eff})
    out.to_csv("/home/user/bonds/data/results/nova17_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova17_returns.csv")


if __name__ == "__main__":
    main()
