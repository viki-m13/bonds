"""Final NOVA ensemble — post Chronos-pivot.

Sleeves (all monthly rebalance, discrete position, no vol scaling):
  - NOVA18_LO : 12-ETF long-only TSMOM (Moskowitz 2012)   OOS 0.79
  - NOVA11    : Chronos-gate + VIX harvest                 OOS 0.48
  - NOVA19    : Low-vol anomaly long-only (Frazzini 2014)  OOS 0.30
  - NOVA20_A  : Turn-of-month seasonality                  OOS 0.47
  - NOVA16    : Chronos RV-forecast rotation               OOS 0.63

ERC inverse-vol on pre-2022 window, evaluate IS/OOS.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from hydra_core import stats


def load(name, col):
    df = pd.read_csv(f"/home/user/bonds/data/results/{name}.csv",
                     parse_dates=[0], index_col=0)
    return df[col]


def main():
    sleeves = {
        "NOVA11":    load("nova11_returns", "NOVA11"),
        "NOVA16":    load("nova16_returns", "NOVA16"),
        "NOVA18_LO": load("nova18_returns", "NOVA18_LO"),
        "NOVA19":    load("nova19_returns", "NOVA19"),
        "NOVA20_A":  load("nova20_returns", "TOM"),
    }
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0)
    df = df.loc["2018-02-01":]
    print("Correlations:")
    print(df.corr().round(2).to_string())
    print()
    CUT = pd.Timestamp("2022-01-01")
    # ERC
    vol = df.loc[:CUT].std() * np.sqrt(252)
    invv = 1 / vol
    w = invv / invv.sum()
    print(f"ERC weights (pre-2022): {w.round(3).to_dict()}\n")

    port = (df * w).sum(axis=1)
    s = stats(port, "NOVA ENSEMBLE (5 sleeves)")
    print(f"{s['label']:34s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  Vol={ss['vol']:>5.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Best-two ensemble (NOVA18_LO + NOVA20_A)
    for combo in [["NOVA18_LO", "NOVA20_A"],
                  ["NOVA18_LO", "NOVA11"],
                  ["NOVA18_LO", "NOVA16", "NOVA20_A"],
                  ["NOVA18_LO", "NOVA11", "NOVA20_A"],
                  ["NOVA18_LO", "NOVA11", "NOVA16", "NOVA20_A"]]:
        sub = df[combo]
        vols = sub.loc[:CUT].std() * np.sqrt(252)
        ws = (1 / vols) / (1 / vols).sum()
        p = (sub * ws).sum(axis=1)
        name = "+".join(combo)
        s_full = stats(p, name)
        s_is = stats(p.loc[:CUT], "IS")
        s_oos = stats(p.loc[CUT:], "OOS")
        print(f"{name:60s} SR={s_full['sharpe']:.2f}  IS={s_is['sharpe']:.2f}  "
              f"OOS={s_oos['sharpe']:.2f}  Vol={s_full['vol']:.1f}%")

    # Annual for full 5-sleeve
    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (5-sleeve):")
    print(ann.to_string())

    out = pd.DataFrame({"ENSEMBLE": port}).join(df, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova_ensemble2_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova_ensemble2_returns.csv")


if __name__ == "__main__":
    main()
