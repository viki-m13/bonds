"""Final NOVA ensemble — including intraday-data-gated overnight drift.

Sleeves (all monthly/weekly/daily discrete, no vol scaling):
  NOVA18_LO : 12-ETF long-only TSMOM                      OOS 0.79
  NOVA23    : RV-gated overnight drift (5-min intraday)   OOS 0.99  ⭐
  NOVA11    : Chronos dispersion gate + VIX harvest       OOS 0.48
  NOVA16    : Chronos RV-forecast rotation                OOS 0.63
  NOVA20_A  : Turn-of-month SPY/BIL                       OOS 0.47
  NOVA21_LO : Residual momentum long-only (tilt)          OOS 0.65

ERC inverse-vol weights on pre-2022 sub-sample."""
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
        "NOVA11":    load("nova11_returns",  "NOVA11"),
        "NOVA16":    load("nova16_returns",  "NOVA16"),
        "NOVA18_LO": load("nova18_returns",  "NOVA18_LO"),
        "NOVA20_A":  load("nova20_returns",  "TOM"),
        "NOVA21_LO": load("nova21_returns",  "NOVA21_LO"),
        "NOVA23":    load("nova23_returns",  "NOVA23"),
    }
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0)
    df = df.loc["2018-02-01":]
    print("Correlations:")
    print(df.corr().round(2).to_string())
    print()
    CUT = pd.Timestamp("2022-01-01")

    # Per-sleeve stats
    for c in df.columns:
        x = df[c]
        sf = stats(x, c)
        si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:12s} SR_full={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  "
              f"MDD={sf['mdd']:>7.2f}%")
    print()

    vols = df.loc[:CUT].std() * np.sqrt(252)
    invv = 1 / vols
    w = invv / invv.sum()
    print(f"ERC weights (pre-2022): {w.round(3).to_dict()}\n")

    port = (df * w).sum(axis=1)
    s = stats(port, "6-SLEEVE ENSEMBLE (ERC)")
    print(f"{s['label']:34s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  Vol={ss['vol']:>5.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Combo search: try subsets
    print("\nSubsets (ERC pre-2022 weights):")
    combos = [
        ["NOVA18_LO", "NOVA23"],
        ["NOVA18_LO", "NOVA23", "NOVA20_A"],
        ["NOVA18_LO", "NOVA23", "NOVA11"],
        ["NOVA18_LO", "NOVA23", "NOVA16"],
        ["NOVA18_LO", "NOVA23", "NOVA20_A", "NOVA11"],
        ["NOVA18_LO", "NOVA23", "NOVA20_A", "NOVA16"],
        ["NOVA18_LO", "NOVA23", "NOVA20_A", "NOVA11", "NOVA16"],
        ["NOVA18_LO", "NOVA23", "NOVA20_A", "NOVA11", "NOVA16", "NOVA21_LO"],
    ]
    for combo in combos:
        sub = df[combo]
        vs = sub.loc[:CUT].std() * np.sqrt(252)
        ws = (1 / vs) / (1 / vs).sum()
        p = (sub * ws).sum(axis=1)
        s_full = stats(p, "")
        s_is = stats(p.loc[:CUT], "")
        s_oos = stats(p.loc[CUT:], "")
        name = "+".join(s.replace("NOVA", "") for s in combo)
        print(f"  [{name:50s}] SR={s_full['sharpe']:>5.2f}  "
              f"IS={s_is['sharpe']:>5.2f}  OOS={s_oos['sharpe']:>5.2f}  "
              f"Vol={s_full['vol']:>4.1f}%  MDD={s_full['mdd']:>7.2f}%")

    # Annual for 6-sleeve
    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (6-sleeve):")
    print(ann.to_string())

    out = pd.DataFrame({"ENSEMBLE6": port}).join(df, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova_ensemble3_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova_ensemble3_returns.csv")


if __name__ == "__main__":
    main()
