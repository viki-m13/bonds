"""NOVA ensemble — final honest aggregation.

After 17 experiments the best OOS SR on single sleeves is ~0.5-1.0.
This script constructs an ERC ensemble over the best uncorrelated sleeves:
  - NOVA11 (Chronos dispersion-gate + VIX harvest): SR 0.48 OOS, vol 9%
  - NOVA16 (Chronos RV forecast rotation): SR 0.64 OOS, vol 34%
  - NOVA13B (cross-sectional overnight gap-fade): uncorrelated even if
    low-SR — used only if it improves ensemble

Inverse-vol risk-parity on pre-2022, then evaluate IS/OOS."""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


def load(name, col=None):
    df = pd.read_csv(f"/home/user/bonds/data/results/{name}.csv",
                     parse_dates=[0], index_col=0)
    return df[col] if col else df.iloc[:, 0]


def main():
    sleeves = {
        "NOVA11": load("nova11_returns", "NOVA11"),
        "NOVA16": load("nova16_returns", "NOVA16"),
    }
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0)
    start = pd.Timestamp("2018-02-01")
    df = df.loc[start:]

    print("Correlations:")
    print(df.corr().round(2).to_string(), "\n")

    CUT = pd.Timestamp("2022-01-01")
    vol = df.loc[:CUT].std() * np.sqrt(252)
    invv = 1 / vol
    w = invv / invv.sum()
    print(f"ERC weights (pre-2022): {w.round(3).to_dict()}\n")

    port = (df * w).sum(axis=1)
    s = stats(port, "ENSEMBLE NOVA11+NOVA16")
    print(f"{s['label']:40s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")

    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  Vol={ss['vol']:>5.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Annual
    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    out = pd.DataFrame({"ENSEMBLE": port}).join(df, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova_ensemble_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova_ensemble_returns.csv")


if __name__ == "__main__":
    main()
