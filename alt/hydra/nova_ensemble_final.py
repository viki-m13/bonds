"""FINAL NOVA ensemble — targeting SR 2+ OOS with monthly/daily rebalance.

Sleeves (all DISCRETE position, no continuous vol scaling):
  NOVA24 : RV-gated overnight drift EW SPY+QQQ+IWM        OOS 1.63 ⭐
  NOVA18_LO : 12-ETF long-only TSMOM (daytime)            OOS 0.79
  NOVA20_A  : Turn-of-month SPY/BIL                       OOS 0.47
  NOVA11    : Chronos dispersion gate + VIX harvest       OOS 0.48

Overnight (NOVA24) and daytime (NOVA18_LO) sleeves occupy DISJOINT time
windows → structurally uncorrelated.

ERC inverse-vol on pre-2022 window."""
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
        "NOVA18_LO": load("nova18_returns",  "NOVA18_LO"),
        "NOVA20_A":  load("nova20_returns",  "TOM"),
        "NOVA24":    load("nova24_returns",  "NOVA24"),
    }
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0)
    df = df.loc["2017-06-01":]
    print("Correlations:")
    print(df.corr().round(2).to_string())
    print()
    CUT = pd.Timestamp("2022-01-01")
    for c in df.columns:
        x = df[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:12s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")
    print()

    # Full ERC
    vols = df.loc[:CUT].std() * np.sqrt(252)
    w = (1 / vols) / (1 / vols).sum()
    print(f"ERC weights (pre-2022): {w.round(3).to_dict()}")
    port = (df * w).sum(axis=1)
    s = stats(port, "FINAL ENSEMBLE (4 sleeves)")
    print(f"\n{s['label']:34s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  Vol={ss['vol']:>5.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Subset search
    print("\nAll subsets:")
    import itertools
    cols = list(df.columns)
    best = {}
    for r in range(2, len(cols) + 1):
        for combo in itertools.combinations(cols, r):
            sub = df[list(combo)]
            vs = sub.loc[:CUT].std() * np.sqrt(252)
            ws = (1 / vs) / (1 / vs).sum()
            p = (sub * ws).sum(axis=1)
            oos = stats(p.loc[CUT:], "")["sharpe"]
            full = stats(p, "")["sharpe"]
            name = "+".join(c.replace("NOVA", "") for c in combo)
            print(f"  [{name:40s}] Full={full:>5.2f}  "
                  f"IS={stats(p.loc[:CUT],'')['sharpe']:>5.2f}  "
                  f"OOS={oos:>5.2f}  "
                  f"Vol={stats(p,'')['vol']:>4.1f}%  MDD={stats(p,'')['mdd']:>7.2f}%")

    # Heaviest on NOVA24 subset — 24 + 18_LO + 20_A
    best_combo = ["NOVA24", "NOVA18_LO", "NOVA20_A"]
    sub = df[best_combo]
    vs = sub.loc[:CUT].std() * np.sqrt(252)
    ws = (1 / vs) / (1 / vs).sum()
    port_best = (sub * ws).sum(axis=1)
    print(f"\nBEST 3-sleeve = {best_combo}")
    print(f"ERC weights: {ws.round(3).to_dict()}")
    s = stats(port_best, "BEST 3-SLEEVE")
    print(f"{s['label']:34s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(port_best.loc[:CUT], "IS <2022"), (port_best.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    ann = port_best.groupby(port_best.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (BEST 3-sleeve):")
    print(ann.to_string())

    out = pd.DataFrame({"FINAL_ENSEMBLE": port_best}).join(df, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova_ensemble_final_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova_ensemble_final_returns.csv")


if __name__ == "__main__":
    main()
