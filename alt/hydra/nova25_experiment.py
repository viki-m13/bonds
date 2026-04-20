"""NOVA25 — RV-gated overnight drift EW across ALL 7 intraday ETFs.

NOVA23: SPY OOS 0.99.  NOVA24: EW SPY+QQQ+IWM OOS 1.63.
Extending to the full available 5-min intraday universe
(SPY, QQQ, IWM, DIA, GLD, TLT, XLF) should deliver more diversification
across equity segments (broad, tech, small-cap, mega-cap, gold, bond,
financial sector)."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
TC_BPS = 2.0
RV_CUT = 0.15
TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD", "TLT", "XLF"]


def five_min_rv(t):
    df = pd.read_csv(INTRA / f"{t}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["logret"] = np.log(df["close"]).diff()
    fod = df["date"] != df["date"].shift(1)
    df.loc[fod, "logret"] = 0.0
    rv = df.groupby("date")["logret"].apply(lambda x: np.sqrt(np.sum(x ** 2)))
    rv.index = pd.to_datetime(rv.index)
    return rv * np.sqrt(252)


def overnight_returns(t):
    df = pd.read_csv(INTRA / f"{t}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["time"] = df["ts"].dt.time
    px_1555 = df[df["time"] == pd.to_datetime("15:55").time()].set_index(
        "date")["close"]
    px_1555 = px_1555[~px_1555.index.duplicated(keep="first")].sort_index()
    px_1555.index = pd.to_datetime(px_1555.index)
    open_ = df.groupby("date")["open"].first()
    open_.index = pd.to_datetime(open_.index)
    common = px_1555.index.intersection(open_.index)
    return ((open_.loc[common].shift(-1) / px_1555.loc[common]) - 1).dropna()


def sleeve(t, bil_ret):
    rv = five_min_rv(t)
    ovn = overnight_returns(t)
    common = rv.index.intersection(ovn.index).intersection(bil_ret.index)
    rv, ovn, bil = rv.loc[common], ovn.loc[common], bil_ret.loc[common]
    rv20 = rv.rolling(20).mean().shift(1)
    gate = rv20 < RV_CUT
    r = pd.Series(0.0, index=common)
    r[gate] = ovn[gate]
    r[~gate] = bil[~gate]
    ch = (gate != gate.shift(1)).astype(int)
    r = r - ch * (TC_BPS / 1e4)
    return r.rename(f"ovn_{t}")


def main():
    bil = load_etf("BIL").pct_change().fillna(0)
    bil.index = pd.to_datetime(bil.index)

    sleeves = {t: sleeve(t, bil) for t in TICKERS}
    sdf = pd.DataFrame(sleeves).dropna(how="all").fillna(0)
    warm = pd.Timestamp("2016-03-01")
    sdf = sdf.loc[warm:]
    CUT = pd.Timestamp("2022-01-01")

    print("Per-ticker:")
    for c in sdf.columns:
        x = sdf[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:10s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")

    print(f"\nCorrelations:")
    print(sdf.corr().round(2).to_string())

    ew = sdf.mean(axis=1)
    s = stats(ew, "NOVA25 EW 7-ticker ovn")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(ew.loc[:CUT], "IS <2022"), (ew.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    vols = sdf.loc[:CUT].std() * np.sqrt(252)
    w = (1 / vols) / (1 / vols).sum()
    erc = (sdf * w).sum(axis=1)
    s = stats(erc, "NOVA25 ERC 7-ticker ovn")
    print(f"\nERC weights: {w.round(3).to_dict()}")
    print(f"{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(erc.loc[:CUT], "IS <2022"), (erc.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Best equity-only subset (drop GLD, TLT which may hurt)
    for keep in [["SPY","QQQ","IWM","DIA"],
                 ["SPY","QQQ","IWM","GLD"],
                 ["SPY","QQQ","IWM","DIA","GLD"],
                 ["SPY","QQQ","IWM","GLD","XLF"],
                 ["QQQ","IWM","GLD"],
                 ["SPY","QQQ","IWM","DIA","GLD","XLF"]]:
        sub = sdf[keep]
        vs = sub.loc[:CUT].std() * np.sqrt(252)
        ws = (1 / vs) / (1 / vs).sum()
        p = (sub * ws).sum(axis=1)
        full = stats(p, "")
        oos = stats(p.loc[CUT:], "")
        is_ = stats(p.loc[:CUT], "")
        print(f"  {'+'.join(keep):40s} SR={full['sharpe']:.2f}  IS={is_['sharpe']:.2f}  "
              f"OOS={oos['sharpe']:.2f}  Vol={full['vol']:.1f}%  MDD={full['mdd']:.1f}%")

    ann = erc.groupby(erc.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (ERC 7-ticker):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA25": erc}).join(sdf, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova25_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova25_returns.csv")


if __name__ == "__main__":
    main()
