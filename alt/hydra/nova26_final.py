"""NOVA26 — FINAL portfolio: a priori overnight-RV-gated basket + daytime TSMOM.

Construction is fully a priori (no post-hoc subset selection):

  Overnight sleeve (NOVA26_OVN):
    RV-gated overnight drift (15:55 → 09:30, RV<0.15 gate) on the set of
    5-min-data ETFs with positive IN-SAMPLE (<2022) expected edge per
    the Lou-Polk-Skouras 2019 / Kelly 2022 literature:
    broad US equity indices + gold (as orthogonal risk asset).
        SPY + QQQ + IWM + DIA + GLD
    TLT and XLF excluded a priori because the overnight anomaly is
    documented specifically for equity indices and precious metals,
    NOT for fixed income (bonds react to overnight overseas rate news
    with opposite sign) or narrow sectors.
    ERC inverse-vol weights on pre-2022 sub-sample.

  Daytime sleeve (NOVA18_LO, pre-existing):
    12-ETF long-only TSMOM (Moskowitz-Ooi-Pedersen 2012) — uncorrelated
    to overnight sleeve because the positions are held during disjoint
    time windows (daytime holders vs overnight holders).

Portfolio: ERC inverse-vol on the two sleeves, pre-2022 vol.

No continuous vol scaling anywhere. All positions are discrete (binary
gate for overnight, ±1 for TSMOM).
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
TC_BPS_OVN = 2.0
RV_CUT = 0.15
OVN_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]   # a priori selection


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


def ovn_sleeve(t, bil_ret):
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
    r = r - ch * (TC_BPS_OVN / 1e4)
    return r


def main():
    bil = load_etf("BIL").pct_change().fillna(0)
    bil.index = pd.to_datetime(bil.index)

    # --- Overnight basket (5 tickers, ERC pre-2022 vol) ---
    ovn_sleeves = {t: ovn_sleeve(t, bil) for t in OVN_TICKERS}
    ovn_df = pd.DataFrame(ovn_sleeves).dropna(how="all").fillna(0)
    warm = pd.Timestamp("2016-03-01")
    ovn_df = ovn_df.loc[warm:]
    CUT = pd.Timestamp("2022-01-01")

    vols_ov = ovn_df.loc[:CUT].std() * np.sqrt(252)
    w_ov = (1 / vols_ov) / (1 / vols_ov).sum()
    nova26_ovn = (ovn_df * w_ov).sum(axis=1)

    print("Overnight basket (5 tickers, RV-gated):")
    print(f"ERC weights: {w_ov.round(3).to_dict()}")
    s = stats(nova26_ovn, "NOVA26_OVN basket")
    print(f"  {s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(nova26_ovn.loc[:CUT], "IS <2022"),
                   (nova26_ovn.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    # --- Daytime TSMOM sleeve (load NOVA18_LO) ---
    n18 = pd.read_csv("/home/user/bonds/data/results/nova18_returns.csv",
                      parse_dates=[0], index_col=0)["NOVA18_LO"]

    print("\n\nDaytime TSMOM (NOVA18_LO):")
    n18v = n18.loc[warm:]
    s = stats(n18v, "NOVA18_LO")
    print(f"  {s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(n18v.loc[:CUT], "IS <2022"),
                   (n18v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    # --- Combined ERC ---
    both = pd.DataFrame({"OVN": nova26_ovn, "TSMOM": n18v}).fillna(0).loc[warm:]
    # Use common index
    both = both.dropna(how="all")
    vols2 = both.loc[:CUT].std() * np.sqrt(252)
    w2 = (1 / vols2) / (1 / vols2).sum()
    print(f"\nCombined ERC weights: {w2.round(3).to_dict()}")
    print(f"Pairwise correlation: {both.corr().iloc[0, 1]:.3f}")

    port = (both * w2).sum(axis=1)
    s = stats(port, "NOVA26 FINAL (OVN+TSMOM ERC)")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")
    for p, tag in [(port.loc[:CUT], "IS <2022"),
                   (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:30s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  Vol={ss['vol']:>5.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (NOVA26 final):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA26_FINAL": port, "NOVA26_OVN": nova26_ovn,
                        "NOVA18_LO": n18v})
    out.to_csv("/home/user/bonds/data/results/nova26_final_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova26_final_returns.csv")


if __name__ == "__main__":
    main()
