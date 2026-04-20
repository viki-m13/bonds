"""NOVA28 — Weekly-gated overnight + sector-rotation on 5-min ETFs.

Two orthogonal sleeves that require only WEEKLY decision-making:

A) WEEKLY OVERNIGHT (low-TC variant of NOVA24/25):
     Every Monday evening, check RV<0.15 gate for each ETF.
     If gate ON → hold overnight Mon→Tue, Tue→Wed, ..., Fri→next-Mon.
     Gate is only re-evaluated Monday; if gate flips mid-week we still
     hold. This cuts decision frequency 5x at the cost of some signal.
     ETFs: SPY, QQQ, IWM, DIA, GLD (a priori, matches NOVA26).
     TC: 2 bps charged per active overnight (still realistic).

B) WEEKLY RS SECTOR ROTATION (long-only):
     XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLU, XLB — 9 S&P sectors.
     Each Monday: rank by 63-day return. Long top-3, equal weight.
     Hold 1 week. TC 10 bps per rebalance.
     Published effect (Moskowitz-Grinblatt 1999) SR ~0.8-1.0.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
ETF_DIR = Path("/home/user/bonds/data/etfs")
TC_BPS_OVN = 2.0
RV_CUT = 0.15
OVN_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]


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


def weekly_flag(dates):
    """True on Mondays (first trading day of the ISO week)."""
    wk = pd.Series(dates).apply(lambda d: d.isocalendar().week)
    yr = pd.Series(dates).apply(lambda d: d.year)
    key = yr * 100 + wk
    flag = pd.Series(False, index=dates)
    seen = set()
    for i, d in enumerate(dates):
        k = key.iloc[i]
        if k not in seen:
            flag.iloc[i] = True
            seen.add(k)
    return flag


def sleeve_weekly_ovn(t, bil_ret, dates):
    rv = five_min_rv(t)
    ovn = overnight_returns(t)
    common = rv.index.intersection(ovn.index).intersection(bil_ret.index)
    rv, ovn_r, bil_r = rv.loc[common], ovn.loc[common], bil_ret.loc[common]
    rv20 = rv.rolling(20).mean().shift(1)
    # Compute gate daily
    gate_daily = rv20 < RV_CUT
    # Enforce: gate only changes on weekly rebalance days
    wflag = weekly_flag(common)
    gate_weekly = pd.Series(False, index=common)
    current = False
    for i, d in enumerate(common):
        if wflag.iloc[i]:
            current = bool(gate_daily.iloc[i])
        gate_weekly.iloc[i] = current
    r = pd.Series(0.0, index=common)
    r[gate_weekly] = ovn_r[gate_weekly] - TC_BPS_OVN / 1e4
    r[~gate_weekly] = bil_r[~gate_weekly]
    return r


def sleeve_sector_rotation(dates):
    px = {}
    for s in SECTORS:
        p = load_etf(s)
        if p is not None:
            px[s] = p.reindex(dates).ffill()
    P = pd.DataFrame(px).dropna(how="all")
    R = P.pct_change().fillna(0)
    mom = P.pct_change(63).shift(1)   # 63-day momentum
    wflag = weekly_flag(R.index)
    weights = pd.DataFrame(0.0, index=R.index, columns=R.columns)
    current = pd.Series(0.0, index=R.columns)
    for i, d in enumerate(R.index):
        if wflag.iloc[i] and i > 63 and not mom.loc[d].isna().all():
            ranked = mom.loc[d].dropna().sort_values()
            top3 = ranked.index[-3:]
            current = pd.Series(0.0, index=R.columns)
            current[top3] = 1.0 / 3
        weights.iloc[i] = current.values
    w_eff = weights.shift(1).fillna(0)
    port = (w_eff * R).sum(axis=1)
    to = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    port = port - to * (10.0 / 1e4)
    return port


def main():
    bil = load_etf("BIL").pct_change().fillna(0)
    bil.index = pd.to_datetime(bil.index)
    dates_guide = bil.index

    # Sleeve A — weekly overnight
    a_sleeves = {t: sleeve_weekly_ovn(t, bil, dates_guide) for t in OVN_TICKERS}
    adf = pd.DataFrame(a_sleeves).dropna(how="all").fillna(0)
    warm = pd.Timestamp("2016-03-01")
    adf = adf.loc[warm:]
    CUT = pd.Timestamp("2022-01-01")
    vols = adf.loc[:CUT].std() * np.sqrt(252)
    wA = (1 / vols) / (1 / vols).sum()
    portA = (adf * wA).sum(axis=1)
    s = stats(portA, "A: weekly OVN basket")
    print(f"{s['label']:30s} SR={s['sharpe']:>5.2f}  IS="
          f"{stats(portA.loc[:CUT],'')['sharpe']:.2f}  "
          f"OOS={stats(portA.loc[CUT:],'')['sharpe']:.2f}  "
          f"Vol={s['vol']:.2f}%  MDD={s['mdd']:>7.2f}%")

    # Sleeve B — sector rotation
    portB = sleeve_sector_rotation(dates_guide).loc[warm:]
    s = stats(portB, "B: sector rotation top-3 weekly")
    print(f"{s['label']:30s} SR={s['sharpe']:>5.2f}  IS="
          f"{stats(portB.loc[:CUT],'')['sharpe']:.2f}  "
          f"OOS={stats(portB.loc[CUT:],'')['sharpe']:.2f}  "
          f"Vol={s['vol']:.2f}%  MDD={s['mdd']:>7.2f}%")

    # Per-ticker weekly OVN stats
    print("\nPer-ticker weekly OVN:")
    for c in adf.columns:
        x = adf[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:10s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%")

    # Ensemble
    comb = pd.DataFrame({"weekly_OVN": portA, "sector_rot": portB}).fillna(0)
    vols2 = comb.loc[:CUT].std() * np.sqrt(252)
    wE = (1 / vols2) / (1 / vols2).sum()
    port = (comb * wE).sum(axis=1)
    print(f"\nERC weights: {wE.round(3).to_dict()}")
    print(f"Pairwise corr: {comb.corr().iloc[0, 1]:.3f}")
    s = stats(port, "NOVA28 combined 2-sleeve")
    print(f"{s['label']:30s} SR={s['sharpe']:>5.2f}  IS="
          f"{stats(port.loc[:CUT],'')['sharpe']:.2f}  "
          f"OOS={stats(port.loc[CUT:],'')['sharpe']:.2f}  "
          f"Vol={s['vol']:.2f}%  MDD={s['mdd']:>7.2f}%")

    out = pd.DataFrame({"NOVA28": port, "weekly_OVN": portA, "sector_rot": portB})
    out.to_csv("/home/user/bonds/data/results/nova28_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova28_returns.csv")


if __name__ == "__main__":
    main()
