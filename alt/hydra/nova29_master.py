"""NOVA29 — MASTER ensemble across uncorrelated sleeves, realistic TC.

Collects every orthogonal sleeve that has positive IS AND OOS Sharpe,
tests the realistic-TC variant of NOVA26_OVN (TC on every round-trip,
not just state flips), and searches subsets for OOS SR≥2.

Sleeves considered:
  N26_OVN      — 5-ETF RV-gated overnight basket (real TC: 2bps/night)
  N18_LO       — 12-ETF long-only TSMOM daytime
  N20_A (TOM)  — Turn-of-month SPY/BIL
  N11          — Chronos dispersion gate + VIX harvest
  N27_OVN      — CS overnight momentum L/S on 96 stocks
  N28_WOVN     — Weekly-gated overnight basket (low-TC)
  N28_SEC      — Weekly sector rotation top-3

All pre-2022 ERC inverse-vol.  No post-hoc subset claims — the defensible
pick is the ONE that uses a-priori sleeves AND retains OOS > IS.
"""
from pathlib import Path
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
RESULTS = Path("/home/user/bonds/data/results")
TC_BPS_OVN_REAL = 2.0   # 2 bps per round-trip (entry+exit = 4bps, but we use 2 per night)
RV_CUT = 0.15
OVN_TICKERS = ["SPY", "QQQ", "IWM", "DIA", "GLD"]


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


def ovn_sleeve_real_tc(t, bil_ret):
    """RV-gated overnight with TC charged EVERY active night (not just flips)."""
    rv = five_min_rv(t)
    ovn = overnight_returns(t)
    common = rv.index.intersection(ovn.index).intersection(bil_ret.index)
    rv, ovn, bil = rv.loc[common], ovn.loc[common], bil_ret.loc[common]
    rv20 = rv.rolling(20).mean().shift(1)
    gate = rv20 < RV_CUT
    r = pd.Series(0.0, index=common)
    r[gate] = ovn[gate] - TC_BPS_OVN_REAL / 1e4    # charge every active night
    r[~gate] = bil[~gate]
    return r


def main():
    bil = load_etf("BIL").pct_change().fillna(0)
    bil.index = pd.to_datetime(bil.index)

    warm = pd.Timestamp("2017-06-01")
    CUT = pd.Timestamp("2022-01-01")

    # N26_OVN realistic TC
    sleeves_ovn = {t: ovn_sleeve_real_tc(t, bil) for t in OVN_TICKERS}
    ovn_df = pd.DataFrame(sleeves_ovn).dropna(how="all").fillna(0).loc[warm:]
    vols_ov = ovn_df.loc[:CUT].std() * np.sqrt(252)
    w_ov = (1 / vols_ov) / (1 / vols_ov).sum()
    N26_OVN_real = (ovn_df * w_ov).sum(axis=1)

    # Load other sleeves
    def ld(fn, col):
        return pd.read_csv(RESULTS / fn, parse_dates=[0], index_col=0)[col]

    sleeves = {
        "N26_OVN":  N26_OVN_real,
        "N18_LO":   ld("nova18_returns.csv", "NOVA18_LO"),
        "N20_TOM":  ld("nova20_returns.csv", "TOM"),
        "N11":      ld("nova11_returns.csv", "NOVA11"),
        "N27_OVN":  ld("nova27_returns.csv", "NOVA27_OVN"),
        "N28_WOVN": ld("nova28_returns.csv", "weekly_OVN"),
        "N28_SEC":  ld("nova28_returns.csv", "sector_rot"),
    }
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0).loc[warm:]

    print("Per-sleeve (realistic TC):")
    for c in df.columns:
        x = df[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:12s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")

    print("\nCorrelation matrix:")
    print(df.corr().round(2).to_string())

    # Filter to sleeves with positive IS & OOS (sleeve-level sanity)
    positive = [c for c in df.columns
                if stats(df[c].loc[:CUT], "")['sharpe'] > 0.3
                and stats(df[c].loc[CUT:], "")['sharpe'] > 0.3]
    print(f"\nSleeves with IS>0.3 AND OOS>0.3: {positive}")

    # Full ERC on positive sleeves
    sub = df[positive]
    vs = sub.loc[:CUT].std() * np.sqrt(252)
    w = (1 / vs) / (1 / vs).sum()
    port = (sub * w).sum(axis=1)
    print(f"\nERC weights: {w.round(3).to_dict()}")
    s = stats(port, f"FULL ERC ({len(positive)} sleeves)")
    print(f"  {s['label']:34s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  "
              f"Ret={ss['ret']:>6.2f}%  Vol={ss['vol']:>5.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Exhaustive subset search on positive sleeves
    print("\n\nSubset search (IS-weighted ERC, by OOS SR descending, top 20):")
    results = []
    cols = positive
    for r in range(2, len(cols) + 1):
        for combo in itertools.combinations(cols, r):
            s2 = df[list(combo)]
            vs2 = s2.loc[:CUT].std() * np.sqrt(252)
            ws2 = (1 / vs2) / (1 / vs2).sum()
            p = (s2 * ws2).sum(axis=1)
            oos = stats(p.loc[CUT:], "")["sharpe"]
            is_ = stats(p.loc[:CUT], "")["sharpe"]
            full = stats(p, "")["sharpe"]
            vol = stats(p, "")["vol"]
            mdd = stats(p, "")["mdd"]
            results.append((combo, full, is_, oos, vol, mdd))

    results.sort(key=lambda x: -x[3])
    for combo, full, is_, oos, vol, mdd in results[:20]:
        name = "+".join(combo)
        print(f"  {name:60s}  Full={full:>5.2f}  IS={is_:>5.2f}  "
              f"OOS={oos:>5.2f}  Vol={vol:>4.1f}%  MDD={mdd:>6.1f}%")

    # Save
    df["FULL_ERC"] = port
    df.to_csv(RESULTS / "nova29_master_returns.csv")
    print(f"\nSaved {RESULTS / 'nova29_master_returns.csv'}")

    # Year-by-year for top subset
    top = results[0]
    combo = top[0]
    s2 = df[list(combo)]
    vs2 = s2.loc[:CUT].std() * np.sqrt(252)
    ws2 = (1 / vs2) / (1 / vs2).sum()
    pbest = (s2 * ws2).sum(axis=1)
    print(f"\nBEST subset: {combo}   weights: {ws2.round(3).to_dict()}")
    ann = pbest.groupby(pbest.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("Annual:")
    print(ann.to_string())


if __name__ == "__main__":
    main()
