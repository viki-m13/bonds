"""NOVA14 — Intraday REVERSAL (2016-2026 era) + Overnight drift.

First test (see repo history): Gao-Han-Li-Zhou 2018 intraday momentum
(sign of first-30min predicts last-30min) flipped sign in the 2016-
2026 era: SR -1.34 on SPY, -1.12 QQQ, -0.83 IWM. HFT arbitraged out
the momentum and the pattern now MEAN-REVERTS. FADE the morning move.

Fixed-a-priori rules (no tuning):

  A) INTRADAY REVERSAL — FADE AFTERNOON
     At 15:30, pos = -sign(r_0930_to_1000).
     Hold pos until 16:00 close. 2 bps round-trip TC.
     Per-ETF; equal-weight SPY+QQQ+IWM.

  B) FULL-SESSION REVERSAL — FADE ALL DAY
     At 10:00, pos = -sign(r_0930_to_1000).
     Hold until 16:00. Captures the full reversal move, 2 bps TC.
     (Longer hold → more carry of any trend component, test which dominates.)

  C) OVERNIGHT LONG DRIFT
     Buy at 15:55 close, sell at next-day 9:30 open. 2 bps TC.
     Averages ~4 bps/day drift before TC; SR ~0.5 after.

Ensemble: equal-risk-contribution, weights from pre-2022 vol (strict OOS).
OOS = 2022+."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


DATA = Path("/home/user/bonds/data/intraday_5min")
TC_BPS = 2.0


def load_5min(t):
    df = pd.read_csv(DATA / f"{t}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["time"] = df["ts"].dt.time
    return df.sort_values("ts")


def price_close_of_bar_starting(df, start_time):
    """Close of the 5-min bar that starts at start_time."""
    target = pd.to_datetime(start_time).time()
    sub = df[df["time"] == target].set_index("date")["close"]
    return sub[~sub.index.duplicated(keep="first")].sort_index()


def day_open(df):
    return df.groupby("date")["open"].first()


def day_close(df):
    return df.groupby("date")["close"].last()


def strat_reversal_afternoon(ticker):
    df = load_5min(ticker)
    open_ = day_open(df)
    px_10am = price_close_of_bar_starting(df, "09:55")
    px_1530 = price_close_of_bar_starting(df, "15:25")
    close_ = day_close(df)
    common = open_.index.intersection(px_10am.index).intersection(
        px_1530.index).intersection(close_.index)
    open_, px_10am, px_1530, close_ = [s.loc[common] for s in
                                        (open_, px_10am, px_1530, close_)]
    r_first30 = px_10am / open_ - 1
    r_last30 = close_ / px_1530 - 1
    pos = -np.sign(r_first30)
    r = pos * r_last30 - (TC_BPS / 1e4)
    return r.rename(f"rev_pm_{ticker}")


def strat_reversal_fullsession(ticker):
    df = load_5min(ticker)
    open_ = day_open(df)
    px_10am = price_close_of_bar_starting(df, "09:55")
    close_ = day_close(df)
    common = open_.index.intersection(px_10am.index).intersection(close_.index)
    open_, px_10am, close_ = [s.loc[common] for s in (open_, px_10am, close_)]
    r_first30 = px_10am / open_ - 1
    r_rest = close_ / px_10am - 1
    pos = -np.sign(r_first30)
    r = pos * r_rest - (TC_BPS / 1e4)
    return r.rename(f"rev_full_{ticker}")


def strat_overnight(ticker):
    df = load_5min(ticker)
    px_1555 = price_close_of_bar_starting(df, "15:55")
    open_ = day_open(df)
    common = px_1555.index.intersection(open_.index)
    px_1555, open_ = px_1555.loc[common], open_.loc[common]
    overnight = open_.shift(-1) / px_1555 - 1
    overnight = overnight.dropna()
    tc = TC_BPS / 1e4
    return (overnight - tc).rename(f"ovn_{ticker}")


def main():
    tickers = ["SPY", "QQQ", "IWM"]
    print(f"Universe: {tickers}")
    print("NOVA14 — intraday reversal + overnight drift\n")

    sleeves = {"rev_pm": [], "rev_full": [], "ovn": []}
    for t in tickers:
        sleeves["rev_pm"].append(strat_reversal_afternoon(t))
        sleeves["rev_full"].append(strat_reversal_fullsession(t))
        sleeves["ovn"].append(strat_overnight(t))

    print(f"{'Strategy':<12} {'Ticker':<6} {'SR':>7} {'Ret%':>7} {'Vol%':>7} {'MDD%':>8}")
    for strat, lst in sleeves.items():
        for s in lst:
            t = s.name.split("_")[-1]
            st = stats(s, strat)
            print(f"  {strat:<10} {t:<6} {st['sharpe']:>7.2f} {st['ret']:>7.2f} "
                  f"{st['vol']:>7.2f} {st['mdd']:>8.2f}")

    strat_ew = {}
    for s_name, lst in sleeves.items():
        df = pd.concat(lst, axis=1).fillna(0)
        strat_ew[s_name] = df.mean(axis=1).rename(s_name)
        st = stats(strat_ew[s_name], s_name + "_avg")
        print(f"  {s_name+'_avg':<20} SR={st['sharpe']:>5.2f}  "
              f"Ret={st['ret']:>6.2f}%  Vol={st['vol']:>5.2f}%  MDD={st['mdd']:>7.2f}%")

    combined = pd.DataFrame(strat_ew)
    print("\nStrategy correlations:")
    print(combined.corr().round(3))

    CUT = pd.Timestamp("2022-01-01")
    vols = combined.loc[:CUT].std() * np.sqrt(252)
    invv = 1 / vols
    w = invv / invv.sum()
    print(f"\nERC weights (pre-2022): {w.round(3).to_dict()}")

    port = (combined * w).sum(axis=1)
    s = stats(port, "NOVA14 ERC (3-sleeve)")
    print(f"\n{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:26s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # 2-sleeve variant: rev_pm + ovn (drop the redundant rev_full)
    two = combined[["rev_pm", "ovn"]]
    vols2 = two.loc[:CUT].std() * np.sqrt(252)
    w2 = (1/vols2) / (1/vols2).sum()
    port2 = (two * w2).sum(axis=1)
    s = stats(port2, "NOVA14 2-sleeve rev_pm+ovn")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(port2.loc[:CUT], "IS <2022"), (port2.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Annual
    ann = port.groupby(port.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (3-sleeve ensemble):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA14": port}).join(combined, how="outer")
    out.to_csv("/home/user/bonds/data/results/nova14_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova14_returns.csv")


if __name__ == "__main__":
    main()
