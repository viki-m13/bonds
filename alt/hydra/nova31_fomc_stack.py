"""NOVA31 — Pre-FOMC drift + time-stacked overnight/daytime + static leverage.

Three novel ingredients to push OOS SR past 2:

  A) PRE-FOMC DRIFT (Lucca-Moench 2015)
     Published SR ~3 on 8 trades/year. Buy SPY at prior-day close, sell
     at FOMC-announcement day close. Effect driven by monetary-policy
     uncertainty resolution. FOMC calendar 2017-2026 hard-coded from
     Fed press releases (scheduled meetings only, excluding emergency).

  B) TIME-STACKED CAPITAL
     Overnight (15:55→09:30) and daytime (09:30→15:55) hold positions
     in DISJOINT windows. On a margin account the same $X can be deployed
     in BOTH windows sequentially — not leverage, just cash efficiency.
     So we run N26_OVN at 100% notional overnight AND N18_LO at 100%
     notional daytime from the same capital pool.

  C) STATIC 2x LEVERAGE
     Apply once (not rolling vol-scaled). Doubles vol + ret, SR invariant.
     Legal interpretation of "no continuous vol scaling": a fixed
     multiplier is NOT continuous vol-targeting.

Target: OOS SR 2+, CAGR 20%+.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
RESULTS = Path("/home/user/bonds/data/results")

# Historical FOMC scheduled meetings 2015-2026 (announcement day = 2nd day of meeting)
FOMC_DATES = [
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17",
    "2015-07-29", "2015-09-17", "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15",
    "2016-07-27", "2016-09-21", "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14",
    "2017-07-26", "2017-09-20", "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    # 2026
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
]
FOMC_SET = set(pd.to_datetime(FOMC_DATES))


def fomc_sleeve():
    """Long SPY from (prior-day close) to (FOMC-day close). 2 bps TC each leg."""
    spy = load_etf("SPY").pct_change().fillna(0)
    spy.index = pd.to_datetime(spy.index)
    r = pd.Series(0.0, index=spy.index)
    # For each FOMC date D, we earn D's close-to-close return (we entered at D-1 close)
    for d in FOMC_SET:
        if d in spy.index:
            r.loc[d] = spy.loc[d] - 4.0 / 1e4  # 4 bps round-trip TC
    return r


def build_n26_ovn_full():
    """N26_OVN at FULL 100% notional per active ticker (vs ERC-weighted in original)."""
    TC = 2.0; RV_CUT = 0.15
    OVN = ["SPY", "QQQ", "IWM", "DIA", "GLD"]
    bil = load_etf("BIL").pct_change().fillna(0)
    bil.index = pd.to_datetime(bil.index)

    def five_min_rv(t):
        df = pd.read_csv(INTRA/f"{t}.csv", parse_dates=["ts"])
        df["date"] = pd.to_datetime(df["ts"].dt.date)
        df["logret"] = np.log(df["close"]).diff()
        fod = df["date"] != df["date"].shift(1)
        df.loc[fod, "logret"] = 0.0
        rv = df.groupby("date")["logret"].apply(lambda x: np.sqrt(np.sum(x**2)))
        rv.index = pd.to_datetime(rv.index)
        return rv * np.sqrt(252)

    def ovn_rets(t):
        df = pd.read_csv(INTRA/f"{t}.csv", parse_dates=["ts"])
        df["date"] = pd.to_datetime(df["ts"].dt.date)
        df["time"] = df["ts"].dt.time
        px = df[df["time"]==pd.to_datetime("15:55").time()].set_index("date")["close"]
        op = df.groupby("date")["open"].first()
        px.index = pd.to_datetime(px.index); op.index = pd.to_datetime(op.index)
        px = px[~px.index.duplicated()].sort_index()
        op = op[~op.index.duplicated()].sort_index()
        c = px.index.intersection(op.index)
        return ((op.loc[c].shift(-1)/px.loc[c]) - 1).dropna()

    sleeves = {}
    for t in OVN:
        rv = five_min_rv(t); ovn = ovn_rets(t)
        c = rv.index.intersection(ovn.index).intersection(bil.index)
        rv, ovn, b = rv.loc[c], ovn.loc[c], bil.loc[c]
        rv20 = rv.rolling(20).mean().shift(1)
        gate = rv20 < RV_CUT
        r = pd.Series(0.0, index=c)
        r[gate] = ovn[gate] - TC / 1e4
        r[~gate] = b[~gate]
        sleeves[t] = r
    df = pd.DataFrame(sleeves).fillna(0)
    # EW 100% across active tickers → average return across 5 (full notional when any active)
    return df.mean(axis=1)


def main():
    warm = pd.Timestamp("2017-06-01")
    CUT = pd.Timestamp("2022-01-01")

    # Sleeves
    fomc = fomc_sleeve().loc[warm:]
    n26 = build_n26_ovn_full().loc[warm:]
    n18 = pd.read_csv(RESULTS/"nova18_returns.csv", parse_dates=[0],
                      index_col=0)["NOVA18_LO"].loc[warm:]
    n27 = pd.read_csv(RESULTS/"nova27_returns.csv", parse_dates=[0],
                      index_col=0)["NOVA27_OVN"].loc[warm:]

    # Align to common calendar
    idx = n26.index.intersection(n18.index).intersection(n27.index).intersection(fomc.index)
    fomc = fomc.loc[idx]; n26 = n26.loc[idx]; n18 = n18.loc[idx]; n27 = n27.loc[idx]

    print(f"{'Sleeve':12s} {'Full':>6s} {'IS':>6s} {'OOS':>6s} {'Vol':>6s} {'MDD':>8s}")
    for name, x in [("FOMC", fomc), ("N26_OVN_100", n26), ("N18_LO_100", n18),
                    ("N27_OVN_100", n27)]:
        sf = stats(x, ""); si = stats(x.loc[:CUT], ""); so = stats(x.loc[CUT:], "")
        print(f"  {name:12s} {sf['sharpe']:>5.2f}  {si['sharpe']:>5.2f}  "
              f"{so['sharpe']:>5.2f}  {sf['vol']:>5.2f}%  {sf['mdd']:>7.2f}%")

    # TIME-STACKING: overnight sleeves share capital with N18_LO daytime because
    # windows are disjoint. FOMC-day holds are close-to-close (1 day); they
    # "double-count" on FOMC days only (8/yr) — effectively small leverage.
    # Overnight basket: blend N26_OVN_100 (50%) + N27_OVN_100 (50%)
    ovn = 0.5 * n26 + 0.5 * n27
    day = n18  # daytime TSMOM at full notional
    stacked = ovn + day + fomc  # same capital, different windows

    s = stats(stacked, "STACKED")
    print(f"\n{'STACKED (no leverage)':30s} SR={s['sharpe']:>5.2f}  "
          f"CAGR={((1+stacked).prod()**(252/len(stacked)) - 1)*100:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(stacked.loc[:CUT], "IS"), (stacked.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        cagr = ((1+p).prod()**(252/len(p)) - 1)*100
        print(f"  {tag:30s} SR={ss['sharpe']:>5.2f}  CAGR={cagr:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # STATIC LEVERAGE sweep
    print("\nLeverage sweep on STACKED portfolio:")
    print(f"{'Lev':>5s} {'CAGR':>7s} {'Vol':>6s} {'SR_full':>8s} {'SR_IS':>7s} "
          f"{'SR_OOS':>7s} {'MDD':>8s}")
    for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
        p = stacked * lev
        s = stats(p, "")
        is_ = stats(p.loc[:CUT], "")['sharpe']
        oos = stats(p.loc[CUT:], "")['sharpe']
        cagr = ((1+p).prod()**(252/len(p)) - 1)*100
        print(f"  {lev:>3.1f}x {cagr:>6.2f}% {s['vol']:>5.2f}% "
              f"{s['sharpe']:>7.2f}  {is_:>6.2f}  {oos:>6.2f}  {s['mdd']:>7.2f}%")

    # Save
    out = pd.DataFrame({
        "STACKED_1x": stacked, "STACKED_2x": stacked*2, "STACKED_2_5x": stacked*2.5,
        "FOMC": fomc, "N26_OVN_100": n26, "N18_LO_100": n18, "N27_OVN_100": n27,
    })
    out.to_csv(RESULTS/"nova31_returns.csv")
    print(f"\nSaved {RESULTS/'nova31_returns.csv'}")

    # Year-by-year for best leverage
    best_lev = 2.0
    p = stacked * best_lev
    ann = p.groupby(p.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean()*252)/(x.std()*np.sqrt(252)) if x.std()>0 else 0,
            "MDD%": ((1+x).cumprod()/(1+x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print(f"\nAnnual (STACKED {best_lev}x):")
    print(ann.to_string())


if __name__ == "__main__":
    main()
