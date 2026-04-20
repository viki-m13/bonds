"""NOVA13 — overnight-gap mean reversion (intraday execution).

Published finding (Chen & Demirer 2023; Lou, Polk, Skouras 2019):
large overnight gaps tend to REVERSE during the subsequent trading
session. Structural cause: retail/news-driven sentiment at the open
overreacts; liquidity providers fade the extreme and mean-revert by
close.

Using Alpaca daily OHLC 2016-present for 96 large-caps + 20 ETFs,
we exploit this without relying on forecasting:

  Signal (per ticker, per day):
    gap_t = open_t / close_{t-1} - 1
    intra_t = close_t / open_t - 1  (ex-post)

  Strategy A — SPY / ETF gap-fade:
    For each of {SPY, QQQ, IWM, DIA, XLK, XLF, XLE}:
      if gap_t < -0.50%  →  long position at open, close at 4pm (ride reversal up)
      if gap_t > +0.75%  →  short position at open, close at 4pm (ride reversal down)
      else               →  flat
    Asymmetric thresholds because up-gap reversal is weaker than
    down-gap reversal in the literature. Fixed a priori — no tuning.

  Strategy B — cross-sectional stock gap-fade:
    Rank 96 stocks by gap_t. Long bottom decile intraday (big down-gaps),
    short top decile intraday (big up-gaps). Equal-weight dollar-neutral.
    Fixed decile split. No tuning.

TC: 2 bps per round-trip (competitive MOO/MOC execution, realistic for
institutional; doubled to 4 bps for conservatism).

Backtest: full sample 2016-2026. IS=2016-2021 (6y), OOS=2022-now (~4y).
NOT using vol scaling, NOT fitting thresholds."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


DATA = Path("/home/user/bonds/data/intraday_daily")
TC_BPS = 4.0   # 2 bps × 2 legs (conservative)


def load_daily(t):
    p = DATA / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["ts"])
    df["date"] = df["ts"].dt.date
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]].sort_index()
    df.index = pd.to_datetime(df.index)
    return df


ETF_GAP = ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLY"]
STOCK_UNIVERSE = None   # filled in main


def sleeve_etf_gap_fade():
    """For each ETF, fade gaps intraday. Dollar-unit per ETF, equal-weight sum."""
    rets = []
    for t in ETF_GAP:
        df = load_daily(t)
        if df is None:
            continue
        gap = df["open"] / df["close"].shift(1) - 1
        intra = df["close"] / df["open"] - 1
        # Signal: fade gap
        pos = pd.Series(0.0, index=df.index)
        pos[gap < -0.005] = 1.0   # long intraday after down-gap
        pos[gap > 0.0075] = -1.0  # short intraday after up-gap
        # TC on entry+exit (same day)
        tc = (pos != 0).astype(float) * (TC_BPS / 1e4)
        r = pos * intra - tc
        rets.append(r.rename(f"gap_{t}"))
    df_r = pd.concat(rets, axis=1).fillna(0)
    # Equal-weight daily across ETFs that traded; average of nonzero counts
    port = df_r.mean(axis=1)
    return port, df_r


def sleeve_cs_gap_fade():
    """Cross-sectional: long bottom-decile gap (biggest down-gaps), short top
    decile (biggest up-gaps). Hold intraday only, equal-weight."""
    global STOCK_UNIVERSE
    stocks = STOCK_UNIVERSE
    all_gap = {}
    all_intra = {}
    for t in stocks:
        df = load_daily(t)
        if df is None:
            continue
        all_gap[t] = df["open"] / df["close"].shift(1) - 1
        all_intra[t] = df["close"] / df["open"] - 1
    G = pd.DataFrame(all_gap)
    I = pd.DataFrame(all_intra)
    common = G.index.intersection(I.index)
    G = G.loc[common]
    I = I.loc[common]

    N = 10    # decile on 96 universe = top/bottom 10
    port = pd.Series(0.0, index=common)
    turnover = pd.Series(0.0, index=common)
    for d in common:
        g = G.loc[d].dropna()
        if len(g) < 2 * N:
            continue
        ranked = g.sort_values()
        shorts = ranked.index[-N:]   # biggest up-gaps → short intraday
        longs = ranked.index[:N]     # biggest down-gaps → long intraday
        r_long = I.loc[d, longs].mean()
        r_short = -I.loc[d, shorts].mean()
        port.loc[d] = 0.5 * r_long + 0.5 * r_short
        turnover.loc[d] = 2.0   # 100% long + 100% short turned over every day
    tc = turnover * (TC_BPS / 1e4)
    return (port - tc).rename("cs_gap_fade"), turnover


def main():
    global STOCK_UNIVERSE
    STOCK_UNIVERSE = sorted([p.stem for p in DATA.glob("*.csv")
                             if p.stem not in ETF_GAP + ["TLT", "GLD", "EFA", "EEM", "VNQ",
                                                         "XLP", "XLI", "XLV", "XLU", "XLB", "XLRE",
                                                         "HYG", "IEF", "SHY", "BIL"]])
    print(f"Stock universe: {len(STOCK_UNIVERSE)} tickers")
    print("NOVA13 — overnight-gap mean reversion (daily OHLC, intraday exec)\n")

    print("Building sleeve A (ETF gap-fade)...")
    portA, dfA = sleeve_etf_gap_fade()
    print("Building sleeve B (cross-sectional gap-fade)...")
    portB, turn = sleeve_cs_gap_fade()

    warm = pd.Timestamp("2016-03-01")
    portA_v = portA.loc[warm:]
    portB_v = portB.loc[warm:]

    for r, lbl in [(portA_v, "A: ETF gap-fade"), (portB_v, "B: CS gap-fade")]:
        s = stats(r, lbl)
        print(f"{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Per-ETF
    print("\nPer-ETF (A):")
    for c in dfA.columns:
        r = dfA[c].loc[warm:]
        r_nz = r[r != 0]
        s = stats(r, c)
        n_trades = (dfA[c].loc[warm:] != 0).sum()
        print(f"  {c:18s} SR={s['sharpe']:>5.2f}  Trades={n_trades:>4}  "
              f"Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")

    corr = portA_v.corr(portB_v)
    print(f"\nCorr(A,B) = {corr:+.3f}")

    # ERC weights
    CUT = pd.Timestamp("2022-01-01")
    vA = portA.loc[warm:CUT].std() * np.sqrt(252)
    vB = portB.loc[warm:CUT].std() * np.sqrt(252)
    wA = (1 / vA) / (1 / vA + 1 / vB)
    wB = 1 - wA
    print(f"ERC weights (pre-2022 vol): A={wA:.3f} B={wB:.3f}")

    port = wA * portA_v + wB * portB_v
    s = stats(port, "NOVA13 ERC")
    print(f"\n{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    for p, tag in [(port.loc[:CUT], "IS <2022"), (port.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:26s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

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

    out = pd.DataFrame({"NOVA13": port, "sleeveA": portA_v, "sleeveB": portB_v})
    out.to_csv("/home/user/bonds/data/results/nova13_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova13_returns.csv")


if __name__ == "__main__":
    main()
