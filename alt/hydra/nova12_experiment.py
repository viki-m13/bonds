"""NOVA12 — cross-sectional equity long-short on 96 US large-caps.

We've been trapped in ETF-level time-series forecasting (all NOVA2-11
max out at ~0.7 OOS SR). Cross-sectional equity factors are a
structurally different alpha source: 40+ years of academic evidence,
robust OOS across multiple markets.

Two sleeves, both fully mechanical, zero tuning:

  A) 12-1 MOMENTUM (Jegadeesh-Titman 1993)
     Signal: 252-day return skipping last 21 days (avoid reversal).
     Monthly: rank all stocks by signal. Long top decile (~10),
     short bottom decile (~10), equal-weight within leg, dollar-
     neutral, 1-bar lag, 15 bps TC per trade.

  B) 5-DAY SHORT-TERM REVERSAL (Lehmann 1990, Jegadeesh 1990)
     Signal: 5-day return (most recent week).
     Weekly rebalance (Monday): long bottom decile (prev-week losers),
     short top decile (prev-week winners), equal-weight, dollar-neutral.
     15 bps TC each leg — pays for itself if reversal spread > ~12 bps
     per week.

Momentum and ST reversal are KNOWN to be negatively correlated
(reversal contra-trades recent momentum), so combining adds meaningful
diversification even before Chronos/trend sleeves.

Ensemble: equal-RISK-contribution weights, computed on pre-2018 only.
Report full / IS (<2018) / OOS (>=2018). 96 stocks, 2005-01-03..."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


STOCKS_DIR = Path("/home/user/bonds/data/stocks")
TC_BPS = 15.0


def load_stocks():
    tickers = sorted([p.stem for p in STOCKS_DIR.glob("*.csv")])
    frames = {}
    for t in tickers:
        s = pd.read_csv(STOCKS_DIR / f"{t}.csv", parse_dates=["Date"]).set_index("Date")["Close"]
        s = s[~s.index.duplicated(keep="first")].sort_index()
        frames[t] = s
    px = pd.DataFrame(frames).sort_index().ffill()
    rets = px.pct_change()
    return px, rets


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def weekly_first_flag(index):
    """Every Monday (or first trading day of the week)."""
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].isocalendar().week != index[i - 1].isocalendar().week:
            out.iloc[i] = True
    return out


def cross_sectional_longshort(signal, rets, n_leg, rebal_flag):
    """Rank stocks by signal at each rebal point. Long top-n, short
    bottom-n, equal-weight, dollar-neutral. 1-bar lag, TC on turnover."""
    dates = rets.index
    tickers = rets.columns
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)

    last_w = pd.Series(0.0, index=tickers)
    for i, d in enumerate(dates):
        if rebal_flag.iloc[i]:
            s = signal.loc[d]
            valid = s.dropna()
            if len(valid) < 2 * n_leg:
                # not enough, hold last
                weights.iloc[i] = last_w.values
                continue
            ranked = valid.sort_values()
            shorts = ranked.index[:n_leg]
            longs = ranked.index[-n_leg:]
            w = pd.Series(0.0, index=tickers)
            w[longs] = 1.0 / n_leg
            w[shorts] = -1.0 / n_leg
            last_w = w
        weights.iloc[i] = last_w.values

    # 1-bar lag
    w_eff = weights.shift(1).fillna(0)
    # Portfolio return
    port_ret = (w_eff * rets.fillna(0)).sum(axis=1)

    # TC on weight turnover
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    tc = turnover * (TC_BPS / 1e4)
    return port_ret - tc, w_eff


def sleeve_momentum(px, rets, dates):
    # 12-1 momentum: 252d return skipping last 21d
    m = px.shift(21) / px.shift(21 + 252) - 1
    first = monthly_first_flag(pd.Index(dates))
    r, w = cross_sectional_longshort(m, rets, n_leg=10, rebal_flag=first)
    return r.rename("mom_12_1"), w


def sleeve_stReversal(px, rets, dates):
    # 5-day reversal: rank on last 5d return, then flip (bottom is long, top is short)
    # Achieved by signaling NEGATIVE of 5d return (bottom performers get high signal)
    five = px / px.shift(5) - 1
    signal = -five   # negate so BOTTOM decile of 5d return gets HIGHEST rank → long
    first_week = weekly_first_flag(pd.Index(dates))
    r, w = cross_sectional_longshort(signal, rets, n_leg=10, rebal_flag=first_week)
    return r.rename("st_rev_5d"), w


def main():
    print("Loading 96 stocks...")
    px, rets = load_stocks()
    dates = rets.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()}  |  {px.shape[1]} tickers")
    print("NOVA12 — cross-sectional momentum + short-term reversal (equity L/S)\n")

    print("Building sleeve A: 12-1 momentum, monthly, decile L/S...")
    rA, _ = sleeve_momentum(px, rets, dates)

    print("Building sleeve B: 5-day reversal, weekly, decile L/S...")
    rB, _ = sleeve_stReversal(px, rets, dates)

    # Sufficient warmup
    warm = pd.Timestamp("2006-02-01")
    rA_v = rA.loc[warm:]
    rB_v = rB.loc[warm:]

    for r, lbl in [(rA_v, "A: 12-1 momentum L/S"), (rB_v, "B: 5d reversal L/S")]:
        s = stats(r, lbl)
        print(f"{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    # Correlation
    corr = rA_v.corr(rB_v)
    print(f"\nCorr(A,B) = {corr:+.3f}")

    # ERC weights pre-2018
    CUT = pd.Timestamp("2018-01-01")
    vA = rA.loc[warm:CUT].std() * np.sqrt(252)
    vB = rB.loc[warm:CUT].std() * np.sqrt(252)
    wA = (1 / vA) / (1 / vA + 1 / vB)
    wB = 1 - wA
    print(f"ERC weights (pre-2018 vol): A={wA:.3f} B={wB:.3f}")

    port = wA * rA_v + wB * rB_v
    s = stats(port, "NOVA12 ERC")
    print(f"\n{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    for p, tag in [(port.loc[:CUT], "IS <2018"), (port.loc[CUT:], "OOS >=2018")]:
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

    # Save
    out = pd.DataFrame({"NOVA12": port, "momentum": rA_v, "st_reversal": rB_v})
    out.to_csv("/home/user/bonds/data/results/nova12_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova12_returns.csv")


if __name__ == "__main__":
    main()
