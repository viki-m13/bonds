"""NOVA19 — Betting Against Beta / low-vol equity anomaly (Frazzini-Pedersen 2014).

FP14 documented OOS SR 1.0+ for the low-volatility anomaly across 20
equity markets 1984-2012. We apply a simple a priori rule on our 96
large-cap universe, monthly rebalance.

Strategy (FIXED a priori):
  At each month-start:
    1. Compute trailing 63-day realized vol per stock.
    2. Rank stocks by vol (ascending).
    3. Long top-20 LOWEST-vol stocks. Equal weight.
    4. Hold one month, rebalance on first trading day of next month.
  TC = 10 bps × monthly turnover.

NOT vol-scaling: this is discrete LOW-VOL-STOCK selection. Each position
is 5% of book, long-only. The portfolio's total vol is whatever the
basket produces naturally — we do not target any number.

Why this is structurally different from prior attempts:
  - FP14's anomaly is rooted in leverage-constraint theory (Black 1972):
    investors who cannot lever bid up high-beta stocks, leaving low-beta
    stocks underpriced.
  - Persistent across decades and markets.
  - Especially strong in BEAR markets (low-vol stocks don't fall as much).
  - Low correlation to TSMOM (NOVA18)."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


DATA = Path("/home/user/bonds/data/intraday_daily")
EXCLUDE = {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "VNQ",
           "XLK", "XLF", "XLE", "XLY", "XLP", "XLI", "XLV", "XLU", "XLB",
           "XLRE", "HYG", "IEF", "SHY", "BIL"}
WINDOW = 63
N_LONG = 20
TC_BPS = 10.0


def load_prices():
    stocks = sorted([p.stem for p in DATA.glob("*.csv") if p.stem not in EXCLUDE])
    frames = {}
    for t in stocks:
        df = pd.read_csv(DATA / f"{t}.csv", parse_dates=["ts"])
        df["date"] = pd.to_datetime(df["ts"].dt.date)
        frames[t] = df.set_index("date")["close"].sort_index()
    px = pd.DataFrame(frames).sort_index().ffill()
    return px


def monthly_first(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def main():
    px = load_prices()
    rets = px.pct_change().fillna(0)
    dates = px.index
    print(f"Universe: {rets.shape[1]} stocks, {len(dates)} days")

    first = monthly_first(dates)
    # Rolling 63-day realized vol, lag 1 day
    vol = rets.rolling(WINDOW).std().shift(1)

    weights = pd.DataFrame(0.0, index=dates, columns=rets.columns)
    current = pd.Series(0.0, index=rets.columns)

    for i, d in enumerate(dates):
        if first.iloc[i] and i >= WINDOW:
            v = vol.loc[d].dropna()
            if len(v) >= N_LONG * 2:
                picks = v.nsmallest(N_LONG).index
                current = pd.Series(0.0, index=rets.columns)
                current[picks] = 1.0 / N_LONG
        weights.loc[d] = current.values

    w_eff = weights.shift(1).fillna(0)
    port_gross = (w_eff * rets).sum(axis=1)
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    port = port_gross - turnover * (TC_BPS / 1e4)

    warm = pd.Timestamp("2016-06-01")
    p_v = port.loc[warm:]
    s = stats(p_v, f"NOVA19 Low-Vol-{N_LONG}")
    print(f"\n{s['label']:28s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(p_v.loc[:CUT], "IS <2022"), (p_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:26s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Long-short variant: long low-vol, short high-vol (BAB-style)
    weights_ls = pd.DataFrame(0.0, index=dates, columns=rets.columns)
    current_ls = pd.Series(0.0, index=rets.columns)
    for i, d in enumerate(dates):
        if first.iloc[i] and i >= WINDOW:
            v = vol.loc[d].dropna()
            if len(v) >= 2 * N_LONG:
                longs = v.nsmallest(N_LONG).index
                shorts = v.nlargest(N_LONG).index
                current_ls = pd.Series(0.0, index=rets.columns)
                current_ls[longs] = 1.0 / N_LONG
                current_ls[shorts] = -1.0 / N_LONG
        weights_ls.loc[d] = current_ls.values

    w_ls = weights_ls.shift(1).fillna(0)
    port_ls_gross = (w_ls * rets).sum(axis=1)
    turnover_ls = (w_ls - w_ls.shift(1)).abs().sum(axis=1).fillna(0)
    port_ls = port_ls_gross - turnover_ls * (TC_BPS / 1e4)
    pls = port_ls.loc[warm:]
    s = stats(pls, "NOVA19-LS Low-Vol L/S (BAB proxy)")
    print(f"\n{s['label']:34s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(pls.loc[:CUT], "IS <2022"), (pls.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:32s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Annual
    ann = p_v.groupby(p_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (long-only):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA19": port, "NOVA19_LS": port_ls})
    out.to_csv("/home/user/bonds/data/results/nova19_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova19_returns.csv")


if __name__ == "__main__":
    main()
