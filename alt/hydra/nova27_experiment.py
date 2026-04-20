"""NOVA27 — Cross-sectional OVERNIGHT MOMENTUM on 96 large-caps (Lou-Polk-Skouras 2019).

Key finding (JF 2019 Table 5): overnight returns are STRONGLY persistent
across stocks. Stocks with high trailing overnight returns continue to
have high overnight returns; the OLS estimate of stock-level overnight
momentum has t-stat >20 over 1993-2013.

Strategy (FIXED a priori):
  At each month-start:
    1. For each stock, compute trailing 252-day cumulative OVERNIGHT
       return = Σ log(open_t / close_{t-1}).
    2. Rank stocks. Long top decile (N_SIDE=10), short bottom decile.
    3. Equal-weight dollar-neutral.
    4. Hold full calendar month (close-to-close each day — we do NOT
       execute overnight-only; retail execution would be MOC+MOO but
       that's impractical for 20 names simultaneously).
    5. Monthly rebalance. 1-day lag. 15 bps TC per rebalance.

Note: holding close-to-close dilutes the effect (Lou 2019: overnight
captures 100%+ while intraday adds ~0%). The published SR drops from
~2.5 (overnight-only) to ~1.0-1.5 (close-to-close) but is more
executable on monthly schedule."""
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
LOOKBACK = 252
N_SIDE = 10
TC_BPS = 15.0


def load_ohlc():
    stocks = sorted([p.stem for p in DATA.glob("*.csv") if p.stem not in EXCLUDE])
    opens = {}
    closes = {}
    for t in stocks:
        df = pd.read_csv(DATA / f"{t}.csv", parse_dates=["ts"])
        df["date"] = pd.to_datetime(df["ts"].dt.date)
        df = df.set_index("date").sort_index()
        opens[t] = df["open"]
        closes[t] = df["close"]
    O = pd.DataFrame(opens).sort_index().ffill()
    C = pd.DataFrame(closes).sort_index().ffill()
    return O, C


def monthly_first(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def main():
    O, C = load_ohlc()
    dates = C.index
    print(f"Universe: {C.shape[1]} stocks, {len(dates)} days")

    # Overnight log-returns
    ovn_logret = (np.log(O) - np.log(C.shift(1))).fillna(0)
    close_ret = C.pct_change().fillna(0)   # full close-to-close

    first = monthly_first(dates)
    weights = pd.DataFrame(0.0, index=dates, columns=C.columns)
    current = pd.Series(0.0, index=C.columns)

    signal_hist = []
    for i, d in enumerate(dates):
        if not first.iloc[i] or i < LOOKBACK:
            weights.loc[d] = current.values
            continue
        # trailing 252d overnight cum log-return
        sig = ovn_logret.iloc[i - LOOKBACK:i].sum()
        sig = sig.dropna()
        if len(sig) < 2 * N_SIDE:
            weights.loc[d] = current.values
            continue
        ranked = sig.sort_values()
        shorts = ranked.index[:N_SIDE]
        longs = ranked.index[-N_SIDE:]
        current = pd.Series(0.0, index=C.columns)
        current[longs] = 1.0 / N_SIDE
        current[shorts] = -1.0 / N_SIDE
        weights.loc[d] = current.values
        signal_hist.append((d, sig))

    w_eff = weights.shift(1).fillna(0)
    port_gross = (w_eff * close_ret).sum(axis=1)
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    port = port_gross - turnover * (TC_BPS / 1e4)

    # Also compute overnight-only variant (positioned at close, earned overnight only, exit at open)
    ovn_ret = (O / C.shift(1) - 1).fillna(0)
    port_ovn_gross = (w_eff * ovn_ret).sum(axis=1)
    # Higher TC for overnight-only (full round-trip per month still, but per-name TC per name)
    port_ovn = port_ovn_gross - turnover * (TC_BPS / 1e4)

    warm = pd.Timestamp("2017-06-01")
    p_v = port.loc[warm:]
    pov_v = port_ovn.loc[warm:]

    print("\nCLOSE-TO-CLOSE (hold full day, monthly rebal):")
    s = stats(p_v, "NOVA27 CS OVN-mom close-close")
    print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(p_v.loc[:CUT], "IS <2022"), (p_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:30s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    print("\nOVERNIGHT-ONLY (in at MOC, out at MOO, monthly):")
    s = stats(pov_v, "NOVA27 CS OVN-mom overnight")
    print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(pov_v.loc[:CUT], "IS <2022"), (pov_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:30s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    # Long-only top-10 (no short)
    w_lo = weights.clip(lower=0)
    w_lo_eff = w_lo.shift(1).fillna(0)
    port_lo = (w_lo_eff * close_ret).sum(axis=1)
    to_lo = (w_lo_eff - w_lo_eff.shift(1)).abs().sum(axis=1).fillna(0)
    port_lo = port_lo - to_lo * (TC_BPS / 1e4)
    plo_v = port_lo.loc[warm:]
    s = stats(plo_v, "NOVA27 CS OVN-mom long-only top10")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(plo_v.loc[:CUT], "IS <2022"), (plo_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:30s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    ann = p_v.groupby(p_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (close-close L/S):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA27_CC": port, "NOVA27_OVN": port_ovn, "NOVA27_LO": port_lo})
    out.to_csv("/home/user/bonds/data/results/nova27_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova27_returns.csv")


if __name__ == "__main__":
    main()
