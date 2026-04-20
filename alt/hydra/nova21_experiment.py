"""NOVA21 — Residual momentum (Blitz-Huij-Martens 2011), monthly rebal.

Key finding: residual momentum (12-1 momentum on PCA-residualized returns)
delivers ~2x the Sharpe of raw cross-sectional momentum, with dramatically
reduced momentum-crash risk. Published OOS SR 1.0-1.5 on US large-caps
1930-2009.

Method (FIXED a priori):
  At each month-start:
    1. Pool trailing 252 days of returns across 96 stocks.
    2. Fit PCA with K=10 factors (capture market+sector).
    3. Compute residual per stock: ε = r - Σ β_k f_k
    4. Residual 12-1 momentum: sum of daily residuals from t-252 to t-21
       (exclude most recent month to avoid short-term reversal).
    5. Cross-sectional rank. Long top decile (10 stocks), short bottom
       decile (10 stocks). Equal-weight dollar-neutral.
    6. Monthly rebalance, 1-day execution lag, 15 bps TC per rebalance.

Why this should break through the 1.0 OOS ceiling:
  - Market/sector-neutral: immune to beta/factor risk that hurt NOVA12
  - Avoids the Dec-2008/Apr-2009 and Apr-2020 momentum crashes (β loading
    is the main driver of crashes)
  - Low turnover (monthly): TC manageable"""
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
SKIP = 21       # exclude last month
K_FACTORS = 10
N_SIDE = 10
TC_BPS = 15.0


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
    tickers = list(rets.columns)
    print(f"Universe: {len(tickers)} stocks, {len(dates)} days")

    first = monthly_first(dates)
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    current = pd.Series(0.0, index=tickers)

    nm = 0
    for i, d in enumerate(dates):
        if not first.iloc[i] or i < LOOKBACK:
            weights.loc[d] = current.values
            continue
        window = rets.iloc[i - LOOKBACK:i]
        W = window.dropna(axis=1, how="any")
        if W.shape[1] < 20:
            weights.loc[d] = current.values
            continue
        Wc = W - W.mean()
        U, S, Vt = np.linalg.svd(Wc.values, full_matrices=False)
        K = min(K_FACTORS, len(S))
        F = U[:, :K] * S[:K]
        # Residuals: ε = W_c - F @ (F^+ W_c)
        # Least-squares coefficient per stock, residuals collected
        beta, *_ = np.linalg.lstsq(F, Wc.values, rcond=None)
        eps = Wc.values - F @ beta     # (LOOKBACK, Nstocks)
        # 12-1 residual momentum = sum of residuals from t-LOOKBACK to t-SKIP
        sig = eps[:-SKIP, :].sum(axis=0)   # (Nstocks,)
        sig_s = pd.Series(sig, index=W.columns)
        ranked = sig_s.sort_values()
        shorts = ranked.index[:N_SIDE]
        longs = ranked.index[-N_SIDE:]
        current = pd.Series(0.0, index=tickers)
        current[longs] = 1.0 / N_SIDE
        current[shorts] = -1.0 / N_SIDE
        weights.loc[d] = current.values
        nm += 1
    print(f"Rebalances: {nm}")

    w_eff = weights.shift(1).fillna(0)
    port_gross = (w_eff * rets).sum(axis=1)
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    port = port_gross - turnover * (TC_BPS / 1e4)

    warm = pd.Timestamp("2017-06-01")
    p_v = port.loc[warm:]
    s = stats(p_v, "NOVA21 Residual Mom L/S")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(p_v.loc[:CUT], "IS <2022"), (p_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Long-only variant
    w_lo = weights.clip(lower=0)
    # rescale to sum to 1
    # leave at 0.1 per stock ⇒ 1.0 gross long-only
    w_lo_eff = w_lo.shift(1).fillna(0)
    port_lo = (w_lo_eff * rets).sum(axis=1)
    turnover_lo = (w_lo_eff - w_lo_eff.shift(1)).abs().sum(axis=1).fillna(0)
    port_lo = port_lo - turnover_lo * (TC_BPS / 1e4)
    p_lo = port_lo.loc[warm:]
    s = stats(p_lo, "NOVA21 Residual Mom Long-only")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(p_lo.loc[:CUT], "IS <2022"), (p_lo.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"MDD={ss['mdd']:>7.2f}%")

    ann = p_v.groupby(p_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual (L/S):")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA21": port, "NOVA21_LO": port_lo})
    out.to_csv("/home/user/bonds/data/results/nova21_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova21_returns.csv")


if __name__ == "__main__":
    main()
