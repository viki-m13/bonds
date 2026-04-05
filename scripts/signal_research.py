#!/usr/bin/env python3
"""
Signal Research for Bond ETFs
=============================
Tests individual trading signals on bond ETF data with proper backtesting
infrastructure (no forward bias, transaction costs, train/test split).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
TREASURY_PATH = DATA_DIR / "treasury" / "daily_treasury_yields.csv"

# Key tickers
KEY_TICKERS = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "AGG", "BND",
    "TIP", "EMB", "MUB", "VCIT", "VCSH", "MBB", "FLOT", "VGLT", "SPTL", "GOVT",
]

TRANSACTION_COST_BPS = 5
TRAIN_FRAC = 0.60

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_etf(ticker: str) -> pd.DataFrame:
    """Load ETF CSV and return DataFrame with Date index and Adj Close."""
    path = ETF_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    df = df.set_index("Date")
    # Use Close as the adjusted close (the CSVs have Close as the adj field)
    df = df.rename(columns={"Close": "AdjClose"})
    df = df[~df.index.duplicated(keep="first")]
    return df


def load_all_etfs() -> pd.DataFrame:
    """Load adjusted close for all key tickers into a single DataFrame."""
    prices = {}
    for t in KEY_TICKERS:
        df = load_etf(t)
        if not df.empty and "AdjClose" in df.columns:
            prices[t] = df["AdjClose"]
    return pd.DataFrame(prices).sort_index().dropna(how="all")


def load_fred() -> pd.DataFrame:
    df = pd.read_csv(FRED_PATH, parse_dates=["Date"]).sort_values("Date")
    df = df.set_index("Date")
    df = df[~df.index.duplicated(keep="first")]
    # Convert all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_treasury_yields() -> pd.DataFrame:
    df = pd.read_csv(TREASURY_PATH, parse_dates=["Date"]).sort_values("Date")
    df = df.set_index("Date")
    df = df[~df.index.duplicated(keep="first")]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

def compute_metrics(returns: pd.Series, name: str = "") -> dict:
    """Compute performance metrics from a return series (daily)."""
    if returns.empty or returns.std() == 0:
        return {
            "Signal": name, "Sharpe": 0.0, "MaxDD": 0.0, "WinRate": 0.0,
            "AvgTradeRet": 0.0, "NumTrades": 0, "Calmar": 0.0,
            "AnnRet": 0.0, "AnnVol": 0.0,
        }
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win_rate = (returns > 0).mean()
    # Count trades as position changes (approximate from signal changes)
    avg_trade_ret = returns[returns != 0].mean() if (returns != 0).any() else 0.0
    num_trades = int((returns != 0).sum())
    return {
        "Signal": name,
        "AnnRet": round(ann_ret * 100, 2),
        "AnnVol": round(ann_vol * 100, 2),
        "Sharpe": round(sharpe, 3),
        "MaxDD": round(max_dd * 100, 2),
        "WinRate": round(win_rate * 100, 1),
        "AvgTradeRet": round(avg_trade_ret * 10000, 2),  # in bps
        "NumTrades": num_trades,
        "Calmar": round(calmar, 3),
    }


def backtest_signal(
    positions: pd.Series,
    prices: pd.Series | pd.DataFrame,
    cost_bps: float = TRANSACTION_COST_BPS,
    train_frac: float = TRAIN_FRAC,
    signal_name: str = "",
) -> dict:
    """
    Run a backtest given a position series and a price series.

    positions: Series of target weights (e.g., -1, 0, +1) aligned to dates.
               Signal on date t uses only data up to t, trade is executed at
               close of t, return realized from t to t+1.
    prices:    Adjusted close prices (Series or single-column DataFrame).

    Returns dict with train and test metrics.
    """
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] == 1:
            prices = prices.iloc[:, 0]
        else:
            raise ValueError("prices must be a single Series for simple backtest")

    # Align
    common = positions.dropna().index.intersection(prices.dropna().index)
    common = common.sort_values()
    pos = positions.loc[common]
    px = prices.loc[common]

    # Daily returns of the underlying
    asset_ret = px.pct_change()

    # Strategy return: position at t drives return from t to t+1
    # Shift position by 1 to avoid look-ahead
    strat_ret = pos.shift(1) * asset_ret

    # Transaction costs on position changes
    turnover = pos.diff().abs()
    tc = turnover * (cost_bps / 10000)
    strat_ret = strat_ret - tc

    strat_ret = strat_ret.dropna()

    if len(strat_ret) < 60:
        empty = compute_metrics(pd.Series(dtype=float), signal_name)
        return {"train": empty, "test": empty}

    split_idx = int(len(strat_ret) * train_frac)
    train_ret = strat_ret.iloc[:split_idx]
    test_ret = strat_ret.iloc[split_idx:]

    train_m = compute_metrics(train_ret, f"{signal_name} [TRAIN]")
    test_m = compute_metrics(test_ret, f"{signal_name} [TEST]")
    return {"train": train_m, "test": test_m}


def backtest_portfolio_signal(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = TRANSACTION_COST_BPS,
    train_frac: float = TRAIN_FRAC,
    signal_name: str = "",
) -> dict:
    """
    Backtest a portfolio of positions (weights DataFrame: dates x tickers).
    """
    # Align
    common_dates = weights.dropna(how="all").index.intersection(
        prices.dropna(how="all").index
    ).sort_values()
    tickers = [c for c in weights.columns if c in prices.columns]
    if not tickers or len(common_dates) < 60:
        empty = compute_metrics(pd.Series(dtype=float), signal_name)
        return {"train": empty, "test": empty}

    w = weights.loc[common_dates, tickers].fillna(0)
    px = prices.loc[common_dates, tickers]
    asset_ret = px.pct_change()

    # Strategy return: w at t drives return t to t+1
    strat_ret = (w.shift(1) * asset_ret).sum(axis=1)

    # Transaction costs
    turnover = w.diff().abs().sum(axis=1)
    tc = turnover * (cost_bps / 10000)
    strat_ret = strat_ret - tc
    strat_ret = strat_ret.dropna()

    if len(strat_ret) < 60:
        empty = compute_metrics(pd.Series(dtype=float), signal_name)
        return {"train": empty, "test": empty}

    split_idx = int(len(strat_ret) * train_frac)
    train_ret = strat_ret.iloc[:split_idx]
    test_ret = strat_ret.iloc[split_idx:]

    return {
        "train": compute_metrics(train_ret, f"{signal_name} [TRAIN]"),
        "test": compute_metrics(test_ret, f"{signal_name} [TEST]"),
    }


# ===========================================================================
# SIGNAL DEFINITIONS
# ===========================================================================

def signal_yield_curve_slope_momentum(prices: pd.DataFrame, fred: pd.DataFrame) -> dict:
    """
    Signal (a): Go long duration when yield curve is steepening (10Y-2Y spread
    increasing over 21 days), short when flattening.
    Trade TLT and IEF.
    """
    spread = fred["T10Y2Y"].dropna()
    slope_change = spread.diff(21)  # 21-day change in slope

    # Position: +1 if steepening, -1 if flattening
    pos = slope_change.apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0))

    results = {}
    for ticker in ["TLT", "IEF"]:
        if ticker in prices.columns:
            r = backtest_signal(pos, prices[ticker], signal_name=f"YC_Slope_Mom_{ticker}")
            results[f"YC_Slope_Mom_{ticker}"] = r
    return results


def signal_credit_spread_mean_reversion(prices: pd.DataFrame, fred: pd.DataFrame) -> dict:
    """
    Signal (b): When HY OAS z-score (63-day) > +2, go long HYG/JNK.
    When z-score < -1, short. Otherwise flat.
    """
    hy_oas = fred["BAMLH0A0HYM2"].dropna()
    rolling_mean = hy_oas.rolling(63, min_periods=50).mean()
    rolling_std = hy_oas.rolling(63, min_periods=50).std()
    z = (hy_oas - rolling_mean) / rolling_std

    # When spreads are extremely wide (z > 2), go long (expect compression)
    # When spreads tight (z < -1), go short (expect widening)
    pos = pd.Series(0.0, index=z.index)
    pos[z > 2] = 1.0
    pos[z < -1] = -1.0

    results = {}
    for ticker in ["HYG", "JNK"]:
        if ticker in prices.columns:
            r = backtest_signal(pos, prices[ticker], signal_name=f"Credit_MR_{ticker}")
            results[f"Credit_MR_{ticker}"] = r
    return results


def signal_cross_sectional_momentum(prices: pd.DataFrame) -> dict:
    """
    Signal (c): Rank all ETFs by 63-day return. Long top quintile, short
    bottom quintile. Rebalance weekly.
    """
    ret63 = prices.pct_change(63)
    n_tickers = ret63.shape[1]
    quintile_size = max(1, n_tickers // 5)

    weights = pd.DataFrame(0.0, index=ret63.index, columns=ret63.columns)

    # Rebalance weekly (every 5 trading days)
    rebal_dates = ret63.dropna(how="all").index[::5]

    last_w = pd.Series(0.0, index=ret63.columns)
    for dt in ret63.index:
        if dt in rebal_dates:
            row = ret63.loc[dt].dropna()
            if len(row) < 5:
                last_w = pd.Series(0.0, index=ret63.columns)
            else:
                ranks = row.rank(ascending=True)
                n = len(row)
                qs = max(1, n // 5)
                w = pd.Series(0.0, index=ret63.columns)
                longs = ranks[ranks > n - qs].index
                shorts = ranks[ranks <= qs].index
                if len(longs) > 0:
                    w[longs] = 1.0 / len(longs)
                if len(shorts) > 0:
                    w[shorts] = -1.0 / len(shorts)
                last_w = w
        weights.loc[dt] = last_w

    return {
        "XS_Momentum": backtest_portfolio_signal(
            weights, prices, signal_name="XS_Momentum"
        )
    }


def signal_volatility_regime(prices: pd.DataFrame, fred: pd.DataFrame) -> dict:
    """
    Signal (d): VIX above 63-day 75th pctl -> shift to short-duration/quality
    (SHY, VCSH). VIX below 25th pctl -> long-duration/credit (TLT, HYG, EMB).
    Otherwise neutral (equal weight all).
    """
    vix = fred["VIXCLS"].dropna()
    vix_p75 = vix.rolling(63, min_periods=50).quantile(0.75)
    vix_p25 = vix.rolling(63, min_periods=50).quantile(0.25)

    risk_off = ["SHY", "VCSH"]
    risk_on = ["TLT", "HYG", "EMB"]
    all_t = list(set(risk_off + risk_on))
    available = [t for t in all_t if t in prices.columns]

    weights = pd.DataFrame(0.0, index=vix.index, columns=available)

    for dt in vix.index:
        if dt not in vix_p75.index or pd.isna(vix_p75.get(dt)) or pd.isna(vix.get(dt)):
            continue
        v = vix[dt]
        p75 = vix_p75[dt]
        p25 = vix_p25[dt]
        if v > p75:
            # Risk-off: short duration / quality
            tgt = [t for t in risk_off if t in available]
            if tgt:
                for t in tgt:
                    weights.loc[dt, t] = 1.0 / len(tgt)
        elif v < p25:
            # Risk-on: long duration / credit
            tgt = [t for t in risk_on if t in available]
            if tgt:
                for t in tgt:
                    weights.loc[dt, t] = 1.0 / len(tgt)
        else:
            # Neutral
            if available:
                for t in available:
                    weights.loc[dt, t] = 1.0 / len(available)

    return {
        "Vol_Regime": backtest_portfolio_signal(
            weights, prices, signal_name="Vol_Regime"
        )
    }


def signal_carry_momentum_combo(prices: pd.DataFrame) -> dict:
    """
    Signal (e): Approximate carry ~ trailing 12-month total return minus
    price return (here use trailing 12m return as proxy since we only have
    price data). Combine carry rank with 6-month momentum rank.
    Long top tercile of combined rank, short bottom tercile.
    """
    ret_12m = prices.pct_change(252)
    ret_6m = prices.pct_change(126)

    # Carry proxy: we use the difference between 12m return and 6m return
    # as a rough approximation of income/carry component
    carry_proxy = ret_12m - ret_6m

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    # Rebalance monthly (~21 days)
    valid_idx = ret_12m.dropna(how="all").index
    rebal_dates = valid_idx[::21]

    last_w = pd.Series(0.0, index=prices.columns)
    for dt in prices.index:
        if dt in rebal_dates:
            carry_row = carry_proxy.loc[dt].dropna() if dt in carry_proxy.index else pd.Series(dtype=float)
            mom_row = ret_6m.loc[dt].dropna() if dt in ret_6m.index else pd.Series(dtype=float)
            common = carry_row.index.intersection(mom_row.index)
            if len(common) < 5:
                last_w = pd.Series(0.0, index=prices.columns)
            else:
                carry_rank = carry_row[common].rank(pct=True)
                mom_rank = mom_row[common].rank(pct=True)
                combo = (carry_rank + mom_rank) / 2.0
                tercile = max(1, len(common) // 3)
                overall_rank = combo.rank(ascending=True)
                n = len(common)
                w = pd.Series(0.0, index=prices.columns)
                longs = overall_rank[overall_rank > n - tercile].index
                shorts = overall_rank[overall_rank <= tercile].index
                if len(longs) > 0:
                    w[longs] = 1.0 / len(longs)
                if len(shorts) > 0:
                    w[shorts] = -1.0 / len(shorts)
                last_w = w
        weights.loc[dt] = last_w

    return {
        "Carry_Mom": backtest_portfolio_signal(
            weights, prices, signal_name="Carry_Mom"
        )
    }


def signal_term_structure_curvature(prices: pd.DataFrame, fred: pd.DataFrame) -> dict:
    """
    Signal (f): Curvature = 2*5Y - 2Y - 10Y.
    When curvature mean-reverts from extremes, trade the belly (IEF) vs
    wings (SHY + TLT).

    High curvature (z > 1.5) -> belly cheap -> long IEF, short 0.5*SHY + 0.5*TLT
    Low curvature (z < -1.5) -> belly rich -> short IEF, long 0.5*SHY + 0.5*TLT
    """
    dgs5 = fred["DGS5"].dropna()
    dgs2 = fred["DGS2"].dropna()
    dgs10 = fred["DGS10"].dropna()

    curvature = 2 * dgs5 - dgs2 - dgs10
    curvature = curvature.dropna()

    roll_mean = curvature.rolling(63, min_periods=50).mean()
    roll_std = curvature.rolling(63, min_periods=50).std()
    z = (curvature - roll_mean) / roll_std

    tickers_needed = ["IEF", "SHY", "TLT"]
    available = [t for t in tickers_needed if t in prices.columns]
    if len(available) < 3:
        empty = compute_metrics(pd.Series(dtype=float), "Curvature")
        return {"Curvature": {"train": empty, "test": empty}}

    weights = pd.DataFrame(0.0, index=z.index, columns=tickers_needed)

    for dt in z.index:
        zval = z.get(dt, np.nan)
        if pd.isna(zval):
            continue
        if zval > 1.5:
            # Belly cheap -> long IEF, short wings
            weights.loc[dt, "IEF"] = 1.0
            weights.loc[dt, "SHY"] = -0.5
            weights.loc[dt, "TLT"] = -0.5
        elif zval < -1.5:
            # Belly rich -> short IEF, long wings
            weights.loc[dt, "IEF"] = -1.0
            weights.loc[dt, "SHY"] = 0.5
            weights.loc[dt, "TLT"] = 0.5
        # else: flat (0)

    return {
        "Curvature": backtest_portfolio_signal(
            weights, prices, signal_name="Curvature"
        )
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("=" * 80)
    print("BOND ETF SIGNAL RESEARCH")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    prices = load_all_etfs()
    fred = load_fred()
    print(f"  ETF prices: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  FRED data: {fred.shape[0]} rows x {fred.shape[1]} columns")
    print(f"  Available tickers: {list(prices.columns)}")

    # Run all signals
    all_results = {}

    print("\n--- Signal (a): Yield Curve Slope Momentum ---")
    res = signal_yield_curve_slope_momentum(prices, fred)
    all_results.update(res)
    for k, v in res.items():
        print(f"\n  {k}:")
        for period in ["train", "test"]:
            m = v[period]
            print(f"    {period.upper():5s}: Sharpe={m['Sharpe']:+.3f}  AnnRet={m['AnnRet']:+.2f}%  "
                  f"MaxDD={m['MaxDD']:.2f}%  WinRate={m['WinRate']:.1f}%  "
                  f"Calmar={m['Calmar']:.3f}  #Trades={m['NumTrades']}")

    print("\n--- Signal (b): Credit Spread Mean Reversion ---")
    res = signal_credit_spread_mean_reversion(prices, fred)
    all_results.update(res)
    for k, v in res.items():
        print(f"\n  {k}:")
        for period in ["train", "test"]:
            m = v[period]
            print(f"    {period.upper():5s}: Sharpe={m['Sharpe']:+.3f}  AnnRet={m['AnnRet']:+.2f}%  "
                  f"MaxDD={m['MaxDD']:.2f}%  WinRate={m['WinRate']:.1f}%  "
                  f"Calmar={m['Calmar']:.3f}  #Trades={m['NumTrades']}")

    print("\n--- Signal (c): Cross-Sectional Momentum ---")
    res = signal_cross_sectional_momentum(prices)
    all_results.update(res)
    for k, v in res.items():
        print(f"\n  {k}:")
        for period in ["train", "test"]:
            m = v[period]
            print(f"    {period.upper():5s}: Sharpe={m['Sharpe']:+.3f}  AnnRet={m['AnnRet']:+.2f}%  "
                  f"MaxDD={m['MaxDD']:.2f}%  WinRate={m['WinRate']:.1f}%  "
                  f"Calmar={m['Calmar']:.3f}  #Trades={m['NumTrades']}")

    print("\n--- Signal (d): Volatility Regime ---")
    res = signal_volatility_regime(prices, fred)
    all_results.update(res)
    for k, v in res.items():
        print(f"\n  {k}:")
        for period in ["train", "test"]:
            m = v[period]
            print(f"    {period.upper():5s}: Sharpe={m['Sharpe']:+.3f}  AnnRet={m['AnnRet']:+.2f}%  "
                  f"MaxDD={m['MaxDD']:.2f}%  WinRate={m['WinRate']:.1f}%  "
                  f"Calmar={m['Calmar']:.3f}  #Trades={m['NumTrades']}")

    print("\n--- Signal (e): Carry + Momentum Combo ---")
    res = signal_carry_momentum_combo(prices)
    all_results.update(res)
    for k, v in res.items():
        print(f"\n  {k}:")
        for period in ["train", "test"]:
            m = v[period]
            print(f"    {period.upper():5s}: Sharpe={m['Sharpe']:+.3f}  AnnRet={m['AnnRet']:+.2f}%  "
                  f"MaxDD={m['MaxDD']:.2f}%  WinRate={m['WinRate']:.1f}%  "
                  f"Calmar={m['Calmar']:.3f}  #Trades={m['NumTrades']}")

    print("\n--- Signal (f): Term Structure Curvature ---")
    res = signal_term_structure_curvature(prices, fred)
    all_results.update(res)
    for k, v in res.items():
        print(f"\n  {k}:")
        for period in ["train", "test"]:
            m = v[period]
            print(f"    {period.upper():5s}: Sharpe={m['Sharpe']:+.3f}  AnnRet={m['AnnRet']:+.2f}%  "
                  f"MaxDD={m['MaxDD']:.2f}%  WinRate={m['WinRate']:.1f}%  "
                  f"Calmar={m['Calmar']:.3f}  #Trades={m['NumTrades']}")

    # -----------------------------------------------------------------------
    # Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - ALL SIGNALS")
    print("=" * 80)

    rows = []
    for sig_name, res in all_results.items():
        for period in ["train", "test"]:
            m = res[period]
            rows.append({
                "Signal": sig_name,
                "Period": period.upper(),
                "AnnRet%": m["AnnRet"],
                "AnnVol%": m["AnnVol"],
                "Sharpe": m["Sharpe"],
                "MaxDD%": m["MaxDD"],
                "WinRate%": m["WinRate"],
                "AvgRet(bps)": m["AvgTradeRet"],
                "Calmar": m["Calmar"],
                "#Days": m["NumTrades"],
            })

    summary = pd.DataFrame(rows)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:+.3f}" if isinstance(x, float) else str(x))
    print(summary.to_string(index=False))

    # Rank by test Sharpe
    print("\n" + "=" * 80)
    print("RANKING BY TEST SHARPE RATIO")
    print("=" * 80)
    test_only = summary[summary["Period"] == "TEST"].sort_values("Sharpe", ascending=False)
    print(test_only[["Signal", "AnnRet%", "Sharpe", "MaxDD%", "Calmar"]].to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
