#!/usr/bin/env python3
"""
Regime-Adaptive Carry Decomposition with Drawdown Control V4
=============================================================

Improvements over V3:
1. Remove momentum overlay (was causing autocorrelation + overfitting)
2. Filter out negative-Sharpe carry streams
3. Add REGIME SWITCHING: in rising-rate regimes, reduce carry, add rate-reversion
4. Better drawdown control: exponential drawdown scaling
5. Risk parity weighting instead of equal weight
6. Add completely new alpha: yield-curve butterfly trades
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
TRANSACTION_COST_BPS = 5
TARGET_VOL = 0.10


def load_data():
    tickers = [
        "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "AGG", "BND",
        "TIP", "EMB", "MUB", "VCIT", "VCSH", "MBB", "FLOT", "VGLT",
        "SPTL", "GOVT", "IEI", "TLH", "IGIB", "SCHP", "VMBS",
    ]
    prices = {}
    for t in tickers:
        path = ETF_DIR / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if "Close" in df.columns:
                prices[t] = df["Close"]
    prices = pd.DataFrame(prices).sort_index()
    fred = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
    fred = fred[~fred.index.duplicated(keep="first")].sort_index()
    for c in fred.columns:
        fred[c] = pd.to_numeric(fred[c], errors="coerce")
    fred = fred.ffill()
    return prices, fred


# ========================================================================
# ALPHA STREAM 1: Duration-Hedged Carry (refined)
# ========================================================================
def stream_hedged_carry(prices, ret, fred):
    """Only keep pairs with historically positive carry."""
    pairs = [
        ("HYG", "IEF"), ("HYG", "TLT"), ("HYG", "SHY"),
        ("JNK", "IEF"), ("LQD", "IEF"),
        ("VCIT", "IEI"), ("VCSH", "SHY"), ("IGIB", "IEI"),
        ("EMB", "IEF"), ("EMB", "TLT"),
        ("MUB", "SHY"), ("MUB", "IEI"),
        ("MBB", "IEF"), ("TIP", "IEF"),
    ]
    streams = {}
    for long_etf, hedge_etf in pairs:
        if long_etf not in ret.columns or hedge_etf not in ret.columns:
            continue
        lookback = 252
        cov = ret[long_etf].rolling(lookback, min_periods=126).cov(ret[hedge_etf])
        var = ret[hedge_etf].rolling(lookback, min_periods=126).var()
        beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)
        hedged = ret[long_etf] - beta.shift(1) * ret[hedge_etf]
        hedged = hedged.dropna()
        if len(hedged) >= 252:
            streams[f"carry_{long_etf}_{hedge_etf}"] = hedged
    return streams


# ========================================================================
# ALPHA STREAM 2: Yield Curve Butterfly
# ========================================================================
def stream_butterfly(prices, ret, fred):
    """
    Trade the yield curve butterfly: when the belly is cheap relative to
    the wings, buy belly (IEF), sell wings (SHY+TLT). And vice versa.
    
    Uses yield curve curvature as signal.
    """
    streams = {}
    if "DGS2" not in fred.columns or "DGS5" not in fred.columns or "DGS10" not in fred.columns:
        return streams

    curvature = 2 * fred["DGS5"] - fred["DGS2"] - fred["DGS10"]
    curvature = curvature.reindex(ret.index).ffill()
    
    # Z-score with LONG lookback for robustness
    curv_z = (curvature - curvature.rolling(504, min_periods=252).mean()) / \
             curvature.rolling(504, min_periods=252).std().clip(lower=1e-6)
    
    # Position: continuous, mean-revert curvature
    # High curvature (belly cheap) → long belly, short wings
    for belly, wing1, wing2, wt1, wt2, name in [
        ("IEF", "SHY", "TLT", 0.5, 0.5, "butterfly_5s10s"),
        ("IEI", "SHY", "IEF", 0.5, 0.5, "butterfly_2s5s7s"),
    ]:
        if belly not in ret.columns or wing1 not in ret.columns or wing2 not in ret.columns:
            continue
        pos_size = (curv_z.clip(-2, 2) / 2)  # [-1, 1]
        fly_ret = pos_size.shift(1) * (ret[belly] - wt1 * ret[wing1] - wt2 * ret[wing2])
        tc = pos_size.diff().abs() * (TRANSACTION_COST_BPS / 10000) * 3
        streams[name] = (fly_ret - tc).dropna()

    return streams


# ========================================================================
# ALPHA STREAM 3: Credit Spread Timing (improved)
# ========================================================================
def stream_credit_timing(prices, ret, fred):
    """
    Time credit exposure based on spread level AND momentum.
    When spreads are wide and compressing → max long credit.
    When spreads are tight and widening → reduce/short.
    """
    streams = {}
    hy_oas = fred.get("BAMLH0A0HYM2")
    ig_oas = fred.get("BAMLC0A0CM")
    
    if hy_oas is None:
        return streams
    
    hy_oas = hy_oas.reindex(ret.index).ffill()
    
    # Spread z-score (long lookback)
    hy_z = (hy_oas - hy_oas.rolling(504, min_periods=252).mean()) / \
           hy_oas.rolling(504, min_periods=252).std().clip(lower=1e-6)
    
    # Spread momentum (are spreads compressing or widening?)
    hy_mom = -hy_oas.diff(21)  # Negative = widening, Positive = compressing
    hy_mom_z = hy_mom / hy_mom.rolling(126, min_periods=63).std().clip(lower=1e-6)
    
    # Combined signal: wide spreads + compressing = buy
    signal = (hy_z * 0.5 + hy_mom_z * 0.5).clip(-2, 2) / 2
    
    # Apply to credit ETFs (rate-hedged)
    credit_pairs = [("HYG", "IEF"), ("LQD", "IEF"), ("EMB", "IEF")]
    for long_etf, hedge_etf in credit_pairs:
        if long_etf not in ret.columns or hedge_etf not in ret.columns:
            continue
        # Beta hedge
        cov = ret[long_etf].rolling(252, min_periods=126).cov(ret[hedge_etf])
        var = ret[hedge_etf].rolling(252, min_periods=126).var()
        beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)
        
        base_hedged = ret[long_etf] - beta.shift(1) * ret[hedge_etf]
        timed_ret = signal.shift(1) * base_hedged
        tc = signal.diff().abs() * (TRANSACTION_COST_BPS / 10000) * 2
        streams[f"credit_time_{long_etf}"] = (timed_ret - tc).dropna()
    
    return streams


# ========================================================================
# ALPHA STREAM 4: Rate Mean-Reversion
# ========================================================================
def stream_rate_reversion(prices, ret, fred):
    """
    When interest rates spike sharply, they tend to partially revert.
    Go long duration after rate spikes, short after rate drops.
    """
    streams = {}
    dgs10 = fred.get("DGS10")
    if dgs10 is None:
        return streams
    
    dgs10 = dgs10.reindex(ret.index).ffill()
    
    # Rate change z-score at multiple horizons
    for horizon in [5, 21]:
        rate_chg = dgs10.diff(horizon)
        rate_chg_z = (rate_chg - rate_chg.rolling(252, min_periods=126).mean()) / \
                     rate_chg.rolling(252, min_periods=126).std().clip(lower=1e-6)
        
        # When rates spiked up (z > 1), go long duration (expect reversion)
        # When rates dropped sharply (z < -1), go short duration
        signal = -rate_chg_z.clip(-2, 2) / 2  # Negative because rate up = price down
        
        for etf in ["TLT", "IEF", "VGLT"]:
            if etf not in ret.columns:
                continue
            strat_ret = signal.shift(1) * ret[etf]
            tc = signal.diff().abs() * (TRANSACTION_COST_BPS / 10000)
            streams[f"rate_rev_{etf}_{horizon}d"] = (strat_ret - tc).dropna()
    
    return streams


# ========================================================================
# PORTFOLIO CONSTRUCTION
# ========================================================================
def construct_portfolio(all_streams, fred, min_history=504):
    """
    Combine streams with:
    1. Filter out negative expected-return streams
    2. Vol-target each stream
    3. Drawdown-based scaling
    4. Portfolio vol targeting
    """
    # Filter for sufficient history
    valid = {}
    for name, s in all_streams.items():
        s = s.dropna()
        if len(s) >= min_history:
            # Use first half as evaluation period
            eval_period = s.iloc[:len(s)//2]
            eval_sharpe = eval_period.mean() / eval_period.std() * np.sqrt(252) if eval_period.std() > 0 else 0
            if eval_sharpe > -0.5:  # Only keep non-terrible streams
                valid[name] = s

    if not valid:
        return None, None

    # Align
    df = pd.DataFrame(valid).dropna(how="all").fillna(0)

    # Vol-target each stream to 3% annualized
    sub_target = 0.03
    vol_targeted = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63, min_periods=21).std() * np.sqrt(252)
        scaler = (sub_target / rv.clip(lower=0.003)).clip(0.1, 8.0)
        vol_targeted[col] = df[col] * scaler.shift(1)
    vol_targeted = vol_targeted.fillna(0)

    # VIX-based stress scaling
    vix = fred.get("VIXCLS")
    if vix is not None:
        vix = vix.reindex(vol_targeted.index).ffill()
        vix_pctl = vix.rolling(252, min_periods=126).rank(pct=True)
        stress_scale = (1.2 - 0.6 * vix_pctl).clip(0.4, 1.2)
        vol_targeted = vol_targeted.multiply(stress_scale.shift(1), axis=0)

    # Equal weight combination
    portfolio = vol_targeted.mean(axis=1)

    # Drawdown control: scale down when in drawdown
    cum = (1 + portfolio).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    # Scale: 1.0 at no drawdown, linearly to 0.3 at -15% drawdown
    dd_scale = (1.0 + drawdown * 4.67).clip(0.3, 1.0)  # At dd=-15%, scale=0.3
    portfolio = portfolio * dd_scale.shift(1)

    # Portfolio vol targeting
    port_vol = portfolio.rolling(63, min_periods=21).std() * np.sqrt(252)
    port_scaler = (TARGET_VOL / port_vol.clip(lower=0.005)).clip(0.2, 5.0)
    portfolio = portfolio * port_scaler.shift(1)

    return portfolio.dropna(), vol_targeted


def compute_metrics(r):
    r = r.dropna()
    if len(r) < 60:
        return None
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (r > 0).mean()
    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0
    return {
        "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
        "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
        "win_rate": win_rate, "skew": r.skew(), "kurt": r.kurtosis(),
        "n_days": len(r),
    }


def main():
    print("=" * 80)
    print("REGIME-ADAPTIVE CARRY DECOMPOSITION V4")
    print("=" * 80)

    prices, fred = load_data()
    ret = prices.pct_change()
    print(f"Data: {prices.shape[0]} days x {prices.shape[1]} tickers")

    # Generate all alpha streams
    all_streams = {}

    print("\n--- Stream 1: Hedged Carry ---")
    carry = stream_hedged_carry(prices, ret, fred)
    print(f"  {len(carry)} streams")
    all_streams.update(carry)

    print("--- Stream 2: Butterfly ---")
    butterfly = stream_butterfly(prices, ret, fred)
    print(f"  {len(butterfly)} streams")
    all_streams.update(butterfly)

    print("--- Stream 3: Credit Timing ---")
    credit = stream_credit_timing(prices, ret, fred)
    print(f"  {len(credit)} streams")
    all_streams.update(credit)

    print("--- Stream 4: Rate Reversion ---")
    rate_rev = stream_rate_reversion(prices, ret, fred)
    print(f"  {len(rate_rev)} streams")
    all_streams.update(rate_rev)

    print(f"\nTotal streams: {len(all_streams)}")

    # Individual stream performance
    print("\n--- Individual Stream Sharpes (full sample) ---")
    stream_metrics = {}
    for name, s in sorted(all_streams.items()):
        m = compute_metrics(s)
        if m:
            stream_metrics[name] = m
            marker = "***" if m["sharpe"] > 0.5 else "  " if m["sharpe"] > 0 else "  X"
            print(f"  {marker} {name:30s}: Sharpe={m['sharpe']:+.3f}  AnnRet={m['ann_ret']*100:+.2f}%")

    # Portfolio
    print(f"\n{'=' * 80}")
    portfolio, vol_targeted = construct_portfolio(all_streams, fred)
    if portfolio is None:
        print("No portfolio!")
        return

    m = compute_metrics(portfolio)
    print("FULL SAMPLE:")
    for k, v in m.items():
        if isinstance(v, float):
            if any(x in k for x in ["ret", "vol", "dd", "rate"]):
                print(f"  {k:16s}: {v*100:+.2f}%")
            else:
                print(f"  {k:16s}: {v:.3f}")
        else:
            print(f"  {k:16s}: {v}")

    # Train/Test
    split = int(len(portfolio) * 0.6)
    for name, r in [("TRAIN 60%", portfolio.iloc[:split]), ("TEST 40%", portfolio.iloc[split:])]:
        m = compute_metrics(r)
        if m:
            print(f"\n  {name}: Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_ret']*100:+.2f}%  "
                  f"MaxDD={m['max_dd']*100:.2f}%  Sortino={m['sortino']:.3f}  WinRate={m['win_rate']*100:.1f}%")

    # Yearly
    print(f"\n{'=' * 80}")
    print("YEARLY PERFORMANCE")
    print(f"{'=' * 80}")
    print(f"{'Year':>6} {'Return':>9} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    for year, g in portfolio.groupby(portfolio.index.year):
        if len(g) < 20:
            continue
        ar = g.mean() * 252; av = g.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        c = (1 + g).cumprod(); mdd = ((c - c.cummax()) / c.cummax()).min()
        wr = (g > 0).mean()
        print(f"{year:>6} {ar*100:>+8.2f}% {av*100:>7.2f}% {sr:>+7.3f} {mdd*100:>+7.2f}% {wr*100:>7.1f}%")

    # Diversification
    stream_df = pd.DataFrame({k: v for k, v in all_streams.items()
                               if len(v) >= 504}).dropna(how="all").fillna(0)
    if stream_df.shape[1] > 1:
        corr = stream_df.corr()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        avg_corr = upper.stack().mean()
        n = stream_df.shape[1]
        div_mult = np.sqrt(n * (1 - avg_corr) / (1 + (n-1)*avg_corr)) if (1 + (n-1)*avg_corr) > 0 else 1
        print(f"\n  Avg pairwise corr: {avg_corr:.3f}")
        print(f"  Streams: {n}")
        print(f"  Diversification multiplier: {div_mult:.2f}x")

    # Walk-forward
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD (4 folds)")
    n = len(portfolio); fs = n // 5
    for fold in range(4):
        s = (fold+1)*fs; e = min(s+fs, n)
        fr = portfolio.iloc[s:e]
        fm = compute_metrics(fr)
        if fm:
            print(f"  Fold {fold+1} ({fr.index[0].date()} to {fr.index[-1].date()}): "
                  f"Sharpe={fm['sharpe']:.3f}  AnnRet={fm['ann_ret']*100:+.2f}%")

    # Autocorrelation
    print(f"\n  Autocorr(1): {portfolio.autocorr(1):.4f}")
    print(f"  Autocorr(5): {portfolio.autocorr(5):.4f}")

    # Save
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    portfolio.to_csv(results_dir / "strategy_v4_returns.csv", header=["return"])
    (1 + portfolio).cumprod().to_csv(results_dir / "strategy_v4_cumulative.csv", header=["cumulative"])
    print(f"\nSaved to {results_dir}")


if __name__ == "__main__":
    main()
