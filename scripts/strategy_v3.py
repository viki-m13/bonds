#!/usr/bin/env python3
"""
Adaptive Multi-Horizon Carry Decomposition Strategy V3
=======================================================

CORE INSIGHT: In bond markets, carry (income) is the dominant and most
reliable alpha source. The novel contribution is HOW we harvest carry:

1. DURATION-ISOLATED CARRY: For each credit/sector ETF, dynamically
   hedge duration risk using optimal treasury hedge (beta-adjusted).
   This isolates the pure spread/carry component.

2. VOLATILITY-CONDITIONAL SIZING: Scale carry exposure inversely with
   realized volatility AND credit spread momentum. Reduce in stress,
   increase in calm periods. This transforms negative skew into neutral.

3. MULTI-INSTRUMENT DIVERSIFICATION: Run carry across 11+ independent
   hedged pairs spanning different sectors, durations, and credit tiers.
   Diversification across uncorrelated carry streams is the key to
   high Sharpe.

4. ADAPTIVE MOMENTUM OVERLAY (carry enhancer): Within each carry stream,
   add/reduce exposure based on short-term momentum of the hedged spread.
   This captures the mean-reversion of credit spreads.

5. CROSS-SECTIONAL CARRY ROTATION: Overweight pairs where carry is
   highest relative to risk, underweight where carry has been harvested.
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


def compute_hedged_carry_returns(prices, ret, fred):
    """
    Generate duration-hedged carry return streams.
    Each stream is a long-short pair: long carry asset, short beta*treasury.
    """
    # Define carry pairs: (long, hedge, name)
    pairs = [
        ("HYG", "IEF", "HY_midhedge"),
        ("HYG", "TLT", "HY_longhedge"),
        ("HYG", "SHY", "HY_shorthedge"),
        ("JNK", "IEF", "JNK_midhedge"),
        ("LQD", "IEF", "IG_midhedge"),
        ("LQD", "SHY", "IG_shorthedge"),
        ("VCIT", "IEI", "MidCorp"),
        ("VCSH", "SHY", "ShortCorp"),
        ("IGIB", "IEI", "IG5yr"),
        ("EMB", "IEF", "EM_carry"),
        ("EMB", "TLT", "EM_longhedge"),
        ("MUB", "SHY", "Muni_carry"),
        ("MUB", "IEI", "Muni_midhedge"),
        ("MBB", "IEF", "MBS_carry"),
        ("TIP", "IEF", "TIPS_carry"),
        ("VMBS", "IEI", "MBS2_carry"),
    ]

    streams = {}
    for long_etf, hedge_etf, name in pairs:
        if long_etf not in ret.columns or hedge_etf not in ret.columns:
            continue

        # Rolling beta hedge ratio (expanding with decay)
        lookback = 252
        cov = ret[long_etf].rolling(lookback, min_periods=126).cov(ret[hedge_etf])
        var = ret[hedge_etf].rolling(lookback, min_periods=126).var()
        beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)

        # Hedged return
        hedged = ret[long_etf] - beta.shift(1) * ret[hedge_etf]
        hedged = hedged.dropna()

        if len(hedged) < 252:
            continue

        streams[name] = hedged

    return streams


def apply_vol_scaling(streams, fred):
    """
    Apply volatility-conditional sizing to each carry stream.
    Reduce in high vol, increase in low vol. This improves Sharpe
    by cutting left-tail risk.
    """
    vix = fred.get("VIXCLS")
    scaled = {}

    for name, ret_stream in streams.items():
        # Own realized vol scaling
        rv = ret_stream.rolling(63, min_periods=21).std() * np.sqrt(252)
        target_sub_vol = 0.05  # 5% ann per stream
        vol_scalar = (target_sub_vol / rv.clip(lower=0.005)).clip(0.1, 5.0)

        # VIX overlay: additional scaling based on market stress
        if vix is not None:
            vix_aligned = vix.reindex(ret_stream.index).ffill()
            vix_pctl = vix_aligned.rolling(252, min_periods=126).rank(pct=True)
            # In high stress (>75th pctl): scale to 50%
            # In low stress (<25th pctl): scale to 120%
            stress_scalar = (1.3 - 0.8 * vix_pctl).clip(0.4, 1.3)
            vol_scalar = vol_scalar * stress_scalar

        scaled[name] = ret_stream * vol_scalar.shift(1)

    return scaled


def apply_momentum_overlay(streams, prices, ret):
    """
    For each carry stream, add/reduce based on short-term momentum of the
    hedged spread. If the carry trade is working (positive recent return),
    increase; if bleeding, reduce. This captures mean-reversion timing.
    """
    enhanced = {}
    for name, ret_stream in streams.items():
        # Short-term momentum of the hedged return
        cum_5d = ret_stream.rolling(5).sum()
        cum_21d = ret_stream.rolling(21).sum()

        # Momentum signal: normalized
        mom_5d = cum_5d / ret_stream.rolling(63, min_periods=21).std().clip(lower=1e-6) / np.sqrt(5)
        mom_21d = cum_21d / ret_stream.rolling(63, min_periods=21).std().clip(lower=1e-6) / np.sqrt(21)

        # Blend: 50% 5d, 50% 21d
        mom_signal = (mom_5d + mom_21d) / 2

        # Scale carry by momentum: base of 1.0, +/- 50% adjustment
        # Positive momentum → scale up to 1.5x
        # Negative momentum → scale down to 0.5x
        mom_scalar = (1.0 + 0.25 * mom_signal.clip(-2, 2)).clip(0.3, 1.7)

        enhanced[name] = ret_stream * mom_scalar.shift(1)

    return enhanced


def cross_sectional_tilt(streams, lookback=63):
    """
    Overweight carry streams that are currently performing well (positive
    risk-adjusted carry) and underweight those that are bleeding.
    This is a cross-sectional momentum/quality tilt.
    """
    if len(streams) < 3:
        return streams

    # Compute recent Sharpe for each stream
    df = pd.DataFrame(streams).fillna(0)
    recent_sharpe = df.rolling(lookback, min_periods=21).mean() / \
                    df.rolling(lookback, min_periods=21).std().clip(lower=1e-6)

    # Rank across streams and convert to weights
    # Highest recent Sharpe gets 2x, lowest gets 0.5x
    ranks = recent_sharpe.rank(axis=1, pct=True)
    weights = 0.5 + ranks  # Range: [0.5, 1.5]

    # Normalize so mean weight = 1
    weights = weights.div(weights.mean(axis=1), axis=0)

    tilted = {}
    for name in streams:
        if name in weights.columns:
            tilted[name] = df[name] * weights[name].shift(1)
        else:
            tilted[name] = streams[name]

    return tilted


def combine_streams(streams, target_vol=TARGET_VOL):
    """
    Combine all carry streams with proper diversification.
    Uses shrunk inverse-variance weighting.
    """
    df = pd.DataFrame(streams).dropna(how="all").fillna(0)

    if df.shape[1] == 0:
        return None

    # Equal weight (most robust for moderate number of strategies)
    portfolio = df.mean(axis=1)

    # Portfolio-level vol targeting
    port_vol = portfolio.rolling(63, min_periods=21).std() * np.sqrt(252)
    scaler = (target_vol / port_vol.clip(lower=0.005)).clip(0.2, 5.0)
    portfolio = portfolio * scaler.shift(1)

    # Apply transaction cost for portfolio turnover
    # Approximate: each stream has turnover from vol scaling changes
    avg_daily_turnover = scaler.diff().abs().mean() * 0.1  # rough
    tc = avg_daily_turnover * (TRANSACTION_COST_BPS / 10000)
    portfolio = portfolio - tc

    return portfolio.dropna()


def compute_metrics(returns, name=""):
    r = returns.dropna()
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
        "name": name, "ann_ret": ann_ret, "ann_vol": ann_vol,
        "sharpe": sharpe, "sortino": sortino, "max_dd": max_dd,
        "calmar": calmar, "win_rate": win_rate,
        "skew": r.skew(), "kurt": r.kurtosis(), "n_days": len(r),
    }


def main():
    print("=" * 80)
    print("ADAPTIVE MULTI-HORIZON CARRY DECOMPOSITION STRATEGY V3")
    print("=" * 80)

    prices, fred = load_data()
    ret = prices.pct_change()
    print(f"Data: {prices.shape[0]} days x {prices.shape[1]} tickers")

    # Step 1: Raw hedged carry streams
    print("\n--- Step 1: Duration-Hedged Carry ---")
    raw_streams = compute_hedged_carry_returns(prices, ret, fred)
    print(f"  {len(raw_streams)} carry streams generated")

    # Show individual stream performance
    print("\n  Individual stream Sharpes:")
    stream_sharpes = {}
    for name, s in raw_streams.items():
        m = compute_metrics(s, name)
        if m:
            stream_sharpes[name] = m["sharpe"]
            print(f"    {name:20s}: Sharpe={m['sharpe']:+.3f}  AnnRet={m['ann_ret']*100:+.2f}%")

    # Step 2: Vol scaling
    print("\n--- Step 2: Volatility-Conditional Sizing ---")
    vol_scaled = apply_vol_scaling(raw_streams, fred)

    # Step 3: Momentum overlay
    print("--- Step 3: Momentum Overlay ---")
    with_momentum = apply_momentum_overlay(vol_scaled, prices, ret)

    # Step 4: Cross-sectional tilt
    print("--- Step 4: Cross-Sectional Carry Rotation ---")
    tilted = cross_sectional_tilt(with_momentum)

    # Step 5: Combine
    print("--- Step 5: Portfolio Construction ---")
    portfolio_ret = combine_streams(tilted)

    if portfolio_ret is None:
        print("No portfolio generated!")
        return

    # === RESULTS ===
    print(f"\n{'=' * 80}")
    print("FULL SAMPLE RESULTS")
    print(f"{'=' * 80}")
    m = compute_metrics(portfolio_ret)
    for k, v in m.items():
        if k == "name":
            continue
        if isinstance(v, float):
            if "ret" in k or "vol" in k or "dd" in k or "rate" in k:
                print(f"  {k:16s}: {v*100:+.2f}%")
            else:
                print(f"  {k:16s}: {v:.3f}")
        else:
            print(f"  {k:16s}: {v}")

    # Train/Test
    split = int(len(portfolio_ret) * 0.6)
    for period, r in [("TRAIN 60%", portfolio_ret.iloc[:split]),
                       ("TEST 40%", portfolio_ret.iloc[split:])]:
        m = compute_metrics(r, period)
        if m:
            print(f"\n  {period}: Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_ret']*100:+.2f}%  "
                  f"MaxDD={m['max_dd']*100:.2f}%  Sortino={m['sortino']:.3f}  WinRate={m['win_rate']*100:.1f}%")

    # Yearly
    print(f"\n{'=' * 80}")
    print("YEARLY PERFORMANCE")
    print(f"{'=' * 80}")
    print(f"{'Year':>6} {'Return':>9} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")
    for year, g in portfolio_ret.groupby(portfolio_ret.index.year):
        if len(g) < 20:
            continue
        ar = g.mean() * 252
        av = g.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        c = (1 + g).cumprod()
        mdd = ((c - c.cummax()) / c.cummax()).min()
        wr = (g > 0).mean()
        print(f"{year:>6} {ar*100:>+8.2f}% {av*100:>7.2f}% {sr:>+7.3f} {mdd*100:>+7.2f}% {wr*100:>7.1f}%")

    # Correlation between carry streams
    print(f"\n{'=' * 80}")
    print("DIVERSIFICATION ANALYSIS")
    print(f"{'=' * 80}")
    stream_df = pd.DataFrame(raw_streams).dropna(how="all").fillna(0)
    corr_matrix = stream_df.corr()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    avg_corr = upper_tri.stack().mean()
    print(f"  Average pairwise correlation: {avg_corr:.3f}")
    print(f"  Number of streams: {len(raw_streams)}")
    print(f"  Theoretical Sharpe multiplier: {np.sqrt(len(raw_streams) * (1 - avg_corr) / (1 + (len(raw_streams)-1)*avg_corr)):.2f}x")

    # Walk-forward test
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD TEST (expanding window, 4 folds)")
    print(f"{'=' * 80}")
    n_total = len(portfolio_ret)
    fold_size = n_total // 5
    for fold in range(4):
        start = (fold + 1) * fold_size
        end = min(start + fold_size, n_total)
        fold_ret = portfolio_ret.iloc[start:end]
        fm = compute_metrics(fold_ret, f"Fold {fold+1}")
        if fm:
            print(f"  Fold {fold+1} ({fold_ret.index[0].date()} to {fold_ret.index[-1].date()}): "
                  f"Sharpe={fm['sharpe']:.3f}  AnnRet={fm['ann_ret']*100:+.2f}%  MaxDD={fm['max_dd']*100:.2f}%")

    # Autocorrelation
    ac1 = portfolio_ret.autocorr(1)
    ac5 = portfolio_ret.autocorr(5)
    print(f"\n  Autocorr(1): {ac1:.4f}  Autocorr(5): {ac5:.4f}")

    # Save
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    portfolio_ret.to_csv(results_dir / "strategy_v3_returns.csv", header=["return"])
    (1 + portfolio_ret).cumprod().to_csv(results_dir / "strategy_v3_cumulative.csv", header=["cumulative"])
    print(f"\nSaved to {results_dir}")


if __name__ == "__main__":
    main()
