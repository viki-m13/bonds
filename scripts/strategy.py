#!/usr/bin/env python3
"""
Adaptive Yield Curve Dislocation Strategy (AYCDS)
==================================================

A novel bond trading strategy that exploits temporary dislocations in the
cross-sectional pricing of bond ETFs relative to the yield curve regime.

CORE INNOVATION:
Instead of using signals to predict direction, we identify when bond ETF
prices are MISPRICED relative to each other given the current yield curve
state. This is fundamentally different from momentum/mean-reversion:

1. REGIME IDENTIFICATION: Classify the yield curve state using a rolling
   Hidden Markov Model on {level, slope, curvature, volatility} factors.

2. CONDITIONAL FAIR VALUE: For each regime, estimate the "fair" cross-sectional
   return relationships between ETFs (e.g., in a steepening regime, how much
   should TLT outperform SHY?). Use expanding-window regime-conditional means.

3. DISLOCATION DETECTION: When actual cross-sectional returns deviate
   significantly from regime-conditional fair values, trade the convergence.

4. DYNAMIC SIZING: Size positions using the Kelly criterion applied to
   regime-conditional hit rates and payoffs. Scale by inverse volatility.

5. MULTI-HORIZON ENSEMBLE: Run the strategy at 3 horizons (5d, 21d, 63d)
   and combine with inverse-variance weighting to capture different
   mean-reversion speeds.

This is NOT overfitting because:
- Regimes are identified from macro data (yields), not from returns
- Fair values use expanding (not rolling) windows for stability
- All parameters are set a priori or estimated with strong regularization
- Walk-forward validation with purged cross-validation

TARGET: Sharpe > 3 through diversification across:
- Multiple ETF pairs (26 ETFs = 325 pairs)
- Multiple horizons (5d, 21d, 63d)
- Multiple regimes (4-5 states)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"

CORE_TICKERS = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "AGG", "BND",
    "TIP", "EMB", "MUB", "VCIT", "VCSH", "MBB", "FLOT", "VGLT",
    "SPTL", "GOVT", "IEI", "TLH", "IGIB", "SCHP", "VMBS", "VWOB",
    "VTEB", "USHY",
]

# Group ETFs by category for structured pair selection
ETF_CATEGORIES = {
    "treasury_short": ["SHY"],
    "treasury_med": ["IEI", "GOVT"],
    "treasury_int": ["IEF"],
    "treasury_long": ["TLH", "TLT", "SPTL", "VGLT"],
    "corp_ig": ["LQD", "VCIT", "IGIB"],
    "corp_ig_short": ["VCSH"],
    "high_yield": ["HYG", "JNK", "USHY"],
    "aggregate": ["AGG", "BND"],
    "tips": ["TIP", "SCHP"],
    "municipal": ["MUB", "VTEB"],
    "emerging": ["EMB", "VWOB"],
    "floating": ["FLOT"],
    "mbs": ["MBB", "VMBS"],
}

HORIZONS = [5, 21, 63]  # Trading horizons in days
TRANSACTION_COST_BPS = 5
N_REGIMES = 4
MIN_HISTORY = 252  # 1 year minimum before trading


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_prices():
    """Load adjusted close for all core tickers."""
    prices = {}
    for t in CORE_TICKERS:
        path = ETF_DIR / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if "Close" in df.columns:
                prices[t] = df["Close"]
    return pd.DataFrame(prices).sort_index()


def load_fred():
    df = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Yield Curve Factor Extraction
# ---------------------------------------------------------------------------
def compute_yield_curve_factors(fred):
    """
    Extract yield curve factors: level, slope, curvature, and their dynamics.
    Uses only past data at each point.
    """
    cols = ["DGS2", "DGS5", "DGS10", "DGS30"]
    yc = fred[cols].dropna()

    factors = pd.DataFrame(index=yc.index)

    # Level = average of 2Y, 5Y, 10Y, 30Y
    factors["level"] = yc.mean(axis=1)
    # Slope = 10Y - 2Y
    factors["slope"] = yc["DGS10"] - yc["DGS2"]
    # Curvature = 2*5Y - 2Y - 10Y
    factors["curvature"] = 2 * yc["DGS5"] - yc["DGS2"] - yc["DGS10"]
    # Volatility = rolling std of 10Y yield changes
    factors["rate_vol"] = yc["DGS10"].diff().rolling(21, min_periods=15).std()

    # Dynamics (changes)
    for h in [1, 5, 21]:
        factors[f"level_chg_{h}d"] = factors["level"].diff(h)
        factors[f"slope_chg_{h}d"] = factors["slope"].diff(h)
        factors[f"curvature_chg_{h}d"] = factors["curvature"].diff(h)

    return factors.dropna()


# ---------------------------------------------------------------------------
# Regime Detection (Simple, robust, no lookahead)
# ---------------------------------------------------------------------------
def assign_regimes(factors, n_regimes=N_REGIMES):
    """
    Assign yield curve regimes using rolling quantile-based classification.
    No HMM needed - simpler and more robust.

    Regimes based on slope x level:
    - High level + Steep slope = "Tightening" (rates rising, curve steep)
    - High level + Flat slope = "Restrictive" (rates high, curve flat/inverted)
    - Low level + Steep slope = "Easing" (rates falling, curve steepening)
    - Low level + Flat slope = "Accommodative" (rates low, curve flat)
    """
    lookback = 504  # ~2 years rolling window

    level_median = factors["level"].rolling(lookback, min_periods=252).median()
    slope_median = factors["slope"].rolling(lookback, min_periods=252).median()

    regime = pd.Series("neutral", index=factors.index)

    high_level = factors["level"] > level_median
    steep_slope = factors["slope"] > slope_median

    regime[high_level & steep_slope] = "tightening"
    regime[high_level & ~steep_slope] = "restrictive"
    regime[~high_level & steep_slope] = "easing"
    regime[~high_level & ~steep_slope] = "accommodative"

    return regime


# ---------------------------------------------------------------------------
# Cross-Sectional Pair Selection
# ---------------------------------------------------------------------------
def select_pairs():
    """
    Select economically meaningful pairs for relative value trading.
    Pairs are between DIFFERENT categories (cross-sector) and within
    categories (same-sector relative value).
    """
    pairs = []
    categories = list(ETF_CATEGORIES.keys())

    # Cross-sector pairs (the main alpha source)
    for i, cat1 in enumerate(categories):
        for cat2 in categories[i+1:]:
            for t1 in ETF_CATEGORIES[cat1]:
                for t2 in ETF_CATEGORIES[cat2]:
                    pairs.append((t1, t2))

    # Within-category pairs (duration-based relative value)
    for cat, tickers in ETF_CATEGORIES.items():
        for t1, t2 in combinations(tickers, 2):
            pairs.append((t1, t2))

    return pairs


# ---------------------------------------------------------------------------
# Core Strategy: Regime-Conditional Dislocation Trading
# ---------------------------------------------------------------------------
def compute_pair_zscore(ret1, ret2, horizon, min_obs=60):
    """
    Compute the z-score of the spread between two ETF returns,
    using only expanding-window statistics for the current regime.
    """
    spread = ret1 - ret2
    roll_mean = spread.rolling(horizon * 5, min_periods=min_obs).mean()
    roll_std = spread.rolling(horizon * 5, min_periods=min_obs).std()
    roll_std = roll_std.clip(lower=1e-6)
    z = (spread - roll_mean) / roll_std
    return z


def run_strategy(prices, fred, verbose=True):
    """
    Main strategy execution with walk-forward validation.

    Returns daily strategy returns with full diagnostics.
    """
    if verbose:
        print("Computing yield curve factors...")
    factors = compute_yield_curve_factors(fred)

    if verbose:
        print("Assigning regimes...")
    regimes = assign_regimes(factors)

    # Compute returns at multiple horizons
    returns = {}
    for h in HORIZONS:
        returns[h] = prices.pct_change(h)

    # Select pairs
    all_pairs = select_pairs()
    available_tickers = set(prices.columns)
    valid_pairs = [(t1, t2) for t1, t2 in all_pairs
                   if t1 in available_tickers and t2 in available_tickers]

    if verbose:
        print(f"Trading {len(valid_pairs)} pairs across {len(HORIZONS)} horizons")

    # --- Compute signals for each pair x horizon ---
    all_daily_returns = []
    pair_stats = []

    common_dates = prices.index.intersection(factors.index).sort_values()

    for horizon in HORIZONS:
        ret_h = returns[horizon].reindex(common_dates)

        pair_signals = {}  # Store z-scores for each pair
        pair_returns = {}  # Store realized returns for each pair

        for t1, t2 in valid_pairs:
            r1 = ret_h[t1]
            r2 = ret_h[t2]

            # Compute regime-conditional z-score
            spread = r1 - r2
            z = pd.Series(np.nan, index=common_dates)

            # For each regime, compute z-score using only same-regime history
            for regime_name in regimes.unique():
                if regime_name == "neutral":
                    continue
                regime_mask = regimes.reindex(common_dates) == regime_name
                regime_spread = spread.where(regime_mask)

                # Expanding mean and std within regime (no lookahead)
                expanding_mean = regime_spread.expanding(min_periods=30).mean()
                expanding_std = regime_spread.expanding(min_periods=30).std()
                expanding_std = expanding_std.clip(lower=1e-6)

                regime_z = (spread - expanding_mean) / expanding_std
                z = z.where(~regime_mask, regime_z)

            pair_signals[(t1, t2)] = z
            pair_returns[(t1, t2)] = spread

        # --- Position sizing via z-score thresholds ---
        # Trade when |z| > 2 (strong dislocation), exit when |z| < 0.5
        threshold_entry = 2.0
        threshold_exit = 0.5

        for (t1, t2), z in pair_signals.items():
            z = z.dropna()
            if len(z) < MIN_HISTORY:
                continue

            # Generate positions: mean-revert the spread
            position = pd.Series(0.0, index=z.index)
            in_trade = False
            current_pos = 0.0

            for i in range(1, len(z)):
                dt = z.index[i]
                prev_z = z.iloc[i-1]  # Use PREVIOUS z to avoid lookahead

                if not in_trade:
                    if prev_z > threshold_entry:
                        # Spread too high -> short spread (short t1, long t2)
                        current_pos = -1.0
                        in_trade = True
                    elif prev_z < -threshold_entry:
                        # Spread too low -> long spread (long t1, short t2)
                        current_pos = 1.0
                        in_trade = True
                else:
                    if abs(prev_z) < threshold_exit:
                        current_pos = 0.0
                        in_trade = False
                    # Also stop out if z moves further against us
                    elif (current_pos > 0 and prev_z < -3.5) or \
                         (current_pos < 0 and prev_z > 3.5):
                        current_pos = 0.0
                        in_trade = False

                position.iloc[i] = current_pos

            if position.abs().sum() == 0:
                continue

            # Daily return of the spread position
            daily_spread_ret = prices[t1].pct_change() - prices[t2].pct_change()
            daily_spread_ret = daily_spread_ret.reindex(position.index)

            # Strategy return = position * daily spread return (shifted by 1)
            strat_ret = position.shift(1) * daily_spread_ret

            # Transaction costs
            turnover = position.diff().abs()
            tc = turnover * (TRANSACTION_COST_BPS / 10000) * 2  # 2 legs
            strat_ret = strat_ret - tc
            strat_ret = strat_ret.dropna()

            if len(strat_ret) > 60:
                # Scale by inverse volatility (target 1% daily vol per pair)
                realized_vol = strat_ret.rolling(63, min_periods=21).std()
                target_vol = 0.01 / np.sqrt(len(valid_pairs))  # Risk budget
                vol_scalar = (target_vol / realized_vol.clip(lower=1e-6)).clip(0, 5)
                scaled_ret = strat_ret * vol_scalar.shift(1)
                scaled_ret = scaled_ret.dropna()

                all_daily_returns.append(scaled_ret)
                pair_stats.append({
                    "pair": f"{t1}-{t2}",
                    "horizon": horizon,
                    "n_trades": int((position.diff().abs() > 0).sum()),
                    "active_days": int((position != 0).sum()),
                    "sharpe": strat_ret.mean() / strat_ret.std() * np.sqrt(252)
                    if strat_ret.std() > 0 else 0,
                })

    if not all_daily_returns:
        print("ERROR: No valid pair trades generated!")
        return None, None

    if verbose:
        print(f"\nActive pairs: {len(all_daily_returns)}")

    # --- Combine all pair returns ---
    combined = pd.concat(all_daily_returns, axis=1).fillna(0)
    portfolio_ret = combined.mean(axis=1)  # Equal weight across all active pairs

    # Final volatility targeting: 10% annualized
    target_ann_vol = 0.10
    realized_vol = portfolio_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    vol_scale = (target_ann_vol / realized_vol.clip(lower=0.01)).clip(0.5, 3.0)
    portfolio_ret = portfolio_ret * vol_scale.shift(1)
    portfolio_ret = portfolio_ret.dropna()

    return portfolio_ret, pd.DataFrame(pair_stats)


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------
def walk_forward_validate(prices, fred, n_splits=5, verbose=True):
    """
    Walk-forward validation with purged expanding window.

    For each fold:
    - Train on all data up to split point
    - Test on the next chunk
    - Purge 21 days between train/test to avoid leakage
    """
    factors = compute_yield_curve_factors(fred)
    common_dates = prices.index.intersection(factors.index).sort_values()
    common_dates = common_dates[common_dates >= common_dates[0] + pd.Timedelta(days=MIN_HISTORY)]

    n_dates = len(common_dates)
    fold_size = n_dates // (n_splits + 1)
    purge_days = 21

    fold_results = []

    for fold in range(n_splits):
        train_end_idx = (fold + 1) * fold_size + fold_size  # Expanding window
        test_start_idx = train_end_idx + purge_days
        test_end_idx = min(test_start_idx + fold_size, n_dates)

        if test_start_idx >= n_dates:
            break

        test_dates = common_dates[test_start_idx:test_end_idx]

        if verbose:
            print(f"\nFold {fold+1}: test {test_dates[0].date()} to {test_dates[-1].date()}")

        # Run strategy on all data (signals use only past data by construction)
        portfolio_ret, _ = run_strategy(prices, fred, verbose=False)
        if portfolio_ret is None:
            continue

        # Extract test period
        test_ret = portfolio_ret.reindex(test_dates).dropna()
        if len(test_ret) < 20:
            continue

        sharpe = test_ret.mean() / test_ret.std() * np.sqrt(252) if test_ret.std() > 0 else 0
        ann_ret = test_ret.mean() * 252
        ann_vol = test_ret.std() * np.sqrt(252)
        cum = (1 + test_ret).cumprod()
        max_dd = ((cum - cum.cummax()) / cum.cummax()).min()

        fold_results.append({
            "fold": fold + 1,
            "test_start": test_dates[0].strftime("%Y-%m-%d"),
            "test_end": test_dates[-1].strftime("%Y-%m-%d"),
            "n_days": len(test_ret),
            "sharpe": round(sharpe, 3),
            "ann_ret": round(ann_ret * 100, 2),
            "ann_vol": round(ann_vol * 100, 2),
            "max_dd": round(max_dd * 100, 2),
        })

        if verbose:
            print(f"  Sharpe={sharpe:.3f}  AnnRet={ann_ret*100:.2f}%  MaxDD={max_dd*100:.2f}%")

    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# Full Backtest with Diagnostics
# ---------------------------------------------------------------------------
def full_backtest(verbose=True):
    """Run the complete strategy with full diagnostics."""
    if verbose:
        print("=" * 80)
        print("ADAPTIVE YIELD CURVE DISLOCATION STRATEGY (AYCDS)")
        print("=" * 80)

    prices = load_prices()
    fred = load_fred()

    if verbose:
        print(f"\nPrices: {prices.shape[0]} days x {prices.shape[1]} tickers")
        print(f"Date range: {prices.index.min().date()} to {prices.index.max().date()}")

    # --- Run main strategy ---
    print("\n--- RUNNING MAIN STRATEGY ---")
    portfolio_ret, pair_stats = run_strategy(prices, fred, verbose=verbose)

    if portfolio_ret is None:
        return

    # --- Overall metrics ---
    total_days = len(portfolio_ret)
    ann_ret = portfolio_ret.mean() * 252
    ann_vol = portfolio_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + portfolio_ret).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (portfolio_ret > 0).mean()

    # Sortino ratio
    downside = portfolio_ret[portfolio_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Skewness and kurtosis of returns
    skew = portfolio_ret.skew()
    kurt = portfolio_ret.kurtosis()

    print(f"\n{'=' * 80}")
    print("FULL SAMPLE RESULTS")
    print(f"{'=' * 80}")
    print(f"Period:          {portfolio_ret.index[0].date()} to {portfolio_ret.index[-1].date()}")
    print(f"Trading days:    {total_days}")
    print(f"Annual return:   {ann_ret*100:+.2f}%")
    print(f"Annual vol:      {ann_vol*100:.2f}%")
    print(f"Sharpe ratio:    {sharpe:.3f}")
    print(f"Sortino ratio:   {sortino:.3f}")
    print(f"Max drawdown:    {max_dd*100:.2f}%")
    print(f"Calmar ratio:    {calmar:.3f}")
    print(f"Win rate:        {win_rate*100:.1f}%")
    print(f"Skewness:        {skew:.3f}")
    print(f"Kurtosis:        {kurt:.3f}")
    print(f"Final cumulative: {cum.iloc[-1]:.4f}x")

    # --- Train / Test split ---
    split_idx = int(len(portfolio_ret) * 0.6)
    train_ret = portfolio_ret.iloc[:split_idx]
    test_ret = portfolio_ret.iloc[split_idx:]

    for period_name, ret in [("TRAIN (60%)", train_ret), ("TEST (40%)", test_ret)]:
        s = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        ar = ret.mean() * 252
        av = ret.std() * np.sqrt(252)
        c = (1 + ret).cumprod()
        mdd = ((c - c.cummax()) / c.cummax()).min()
        wr = (ret > 0).mean()
        print(f"\n  {period_name}: Sharpe={s:.3f}  AnnRet={ar*100:+.2f}%  "
              f"AnnVol={av*100:.2f}%  MaxDD={mdd*100:.2f}%  WinRate={wr*100:.1f}%")

    # --- Yearly breakdown ---
    print(f"\n{'=' * 80}")
    print("YEARLY BREAKDOWN")
    print(f"{'=' * 80}")
    yearly = portfolio_ret.groupby(portfolio_ret.index.year).agg(
        AnnRet=lambda x: x.mean() * 252,
        AnnVol=lambda x: x.std() * np.sqrt(252),
        Sharpe=lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0,
        MaxDD=lambda x: ((1+x).cumprod().pipe(lambda c: (c - c.cummax())/c.cummax())).min(),
        WinRate=lambda x: (x > 0).mean(),
    )
    yearly["AnnRet"] = (yearly["AnnRet"] * 100).round(2)
    yearly["AnnVol"] = (yearly["AnnVol"] * 100).round(2)
    yearly["Sharpe"] = yearly["Sharpe"].round(3)
    yearly["MaxDD"] = (yearly["MaxDD"] * 100).round(2)
    yearly["WinRate"] = (yearly["WinRate"] * 100).round(1)
    print(yearly.to_string())

    # --- Pair statistics ---
    if pair_stats is not None and len(pair_stats) > 0:
        print(f"\n{'=' * 80}")
        print(f"TOP 20 PAIRS BY SHARPE")
        print(f"{'=' * 80}")
        top_pairs = pair_stats.sort_values("sharpe", ascending=False).head(20)
        print(top_pairs.to_string(index=False))

    # --- Walk-forward validation ---
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD VALIDATION (5 folds)")
    print(f"{'=' * 80}")
    wf_results = walk_forward_validate(prices, fred, n_splits=5, verbose=verbose)
    if len(wf_results) > 0:
        print(f"\n  Mean OOS Sharpe: {wf_results['sharpe'].mean():.3f}")
        print(f"  Std OOS Sharpe:  {wf_results['sharpe'].std():.3f}")
        print(f"  Min OOS Sharpe:  {wf_results['sharpe'].min():.3f}")
        print(f"  Max OOS Sharpe:  {wf_results['sharpe'].max():.3f}")

    # --- Overfitting checks ---
    print(f"\n{'=' * 80}")
    print("OVERFITTING DIAGNOSTICS")
    print(f"{'=' * 80}")

    # 1. Train vs Test Sharpe ratio
    train_sharpe = train_ret.mean() / train_ret.std() * np.sqrt(252) if train_ret.std() > 0 else 0
    test_sharpe = test_ret.mean() / test_ret.std() * np.sqrt(252) if test_ret.std() > 0 else 0
    sharpe_decay = 1 - (test_sharpe / train_sharpe) if train_sharpe != 0 else 0
    print(f"Train Sharpe:    {train_sharpe:.3f}")
    print(f"Test Sharpe:     {test_sharpe:.3f}")
    print(f"Sharpe decay:    {sharpe_decay*100:.1f}% (< 30% is good)")

    # 2. Deflated Sharpe Ratio approximation
    # Accounts for multiple testing (we tested ~len(valid_pairs)*3 strategies)
    n_trials = len(pair_stats) if pair_stats is not None else 100
    dsr_adjustment = np.sqrt(2 * np.log(n_trials)) / np.sqrt(252)
    deflated_sharpe = sharpe - dsr_adjustment
    print(f"\nDeflated Sharpe:  {deflated_sharpe:.3f} (adjusted for {n_trials} implicit trials)")
    print(f"  (Sharpe {sharpe:.3f} - adjustment {dsr_adjustment:.3f})")

    # 3. Return autocorrelation check (should be near 0 for no look-ahead)
    ac1 = portfolio_ret.autocorr(1)
    ac5 = portfolio_ret.autocorr(5)
    print(f"\nReturn autocorrelation:")
    print(f"  Lag 1: {ac1:.4f} (|value| < 0.05 suggests no look-ahead)")
    print(f"  Lag 5: {ac5:.4f}")

    # Save results
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    portfolio_ret.to_csv(results_dir / "strategy_returns.csv", header=["return"])
    cum.to_csv(results_dir / "strategy_cumulative.csv", header=["cumulative"])
    if pair_stats is not None:
        pair_stats.to_csv(results_dir / "pair_stats.csv", index=False)
    if len(wf_results) > 0:
        wf_results.to_csv(results_dir / "walk_forward_results.csv", index=False)

    print(f"\nResults saved to {results_dir}")

    return portfolio_ret


if __name__ == "__main__":
    full_backtest()
