#!/usr/bin/env python3
"""
Multi-Factor Bond Arbitrage Strategy V2
========================================

Novel approach: Decompose bond ETF returns into systematic risk factors,
then harvest multiple independent alpha streams simultaneously.

ARCHITECTURE:
- Sleeve 1: RATE-HEDGED CARRY (long credit/EM/muni, hedge duration with treasuries)
- Sleeve 2: CROSS-SECTIONAL VALUE (long cheapest spread-to-model, short richest)
- Sleeve 3: ADAPTIVE MOMENTUM (time-series momentum with volatility scaling)
- Sleeve 4: DISPERSION HARVESTING (sell high cross-sectional vol, buy low)
- Sleeve 5: YIELD CURVE REGIME ALPHA (trade duration based on curve dynamics)
- Sleeve 6: CREDIT CYCLE TIMING (rotate credit quality based on spread dynamics)

Each sleeve is independently vol-targeted and combined with inverse-correlation
weighting for maximum diversification benefit.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"

CORE_TICKERS = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "AGG", "BND",
    "TIP", "EMB", "MUB", "VCIT", "VCSH", "MBB", "FLOT", "VGLT",
    "SPTL", "GOVT", "IEI", "TLH", "IGIB", "SCHP", "VMBS",
]

TRANSACTION_COST_BPS = 5
TARGET_VOL = 0.10  # 10% annualized portfolio vol


def load_data():
    prices = {}
    volumes = {}
    for t in CORE_TICKERS:
        path = ETF_DIR / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
            df = df[~df.index.duplicated(keep="first")].sort_index()
            if "Close" in df.columns:
                prices[t] = df["Close"]
            if "Volume" in df.columns:
                volumes[t] = df["Volume"]

    prices = pd.DataFrame(prices).sort_index()
    volumes = pd.DataFrame(volumes).sort_index()

    fred = pd.read_csv(FRED_PATH, parse_dates=["Date"]).set_index("Date")
    fred = fred[~fred.index.duplicated(keep="first")].sort_index()
    for c in fred.columns:
        fred[c] = pd.to_numeric(fred[c], errors="coerce")
    fred = fred.ffill()

    return prices, volumes, fred


# ========================================================================
# SLEEVE 1: RATE-HEDGED CARRY
# ========================================================================
def sleeve_carry(prices, ret, fred):
    """
    Long high-carry bonds (credit, EM, muni), hedge duration risk.
    Use rolling beta to treasury ETFs for hedging.
    Multiple independent carry streams.
    """
    carry_pairs = [
        ("HYG", "IEF", "HY_carry"),
        ("JNK", "IEF", "HY2_carry"),
        ("LQD", "IEF", "IG_carry"),
        ("EMB", "IEF", "EM_carry"),
        ("MUB", "SHY", "Muni_carry"),
        ("VCIT", "IEI", "MidCorp_carry"),
        ("VCSH", "SHY", "ShortCorp_carry"),
        ("HYG", "TLT", "HY_longhedge"),
        ("MBB", "IEF", "MBS_carry"),
        ("TIP", "IEF", "TIPS_carry"),
        ("IGIB", "IEI", "IG5_carry"),
    ]

    sleeve_returns = {}
    for long_etf, hedge_etf, name in carry_pairs:
        if long_etf not in ret.columns or hedge_etf not in ret.columns:
            continue

        lookback = 252
        # Rolling beta for hedge ratio
        cov = ret[long_etf].rolling(lookback, min_periods=126).cov(ret[hedge_etf])
        var = ret[hedge_etf].rolling(lookback, min_periods=126).var()
        beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)

        # Hedged return: long carry, short beta*hedge
        hedged = ret[long_etf] - beta.shift(1) * ret[hedge_etf]

        # Volatility regime filter: reduce exposure in high-vol
        if "VIXCLS" in fred.columns:
            vix = fred["VIXCLS"].reindex(hedged.index).ffill()
            vix_pctl = vix.rolling(252, min_periods=126).rank(pct=True)
            # Scale: 1.0 at low vol, 0.3 at high vol
            vol_scale = (1.3 - vix_pctl).clip(0.3, 1.0)
            hedged = hedged * vol_scale.shift(1)

        sleeve_returns[name] = hedged

    return sleeve_returns


# ========================================================================
# SLEEVE 2: CROSS-SECTIONAL VALUE
# ========================================================================
def sleeve_value(prices, ret, fred):
    """
    Cross-sectional value: rank ETFs by deviation from fair value.
    Fair value = what the ETF 'should' return given current yield environment.
    Long the undervalued, short the overvalued.
    """
    # Use credit spread as value signal for credit ETFs
    credit_etfs = ["LQD", "HYG", "JNK", "VCIT", "VCSH", "IGIB", "EMB"]
    available = [t for t in credit_etfs if t in ret.columns]

    if len(available) < 3:
        return {}

    # Value signal: recent return relative to what spread changes predict
    # If spreads widened (bad) but ETF didn't fall as much = cheap
    sleeve_returns = {}

    for spread_col, etf_group, name in [
        ("BAMLH0A0HYM2", ["HYG", "JNK"], "HY_value"),
        ("BAMLC0A0CM", ["LQD", "VCIT", "IGIB"], "IG_value"),
    ]:
        if spread_col not in fred.columns:
            continue
        etfs = [t for t in etf_group if t in ret.columns]
        if len(etfs) < 2:
            continue

        spread = fred[spread_col].reindex(ret.index).ffill()
        spread_chg = spread.diff(21)  # 21-day spread change

        for horizon in [21, 63]:
            # Expected return given spread change (rolling regression)
            for etf in etfs:
                etf_ret_h = ret[etf].rolling(horizon).sum()
                lookback = 504  # 2 years

                # Rolling correlation as proxy for expected sensitivity
                sensitivity = etf_ret_h.rolling(lookback, min_periods=252).corr(
                    spread_chg.rolling(horizon).sum()
                )

                # Residual = actual return - expected
                expected = sensitivity.shift(1) * spread_chg.rolling(horizon).sum()
                residual = etf_ret_h - expected

                # Z-score of residual
                res_z = (residual - residual.rolling(lookback, min_periods=126).mean()) / \
                        residual.rolling(lookback, min_periods=126).std().clip(lower=1e-6)

                # Position: mean-revert residual (if ETF underperformed vs spread model, go long)
                pos = -res_z.clip(-2, 2) / 2  # Continuous sizing, capped
                strat_ret = pos.shift(1) * ret[etf]

                # Transaction cost
                tc = pos.diff().abs() * (TRANSACTION_COST_BPS / 10000)
                strat_ret = strat_ret - tc

                sleeve_returns[f"{name}_{etf}_{horizon}d"] = strat_ret.dropna()

    return sleeve_returns


# ========================================================================
# SLEEVE 3: ADAPTIVE MOMENTUM
# ========================================================================
def sleeve_momentum(prices, ret, fred):
    """
    Time-series momentum with volatility targeting and adaptive lookback.
    Key insight: use MULTIPLE lookback windows and average (more robust).
    """
    sleeve_returns = {}
    lookbacks = [21, 63, 126, 252]  # 1mo, 3mo, 6mo, 12mo

    # All tradeable ETFs
    tradeable = [t for t in ret.columns if t in CORE_TICKERS]

    for etf in tradeable:
        signals = []
        for lb in lookbacks:
            # Time-series momentum signal: sign of past return, vol-adjusted
            past_ret = ret[etf].rolling(lb, min_periods=int(lb*0.7)).mean() * np.sqrt(252)
            past_vol = ret[etf].rolling(lb, min_periods=int(lb*0.7)).std() * np.sqrt(252)
            risk_adj_mom = past_ret / past_vol.clip(lower=0.01)
            signals.append(risk_adj_mom)

        # Average across lookbacks (more robust than single lookback)
        combined_signal = pd.concat(signals, axis=1).mean(axis=1)

        # Position: continuous, capped at [-1, 1]
        pos = combined_signal.clip(-2, 2) / 2

        strat_ret = pos.shift(1) * ret[etf]
        tc = pos.diff().abs() * (TRANSACTION_COST_BPS / 10000)
        strat_ret = strat_ret - tc

        sleeve_returns[f"tsmom_{etf}"] = strat_ret.dropna()

    return sleeve_returns


# ========================================================================
# SLEEVE 4: DISPERSION HARVESTING
# ========================================================================
def sleeve_dispersion(prices, ret, fred):
    """
    Novel signal: trade the dispersion of bond ETF returns.
    When cross-sectional dispersion is high, individual mean-reversion is stronger.
    When low, momentum is stronger. Adaptively switch.
    """
    tradeable = [t for t in ["TLT", "IEF", "LQD", "HYG", "EMB", "TIP", "MUB", "AGG"]
                 if t in ret.columns]

    if len(tradeable) < 4:
        return {}

    # Cross-sectional dispersion
    xs_ret = ret[tradeable]
    dispersion = xs_ret.std(axis=1)
    disp_pctl = dispersion.rolling(252, min_periods=126).rank(pct=True)

    sleeve_returns = {}

    for etf in tradeable:
        # High dispersion → mean reversion (go against recent move)
        # Low dispersion → momentum (go with recent move)
        mr_signal = -ret[etf].rolling(5).mean()  # 5-day mean reversion
        mom_signal = ret[etf].rolling(63).mean()  # 63-day momentum

        # Blend based on dispersion percentile
        # High disp (>0.7) → more MR weight. Low disp (<0.3) → more mom weight
        mr_weight = disp_pctl.clip(0, 1)
        mom_weight = 1 - mr_weight

        blended = (mr_weight.shift(1) * mr_signal + mom_weight.shift(1) * mom_signal)
        # Normalize
        blended_z = blended / blended.rolling(63, min_periods=21).std().clip(lower=1e-6)
        pos = blended_z.clip(-2, 2) / 2

        strat_ret = pos.shift(1) * ret[etf]
        tc = pos.diff().abs() * (TRANSACTION_COST_BPS / 10000)
        strat_ret = strat_ret - tc
        sleeve_returns[f"disp_{etf}"] = strat_ret.dropna()

    return sleeve_returns


# ========================================================================
# SLEEVE 5: YIELD CURVE REGIME ALPHA
# ========================================================================
def sleeve_yc_regime(prices, ret, fred):
    """
    Trade duration exposure based on yield curve factor dynamics.
    Uses PCA-like decomposition: level, slope, curvature changes.
    """
    sleeve_returns = {}

    if "DGS2" not in fred.columns or "DGS10" not in fred.columns:
        return {}

    slope = (fred["DGS10"] - fred["DGS2"]).reindex(ret.index).ffill()
    level = fred["DGS10"].reindex(ret.index).ffill()

    # Duration ETFs by sensitivity
    etf_durations = {
        "SHY": 2, "IEI": 5, "IEF": 7.5, "TLH": 15, "TLT": 20, "VGLT": 22,
    }

    for etf, dur in etf_durations.items():
        if etf not in ret.columns:
            continue

        # Signal 1: Slope momentum → when steepening, long-duration benefits
        slope_mom = slope.diff(21)
        pos1 = slope_mom / slope_mom.rolling(63, min_periods=21).std().clip(lower=1e-6)
        pos1 = pos1.clip(-2, 2) / 2
        # Scale by duration (more for longer duration ETFs)
        pos1 = pos1 * (dur / 10)

        # Signal 2: Level mean-reversion → when rates spike up, long bonds (expect reversion)
        level_z = (level - level.rolling(252, min_periods=126).mean()) / \
                  level.rolling(252, min_periods=126).std().clip(lower=1e-6)
        pos2 = level_z.clip(-2, 2) / 2  # High rates → long (mean revert)

        # Combine signals
        pos = (pos1 + pos2) / 2

        strat_ret = pos.shift(1) * ret[etf]
        tc = pos.diff().abs() * (TRANSACTION_COST_BPS / 10000)
        strat_ret = strat_ret - tc
        sleeve_returns[f"yc_{etf}"] = strat_ret.dropna()

    return sleeve_returns


# ========================================================================
# SLEEVE 6: CREDIT CYCLE TIMING
# ========================================================================
def sleeve_credit_cycle(prices, ret, fred):
    """
    Rotate between credit quality tiers based on credit cycle position.
    """
    sleeve_returns = {}

    ig_oas = fred.get("BAMLC0A0CM")
    hy_oas = fred.get("BAMLH0A0HYM2")

    if ig_oas is None or hy_oas is None:
        return {}

    ig_oas = ig_oas.reindex(ret.index).ffill()
    hy_oas = hy_oas.reindex(ret.index).ffill()

    # Spread ratio: HY/IG spread ratio as cycle indicator
    spread_ratio = hy_oas / ig_oas.clip(lower=0.01)
    sr_z = (spread_ratio - spread_ratio.rolling(504, min_periods=252).mean()) / \
           spread_ratio.rolling(504, min_periods=252).std().clip(lower=1e-6)

    # Spread momentum: are spreads compressing or widening?
    hy_mom = -hy_oas.diff(21)  # Negative because lower spread = better for credit
    hy_mom_z = hy_mom / hy_mom.rolling(63, min_periods=21).std().clip(lower=1e-6)

    # Trade: when spreads are wide AND compressing → long credit
    # When spreads are tight AND widening → short credit
    credit_signal = (sr_z + hy_mom_z) / 2

    credit_longs = ["HYG", "JNK", "LQD", "VCIT", "EMB"]
    credit_shorts = ["SHY", "FLOT", "GOVT"]

    for etf in credit_longs:
        if etf not in ret.columns:
            continue
        pos = credit_signal.clip(-2, 2) / 2
        strat_ret = pos.shift(1) * ret[etf]
        tc = pos.diff().abs() * (TRANSACTION_COST_BPS / 10000)
        sleeve_returns[f"cc_long_{etf}"] = (strat_ret - tc).dropna()

    for etf in credit_shorts:
        if etf not in ret.columns:
            continue
        pos = -credit_signal.clip(-2, 2) / 2
        strat_ret = pos.shift(1) * ret[etf]
        tc = pos.diff().abs() * (TRANSACTION_COST_BPS / 10000)
        sleeve_returns[f"cc_short_{etf}"] = (strat_ret - tc).dropna()

    return sleeve_returns


# ========================================================================
# PORTFOLIO CONSTRUCTION
# ========================================================================
def construct_portfolio(all_sleeve_returns, min_history=252):
    """
    Combine all sleeve returns with:
    1. Vol-targeting per sub-strategy
    2. Inverse-variance weighting across sub-strategies
    3. Portfolio-level vol targeting
    """
    # Filter for sufficient history
    valid = {k: v for k, v in all_sleeve_returns.items()
             if len(v.dropna()) >= min_history}

    if not valid:
        print("No valid strategies!")
        return None

    # Align all to common dates
    all_df = pd.DataFrame(valid)
    all_df = all_df.dropna(how="all")

    # Step 1: Vol-target each sub-strategy to 5% annualized
    sub_target_vol = 0.05
    vol_targeted = pd.DataFrame(index=all_df.index)

    for col in all_df.columns:
        s = all_df[col].dropna()
        realized_vol = s.rolling(63, min_periods=21).std() * np.sqrt(252)
        scaler = (sub_target_vol / realized_vol.clip(lower=0.005)).clip(0.1, 10)
        vol_targeted[col] = s * scaler.shift(1)

    vol_targeted = vol_targeted.fillna(0)

    # Step 2: Inverse-variance weighting with shrinkage
    # Use rolling correlations to diversify
    lookback = 252
    portfolio_ret = pd.Series(0.0, index=vol_targeted.index)
    n_strats = vol_targeted.shape[1]

    if n_strats <= 1:
        portfolio_ret = vol_targeted.iloc[:, 0] if n_strats == 1 else portfolio_ret
    else:
        # Simple equal weight (most robust, avoids estimation error in correlations)
        portfolio_ret = vol_targeted.mean(axis=1)

    # Step 3: Portfolio-level vol targeting
    port_vol = portfolio_ret.rolling(63, min_periods=21).std() * np.sqrt(252)
    port_scaler = (TARGET_VOL / port_vol.clip(lower=0.005)).clip(0.2, 5.0)
    portfolio_ret = portfolio_ret * port_scaler.shift(1)

    return portfolio_ret.dropna()


# ========================================================================
# METRICS & DIAGNOSTICS
# ========================================================================
def compute_metrics(returns, name=""):
    if returns is None or len(returns) < 60:
        return {}
    r = returns.dropna()
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + r).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    win_rate = (r > 0).mean()
    downside_vol = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    skew = r.skew()
    kurt = r.kurtosis()

    return {
        "name": name,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": win_rate,
        "skew": skew,
        "kurt": kurt,
        "n_days": len(r),
    }


def print_metrics(m, prefix=""):
    if not m:
        print(f"{prefix}No metrics available")
        return
    print(f"{prefix}Annual return:   {m['ann_ret']*100:+.2f}%")
    print(f"{prefix}Annual vol:      {m['ann_vol']*100:.2f}%")
    print(f"{prefix}Sharpe ratio:    {m['sharpe']:.3f}")
    print(f"{prefix}Sortino ratio:   {m['sortino']:.3f}")
    print(f"{prefix}Max drawdown:    {m['max_dd']*100:.2f}%")
    print(f"{prefix}Calmar ratio:    {m['calmar']:.3f}")
    print(f"{prefix}Win rate:        {m['win_rate']*100:.1f}%")
    print(f"{prefix}Skewness:        {m['skew']:.3f}")
    print(f"{prefix}Kurtosis:        {m['kurt']:.3f}")
    print(f"{prefix}Trading days:    {m['n_days']}")


# ========================================================================
# MAIN
# ========================================================================
def main():
    print("=" * 80)
    print("MULTI-FACTOR BOND ARBITRAGE STRATEGY V2")
    print("=" * 80)

    prices, volumes, fred = load_data()
    ret = prices.pct_change()

    print(f"\nData: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print(f"Range: {prices.index.min().date()} to {prices.index.max().date()}")

    # --- Run all sleeves ---
    all_returns = {}

    print("\n--- Sleeve 1: Rate-Hedged Carry ---")
    carry = sleeve_carry(prices, ret, fred)
    print(f"  {len(carry)} sub-strategies")
    all_returns.update(carry)

    print("--- Sleeve 2: Cross-Sectional Value ---")
    value = sleeve_value(prices, ret, fred)
    print(f"  {len(value)} sub-strategies")
    all_returns.update(value)

    print("--- Sleeve 3: Adaptive Momentum ---")
    momentum = sleeve_momentum(prices, ret, fred)
    print(f"  {len(momentum)} sub-strategies")
    all_returns.update(momentum)

    print("--- Sleeve 4: Dispersion Harvesting ---")
    dispersion = sleeve_dispersion(prices, ret, fred)
    print(f"  {len(dispersion)} sub-strategies")
    all_returns.update(dispersion)

    print("--- Sleeve 5: Yield Curve Regime ---")
    yc = sleeve_yc_regime(prices, ret, fred)
    print(f"  {len(yc)} sub-strategies")
    all_returns.update(yc)

    print("--- Sleeve 6: Credit Cycle Timing ---")
    cc = sleeve_credit_cycle(prices, ret, fred)
    print(f"  {len(cc)} sub-strategies")
    all_returns.update(cc)

    print(f"\nTotal sub-strategies: {len(all_returns)}")

    # --- Sleeve-level performance ---
    print(f"\n{'=' * 80}")
    print("SLEEVE-LEVEL PERFORMANCE")
    print(f"{'=' * 80}")

    sleeve_groups = {
        "Carry": carry, "Value": value, "Momentum": momentum,
        "Dispersion": dispersion, "YC_Regime": yc, "Credit_Cycle": cc,
    }

    for sleeve_name, sleeve_dict in sleeve_groups.items():
        if not sleeve_dict:
            continue
        # Combine sub-strats within sleeve
        df = pd.DataFrame(sleeve_dict).dropna(how="all").fillna(0)
        sleeve_ret = df.mean(axis=1)
        m = compute_metrics(sleeve_ret, sleeve_name)
        if m:
            print(f"\n  {sleeve_name}: Sharpe={m['sharpe']:.3f}  "
                  f"AnnRet={m['ann_ret']*100:+.2f}%  "
                  f"MaxDD={m['max_dd']*100:.2f}%")

    # --- Construct portfolio ---
    print(f"\n{'=' * 80}")
    print("PORTFOLIO CONSTRUCTION")
    print(f"{'=' * 80}")

    portfolio_ret = construct_portfolio(all_returns)
    if portfolio_ret is None:
        return

    # Full sample metrics
    full_m = compute_metrics(portfolio_ret, "Full Sample")
    print("\n--- FULL SAMPLE ---")
    print_metrics(full_m, "  ")

    # Train/Test split
    split = int(len(portfolio_ret) * 0.6)
    train_ret = portfolio_ret.iloc[:split]
    test_ret = portfolio_ret.iloc[split:]

    train_m = compute_metrics(train_ret, "Train")
    test_m = compute_metrics(test_ret, "Test")

    print("\n--- TRAIN (60%) ---")
    print_metrics(train_m, "  ")
    print("\n--- TEST (40%) ---")
    print_metrics(test_m, "  ")

    # Yearly breakdown
    print(f"\n{'=' * 80}")
    print("YEARLY PERFORMANCE")
    print(f"{'=' * 80}")
    print(f"{'Year':>6} {'AnnRet':>10} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'WinRate':>8}")

    for year, group in portfolio_ret.groupby(portfolio_ret.index.year):
        if len(group) < 20:
            continue
        ar = group.mean() * 252
        av = group.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        cum = (1 + group).cumprod()
        mdd = ((cum - cum.cummax()) / cum.cummax()).min()
        wr = (group > 0).mean()
        print(f"{year:>6} {ar*100:>+9.2f}% {av*100:>7.2f}% {sr:>+7.3f} {mdd*100:>+7.2f}% {wr*100:>7.1f}%")

    # Overfitting diagnostics
    print(f"\n{'=' * 80}")
    print("OVERFITTING DIAGNOSTICS")
    print(f"{'=' * 80}")
    if train_m and test_m:
        decay = 1 - (test_m["sharpe"] / train_m["sharpe"]) if train_m["sharpe"] != 0 else 0
        print(f"  Train Sharpe:  {train_m['sharpe']:.3f}")
        print(f"  Test Sharpe:   {test_m['sharpe']:.3f}")
        print(f"  Sharpe decay:  {decay*100:.1f}%")

    ac1 = portfolio_ret.autocorr(1)
    ac5 = portfolio_ret.autocorr(5)
    print(f"  Autocorr(1):   {ac1:.4f}")
    print(f"  Autocorr(5):   {ac5:.4f}")

    # Rolling Sharpe
    rolling_sr = portfolio_ret.rolling(252).mean() / portfolio_ret.rolling(252).std() * np.sqrt(252)
    print(f"  Rolling 1Y Sharpe - Mean: {rolling_sr.mean():.3f}, Std: {rolling_sr.std():.3f}")
    print(f"  Rolling 1Y Sharpe - Min: {rolling_sr.min():.3f}, Max: {rolling_sr.max():.3f}")
    print(f"  Pct of time Sharpe > 0: {(rolling_sr > 0).mean()*100:.1f}%")

    # Save results
    results_dir = DATA_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    portfolio_ret.to_csv(results_dir / "strategy_v2_returns.csv", header=["return"])
    (1 + portfolio_ret).cumprod().to_csv(results_dir / "strategy_v2_cumulative.csv",
                                         header=["cumulative"])

    print(f"\nResults saved to {results_dir}")


if __name__ == "__main__":
    main()
