#!/usr/bin/env python3
"""
Factor-Residual Carry Arbitrage Strategy V5
=============================================

TWO INDEPENDENT ALPHA ENGINES:

ENGINE 1: CARRY (proven, Sharpe ~1.0 per V3)
  Duration-hedged carry across 14 sector pairs.
  Vol-conditioned + VIX scaling.

ENGINE 2: FACTOR-RESIDUAL MEAN REVERSION (novel)
  Model each ETF's return as a function of yield curve factor changes:
    R_etf = β_level * ΔLevel + β_slope * ΔSlope + β_curvature * ΔCurvature + ε
  
  The residual ε represents ETF-specific mispricing relative to rates.
  Key insight: bond ETFs have arbitrage mechanisms (creation/redemption)
  that force convergence to NAV. When ε is large and positive (ETF
  outperformed rate model), it will mean-revert (sell signal).
  
  This is analogous to equity stat-arb factor models but applied to
  fixed income ETFs using yield curve factors instead of equity factors.
  
  CRITICAL: β is estimated using EXPANDING window (no lookahead).
  Residuals are z-scored using PAST data only.

ENGINE 3: CROSS-INSTRUMENT MEAN REVERSION
  Within highly correlated ETF groups (e.g., AGG/BND, TLT/VGLT/SPTL),
  trade the ratio when it deviates from its rolling mean.
  Use Ornstein-Uhlenbeck parameters to determine optimal entry/exit.

PORTFOLIO: Inverse-variance weighted combination with vol targeting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("/home/user/bonds/data")
ETF_DIR = DATA_DIR / "etfs"
FRED_PATH = DATA_DIR / "fred" / "_combined_fred.csv"
TC_BPS = 5
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
# ENGINE 1: CARRY
# ========================================================================
def engine_carry(ret, fred):
    """Duration-hedged carry with vol conditioning."""
    pairs = [
        ("HYG", "IEF"), ("HYG", "TLT"), ("HYG", "SHY"),
        ("JNK", "IEF"), ("LQD", "IEF"),
        ("VCIT", "IEI"), ("VCSH", "SHY"), ("IGIB", "IEI"),
        ("EMB", "IEF"), ("EMB", "TLT"),
        ("MUB", "SHY"), ("MUB", "IEI"),
        ("MBB", "IEF"), ("TIP", "IEF"),
    ]
    vix = fred.get("VIXCLS")
    streams = {}
    for long_e, hedge_e in pairs:
        if long_e not in ret.columns or hedge_e not in ret.columns:
            continue
        lb = 252
        cov = ret[long_e].rolling(lb, min_periods=126).cov(ret[hedge_e])
        var = ret[hedge_e].rolling(lb, min_periods=126).var()
        beta = (cov / var.clip(lower=1e-8)).clip(-3, 3)
        hedged = ret[long_e] - beta.shift(1) * ret[hedge_e]

        # Vol-target per stream
        rv = hedged.rolling(63, min_periods=21).std() * np.sqrt(252)
        vol_scale = (0.05 / rv.clip(lower=0.005)).clip(0.1, 5.0)

        # VIX conditioning
        if vix is not None:
            vix_a = vix.reindex(hedged.index).ffill()
            vix_p = vix_a.rolling(252, min_periods=126).rank(pct=True)
            stress = (1.2 - 0.6 * vix_p).clip(0.4, 1.2)
            vol_scale = vol_scale * stress

        scaled = hedged * vol_scale.shift(1)
        if len(scaled.dropna()) >= 252:
            streams[f"carry_{long_e}_{hedge_e}"] = scaled.dropna()

    return streams


# ========================================================================
# ENGINE 2: FACTOR-RESIDUAL MEAN REVERSION
# ========================================================================
def engine_factor_residual(prices, ret, fred):
    """
    Model ETF returns as function of yield curve factors.
    Trade the residual (mispricing) via mean reversion.
    """
    streams = {}

    # Yield curve factors
    cols_needed = ["DGS2", "DGS5", "DGS10", "DGS30"]
    if not all(c in fred.columns for c in cols_needed):
        return streams

    yc = fred[cols_needed].dropna()
    level = yc.mean(axis=1)
    slope = yc["DGS10"] - yc["DGS2"]
    curvature = 2 * yc["DGS5"] - yc["DGS2"] - yc["DGS10"]

    # Factor changes (1-day)
    d_level = level.diff()
    d_slope = slope.diff()
    d_curv = curvature.diff()

    factors = pd.DataFrame({
        "d_level": d_level, "d_slope": d_slope, "d_curv": d_curv
    }).dropna()

    tradeable = [t for t in ret.columns if t in prices.columns]

    for etf in tradeable:
        etf_ret = ret[etf].reindex(factors.index).dropna()
        common = etf_ret.index.intersection(factors.index).sort_values()
        if len(common) < 504:
            continue

        y = etf_ret.loc[common]
        X = factors.loc[common]

        # Expanding window regression (no lookahead)
        residuals = pd.Series(np.nan, index=common)
        min_train = 252

        # Vectorized expanding regression using cumulative sums
        n = len(common)
        Xm = X.values
        ym = y.values

        # Cumulative XtX and Xty for expanding window
        XtX_cum = np.zeros((n, 3, 3))
        Xty_cum = np.zeros((n, 3))

        XtX_cum[0] = np.outer(Xm[0], Xm[0])
        Xty_cum[0] = Xm[0] * ym[0]

        for i in range(1, n):
            XtX_cum[i] = XtX_cum[i-1] + np.outer(Xm[i], Xm[i])
            Xty_cum[i] = Xty_cum[i-1] + Xm[i] * ym[i]

        for i in range(min_train, n):
            XtX = XtX_cum[i-1]  # Use data up to i-1
            Xty = Xty_cum[i-1]
            try:
                # Ridge regression (regularized)
                beta = np.linalg.solve(XtX + 0.01 * np.eye(3), Xty)
                predicted = Xm[i] @ beta
                residuals.iloc[i] = ym[i] - predicted
            except np.linalg.LinAlgError:
                continue

        residuals = residuals.dropna()
        if len(residuals) < 252:
            continue

        # Z-score the residual using expanding window
        res_mean = residuals.expanding(min_periods=126).mean()
        res_std = residuals.expanding(min_periods=126).std().clip(lower=1e-6)
        res_z = (residuals - res_mean) / res_std

        # Position: mean-revert the residual
        # Positive residual = ETF outperformed model → sell (expect reversion)
        # Use 5-day smoothed z-score for less noise
        smooth_z = res_z.rolling(5).mean()
        pos = -smooth_z.clip(-2, 2) / 2

        strat_ret = pos.shift(1) * ret[etf].reindex(pos.index)
        tc = pos.diff().abs() * (TC_BPS / 10000)
        result = (strat_ret - tc).dropna()

        if len(result) >= 252:
            streams[f"factor_res_{etf}"] = result

    return streams


# ========================================================================
# ENGINE 3: CROSS-INSTRUMENT MEAN REVERSION
# ========================================================================
def engine_cross_instrument(prices, ret):
    """
    Trade mean-reversion between highly correlated ETF pairs within categories.
    Uses log-price ratio with Ornstein-Uhlenbeck calibration.
    """
    streams = {}

    # Highly correlated pairs (same index/similar)
    pairs = [
        ("TLT", "VGLT"), ("TLT", "SPTL"), ("VGLT", "SPTL"),
        ("AGG", "BND"),
        ("HYG", "JNK"),
        ("LQD", "VCIT"),
        ("MBB", "VMBS"),
        ("MUB", "VTEB"),
        ("TIP", "SCHP"),
    ]

    for t1, t2 in pairs:
        if t1 not in prices.columns or t2 not in prices.columns:
            continue

        # Log price ratio
        p1 = prices[t1].dropna()
        p2 = prices[t2].dropna()
        common = p1.index.intersection(p2.index).sort_values()
        if len(common) < 504:
            continue

        log_ratio = np.log(p1.loc[common] / p2.loc[common])

        # Calibrate OU process using expanding window
        # dS = theta * (mu - S) * dt + sigma * dW
        # Use lagged regression: S(t) - S(t-1) = a + b * S(t-1)
        # theta = -b, mu = -a/b

        lookback = 252
        z_scores = pd.Series(np.nan, index=common)

        for i in range(lookback, len(common)):
            window = log_ratio.iloc[i-lookback:i]
            delta = window.diff().dropna()
            lag = window.shift(1).dropna()
            common_idx = delta.index.intersection(lag.index)
            if len(common_idx) < 100:
                continue

            d = delta.loc[common_idx].values
            l = lag.loc[common_idx].values

            # OLS: delta = a + b * lag
            n_obs = len(d)
            l_mean = l.mean()
            d_mean = d.mean()
            b = np.sum((l - l_mean) * (d - d_mean)) / (np.sum((l - l_mean)**2) + 1e-10)
            a = d_mean - b * l_mean

            if b >= 0:  # No mean reversion
                continue

            theta = -b * 252  # Annualized
            mu = -a / b
            sigma = np.std(d - a - b * l) * np.sqrt(252)

            # Z-score relative to OU equilibrium
            current = log_ratio.iloc[i]
            eq_std = sigma / np.sqrt(2 * theta) if theta > 0 else sigma
            z = (current - mu) / max(eq_std, 1e-6)
            z_scores.iloc[i] = z

        z_scores = z_scores.dropna()
        if len(z_scores) < 252:
            continue

        # Trade: mean-revert the ratio
        pos = -z_scores.clip(-2, 2) / 2
        spread_ret = ret[t1].reindex(pos.index) - ret[t2].reindex(pos.index)
        strat_ret = pos.shift(1) * spread_ret
        tc = pos.diff().abs() * (TC_BPS / 10000) * 2
        result = (strat_ret - tc).dropna()

        if len(result) >= 252:
            streams[f"xpair_{t1}_{t2}"] = result

    return streams


# ========================================================================
# PORTFOLIO CONSTRUCTION & MAIN
# ========================================================================
def compute_metrics(r):
    r = r.dropna()
    if len(r) < 60: return None
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    cum = (1 + r).cumprod()
    mdd = ((cum - cum.cummax()) / cum.cummax()).min()
    cal = ar / abs(mdd) if mdd != 0 else 0
    wr = (r > 0).mean()
    ds = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else av
    sortino = ar / ds if ds > 0 else 0
    return {"ann_ret": ar, "ann_vol": av, "sharpe": sr, "sortino": sortino,
            "max_dd": mdd, "calmar": cal, "win_rate": wr,
            "skew": r.skew(), "kurt": r.kurtosis(), "n_days": len(r)}


def construct_portfolio(all_streams, min_history=504):
    valid = {}
    for name, s in all_streams.items():
        s = s.dropna()
        if len(s) >= min_history:
            valid[name] = s
    if not valid:
        return None

    df = pd.DataFrame(valid).dropna(how="all").fillna(0)

    # Vol-target each stream to 3%
    vol_t = pd.DataFrame(index=df.index)
    for col in df.columns:
        rv = df[col].rolling(63, min_periods=21).std() * np.sqrt(252)
        sc = (0.03 / rv.clip(lower=0.003)).clip(0.1, 8.0)
        vol_t[col] = df[col] * sc.shift(1)
    vol_t = vol_t.fillna(0)

    # Equal weight
    portfolio = vol_t.mean(axis=1)

    # Drawdown control
    cum = (1 + portfolio).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    dd_scale = (1.0 + dd * 5.0).clip(0.25, 1.0)
    portfolio = portfolio * dd_scale.shift(1)

    # Portfolio vol target
    pv = portfolio.rolling(63, min_periods=21).std() * np.sqrt(252)
    ps = (TARGET_VOL / pv.clip(lower=0.005)).clip(0.2, 5.0)
    portfolio = portfolio * ps.shift(1)

    return portfolio.dropna()


def main():
    print("=" * 80)
    print("FACTOR-RESIDUAL CARRY ARBITRAGE V5")
    print("=" * 80)

    prices, fred = load_data()
    ret = prices.pct_change()
    print(f"Data: {prices.shape[0]} days x {prices.shape[1]} tickers")

    all_streams = {}

    # Engine 1: Carry
    print("\n--- Engine 1: Carry ---")
    carry = engine_carry(ret, fred)
    print(f"  {len(carry)} streams")
    all_streams.update(carry)

    # Engine 2: Factor Residual
    print("--- Engine 2: Factor Residual ---")
    factor_res = engine_factor_residual(prices, ret, fred)
    print(f"  {len(factor_res)} streams")
    all_streams.update(factor_res)

    # Engine 3: Cross-Instrument
    print("--- Engine 3: Cross-Instrument ---")
    xpair = engine_cross_instrument(prices, ret)
    print(f"  {len(xpair)} streams")
    all_streams.update(xpair)

    print(f"\nTotal: {len(all_streams)} streams")

    # Per-engine performance
    print(f"\n{'=' * 80}")
    print("ENGINE-LEVEL PERFORMANCE")
    print(f"{'=' * 80}")
    for eng_name, eng_dict in [("Carry", carry), ("Factor_Residual", factor_res),
                                ("Cross_Instrument", xpair)]:
        if not eng_dict:
            print(f"  {eng_name}: no streams")
            continue
        edf = pd.DataFrame(eng_dict).dropna(how="all").fillna(0)
        er = edf.mean(axis=1)
        m = compute_metrics(er)
        if m:
            print(f"  {eng_name}: Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_ret']*100:+.2f}%  "
                  f"MaxDD={m['max_dd']*100:.2f}%")

    # Individual streams
    print(f"\n--- Top 10 Individual Streams ---")
    stream_m = {}
    for name, s in all_streams.items():
        m = compute_metrics(s)
        if m:
            stream_m[name] = m
    sorted_streams = sorted(stream_m.items(), key=lambda x: x[1]["sharpe"], reverse=True)
    for name, m in sorted_streams[:10]:
        print(f"  {name:35s}: Sharpe={m['sharpe']:+.3f}  AnnRet={m['ann_ret']*100:+.2f}%")
    print(f"  ... (showing top 10 of {len(sorted_streams)})")

    # Portfolio
    print(f"\n{'=' * 80}")
    print("PORTFOLIO RESULTS")
    print(f"{'=' * 80}")
    portfolio = construct_portfolio(all_streams)
    if portfolio is None:
        print("No portfolio!"); return

    m = compute_metrics(portfolio)
    for k, v in m.items():
        if isinstance(v, float):
            if any(x in k for x in ["ret", "vol", "dd", "rate"]):
                print(f"  {k:16s}: {v*100:+.2f}%")
            else:
                print(f"  {k:16s}: {v:.3f}")
        else:
            print(f"  {k:16s}: {v}")

    # Train/Test
    sp = int(len(portfolio) * 0.6)
    for nm, r in [("TRAIN 60%", portfolio.iloc[:sp]), ("TEST 40%", portfolio.iloc[sp:])]:
        m = compute_metrics(r)
        if m:
            print(f"\n  {nm}: Sharpe={m['sharpe']:.3f}  AnnRet={m['ann_ret']*100:+.2f}%  "
                  f"MaxDD={m['max_dd']*100:.2f}%  Sortino={m['sortino']:.3f}")

    # Yearly
    print(f"\n{'=' * 80}")
    print("YEARLY")
    print(f"{'=' * 80}")
    print(f"{'Year':>6} {'Ret':>9} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8}")
    for yr, g in portfolio.groupby(portfolio.index.year):
        if len(g) < 20: continue
        ar = g.mean()*252; av = g.std()*np.sqrt(252)
        sr = ar/av if av > 0 else 0
        c = (1+g).cumprod(); mdd = ((c-c.cummax())/c.cummax()).min()
        print(f"{yr:>6} {ar*100:>+8.2f}% {av*100:>7.2f}% {sr:>+7.3f} {mdd*100:>+7.2f}%")

    # Diversification
    sdf = pd.DataFrame({k:v for k,v in all_streams.items()
                         if len(v.dropna())>=504}).dropna(how="all").fillna(0)
    if sdf.shape[1] > 1:
        cr = sdf.corr()
        up = cr.where(np.triu(np.ones(cr.shape), k=1).astype(bool))
        ac = up.stack().mean()
        n = sdf.shape[1]
        dm = np.sqrt(n*(1-ac)/(1+(n-1)*ac)) if (1+(n-1)*ac) > 0 else 1
        print(f"\n  Avg corr: {ac:.3f}  Streams: {n}  Div multiplier: {dm:.2f}x")

    # Walk-forward
    print(f"\n{'=' * 80}")
    print("WALK-FORWARD (5 folds)")
    nt = len(portfolio); fs = nt // 6
    for fold in range(5):
        s = (fold+1)*fs; e = min(s+fs, nt)
        fr = portfolio.iloc[s:e]
        fm = compute_metrics(fr)
        if fm:
            print(f"  Fold {fold+1} ({fr.index[0].date()} to {fr.index[-1].date()}): "
                  f"Sharpe={fm['sharpe']:.3f}  AnnRet={fm['ann_ret']*100:+.2f}%")

    # Autocorrelation
    print(f"\n  Autocorr(1): {portfolio.autocorr(1):.4f}")
    print(f"  Autocorr(5): {portfolio.autocorr(5):.4f}")

    # Save
    rd = DATA_DIR / "results"; rd.mkdir(exist_ok=True)
    portfolio.to_csv(rd/"strategy_v5_returns.csv", header=["return"])
    (1+portfolio).cumprod().to_csv(rd/"strategy_v5_cumulative.csv", header=["cumulative"])
    print(f"\nSaved to {rd}")


if __name__ == "__main__":
    main()
