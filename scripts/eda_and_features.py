#!/usr/bin/env python3
"""
Bond Market EDA & Feature Engineering
======================================
Loads all bond ETF, treasury, and FRED data. Computes a comprehensive feature
matrix for a systematic bond trading strategy and saves to parquet.
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

# ── 26 core (non-iBond) ETFs ────────────────────────────────────────────────
CORE_ETFS = [
    "SHY", "IEI", "IEF", "TLH", "TLT", "GOVT", "SPTL", "VGLT",   # treasury
    "LQD", "VCIT", "VCSH", "IGIB",                                   # IG corp
    "HYG", "JNK", "USHY",                                            # HY
    "AGG", "BND",                                                     # aggregate
    "TIP", "SCHP",                                                    # TIPS
    "MUB", "VTEB",                                                    # muni
    "EMB", "VWOB",                                                    # EM
    "FLOT",                                                           # floating
    "MBB", "VMBS",                                                    # MBS
]

# Category mapping for relative-value features
CATEGORY = {}
meta = pd.read_csv(os.path.join(DATA, "etfs", "_metadata.csv"))
for _, row in meta.iterrows():
    CATEGORY[row["ticker"]] = row["category"]


# ============================================================================
# 1. Load ETF data
# ============================================================================
def load_etfs():
    """Load Close and Volume for core ETFs, aligned by date."""
    close_dict, volume_dict = {}, {}
    for tk in CORE_ETFS:
        fp = os.path.join(DATA, "etfs", f"{tk}.csv")
        if not os.path.exists(fp):
            print(f"  [WARN] missing {tk}.csv")
            continue
        df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
        # Use Close (already adjusted based on data inspection)
        close_dict[tk] = df["Close"]
        volume_dict[tk] = df["Volume"]
    close = pd.DataFrame(close_dict).sort_index()
    volume = pd.DataFrame(volume_dict).sort_index()
    return close, volume


# ============================================================================
# 2. Load treasury yield data
# ============================================================================
def load_treasury():
    fp = os.path.join(DATA, "treasury", "daily_treasury_yields.csv")
    df = pd.read_csv(fp)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    # Standardise column names
    rename = {
        "1 Mo": "y1m", "1.5 Month": "y1_5m", "2 Mo": "y2m",
        "3 Mo": "y3m", "4 Mo": "y4m", "6 Mo": "y6m",
        "1 Yr": "y1y", "2 Yr": "y2y", "3 Yr": "y3y",
        "5 Yr": "y5y", "7 Yr": "y7y", "10 Yr": "y10y",
        "20 Yr": "y20y", "30 Yr": "y30y",
    }
    df = df.rename(columns=rename)
    df = df[[c for c in rename.values() if c in df.columns]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill()
    return df


def load_real_yields():
    fp = os.path.join(DATA, "treasury", "daily_treasury_real_yields.csv")
    df = pd.read_csv(fp)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    rename = {"5 YR": "ry5y", "7 YR": "ry7y", "10 YR": "ry10y",
              "20 YR": "ry20y", "30 YR": "ry30y"}
    df = df.rename(columns=rename)
    df = df[[c for c in rename.values() if c in df.columns]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill()
    return df


# ============================================================================
# 3. Load FRED data
# ============================================================================
def load_fred():
    fp = os.path.join(DATA, "fred", "_combined_fred.csv")
    df = pd.read_csv(fp, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.ffill()
    return df


# ============================================================================
# Feature computation helpers
# ============================================================================
def log_returns(prices, periods):
    """Compute log returns over multiple horizons. Returns dict of DataFrames."""
    out = {}
    for p in periods:
        out[f"ret_{p}d"] = np.log(prices / prices.shift(p))
    return out


def realised_vol(log_ret_1d, windows):
    out = {}
    for w in windows:
        out[f"rvol_{w}d"] = log_ret_1d.rolling(w).std() * np.sqrt(252)
    return out


def momentum_signals(ret_dict, close):
    """Cross-sectional rank momentum and time-series momentum."""
    out = {}
    for key in ["ret_21d", "ret_63d", "ret_126d", "ret_252d"]:
        if key not in ret_dict:
            continue
        r = ret_dict[key]
        # Time-series momentum: sign of own return
        out[f"tsmom_{key}"] = np.sign(r)
        # Cross-sectional rank (0-1)
        out[f"xsrank_{key}"] = r.rank(axis=1, pct=True)
    return out


def mean_reversion_signals(close, windows=[21, 63, 126]):
    out = {}
    for w in windows:
        rolling_mean = close.rolling(w).mean()
        rolling_std = close.rolling(w).std()
        out[f"zscore_{w}d"] = (close - rolling_mean) / rolling_std
    return out


def volume_signals(volume, windows=[5, 21]):
    out = {}
    for w in windows:
        roll = volume.rolling(w).mean()
        out[f"relvol_{w}d"] = volume / roll
    # Volume momentum: short-term vs long-term average
    if 5 in windows and 21 in windows:
        out["vol_mom_5_21"] = volume.rolling(5).mean() / volume.rolling(21).mean()
    return out


def yield_curve_features(tsy, windows=[1, 5, 21]):
    """Compute level, slope, curvature from treasury yields."""
    out = {}
    # Key tenors for curve features
    key_cols = [c for c in ["y2y", "y3y", "y5y", "y7y", "y10y", "y20y", "y30y"] if c in tsy.columns]

    # Level: average of available yields
    out["yc_level"] = tsy[key_cols].mean(axis=1)

    # Slope
    if "y10y" in tsy.columns and "y2y" in tsy.columns:
        out["yc_slope_10y2y"] = tsy["y10y"] - tsy["y2y"]
    if "y10y" in tsy.columns and "y3m" in tsy.columns:
        out["yc_slope_10y3m"] = tsy["y10y"] - tsy["y3m"]

    # Curvature: 2*5yr - 2yr - 10yr
    if all(c in tsy.columns for c in ["y2y", "y5y", "y10y"]):
        out["yc_curvature"] = 2 * tsy["y5y"] - tsy["y2y"] - tsy["y10y"]

    # Changes in level/slope/curvature
    base_keys = [k for k in out.keys()]  # snapshot current keys
    for bk in base_keys:
        s = out[bk]
        for w in windows:
            out[f"{bk}_chg{w}d"] = s - s.shift(w)

    return pd.DataFrame(out)


def credit_spread_features(fred, windows=[1, 5, 21]):
    out = {}
    # IG OAS
    if "BAMLC0A0CM" in fred.columns:
        ig = fred["BAMLC0A0CM"]
        out["ig_oas"] = ig
        for w in windows:
            out[f"ig_oas_chg{w}d"] = ig - ig.shift(w)
        out["ig_oas_zscore_63"] = (ig - ig.rolling(63).mean()) / ig.rolling(63).std()

    # HY OAS
    if "BAMLH0A0HYM2" in fred.columns:
        hy = fred["BAMLH0A0HYM2"]
        out["hy_oas"] = hy
        for w in windows:
            out[f"hy_oas_chg{w}d"] = hy - hy.shift(w)
        out["hy_oas_zscore_63"] = (hy - hy.rolling(63).mean()) / hy.rolling(63).std()

    # Spread ratio
    if "ig_oas" in out and "hy_oas" in out:
        out["hy_ig_ratio"] = out["hy_oas"] / out["ig_oas"].replace(0, np.nan)

    # BBB spread
    if "BAMLC0A4CBBB" in fred.columns:
        bbb = fred["BAMLC0A4CBBB"]
        out["bbb_oas"] = bbb
        out["bbb_oas_zscore_63"] = (bbb - bbb.rolling(63).mean()) / bbb.rolling(63).std()

    # AAA spread
    if "BAMLC0A1CAAA" in fred.columns:
        aaa = fred["BAMLC0A1CAAA"]
        out["aaa_oas"] = aaa

    return pd.DataFrame(out)


def cross_asset_features(fred, windows=[1, 5, 21]):
    out = {}
    if "VIXCLS" in fred.columns:
        vix = fred["VIXCLS"]
        out["vix"] = vix
        for w in windows:
            out[f"vix_chg{w}d"] = vix - vix.shift(w)
        out["vix_zscore_63"] = (vix - vix.rolling(63).mean()) / vix.rolling(63).std()

    if "DTWEXBGS" in fred.columns:
        dx = fred["DTWEXBGS"]
        out["dollar_idx"] = dx
        for w in windows:
            out[f"dollar_chg{w}d"] = dx.pct_change(w)

    return pd.DataFrame(out)


def relative_value_features(ret_1d, category_map):
    """ETF return relative to its category average."""
    cats = {}
    for tk in ret_1d.columns:
        cat = category_map.get(tk, "other")
        cats.setdefault(cat, []).append(tk)

    rv = pd.DataFrame(index=ret_1d.index, columns=ret_1d.columns, dtype=float)
    for cat, tickers in cats.items():
        if len(tickers) < 2:
            rv[tickers] = 0.0
            continue
        cat_mean = ret_1d[tickers].mean(axis=1)
        for tk in tickers:
            rv[tk] = ret_1d[tk] - cat_mean
    return rv


def carry_proxy(close, volume):
    """
    Approximate dividend/coupon carry from price-only vs adj-close behaviour.
    Since we only have 'Close' (which IS adjusted), we proxy carry as the
    negative of the drift-adjusted return — i.e., the portion of total return
    not explained by price change over 252d.
    A simpler and more robust proxy: trailing 252d return minus trailing 252d
    price return ≈ income return.  With only adjusted close available, we use
    the smooth rolling 252d return level as an income proxy.
    """
    # Use annualised trailing return as a carry signal (higher = more carry)
    ret252 = np.log(close / close.shift(252))
    vol252 = np.log(close / close.shift(1)).rolling(252).std() * np.sqrt(252)
    # Carry = excess return per unit vol (Sharpe-like carry proxy)
    carry = ret252 / vol252.replace(0, np.nan)
    return carry


# ============================================================================
# Main pipeline
# ============================================================================
def main():
    print("=" * 72)
    print("Bond Market EDA & Feature Engineering")
    print("=" * 72)

    # ── Load data ────────────────────────────────────────────────────────
    print("\n[1] Loading data...")
    close, volume = load_etfs()
    print(f"    ETFs loaded: {close.shape[1]} tickers, "
          f"{close.shape[0]} dates ({close.index[0].date()} to {close.index[-1].date()})")

    tsy = load_treasury()
    print(f"    Treasury yields: {tsy.shape[1]} tenors, {tsy.shape[0]} dates")

    real_y = load_real_yields()
    print(f"    Real yields: {real_y.shape[1]} tenors, {real_y.shape[0]} dates")

    fred = load_fred()
    print(f"    FRED series: {fred.shape[1]} columns, {fred.shape[0]} dates")

    # ── ETF return features ──────────────────────────────────────────────
    print("\n[2] Computing ETF features...")
    return_periods = [1, 5, 21, 63, 126, 252]
    ret_dict = log_returns(close, return_periods)
    ret_1d = ret_dict["ret_1d"]

    rvol_dict = realised_vol(ret_1d, [5, 21, 63])
    mom_dict = momentum_signals(ret_dict, close)
    mr_dict = mean_reversion_signals(close, [21, 63, 126])
    vol_dict = volume_signals(volume, [5, 21])

    # Relative value
    rv = relative_value_features(ret_1d, CATEGORY)

    # Carry proxy
    carry = carry_proxy(close, volume)

    # ── Stack ETF-level features into long format ────────────────────────
    # Panel: (date, ticker) → features
    print("\n[3] Building panel dataset...")
    all_dates = close.index
    tickers = list(close.columns)

    panels = []

    # Returns
    for name, df in ret_dict.items():
        s = df.stack()
        s.name = name
        panels.append(s)

    # Realised vol
    for name, df in rvol_dict.items():
        s = df.stack()
        s.name = name
        panels.append(s)

    # Momentum
    for name, df in mom_dict.items():
        s = df.stack()
        s.name = name
        panels.append(s)

    # Mean reversion
    for name, df in mr_dict.items():
        s = df.stack()
        s.name = name
        panels.append(s)

    # Volume signals
    for name, df in vol_dict.items():
        s = df.stack()
        s.name = name
        panels.append(s)

    # Relative value
    rv_s = rv.stack()
    rv_s.name = "rel_value_1d"
    panels.append(rv_s)

    # Carry
    carry_s = carry.stack()
    carry_s.name = "carry_proxy"
    panels.append(carry_s)

    panel = pd.concat(panels, axis=1)
    panel.index.names = ["Date", "Ticker"]
    print(f"    Panel shape (before macro merge): {panel.shape}")

    # ── Macro / yield-curve / credit features (date-level) ───────────────
    print("\n[4] Computing macro features...")
    yc = yield_curve_features(tsy)
    cr = credit_spread_features(fred)
    ca = cross_asset_features(fred)

    macro = yc.join(cr, how="outer").join(ca, how="outer")
    macro = macro.ffill()
    print(f"    Macro features: {macro.shape[1]} columns")

    # ── Merge macro onto panel ───────────────────────────────────────────
    print("\n[5] Merging panel + macro...")
    panel_reset = panel.reset_index()
    macro_reset = macro.reset_index().rename(columns={"index": "Date"})
    # Ensure Date columns are same type
    panel_reset["Date"] = pd.to_datetime(panel_reset["Date"])
    macro_reset["Date"] = pd.to_datetime(macro_reset["Date"])

    merged = panel_reset.merge(macro_reset, on="Date", how="left")
    merged = merged.set_index(["Date", "Ticker"]).sort_index()

    # ── Drop rows with insufficient history ──────────────────────────────
    # Require at least ret_252d to be non-null (= 252 trading days of data)
    pre = len(merged)
    merged = merged.dropna(subset=["ret_252d"])
    post = len(merged)
    print(f"    Dropped {pre - post} rows with insufficient history "
          f"({post} remaining)")

    # Forward-fill macro columns that may have NaN on ETF trading days
    macro_cols = list(macro.columns)
    merged[macro_cols] = merged[macro_cols].groupby(level="Ticker").ffill()

    # ── Summary statistics ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("SUMMARY STATISTICS")
    print("=" * 72)

    n_dates = merged.index.get_level_values("Date").nunique()
    n_tickers = merged.index.get_level_values("Ticker").nunique()
    print(f"\nFinal panel: {merged.shape[0]} rows  |  "
          f"{n_dates} dates  |  {n_tickers} tickers  |  "
          f"{merged.shape[1]} features")

    date_range = (merged.index.get_level_values("Date").min().date(),
                  merged.index.get_level_values("Date").max().date())
    print(f"Date range: {date_range[0]} to {date_range[1]}")

    print(f"\nFeature columns ({merged.shape[1]}):")
    for i, c in enumerate(merged.columns):
        print(f"  {i+1:3d}. {c}")

    print("\n── Descriptive stats (ETF-level features, sampled) ──")
    sample_cols = [c for c in ["ret_1d", "ret_21d", "ret_252d",
                                "rvol_21d", "rvol_63d",
                                "zscore_63d", "carry_proxy",
                                "rel_value_1d"] if c in merged.columns]
    print(merged[sample_cols].describe().round(4).to_string())

    print("\n── Descriptive stats (macro features, sampled) ──")
    macro_sample = [c for c in ["yc_level", "yc_slope_10y2y", "yc_curvature",
                                 "ig_oas", "hy_oas", "hy_ig_ratio",
                                 "vix", "dollar_idx"] if c in merged.columns]
    # Take one ticker slice to show macro stats
    first_tk = merged.index.get_level_values("Ticker")[0]
    macro_slice = merged.loc[(slice(None), first_tk), macro_sample].droplevel("Ticker")
    print(macro_slice.describe().round(4).to_string())

    # ── Correlation highlights ───────────────────────────────────────────
    print("\n── Correlation highlights ──")
    # Cross-sectional average of ETF features by date, then correlate with macro
    daily_avg = merged.groupby(level="Date")[sample_cols].mean()
    combined = daily_avg.join(macro_slice[macro_sample], how="inner")
    corr = combined.corr()
    # Show macro vs ETF correlations
    print("\nCorrelation of daily cross-sectional mean ETF features vs macro:")
    for mc in macro_sample:
        if mc not in corr.columns:
            continue
        top = corr[mc].drop(macro_sample, errors="ignore").abs().nlargest(3)
        items = ", ".join([f"{k}: {corr[mc][k]:+.3f}" for k in top.index])
        print(f"  {mc:20s} → {items}")

    # ── Pairwise return correlations across ETFs ─────────────────────────
    print("\n── Pairwise 21d return correlations (top-10 most correlated pairs) ──")
    ret21_wide = ret_dict["ret_21d"].dropna(how="all")
    pair_corr = ret21_wide.corr()
    # Extract upper triangle
    mask = np.triu(np.ones_like(pair_corr, dtype=bool), k=1)
    pairs = pair_corr.where(mask).stack().dropna().sort_values(ascending=False)
    for (a, b), v in pairs.head(10).items():
        print(f"  {a:5s} - {b:5s}: {v:.4f}")

    print("\n── Least correlated pairs ──")
    for (a, b), v in pairs.tail(5).items():
        print(f"  {a:5s} - {b:5s}: {v:.4f}")

    # ── Missing data report ──────────────────────────────────────────────
    print("\n── Missing data (% NaN per feature) ──")
    pct_nan = merged.isna().mean().sort_values(ascending=False)
    nonzero = pct_nan[pct_nan > 0]
    if len(nonzero) == 0:
        print("  No missing values!")
    else:
        for c, v in nonzero.head(15).items():
            print(f"  {c:30s}: {v*100:6.2f}%")
        if len(nonzero) > 15:
            print(f"  ... and {len(nonzero)-15} more features with missing data")

    # ── Save ─────────────────────────────────────────────────────────────
    out_path = os.path.join(DATA, "features.parquet")
    try:
        merged.to_parquet(out_path)
        print(f"\n[OK] Saved feature matrix to {out_path}")
    except Exception:
        out_path = os.path.join(DATA, "features.csv")
        merged.to_csv(out_path)
        print(f"\n[OK] Saved feature matrix to {out_path} (CSV fallback)")

    print(f"     Shape: {merged.shape}")
    print("=" * 72)
    print("Done.")
    return merged


if __name__ == "__main__":
    main()
