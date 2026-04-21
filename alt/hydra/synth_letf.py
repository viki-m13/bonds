"""Synthetic LETF series — extend LETF history back pre-inception.

A daily LETF return is well-approximated (Avellaneda-Zhang 2010) by:
    r_letf_t = L × r_underlying_t - (L - 1) × rf_t / 252 - expense / 252
where:
    L            = leverage (2 for UGL, 3 for UPRO/TQQQ/TMF)
    rf_t         = fed-funds overnight rate at day t
    expense      = annual expense ratio (decimal)

For dates where the ACTUAL LETF trades, we use the actual LETF return.
For dates BEFORE inception, we use the synthetic formula above with data
from the underlying and FRED's FEDFUNDS series.

Usage:
    series = build_synth_letf("UPRO")   # returns a price series indexed by date
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF_DIR = ROOT / "data/etfs"
FRED_DIR = ROOT / "data/fred"


LETF_SPEC = {
    # letf    : (underlying, leverage, annual expense ratio)
    "UPRO": ("SPY", 3, 0.0091),
    "TQQQ": ("QQQ", 3, 0.0084),
    "TMF":  ("TLT", 3, 0.0106),
    "UGL":  ("GLD", 2, 0.0095),
}


def _load_etf(ticker: str) -> pd.Series:
    df = pd.read_csv(ETF_DIR / f"{ticker}.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    return df[col].astype(float).dropna()


def _load_fedfunds() -> pd.Series:
    f = FRED_DIR / "FEDFUNDS.csv"
    df = pd.read_csv(f, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    s = df["FEDFUNDS"].astype(float) / 100.0
    return s


def _calibration_drag(letf_ticker: str, years: int = 3) -> float:
    """Compute annual drag needed so uncorrected-synth matches real over first
    `years` years of real LETF history. This captures missing swap spread,
    borrow costs, dividend-timing noise and trading frictions that the simple
    formula doesn't model. Applied as a DAILY drag = drag_ann / 252 subtracted
    from synth returns pre-inception."""
    real_px = _load_etf(letf_ticker)
    real_ret = real_px.pct_change().dropna()
    synth_uncorr = _synth_letf_returns_uncorrected(letf_ticker)
    end = real_ret.index[0] + pd.DateOffset(years=years)
    idx = real_ret.loc[:end].index.intersection(synth_uncorr.index)
    if len(idx) < 100:
        return 0.0
    gap = (synth_uncorr.loc[idx].mean() - real_ret.loc[idx].mean()) * 252
    return float(gap)  # positive gap means synth is too high -> subtract it


def _synth_letf_returns_uncorrected(letf_ticker: str) -> pd.Series:
    und, L, exp_ratio = LETF_SPEC[letf_ticker]
    und_px = _load_etf(und)
    und_ret = und_px.pct_change().dropna()
    rf_monthly = _load_fedfunds()
    rf = rf_monthly.reindex(und_ret.index, method="ffill").fillna(0.02)
    daily_rf = rf / 252.0
    return L * und_ret - (L - 1) * daily_rf - exp_ratio / 252.0


def synth_letf_returns(letf_ticker: str) -> pd.Series:
    """Daily returns of synthetic LETF, calibrated so 3-yr real overlap matches."""
    raw = _synth_letf_returns_uncorrected(letf_ticker)
    drag_ann = _calibration_drag(letf_ticker)
    return raw - drag_ann / 252.0


def build_synth_letf(letf_ticker: str) -> pd.Series:
    """Return a price series: real LETF from inception onward, synthetic before.

    NAV normalized so real LETF's first-day price = real LETF's first-day
    price (by construction we always start synthetic at NAV 1.0, then splice
    to actual LETF at its inception using actual LETF's first price).
    """
    real_px = _load_etf(letf_ticker)
    synth_ret = synth_letf_returns(letf_ticker)

    # Use synth returns BEFORE real LETF inception; real returns after.
    real_first = real_px.index[0]
    synth_pre = synth_ret.loc[:real_first - pd.Timedelta(days=1)]
    real_ret = real_px.pct_change().dropna().loc[real_first:]

    # Splice: start synth at 1.0, accumulate through pre-period, then level-shift
    # so that at the splice the NAV equals real_px.iloc[0].
    synth_nav = (1 + synth_pre).cumprod()
    if len(synth_nav) == 0:
        return real_px
    splice_value = real_px.iloc[0]  # first actual-LETF price
    pre_nav = synth_nav * (splice_value / synth_nav.iloc[-1])
    # Scale so the last synth day → splice_value / (1 + first real return)
    # Actually: we want the day BEFORE real inception = splice_value / (1 + real_ret.iloc[0])
    # so that applying real_ret.iloc[0] gives real_px.iloc[0].
    first_real_ret = real_ret.iloc[0]
    target_pre_last = real_px.iloc[0] / (1 + first_real_ret)
    pre_nav = synth_nav * (target_pre_last / synth_nav.iloc[-1])

    combined = pd.concat([pre_nav, real_px])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def check_correlation(letf_ticker: str, overlap_years: int = 3):
    """Sanity check: synth vs real returns correlation on the first N years of real."""
    und, L, exp_ratio = LETF_SPEC[letf_ticker]
    real_px = _load_etf(letf_ticker)
    synth_ret = synth_letf_returns(letf_ticker)
    real_ret = real_px.pct_change().dropna()
    end = real_ret.index[0] + pd.DateOffset(years=overlap_years)
    s = synth_ret.loc[real_ret.index[0]:end]
    r = real_ret.loc[real_ret.index[0]:end]
    idx = s.index.intersection(r.index)
    if len(idx) < 20:
        return None
    corr = np.corrcoef(s.loc[idx], r.loc[idx])[0, 1]
    tracking = (r.loc[idx] - s.loc[idx]).std() * np.sqrt(252)
    return {
        "letf": letf_ticker,
        "overlap_days": len(idx),
        "corr": round(float(corr), 4),
        "annual_tracking_err": round(float(tracking * 100), 3),
        "ann_ret_real": round(float(r.loc[idx].mean() * 252 * 100), 2),
        "ann_ret_synth": round(float(s.loc[idx].mean() * 252 * 100), 2),
    }


if __name__ == "__main__":
    print("Synthetic LETF sanity check (first 3 years of real LETF):")
    print()
    print(f"{'LETF':6s} {'days':>6s} {'corr':>7s} {'TE(ann%)':>9s} "
          f"{'real%':>7s} {'synth%':>7s}")
    for t in LETF_SPEC:
        c = check_correlation(t)
        if c is None:
            print(f"{t}: insufficient overlap")
            continue
        print(f"{c['letf']:6s} {c['overlap_days']:>6d} {c['corr']:>7.4f} "
              f"{c['annual_tracking_err']:>8.2f} "
              f"{c['ann_ret_real']:>7.2f} {c['ann_ret_synth']:>7.2f}")

    print()
    print("Splice results (first and last 2 prices):")
    for t in LETF_SPEC:
        s = build_synth_letf(t)
        print(f"\n{t}: {s.index[0].date()} .. {s.index[-1].date()}  ({len(s)} days)")
        print(f"  first: {s.iloc[0]:.4f}   last: {s.iloc[-1]:.4f}")
