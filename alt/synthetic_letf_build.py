"""Synthetic LETF history builder — pre-inception extension to 2005-01-03.

Formula:
    r_letf_synth[t] = L * r_underlying[t]
                      - (L - 1) * FEDFUNDS[t] / 252
                      - 0.0090 / 252             # expense + spread

Produces synthetic (Date, Open, Close) history prior to each LETF's real
inception date and merges it with the real history on/after inception.

Outputs:
    /home/user/bonds/data/etfs_extended/<TICKER>.csv
    /home/user/bonds/data/results/synthetic_letf_history.pkl  (dict of DF)

Also validates correlation with real LETF data where overlap exists.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF_DIR = ROOT / "data/etfs"
FRED_DIR = ROOT / "data/fred"
EXT_DIR = ROOT / "data/etfs_extended"
RES_DIR = ROOT / "data/results"
EXT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2005-01-03")

# Ticker -> (leverage, underlying)
LETF_MAP = {
    "UPRO":  (3.0, "SPY"),
    "SSO":   (2.0, "SPY"),
    "TQQQ":  (3.0, "QQQ"),
    "QLD":   (2.0, "QQQ"),
    "SOXL":  (3.0, "SMH"),
    "TECL":  (3.0, "XLK"),
    "FAS":   (3.0, "XLF"),
    "ERX":   (2.0, "XLE"),
    "DRN":   (3.0, "VNQ"),
    "EDC":   (3.0, "EEM"),
    "YINN":  (3.0, "FXI"),
    "UCO":   (2.0, "USO"),   # USO starts 2006-04-10
    "UGL":   (2.0, "GLD"),
    "NUGT":  (3.0, "GDX"),   # GDX not in dataset — will warn & skip synth
    "TMF":   (3.0, "TLT"),
    "UBT":   (2.0, "TLT"),
    "TYD":   (3.0, "IEF"),
}

EXPENSE_ANN = 0.0090  # 90 bps expense + spread drag
TRADING_DAYS = 252


def load_prices(ticker: str) -> pd.DataFrame | None:
    fp = ETF_DIR / f"{ticker}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"]).set_index("Date")
    return df[["Open", "Close"]].astype(float)


def load_fedfunds(idx: pd.DatetimeIndex) -> pd.Series:
    """Return FEDFUNDS rate as decimal (e.g. 0.05 for 5%), daily ffill."""
    df = pd.read_csv(FRED_DIR / "FEDFUNDS.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.set_index("Date")["FEDFUNDS"].astype(float) / 100.0
    return df.reindex(idx).ffill().fillna(0.02)


def build_synth(ticker: str, L: float, under: str) -> tuple[pd.DataFrame | None, dict]:
    """Return DataFrame with Date index and Open, Close columns combining synth + real.

    Returns (df, info_dict). df may be None if underlying missing.
    """
    info = {"ticker": ticker, "L": L, "under": under}

    real = load_prices(ticker)
    under_df = load_prices(under)
    if real is None:
        info["error"] = f"real LETF {ticker} missing"
        return None, info
    if under_df is None:
        # GDX missing — cannot build synth; just use real data
        info["error"] = f"underlying {under} missing; using real only"
        return real.copy(), info

    real_start = real.index.min()
    info["real_start"] = str(real_start.date())
    info["under_start"] = str(under_df.index.min().date())

    # Build synthetic on dates where underlying exists but before real inception
    idx = under_df.index.sort_values()
    idx = idx[idx >= START_DATE]

    ff = load_fedfunds(idx)
    daily_rf = ff / TRADING_DAYS
    daily_exp = EXPENSE_ANN / TRADING_DAYS

    o = under_df.loc[idx, "Open"]
    c = under_df.loc[idx, "Close"]
    r_close = c.pct_change()
    r_open = o.pct_change()

    r_letf_close = L * r_close - (L - 1.0) * daily_rf - daily_exp
    r_letf_open = L * r_open - (L - 1.0) * daily_rf - daily_exp

    # Cumulate starting from a base of 100 at the first valid date
    # We only need dates strictly before real_start.
    synth_idx = idx[idx < real_start]

    if len(synth_idx) == 0:
        info["synth_days"] = 0
        return real.copy(), info

    # Run cumulative through the full underlying index starting 2005 so the
    # level is well-defined; but we'll splice real after inception to avoid
    # mismatches in magnitude.
    synth_close = pd.Series(index=idx, dtype=float)
    synth_open = pd.Series(index=idx, dtype=float)
    synth_close.iloc[0] = 100.0
    synth_open.iloc[0] = 100.0
    for i in range(1, len(idx)):
        rc = r_letf_close.iloc[i]
        ro = r_letf_open.iloc[i]
        if not np.isfinite(rc):
            rc = 0.0
        if not np.isfinite(ro):
            ro = 0.0
        synth_close.iloc[i] = synth_close.iloc[i - 1] * (1.0 + rc)
        synth_open.iloc[i] = synth_open.iloc[i - 1] * (1.0 + ro)

    # Scale synthetic pre-inception so that its last synthetic value aligns
    # to the first real day's open (no level jump).
    first_real_day = real.loc[real_start]
    # match on close at the last synth date (the day before real_start)
    last_synth_date = synth_idx[-1]
    synth_pre = pd.DataFrame({
        "Open": synth_open.loc[synth_idx].values,
        "Close": synth_close.loc[synth_idx].values,
    }, index=synth_idx)

    # scale by close to continuity with the real open on first real day
    scale_c = float(first_real_day["Open"]) / float(synth_pre["Close"].iloc[-1])
    synth_pre_scaled = synth_pre * scale_c

    # Calibration: correlate daily returns (log) between synth and real on the
    # overlap window (first N=min(3y, available) after real_start)
    corr_close = np.nan
    corr_open = np.nan
    overlap_days = 0
    overlap = idx.intersection(real.index)
    if len(overlap) >= 30:
        # limit to first 3y after real_start
        end3y = overlap.min() + pd.Timedelta(days=3 * 365)
        ov = overlap[(overlap >= overlap.min()) & (overlap <= end3y)]
        if len(ov) >= 30:
            # Build full synth series on the overlap period using the same formula
            ro_u = r_open.loc[ov]
            rc_u = r_close.loc[ov]
            ff_ov = ff.reindex(ov)
            r_l_c = L * rc_u - (L - 1.0) * ff_ov / TRADING_DAYS - EXPENSE_ANN / TRADING_DAYS
            r_l_o = L * ro_u - (L - 1.0) * ff_ov / TRADING_DAYS - EXPENSE_ANN / TRADING_DAYS

            real_ov = real.reindex(ov)
            real_rc = real_ov["Close"].pct_change()
            real_ro = real_ov["Open"].pct_change()

            corr_close = float(pd.Series(r_l_c).corr(real_rc))
            corr_open = float(pd.Series(r_l_o).corr(real_ro))
            overlap_days = len(ov)

    info["synth_days"] = int(len(synth_idx))
    info["corr_close_overlap3y"] = None if np.isnan(corr_close) else round(corr_close, 4)
    info["corr_open_overlap3y"] = None if np.isnan(corr_open) else round(corr_open, 4)
    info["overlap_days"] = overlap_days

    # Combine: synthetic pre-inception + real on/after inception
    real_kept = real[real.index >= real_start]
    combined = pd.concat([synth_pre_scaled, real_kept])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined, info


def main():
    histories: dict[str, pd.DataFrame] = {}
    summary = []
    for ticker, (L, under) in LETF_MAP.items():
        df, info = build_synth(ticker, L, under)
        if df is not None:
            histories[ticker] = df
            out_path = EXT_DIR / f"{ticker}.csv"
            # Keep existing CSV schema (Date,Close,High,Low,Open,Volume)
            out = df.reset_index().rename(columns={"index": "Date"})
            out["High"] = out[["Open", "Close"]].max(axis=1)
            out["Low"] = out[["Open", "Close"]].min(axis=1)
            out["Volume"] = 0
            out = out[["Date", "Close", "High", "Low", "Open", "Volume"]]
            out.to_csv(out_path, index=False)
            info["out"] = str(out_path)
            info["start"] = str(df.index.min().date())
        summary.append(info)
        cc = info.get("corr_close_overlap3y")
        oo = info.get("corr_open_overlap3y")
        print(f"{ticker:5s} L={L} under={under:5s} "
              f"synth_days={info.get('synth_days','-'):>4} "
              f"corr_close={cc} corr_open={oo} "
              f"err={info.get('error','')}")

    # Pickle combined
    with open(RES_DIR / "synthetic_letf_history.pkl", "wb") as fh:
        pickle.dump(histories, fh)

    # JSON summary of correlations
    (RES_DIR / "synthetic_letf_calibration.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nSaved {len(histories)} extended histories to {EXT_DIR}")
    print(f"Pickle: {RES_DIR/'synthetic_letf_history.pkl'}")
    print(f"Calibration JSON: {RES_DIR/'synthetic_letf_calibration.json'}")


if __name__ == "__main__":
    main()
