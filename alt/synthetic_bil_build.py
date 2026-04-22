"""Synthetic BIL (1-3mo T-Bill) pre-2007-05-30 using FEDFUNDS as proxy.

BIL accrues roughly FEDFUNDS - 15bps/y. We build Open/Close as a level that
accrues daily at ff/252 between 2005-01-03 and 2007-05-29, then splice to
real BIL from 2007-05-30 onwards (rescaled so levels match).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/home/user/bonds")
ETF_DIR = ROOT / "data/etfs"
EXT_DIR = ROOT / "data/etfs_extended"
FRED_DIR = ROOT / "data/fred"

START = pd.Timestamp("2005-01-03")


def main():
    real = pd.read_csv(ETF_DIR / "BIL.csv", parse_dates=["Date"]).sort_values("Date").set_index("Date")
    real = real[["Open", "Close"]].astype(float)
    first_real = real.index.min()

    # Daily index = business days between START and first_real - 1
    ff = pd.read_csv(FRED_DIR / "FEDFUNDS.csv", parse_dates=["Date"]).set_index("Date")["FEDFUNDS"].astype(float) / 100.0
    bdates = pd.bdate_range(START, first_real - pd.Timedelta(days=1))
    ff_d = ff.reindex(bdates).ffill()
    if ff_d.isna().any():
        ff_d = ff_d.fillna(0.02)
    # BIL drag ~ 15 bps
    daily = (ff_d - 0.0015) / 252
    level = (1.0 + daily).cumprod()
    # Scale so end of synth matches real open on first_real
    first_open = float(real.loc[first_real, "Open"])
    scale = first_open / level.iloc[-1]
    synth = pd.DataFrame({
        "Open": level.values * scale,
        "Close": level.values * scale,
    }, index=bdates)

    # Combine
    combined = pd.concat([synth, real[real.index >= first_real]])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    out = combined.reset_index().rename(columns={"index": "Date"})
    out["High"] = out[["Open", "Close"]].max(axis=1)
    out["Low"] = out[["Open", "Close"]].min(axis=1)
    out["Volume"] = 0
    out = out[["Date", "Close", "High", "Low", "Open", "Volume"]]
    out.to_csv(EXT_DIR / "BIL.csv", index=False)
    print(f"BIL extended written: {EXT_DIR/'BIL.csv'}  ({len(combined)} rows, starts {combined.index.min().date()})")


if __name__ == "__main__":
    main()
