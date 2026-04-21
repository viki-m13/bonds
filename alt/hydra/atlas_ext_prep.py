"""Extended-history price table for ATLAS.

Replaces letf_tsmom.prep() with a version that:
  - splices SYNTHETIC LETFs (underlying × leverage − financing − expense) onto
    real LETFs for dates pre-inception (via synth_letf.build_synth_letf)
  - splices a SYNTHETIC BIL (cumulative DGS1MO / 252) onto real BIL for
    dates pre-inception (2005-2007)

Common start date is set by the latest-starting UNDERLYING (GLD: 2004-11-18,
so usable start ≈ 2005-01-03 after 1-month warm-up).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from hydra_core import load_etf
from synth_letf import build_synth_letf


ROOT = Path("/home/user/bonds")
FRED_DIR = ROOT / "data/fred"

PAIRS = [("SPY", "UPRO"), ("QQQ", "TQQQ"), ("TLT", "TMF"), ("GLD", "UGL")]
BIL = "BIL"
START = "2005-01-03"


def build_synth_bil() -> pd.Series:
    """Synthesize BIL price from 1-month Treasury yield for pre-BIL dates."""
    real_bil = load_etf("BIL")
    df = pd.read_csv(FRED_DIR / "DGS1MO.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    y = pd.to_numeric(df["DGS1MO"], errors="coerce") / 100.0  # annual yield
    y = y.ffill().dropna()
    # Daily return = y/252, compound
    daily = y / 252.0
    # Align daily to business days
    bdays = pd.date_range(daily.index[0], daily.index[-1], freq="B")
    daily = daily.reindex(bdays, method="ffill").fillna(0)
    synth_nav = (1 + daily).cumprod()
    # Splice: set synth NAV at day before real BIL inception = real BIL first / (1 + first real ret)
    real_first = real_bil.index[0]
    real_ret = real_bil.pct_change().dropna()
    if len(real_ret) == 0:
        return real_bil
    first_real_ret = real_ret.iloc[0]
    target_pre_last = real_bil.iloc[0] / (1 + first_real_ret)
    pre = synth_nav.loc[:real_first - pd.Timedelta(days=1)]
    if len(pre) == 0:
        return real_bil
    pre = pre * (target_pre_last / pre.iloc[-1])
    combined = pd.concat([pre, real_bil])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined


def extended_prep() -> pd.DataFrame:
    """Return price DataFrame with columns: underlyings + synth-spliced LETFs + BIL."""
    frames = {}
    for und, letf in PAIRS:
        frames[und] = load_etf(und)
        frames[letf] = build_synth_letf(letf)
    frames[BIL] = build_synth_bil()
    px = pd.DataFrame(frames).sort_index()
    letfs = [l for _, l in PAIRS]
    px = px.dropna(subset=letfs + [BIL], how="any")
    px = px.loc[START:]
    return px


if __name__ == "__main__":
    px = extended_prep()
    print(f"Extended-history universe: {px.index[0].date()} .. {px.index[-1].date()} "
          f"({len(px)} rows, {len(px) / 252:.1f} years)")
    print()
    print("First 3 rows:")
    print(px.head(3).round(3))
    print("\nLast 3 rows:")
    print(px.tail(3).round(3))
