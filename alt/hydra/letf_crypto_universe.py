"""Step 7 — build merged LETF + crypto price/return tables.

Crypto trades 7d/week; LETFs trade business days only. We reindex crypto
prices to the LETF business-day calendar (Mon-Fri). pct_change then gives
the Fri→Mon ("weekend") return as the Monday crypto return, which is the
right convention when trading crypto alongside stocks (you only rebalance
on business days anyway).

Two windows:
  BTC-only   : 2015-01-01 onward   (all 17 LETFs active; BTC has 10+ yrs)
  BTC + ETH  : 2018-01-01 onward   (~8 yrs; ETH first listed Nov 2017)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf
from letf_universe import LETF_LONG_2011


def load_with_crypto(extra_crypto, start=None):
    """Load LETF_LONG_2011 + crypto tickers, aligning crypto to LETF biz-day index."""
    frames = {}
    for t in LETF_LONG_2011:
        s = load_etf(t)
        if s is not None:
            frames[t] = s
    letf_df = pd.DataFrame(frames).sort_index()
    biz_idx = letf_df.dropna(how="any").index

    # Reindex crypto to biz-day index. We use the crypto close ON the biz date.
    # If a biz-date has no crypto entry (shouldn't happen since crypto trades
    # daily), ffill.
    for c in extra_crypto:
        cs = load_etf(c)
        if cs is None:
            continue
        # Align crypto to biz index using nearest-before (Fri carries over if
        # biz date somehow missing).
        cs_biz = cs.reindex(biz_idx, method="ffill")
        letf_df[c] = cs_biz

    df = letf_df.loc[biz_idx].dropna(how="any")
    if start is not None:
        df = df.loc[start:]
    return df


def build_window(extra_crypto, start):
    px = load_with_crypto(extra_crypto, start=start)
    rets = px.pct_change().fillna(0)
    print(f"Window: {rets.index[0].date()} .. {rets.index[-1].date()} "
          f"({len(rets)} days); cols: {list(rets.columns)}")
    return px, rets


if __name__ == "__main__":
    print("=== LETF only (2011) ===")
    px1, r1 = build_window([], "2011-01-01")
    print("\n=== LETF + BTC (2015) ===")
    px2, r2 = build_window(["BTC_USD"], "2015-01-01")
    print("\n=== LETF + BTC + ETH (2018) ===")
    px3, r3 = build_window(["BTC_USD", "ETH_USD"], "2018-01-01")
