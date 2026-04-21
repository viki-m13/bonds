"""Priority 3 — vol-neutral benchmarks.

Comparing LETF strategies against 1x SPY is unfair: LETFs run at 30-60%
annualised vol, SPY at ~17%.  A fair Sharpe comparison requires benchmarks
that LIVE IN THE SAME VOL REGIME.

Benchmarks:
  a. SSO buy-hold (2x SPY)            -- ann vol ~30%
  b. UPRO buy-hold (3x SPY)           -- ann vol ~50%
  c. QLD buy-hold (2x QQQ)            -- ann vol ~35%
  d. TQQQ buy-hold (3x QQQ)           -- ann vol ~55%
  e. SPY/TLT 60/40 levered 2x (HFEA-lite via SSO/UBT)
  f. Equal-weight all 17 LETFs (the 'no-selection' LETF portfolio)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import (common_window_returns, run_backtest, summarise, w_fixed)
from letf_crypto_universe import load_with_crypto
from letf_universe import LETF_LONG_2011
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")


def main():
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    rows = []
    # Single-LETF buy-hold
    for t in ["SSO", "UPRO", "QLD", "TQQQ", "TMF", "UBT", "SOXL", "TECL", "UGL"]:
        s = load_etf(t)
        if s is None: continue
        s = s.loc["2011-01-01":"2026-12-31"]
        r = s.pct_change().fillna(0)
        sm = summarise(r, f"{t} buy-hold")
        rows.append(sm)

    # 60/40 with LETFs
    combos = {
        "SSO/UBT 60/40":   {"SSO":0.6, "UBT":0.4},
        "SSO/TMF 60/40":   {"SSO":0.6, "TMF":0.4},
        "QLD/TMF 60/40":   {"QLD":0.6, "TMF":0.4},
        "UPRO/TMF 55/45":  {"UPRO":0.55, "TMF":0.45},
        "EW-all17":        {t: 1/len(LETF_LONG_2011) for t in LETF_LONG_2011},
    }
    for name, w in combos.items():
        r, _ = run_backtest(rets, w_fixed(w), rebal_days=21, exec_lag=1)
        rows.append(summarise(r, name))

    # SPY/TLT 60/40 (unlevered, reference)
    spy = load_etf("SPY").pct_change().fillna(0)
    tlt = load_etf("TLT").pct_change().fillna(0)
    ref = pd.concat([spy, tlt], axis=1).dropna()
    ref.columns = ["SPY","TLT"]
    ref = ref.loc["2011-01-01":"2026-12-31"]
    r = 0.6 * ref["SPY"] + 0.4 * ref["TLT"]
    rows.append(summarise(r, "SPY/TLT 60/40 (1x)"))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_vol_benchmarks.csv", index=False)
    print(df.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
