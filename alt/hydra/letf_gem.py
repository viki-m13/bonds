"""Priority 2e — Dual momentum / GEM (Antonacci).

Global Equity Momentum (Antonacci 2015):
  - If SPY 12m > BIL 12m (absolute mom): go equity sleeve
    - Within equity, pick whichever has higher 12m return: SPY or ACWX/EFA (intl)
  - Else: go long bonds (AGG/TLT)

For our LETF extension:
  - Absolute mom on SPY vs BIL: if SPY wins, go UPRO
  - (Alternative) equity picks between UPRO, TQQQ, SOXL: whichever underlying
    (SPY, QQQ, SOXX) has highest 12-1 month return
  - Bond sleeve: TMF (use TLT 12m vs BIL 12m for the switch)

PRE-REGISTERED: lookback 252d, skip 21d (12-1), monthly rebal.
Three variants:
  (a) GEM-lite: SPY-vs-BIL -> UPRO or TMF (if SPY vs BIL fails equity absolute)
  (b) GEM-3: among {SPY, QQQ, SOXX} with absolute-gate, -> leveraged variant
  (c) GEM-5: among {SPY, QQQ, SOXX, TLT, GLD} — "asset-class momentum"
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import summarise
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")


UND_TO_LETF = {
    "SPY": "UPRO",
    "QQQ": "TQQQ",
    "TLT": "TMF",
    "GLD": "UGL",
}


def load_prices(tickers):
    frames = {}
    for t in tickers:
        s = load_etf(t)
        if s is not None:
            frames[t] = s
    return pd.DataFrame(frames).sort_index()


def gem_backtest(
        px, underlyings, lookback=252, skip=21,
        rebal_days=21, tc_bps=15,
        cash="BIL", bond_sleeve="TLT", bond_letf="TMF"):
    """Absolute+relative momentum -> leveraged expression.

    For each rebal date:
      1. Compute 12-1 month return of each underlying and of cash (BIL).
      2. If max underlying 12-1 return > cash 12-1 return:
           pick best underlying -> go 100% into its LETF
         Else:
           if bond 12-1 > cash: go 100% TMF
           else: go 100% BIL
    """
    n = len(px)
    idx = px.index
    all_letfs = list(UND_TO_LETF.values()) + [cash, bond_letf]
    cols = [c for c in px.columns if c in all_letfs + underlyings + [bond_sleeve, cash]]
    px = px[cols]
    rets = px.pct_change().fillna(0)
    W = pd.DataFrame(0.0, index=idx, columns=px.columns)

    for i in range(0, n, rebal_days):
        if i < lookback + 10:
            W.iloc[i, px.columns.get_loc(cash)] = 1.0
            continue
        slc = px.iloc[i - lookback:(i - skip if skip > 0 else i)]
        # 12-1 returns per asset
        mom = (slc.iloc[-1] / slc.iloc[0] - 1)
        cash_mom = mom.get(cash, 0)
        und_mom = mom[underlyings].dropna()
        bond_mom = mom.get(bond_sleeve, 0)
        if len(und_mom) == 0:
            W.iloc[i, px.columns.get_loc(cash)] = 1.0
            continue
        best_und = und_mom.idxmax()
        best_und_mom = und_mom.max()
        if best_und_mom > cash_mom:
            # Go into the LETF version of the winner
            letf = UND_TO_LETF.get(best_und, best_und)
            if letf in px.columns:
                W.iloc[i, px.columns.get_loc(letf)] = 1.0
            else:
                W.iloc[i, px.columns.get_loc(cash)] = 1.0
        elif bond_mom > cash_mom:
            # Defensive: bonds
            if bond_letf in px.columns:
                W.iloc[i, px.columns.get_loc(bond_letf)] = 1.0
            else:
                W.iloc[i, px.columns.get_loc(cash)] = 1.0
        else:
            W.iloc[i, px.columns.get_loc(cash)] = 1.0

    W = W.replace(0, np.nan).ffill().fillna(0)
    W_eff = W.shift(1).fillna(0)
    tc = W_eff.diff().abs().sum(axis=1).fillna(0) * (tc_bps / 1e4)
    port_ret = (W_eff * rets).sum(axis=1) - tc
    return port_ret, W_eff


def main():
    # We need underlyings PLUS LETFs PLUS cash+bond
    tickers = list(UND_TO_LETF.keys()) + list(UND_TO_LETF.values()) + ["BIL", "TLT"]
    px = load_prices(tickers)
    # Common window that includes LETFs
    px = px.dropna(subset=list(UND_TO_LETF.values()), how="any")
    px = px.loc["2011-01-01":]
    print(f"Window: {px.index[0].date()} .. {px.index[-1].date()} "
          f"({len(px)} days)")

    configs = [
        ("GEM-1 (SPY only) lb=252", ["SPY"], 252, 21),
        ("GEM-1 lb=126", ["SPY"], 126, 21),
        ("GEM-2 (SPY/QQQ) lb=252", ["SPY","QQQ"], 252, 21),
        ("GEM-2 lb=126", ["SPY","QQQ"], 126, 21),
        ("GEM-2 lb=63", ["SPY","QQQ"], 63, 21),
        ("GEM-4 (SPY/QQQ/TLT/GLD) lb=252", ["SPY","QQQ","TLT","GLD"], 252, 21),
        ("GEM-4 lb=126", ["SPY","QQQ","TLT","GLD"], 126, 21),
        ("GEM-4 lb=63", ["SPY","QQQ","TLT","GLD"], 63, 21),
    ]
    rows = []
    for name, unds, lb, sk in configs:
        r, _ = gem_backtest(px, unds, lookback=lb, skip=sk)
        s = summarise(r, name)
        rows.append(s)
        print(f"  {name}: CAGR={s['cagr']}% SR={s['sharpe']} MDD={s['mdd']}%")

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_gem.csv", index=False)
    print("\n" + df.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
