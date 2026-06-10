"""MOSAIC candidate — strategy_v10's stream framework MINUS the un-investable
LETF-shorting engines (vdecay_*, bbpair_*).

Keeps: credit carry, dividend carry, defensive equity, commodity carry,
preferred/loan, REIT, intl bond, FX carry, sector hedges, cross-asset TSMOM.
Selection: trailing-Sharpe adaptive weights (causal, 252d window, monthly).
Also tests a no-selection equal-risk version (less overfit risk).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import importlib.util
spec = importlib.util.spec_from_file_location("sv10", ROOT / "scripts/strategy_v10.py")
sv10 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv10)


def sr(r):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 60 and r.std() > 0 else np.nan


def show(r, label):
    r = r.dropna()
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    print(f"  {label:46s} IS={sr(r.loc[:'2018']):5.2f} OOS={sr(r.loc['2019':]):5.2f} "
          f"full={sr(r):5.2f} vol={r.std()*np.sqrt(252)*100:4.1f}% mdd={mdd*100:5.1f}%")
    return r


def main():
    prices, fred = sv10.load_all_data()
    ret = prices.pct_change()
    streams = sv10.generate_all_streams(ret, fred)
    drop = [k for k in streams if k.startswith(("vdecay", "bbpair"))]
    for k in drop:
        del streams[k]
    print(f"streams after dropping LETF shorts: {len(streams)}")

    # 1) original adaptive portfolio (trailing-sharpe selection, causal)
    port, n_active = sv10.adaptive_portfolio(streams, fred)
    show(port, f"MOSAIC adaptive (avg {n_active:.0f} active)")

    # 2) no-selection equal-risk: every stream scaled to 3% vol, mean
    df = pd.DataFrame(streams)
    scaled = pd.DataFrame(index=df.index)
    for c in df.columns:
        rv = df[c].rolling(63, min_periods=21).std() * np.sqrt(252)
        scaled[c] = df[c] * (0.03 / rv.clip(lower=0.003)).clip(0.1, 8).shift(1)
    eqr = scaled.mean(axis=1)
    show(eqr.dropna(), "MOSAIC equal-risk (no selection)")

    # 3) corr with phoenix
    phx = pd.read_csv(ROOT / "data/results/phoenix_production_returns.csv",
                      parse_dates=["Date"]).set_index("Date")["net_ret"]
    for nm, p in [("adaptive", port), ("eq-risk", eqr)]:
        idx = p.dropna().index.intersection(phx.index)
        print(f"  corr(MOSAIC {nm}, PHOENIX) = {np.corrcoef(p.loc[idx], phx.loc[idx])[0,1]:.2f}")

    port.dropna().rename("ret").to_csv(ROOT / "phoenix5/results/mosaic_adaptive.csv")
    eqr.dropna().rename("ret").to_csv(ROOT / "phoenix5/results/mosaic_eqrisk.csv")


if __name__ == "__main__":
    main()
