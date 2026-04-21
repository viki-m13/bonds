"""ATLAS — drawdown-hardened TSMOM LETF strategy.

Step 1 of 6: produce the daily returns CSV so downstream factsheet generators
can consume it without re-running the strategy.

Strategy: TSMOM K=3m tv=15% on {SPY,QQQ,TLT,GLD} via {UPRO,TQQQ,TMF,UGL}
         + DD-throttle tight (-5/-10/-20) with 252d peak window.

Outputs to data/results/atlas_returns.csv: daily returns (Date, ret).
"""
from pathlib import Path
import pandas as pd

from letf_tsmom import tsmom_with_vol_target
from letf_dd_throttle import apply_dd_throttle
from atlas_ext_prep import extended_prep


OUT = Path("/home/user/bonds/data/results")


def build_atlas_returns():
    tsmom_px = extended_prep()
    base, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.15)
    atlas, mult = apply_dd_throttle(
        base,
        peak_window=252,
        dd_start=-0.05, dd_mid=-0.10, dd_floor=-0.20,
        w_start=1.0, w_mid=0.5, w_floor=0.25,
        smooth_days=5, tc_bps=15,
    )
    atlas.name = "ret"
    atlas.index.name = "Date"
    return atlas, mult


def main():
    atlas, mult = build_atlas_returns()
    out_ret = OUT / "atlas_returns.csv"
    atlas.to_csv(out_ret)
    (mult.rename("mult")).to_csv(OUT / "atlas_multiplier.csv")
    print(f"Wrote {out_ret}")
    print(f"  {len(atlas)} daily obs from {atlas.index[0].date()} to {atlas.index[-1].date()}")
    nav = (1 + atlas).cumprod()
    print(f"  NAV 1.0 -> {nav.iloc[-1]:.2f}")
    print(f"  Ann ret: {atlas.mean()*252*100:.2f}%   Ann vol: {atlas.std()*(252**0.5)*100:.2f}%")
    dd = (nav / nav.cummax() - 1)
    print(f"  Max DD:  {dd.min()*100:.2f}%")


if __name__ == "__main__":
    main()
