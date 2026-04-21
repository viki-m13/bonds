"""Phase 3 validation — rolling 2-year Sharpe & MDD of candidates.

Deployability question: how often does a client sitting in this product
see a 2-year window with SR < 0 or MDD > 30%? That's when they quit.

Strategies compared:
  - base:    TSMOM K=3m tv=15%
  - tight:   + DD-throttle tight (-5/-10/-20)
  - wide:    + DD-throttle wide  (-10/-20/-30)   ← default winner
  - SPY:     buy-and-hold SPY (humility)
  - 60/40:   SPY/TLT 60/40 (humility)

Metrics per window:
  - SR, MDD, vol, CAGR
  - % of 2-yr windows with SR<0
  - % of 2-yr windows with MDD worse than -30%
  - worst 2-yr window's MDD
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import common_window_returns
from letf_crypto_universe import load_with_crypto
from letf_tsmom import tsmom_with_vol_target, prep as tsmom_prep
from letf_dd_throttle import apply_dd_throttle
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")
WIN_DAYS = 504  # ~2 years


def rolling_stats(r, win=WIN_DAYS):
    r = r.dropna()
    idx = r.index
    out = []
    for i in range(win, len(r)):
        w = r.iloc[i - win:i]
        nav = (1 + w).cumprod()
        peak = nav.cummax()
        mdd = (nav / peak - 1).min()
        mu = w.mean() * 252
        sd = w.std() * np.sqrt(252)
        sr = mu / sd if sd > 0 else 0.0
        out.append({
            "end": idx[i], "sr": float(sr), "mdd": float(mdd),
            "vol": float(sd), "cagr": float(nav.iloc[-1] ** (252 / len(w)) - 1),
        })
    return pd.DataFrame(out).set_index("end")


def main():
    tsmom_px = tsmom_prep()
    px = load_with_crypto([], start="2011-01-01")
    rets = common_window_returns(px)

    base, _ = tsmom_with_vol_target(tsmom_px, K_months=3, target_vol=0.15)
    tight, _ = apply_dd_throttle(base, dd_start=-0.05, dd_mid=-0.10, dd_floor=-0.20)
    wide, _ = apply_dd_throttle(base)  # defaults: -10/-20/-30

    # Humility benchmarks
    spy = load_etf("SPY").pct_change().dropna()
    tlt = load_etf("TLT").pct_change().dropna()
    idx = spy.index.intersection(tlt.index)
    spy = spy.loc[idx]
    tlt = tlt.loc[idx]
    sixty40 = 0.6 * spy + 0.4 * tlt

    strats = {
        "TSMOM base":      base.reindex(rets.index).fillna(0),
        "TSMOM + DD-wide": wide.reindex(rets.index).fillna(0),
        "TSMOM + DD-tight": tight.reindex(rets.index).fillna(0),
        "SPY":             spy,
        "60/40 SPY/TLT":   sixty40,
    }

    summary_rows = []
    for name, r in strats.items():
        rs = rolling_stats(r)
        if rs.empty:
            continue
        row = {
            "strategy": name,
            "N_windows": len(rs),
            "SR_median": rs["sr"].median(),
            "SR_p10":    rs["sr"].quantile(0.10),
            "SR_p90":    rs["sr"].quantile(0.90),
            "pct_SR_neg":       100 * (rs["sr"] < 0).mean(),
            "MDD_median":       100 * rs["mdd"].median(),
            "MDD_worst":        100 * rs["mdd"].min(),
            "pct_MDD_worse_30": 100 * (rs["mdd"] < -0.30).mean(),
            "pct_MDD_worse_50": 100 * (rs["mdd"] < -0.50).mean(),
            "CAGR_median":      100 * rs["cagr"].median(),
        }
        summary_rows.append(row)
        rs.to_csv(OUT / f"letf_rolling_{name.replace(' ', '_').replace('/', '-').replace('+', 'plus')}.csv")

    df = pd.DataFrame(summary_rows)
    df.to_csv(OUT / "letf_rolling_summary.csv", index=False)
    print("2-year rolling-window deployability (higher is worse for 'pct' cols):")
    print()
    cols_fmt = {
        "strategy": "<20", "N_windows": ">7d",
        "SR_median": ">7.2f", "SR_p10": ">7.2f", "SR_p90": ">7.2f",
        "pct_SR_neg": ">10.1f",
        "MDD_median": ">10.1f", "MDD_worst": ">10.1f",
        "pct_MDD_worse_30": ">10.1f", "pct_MDD_worse_50": ">10.1f",
        "CAGR_median": ">10.1f",
    }
    print(df.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    print("\nReading guide:")
    print("  pct_SR_neg = % of 2yr windows where SR<0 (client feels lost year)")
    print("  pct_MDD_worse_30 = % of windows with drawdown worse than -30% (client panics)")
    print("  MDD_worst = single worst 2-yr window MDD (worst client vintage)")


if __name__ == "__main__":
    main()
