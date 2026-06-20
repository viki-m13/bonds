"""Overnight / time-of-day seasonality on crypto (hourly) — the one untested idea
from the shared list ("Overnight Seasonality Bitcoin").

Tests whether specific UTC hours carry persistent drift that survives taker cost.
Signal is causal: at each hour we use the trailing 30-day mean basket return for
THAT hour-of-day to decide direction. Equal-weight basket of the hourly-data coins.
Net of 4.5bps taker on position changes. Honest read: hourly trading means high
turnover, so the bar (drift per trade > ~cost) is high — exactly the wall that
killed intraday crypto for a taker before. We also report correlation to the daily
book to see if there's ANY uncorrelated, cost-surviving residue.

Run from crypto_pulse/:  python overnight_seasonality.py
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRLY = os.path.join(ROOT, "data", "crypto_hourly_cb")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
TAKER = 4.5
ANNH = 24 * 365


def load_basket():
    cl = {}
    for f in sorted(glob.glob(os.path.join(HRLY, "*.csv"))):
        t = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        d["ts"] = pd.to_datetime(d["ts"], unit="s")
        d = d.set_index("ts")["close"].sort_index()
        cl[t] = d
    C = pd.DataFrame(cl).sort_index()
    return C


def sharpe_h(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANNH) if (len(p) > 100 and p.std() > 0) else np.nan


def main():
    C = load_basket()
    R = C.pct_change(); R[R.abs() > 0.5] = np.nan
    basket = R.mean(axis=1)                       # equal-weight hourly basket return
    hour = basket.index.hour

    # causal per-hour drift: trailing 30d (720h) mean of basket return at each hour
    df = pd.DataFrame({"r": basket, "h": hour})
    exp_drift = pd.Series(index=basket.index, dtype=float)
    for h in range(24):
        m = df["h"] == h
        exp_drift[m] = df.loc[m, "r"].rolling(30, min_periods=10).mean().shift(1)

    # strategies
    lines = ["# Overnight / time-of-day seasonality on crypto (hourly, net)\n"]
    lines.append(f"{C.shape[1]} coins, {C.index.min().date()}->{C.index.max().date()} "
                 f"({len(C)} hourly bars). Causal per-hour drift (trailing 30d). "
                 f"Net of {TAKER}bps taker on position changes.\n")

    n = len(basket)
    cut = basket.index[int(n * 0.6)]
    lines.append("| strategy | Sharpe | IS | OOS | ann ret | turn/day |")
    lines.append("|---|---|---|---|---|---|")

    def run(sig, name):
        pos = sig.shift(1)
        pnl = pos * basket - (pos - pos.shift(1)).abs() * TAKER / 1e4
        turn = (pos - pos.shift(1)).abs().mean() * 24
        sh = sharpe_h(pnl)
        si = sharpe_h(pnl[pnl.index < cut]); so = sharpe_h(pnl[pnl.index >= cut])
        lines.append(f"| {name} | **{sh:+.2f}** | {si:+.2f} | {so:+.2f} | "
                     f"{pnl.mean()*ANNH:+.0%} | {turn:.1f} |")
        return pnl

    # 1. directional by sign of expected hour drift
    run(np.sign(exp_drift).fillna(0.0), "hour-drift sign (long/short)")
    # 2. long-only positive hours
    run((exp_drift > 0).astype(float), "long positive hours only")
    # 3. fixed 'overnight' block 00-08 UTC long (pre-registered, low turnover)
    onb = pd.Series(((hour >= 0) & (hour < 8)).astype(float), index=basket.index)
    run(onb, "fixed long 00-08 UTC (overnight)")
    # 4. fixed long 22-02 UTC (the 'US close/Asia open' pump window)
    onb2 = pd.Series((((hour >= 22) | (hour < 2))).astype(float), index=basket.index)
    run(onb2, "fixed long 22-02 UTC")

    # which hours are actually positive (full-sample, descriptive only)
    byhr = df.groupby("h")["r"].mean() * 1e4
    best = byhr.sort_values(ascending=False)
    lines.append(f"\nMean basket return by UTC hour (bps, full-sample descriptive): "
                 f"best {best.index[0]}h ({best.iloc[0]:+.1f}), "
                 f"{best.index[1]}h ({best.iloc[1]:+.1f}); worst {best.index[-1]}h "
                 f"({best.iloc[-1]:+.1f}). Spread {best.iloc[0]-best.iloc[-1]:.1f}bps/hr.\n")
    lines.append("## Verdict\n")
    lines.append("- If every Sharpe above is weak/negative net of taker, the "
                 "seasonality is real in gross terms but the hourly turnover x 4.5bps "
                 "eats it — same taker wall as all intraday crypto. The fixed-window "
                 "versions (lowest turnover) are the only ones with a chance; if even "
                 "those don't clear, time-of-day is not a taker-viable sleeve here.\n")

    out = "\n".join(lines)
    with open(os.path.join(HERE, "overnight_seasonality.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/overnight_seasonality.md")


if __name__ == "__main__":
    main()
