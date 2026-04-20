"""NOVA23 — Regime-conditional OVERNIGHT drift (Lou-Polk-Skouras 2019,
Kelly 2022), gated by intraday realized-vol regime from 5-min bars.

Published finding: the entire US equity drift historically occurs between
the close and the next open (15:55 → 09:30). However, post-2016 the
effect has decayed on average, BUT remains strong conditional on being
in a LOW-VOL regime. High-vol regimes have negative overnight drift
(liquidity-driven fire sales at open).

Strategy (FIXED a priori):
  For each trading day:
    1. Compute trailing 20-day realized volatility (annualised) from
       5-min bars on SPY.
    2. Gate:  rv_20d < 0.15  →  hold long SPY overnight (15:55 → 09:30)
              rv_20d ≥ 0.15  →  cash (BIL overnight)
    3. Intraday (09:30 → 15:55) always cash.
    4. 2 bps round-trip TC on overnight legs only.

Uses:
  - 5-min intraday bars (for exact RV)
  - 15:55 close price (for overnight entry)
  - 09:30 open price (for overnight exit)
  - BIL daily proxy for off-day carry

Low TC, no leverage, no short, no vol scaling — a DISCRETE
low-vol-regime-on / high-vol-regime-off calendar anomaly."""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import load_etf, stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
TC_BPS = 2.0
RV_CUT = 0.15


def five_min_rv(ticker):
    df = pd.read_csv(INTRA / f"{ticker}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["logret"] = np.log(df["close"]).diff()
    first_of_day = df["date"] != df["date"].shift(1)
    df.loc[first_of_day, "logret"] = 0.0
    rv = df.groupby("date")["logret"].apply(lambda x: np.sqrt(np.sum(x ** 2)))
    rv.index = pd.to_datetime(rv.index)
    return rv * np.sqrt(252)


def overnight_returns(ticker):
    df = pd.read_csv(INTRA / f"{ticker}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["time"] = df["ts"].dt.time
    px_1555 = df[df["time"] == pd.to_datetime("15:55").time()].set_index("date")["close"]
    px_1555 = px_1555[~px_1555.index.duplicated(keep="first")].sort_index()
    px_1555.index = pd.to_datetime(px_1555.index)
    open_ = df.groupby("date")["open"].first()
    open_.index = pd.to_datetime(open_.index)
    common = px_1555.index.intersection(open_.index)
    px_1555 = px_1555.loc[common]
    open_ = open_.loc[common]
    # Overnight return on day t = (open_{t+1} / close_{t}) - 1, indexed at t+1
    ovn = (open_.shift(-1) / px_1555) - 1
    return ovn.dropna()


def main():
    rv = five_min_rv("SPY")
    ovn = overnight_returns("SPY")
    bil = load_etf("BIL")
    bil_ret = bil.pct_change().fillna(0)
    bil_ret.index = pd.to_datetime(bil_ret.index)

    # Align on overlapping dates
    common = rv.index.intersection(ovn.index).intersection(bil_ret.index)
    rv = rv.loc[common]
    ovn = ovn.loc[common]
    bil_ret = bil_ret.loc[common]

    rv20 = rv.rolling(20).mean().shift(1)   # use info up to t-1
    gate = (rv20 < RV_CUT)                  # True = go long overnight
    gate_prev = gate.shift(1)

    # On day t: if gate_{t-1} True we entered overnight yesterday ⇒ realized
    # this morning. Payoff is ovn_{t-1} (from shift definition). We already
    # built ovn on day-of-realization basis (t = when open realized).
    # So r_t = ovn_t if gate_{t} True (gate computed with info up to t-1), else 0
    r = pd.Series(0.0, index=common)
    r[gate] = ovn[gate]
    r[~gate] = bil_ret[~gate]   # cash in BIL on non-active nights (small)

    # TC on entry/exit (any time state changes)
    changes = (gate != gate.shift(1)).astype(int)
    r = r - changes * (TC_BPS / 1e4)

    warm = pd.Timestamp("2016-03-01")
    r_v = r.loc[warm:]
    exposure = gate.loc[warm:].mean()
    print(f"NOVA23 — conditional overnight drift, RV<{RV_CUT} gate")
    print(f"Long-overnight exposure: {exposure * 100:.1f}% of days")

    s = stats(r_v, "NOVA23 cond-overnight")
    print(f"\n{s['label']:30s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(r_v.loc[:CUT], "IS <2022"), (r_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Grid over RV_CUT to see sensitivity
    print("\nRV_CUT sensitivity (full sample SR):")
    for cut in [0.10, 0.12, 0.13, 0.15, 0.18, 0.20, 0.25]:
        gate_c = (rv20 < cut)
        rc = pd.Series(0.0, index=common)
        rc[gate_c] = ovn[gate_c]
        rc[~gate_c] = bil_ret[~gate_c]
        ch = (gate_c != gate_c.shift(1)).astype(int)
        rc = rc - ch * (TC_BPS / 1e4)
        rcv = rc.loc[warm:]
        is_s = stats(rcv.loc[:CUT], "")["sharpe"]
        oos_s = stats(rcv.loc[CUT:], "")["sharpe"]
        full = stats(rcv, "")
        print(f"  RV<{cut:.2f}: exposure={gate_c.loc[warm:].mean()*100:>5.1f}%  "
              f"Full={full['sharpe']:>5.2f}  IS={is_s:>5.2f}  OOS={oos_s:>5.2f}  "
              f"Vol={full['vol']:>5.2f}%")

    # Also the unconditional overnight drift as benchmark
    bench = ovn.loc[warm:].copy()
    bs = stats(bench, "UNCOND overnight drift")
    print(f"\nUnconditional {bs['label']:22s} SR={bs['sharpe']:>5.2f}  "
          f"Ret={bs['ret']:>6.2f}%  Vol={bs['vol']:>5.2f}%")
    for p, tag in [(bench.loc[:CUT], "IS"), (bench.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        print(f"  Uncond {ss['label']}: SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%")

    ann = r_v.groupby(r_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA23": r})
    out.to_csv("/home/user/bonds/data/results/nova23_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova23_returns.csv")


if __name__ == "__main__":
    main()
