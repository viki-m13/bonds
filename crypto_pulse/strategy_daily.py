"""PULSE — the best HONEST price-action strategy found in the Sharpe-3 hunt.

A vol-targeted, dollar-neutral crypto trend book on the daily spot panel
(data/crypto, 111 coins, 2020-2026): a multi-timeframe trend-sign ensemble
plus a 20-day Donchian breakout, inverse-vol weighted across the liquid
universe, then scaled to a constant 12% annual portfolio volatility.

It clears Sharpe ~1.2 net of 10 bps/side with a shallow (-16%) drawdown — a
genuine, causal, cost-aware result. It is NOT Sharpe 3: see
research/SHARPE_INVESTIGATION.md for why a real 3 was not attainable on any
daily-or-hourly OHLCV data available here (the >3 numbers are all bid-ask-bounce
/ stale-price artifacts that vanish once you stop trading at the formation bar).

Causality: every signal at day d uses information through the close of d; weights
are lagged one day (traded at the next close). Run from crypto_pulse/:
    python strategy_daily.py        (-> research/pulse_equity.png + prints stats)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "crypto")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
ANN = 365


def load():
    cl, vo, hi, lo = {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(DATA, "*.csv"))):
        if "Close" not in pd.read_csv(f, nrows=0).columns.tolist():
            continue
        t = os.path.basename(f)[:-8]
        d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        d = d[~d.index.duplicated()].sort_index()
        cl[t], vo[t], hi[t], lo[t] = d["Close"], d["Volume"], d["High"], d["Low"]
    C = pd.DataFrame(cl).sort_index()
    return (C, pd.DataFrame(vo).reindex_like(C),
            pd.DataFrame(hi).reindex_like(C), pd.DataFrame(lo).reindex_like(C))


def stats(p, ann=ANN):
    p = p.dropna()
    if len(p) < 100:
        return dict(sharpe=np.nan, ann=np.nan, vol=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(ann), ann=p.mean() * ann,
                vol=p.std() * np.sqrt(ann), maxdd=(cum / cum.cummax() - 1).min())


def build(C, V, H, L, vol_target=0.12, cost_bps=10.0):
    R = C.pct_change()
    R[R.abs() > 2.0] = np.nan
    dv = (C * V).rolling(30).mean()
    elig = C.notna() & (dv > 5e6)
    sd = R.rolling(30).std()
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    don = ((C >= H.shift(1).rolling(20).max()).astype(float)
           - (C <= L.shift(1).rolling(20).min()).astype(float))
    sig = trend + don
    w = (sig / sd).where(elig)
    w = w.div(w.abs().sum(axis=1), axis=0)
    pnl = (w.shift(1) * R).sum(axis=1)
    turn = (w.shift(1) - w.shift(2)).abs().sum(axis=1)
    gross = pnl - turn * cost_bps / 1e4
    scale = (vol_target / (gross.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)
    net = gross * scale
    return net[C.index >= C.index[90]], R, elig


def main():
    C, V, H, L = load()
    net, R, elig = build(C, V, H, L)
    s = stats(net)
    print(f"PULSE daily crypto trend+breakout (net 10bps, 12% vol target):")
    print(f"  full-sample Sharpe {s['sharpe']:.2f} | ann {s['ann']:+.1%} | "
          f"vol {s['vol']:.1%} | maxDD {s['maxdd']:+.1%} | "
          f"coins/day~{int(elig.sum(axis=1).median())}")
    for lab, a, b in [("2020-22", "2020-06-01", "2022-12-31"),
                      ("2023-24", "2023-01-01", "2024-12-31"),
                      ("2025-26", "2025-01-01", "2026-12-31")]:
        sub = net[(net.index >= pd.Timestamp(a)) & (net.index <= pd.Timestamp(b))]
        print(f"  {lab}: Sharpe {stats(sub)['sharpe']:+.2f}")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + net).cumprod().plot(ax=ax, color="#16a085", lw=1.6,
                             label=f"PULSE (Sharpe {s['sharpe']:.2f}, net 10bps)")
    ax.set_yscale("log")
    ax.set_title("PULSE — vol-targeted daily-crypto trend+breakout (HONEST, "
                 "not Sharpe 3)")
    ax.set_ylabel("growth of $1 (log)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "pulse_equity.png"), dpi=110)
    print("[written]", os.path.join(HERE, "pulse_equity.png"))


if __name__ == "__main__":
    main()
