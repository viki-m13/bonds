"""Intraday trend trading on intraday bars (the honest shot at a smooth curve).

Builds the construction the literature credits with net Sharpe ~2 (Bui & Nguyen
arXiv 2602.11708; Concretum "Catching Crypto Trends"): a multi-timeframe trend +
breakout book on 4h/6h bars across many liquid coins, per-coin inverse-vol
weighted, ATR-trailing-stop exits, scaled to a CONSTANT portfolio volatility
(the main smoothness lever), net of HL 4.5 bps taker.

Smoothness comes from (1) many near-independent bets (intraday bars × many coins),
(2) constant-vol targeting, (3) trailing stops that cut losers fast. Causality:
signal through close of bar t, traded at next bar's open; costs on turnover.

Data: data/crypto_hourly_cb/*.csv (Coinbase hourly, ~2y, ~28 coins).
Run from crypto_pulse/:  python intraday.py  (-> research/intraday.md +
research/intraday_equity.png)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "crypto_hourly_cb")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
TAKER = 4.5


def load(bar="4h"):
    cl, op, hi, lo, vo = {}, {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(DIR, "*.csv"))):
        t = os.path.basename(f)[:-4]
        d = pd.read_csv(f)
        d["ts"] = pd.to_datetime(d["ts"], unit="s")
        d = d[~d["ts"].duplicated()].set_index("ts").sort_index()
        g = d.resample(bar).agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna()
        cl[t], op[t], hi[t], lo[t], vo[t] = (g["close"], g["open"], g["high"],
                                             g["low"], g["volume"])
    C = pd.DataFrame(cl).sort_index()
    idx = C.index
    f = lambda dd: pd.DataFrame(dd).reindex(idx)
    return C, f(op), f(hi), f(lo), f(vo)


def bars_per_year(bar):
    return {"1h": 24 * 365, "2h": 12 * 365, "4h": 6 * 365, "6h": 4 * 365,
            "8h": 3 * 365, "12h": 2 * 365, "1d": 365}[bar]


def stats(p, ann):
    p = p.dropna()
    if len(p) < 100:
        return dict(sharpe=np.nan, ann=np.nan, vol=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(ann), ann=p.mean() * ann,
                vol=p.std() * np.sqrt(ann), maxdd=(cum / cum.cummax() - 1).min())


def trend_book(C, O, H, L, V, bar="4h", lookbacks=(6, 12, 24, 48, 96),
               don=24, cost=TAKER, vt=0.15, atr_stop=0.0):
    """Continuous multi-timeframe trend + Donchian, inverse-vol, directional,
    portfolio vol-targeted. atr_stop>0 adds a trailing-stop overlay."""
    ann = bars_per_year(bar)
    R = C.pct_change()
    R[R.abs() > 1.0] = np.nan
    dv = (C * V).rolling(24).mean()
    elig = C.notna() & (dv > 2e5)
    sd = R.rolling(48).std()
    sig = sum(np.sign(C / C.shift(k) - 1) for k in lookbacks) / len(lookbacks)
    dch = ((C >= H.shift(1).rolling(don).max()).astype(float)
           - (C <= L.shift(1).rolling(don).min()).astype(float))
    raw = (sig + dch)
    if atr_stop > 0:
        tr = pd.concat([(H - L), (H - C.shift(1)).abs(),
                        (L - C.shift(1)).abs()]).groupby(level=0).max()
        atr = tr.rolling(24).mean()
        # flatten a long if price fell atr_stop*ATR from its trailing-max close
        peak = C.rolling(don, min_periods=1).max()
        trough = C.rolling(don, min_periods=1).min()
        long_ok = (peak - C) < atr_stop * atr
        short_ok = (C - trough) < atr_stop * atr
        raw = raw.where(((raw > 0) & long_ok) | ((raw < 0) & short_ok), 0.0)
    w = (raw / sd).where(elig)
    w = w.div(w.abs().sum(axis=1), axis=0)
    wl = w.shift(1)
    pnl = (wl * R).sum(axis=1)
    turn = (wl - wl.shift(1)).abs().sum(axis=1)
    pre = pnl - turn * cost / 1e4
    scale = (vt / (pre.rolling(int(ann / 12)).std() * np.sqrt(ann))).shift(1).clip(0, 4)
    return (pre * scale), ann, turn


def main():
    lines = ["# Intraday trend on intraday bars — honest smooth-curve attempt\n"]
    lines.append(f"Coinbase hourly resampled to intraday bars, multi-timeframe "
                 f"trend + Donchian, inverse-vol, 15% portfolio vol target, net "
                 f"of {TAKER}bps taker. Causality: signal at close of bar, traded "
                 "next bar. IS=first 60% / OOS=last 40%.\n")
    lines.append("| bar | Sharpe | IS | OOS | ann | vol | maxDD | turn/bar |")
    lines.append("|---|---|---|---|---|---|---|---|")
    best = None
    for bar in ("2h", "4h", "6h", "8h", "12h", "1d"):
        C, O, H, L, V = load(bar)
        if C.shape[0] < 300:
            continue
        pnl, ann, turn = trend_book(C, O, H, L, V, bar=bar)
        idx = pnl.index
        cut = idx[int(len(idx) * 0.6)]
        s = stats(pnl, ann)
        sis = stats(pnl[idx < cut], ann)["sharpe"]
        soos = stats(pnl[idx >= cut], ann)["sharpe"]
        lines.append(f"| {bar} | **{s['sharpe']:+.2f}** | {sis:+.2f} | {soos:+.2f}"
                     f" | {s['ann']:+.1%} | {s['vol']:.0%} | {s['maxdd']:+.1%} | "
                     f"{turn.mean():.2f} |")
        if best is None or (not np.isnan(s["sharpe"]) and s["sharpe"] > best[0]):
            best = (s["sharpe"], bar, pnl, ann, cut)
    lines.append("")

    if best is not None:
        _, bar, pnl, ann, cut = best
        # also test the trailing-stop overlay at the best bar
        C, O, H, L, V = load(bar)
        for stp in (2.0, 3.0):
            ps, _, _ = trend_book(C, O, H, L, V, bar=bar, atr_stop=stp)
            idx = ps.index
            s = stats(ps, ann)
            lines.append(f"- {bar} + {stp:.0f}xATR trailing stop: Sharpe "
                         f"{s['sharpe']:+.2f}, maxDD {s['maxdd']:+.1%}")
        lines.append("")
        s = stats(pnl, ann)
        lines.append(f"## Best: {bar} bars — Sharpe {s['sharpe']:.2f}, ann "
                     f"{s['ann']:+.1%}, vol {s['vol']:.0%}, maxDD {s['maxdd']:+.1%}\n")
        fig, ax = plt.subplots(figsize=(11, 5))
        (1 + pnl.fillna(0)).cumprod().plot(ax=ax, color="#16a085", lw=1.8,
            label=f"intraday trend, {bar} bars (Sharpe {s['sharpe']:.2f}, net "
            f"{TAKER}bps)")
        ax.axvline(cut, color="gray", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_title(f"Intraday trend on {bar} bars — Coinbase ~2y, net of taker "
                     "(IS/OOS split dotted)")
        ax.set_ylabel("growth of $1 (log)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "intraday_equity.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "intraday.md"), "w") as fh:
        fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
