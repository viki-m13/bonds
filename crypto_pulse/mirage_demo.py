"""Reproducible demonstration of the Sharpe-3 MIRAGE on hourly crypto.

The same cross-sectional short-term reversal signal scores Sharpe ~ +17 when you
(impossibly) trade at the very close used to form it, and FLIPS to ~ -6 the
moment you skip one bar before trading. The entire "edge" is bid-ask bounce /
stale-print reversion on a thin venue — untradeable. This is the canonical
reason most published price-action "Sharpe 3+" claims evaporate.

Run from crypto_pulse/ (needs data/crypto_hourly, rebuild with the fetch in
research/SHARPE_INVESTIGATION.md if absent):  python mirage_demo.py
"""
import numpy as np
import pandas as pd

import data_hourly as dh
import backtest as bt

ANN = dh.ANN


def sharpe(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if len(p) > 100 and p.std() > 0 else np.nan


def main():
    P = dh.load()
    C, R = P["close"], P["ret"]
    elig = C.notna() & R.notna() & (C.shift(24 * 30).notna())
    vol = bt.realized_vol(R, 24 * 3)
    realized = C.shift(-1) / C - 1            # close[t]->close[t+1]
    print("Hourly cross-sectional 3h reversal, net 5 bps:")
    print(f"{'execution':38s} {'Sharpe':>8s} {'ann':>8s}")
    for skip, name in [(0, "trade AT formation close (MIRAGE)"),
                       (1, "skip 1 bar (causal, tradeable)"),
                       (2, "skip 2 bars")]:
        sig = -(C / C.shift(3) - 1)
        s = sig.where(elig)
        z = s.sub(s.mean(axis=1), axis=0)
        w = (z / vol).replace([np.inf, -np.inf], np.nan)
        w = w.div(w.abs().sum(axis=1), axis=0)
        wl = w.shift(skip)
        turn = (wl - wl.shift(1)).abs().sum(axis=1)
        pnl = (wl * realized).sum(axis=1) - turn * 5 / 1e4
        pnl = pnl[pnl.index >= pnl.index[24 * 30]]
        print(f"  skip={skip}  {name:30s} {sharpe(pnl):>+8.2f} {pnl.mean()*ANN:>+7.0%}")
    print("\nThe sign flip between skip=0 and skip=1 IS the artifact: nothing")
    print("real survives removing the one bounce bar. (See SHARPE_INVESTIGATION.md)")


if __name__ == "__main__":
    main()
