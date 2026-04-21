"""Invention 3 — compound multi-signal strategy.

Each asset (LETF) gets a CONSENSUS score from 4 orthogonal signals on its
UNDERLYING. Only invest with exposure proportional to #signals-agreeing.

Signals (per underlying U = SPY/QQQ/TLT/GLD):
  s1. Trend:     U close > 200d SMA                       (binary)
  s2. Momentum:  past 63d return > 0                       (binary)
  s3. Vol-filter: realised 21d vol < 1.5× its 252d median  (binary)
  s4. Carry:     21d return > 63d return / 3               (short-term vs. baseline)

For each asset at rebal, score = sum of signals (0..4). Weight asset in its LETF:
  score=4 -> 1/N  (full)
  score=3 -> 0.75/N
  score=2 -> 0.50/N
  score=1 -> 0.25/N
  score=0 -> 0 (parked in BIL)

Cross-asset: N=4 underlyings (SPY/QQQ/TLT/GLD) -> LETF expression via
UPRO/TQQQ/TMF/UGL. Residual capacity parked in BIL.

Pre-registered: vol target 15% annualised at portfolio level (Moreira-Muir
overlay). Monthly rebal, next-day-open execution, 15bps costs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import summarise
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")

PAIRS = [
    ("SPY", "UPRO"),
    ("QQQ", "TQQQ"),
    ("TLT", "TMF"),
    ("GLD", "UGL"),
]
BIL = "BIL"


def load_all():
    frames = {}
    for u, l in PAIRS:
        frames[u] = load_etf(u)
        frames[l] = load_etf(l)
    frames[BIL] = load_etf(BIL)
    px = pd.DataFrame(frames).sort_index()
    px = px.dropna(subset=[l for _, l in PAIRS], how="any")
    px = px.loc["2011-01-01":]
    return px


def signals_for(und_px_hist):
    """Return dict of per-asset 0/1 signal values using history strictly
    before the decision day."""
    out = {}
    for und, letf in PAIRS:
        s = und_px_hist[und].dropna()
        if len(s) < 260:
            out[letf] = 0
            continue
        close = s.iloc[-1]
        sma200 = s.iloc[-200:].mean()
        trend = int(close > sma200)
        # 63d momentum
        mom = int(s.iloc[-1] / s.iloc[-63] - 1 > 0)
        # vol filter: realized 21d vol < 1.5x its 252d median
        rets = s.pct_change().dropna()
        vol21 = rets.iloc[-21:].std() * np.sqrt(252)
        vol_hist = rets.iloc[-252:].rolling(21).std().dropna() * np.sqrt(252)
        vol_med = vol_hist.median() if len(vol_hist) > 10 else vol21
        vol_ok = int(vol21 < 1.5 * vol_med)
        # short-term vs long-term return
        r21 = s.iloc[-1] / s.iloc[-21] - 1
        r63 = s.iloc[-1] / s.iloc[-63] - 1
        carry = int(r21 > r63 / 3)
        score = trend + mom + vol_ok + carry
        out[letf] = score
    return out


def backtest_multisignal(px, rebal_days=21, tc_bps=15,
                          vol_target=0.15, vol_window=126, vol_cap=3.0):
    idx = px.index
    n = len(idx)
    tickers = list(px.columns)
    rets = px.pct_change().fillna(0)
    W = pd.DataFrame(0.0, index=idx, columns=tickers)

    N_A = len(PAIRS)
    for i in range(0, n, rebal_days):
        if i < 260:
            W.iloc[i, tickers.index(BIL)] = 1.0
            continue
        # Signals use history strictly before day i
        scores = signals_for(px.iloc[:i])
        total = sum(scores.values())
        if total == 0:
            W.iloc[i, tickers.index(BIL)] = 1.0
            continue
        # Raw weights (before vol-target)
        raw = {}
        for und, letf in PAIRS:
            raw[letf] = (scores[letf] / 4.0) / N_A
        # Portfolio ex-ante vol at raw weights
        w_vec = np.array([raw.get(t, 0) for t in tickers])
        window = rets.iloc[max(0, i - vol_window):i]
        if len(window) < 20:
            port_vol_ann = 0.5
        else:
            port_vol_ann = float(np.sqrt(w_vec @ window.cov().values @ w_vec.T)
                                 * np.sqrt(252))
        k = min(vol_target / port_vol_ann, vol_cap) if port_vol_ann > 0 else 0
        for letf in [l for _, l in PAIRS]:
            W.iloc[i, tickers.index(letf)] = raw.get(letf, 0) * k
        # Residual -> BIL
        gross = sum(W.iloc[i].values)
        W.iloc[i, tickers.index(BIL)] = max(0, 1 - gross)

    W = W.replace(0, np.nan).ffill().fillna(0)
    W_eff = W.shift(1).fillna(0)
    tc = W_eff.diff().abs().sum(axis=1).fillna(0) * (tc_bps / 1e4)
    port_ret = (W_eff * rets).sum(axis=1) - tc
    return port_ret, W_eff


def main():
    px = load_all()
    print(f"Window: {px.index[0].date()} .. {px.index[-1].date()} "
          f"({len(px)} days)")

    rows = []
    for tv in (0.12, 0.15, 0.20, 0.25, 0.30):
        for vw in (63, 126, 252):
            r, _ = backtest_multisignal(px, vol_target=tv, vol_window=vw)
            s = summarise(r, f"MultiSig tv={int(tv*100)}% vw={vw}d")
            rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_multisignal.csv", index=False)
    print(df.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
