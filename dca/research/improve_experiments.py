"""Idea sweep to honestly improve SUMMIT — technical / risk-adjusted / blended
selection variants, all scored on the full 244-window grid vs the baseline.

Each variant replaces only the RISK-ON ("bull") score; the regime switch and the
quality-rebound bear sleeve are kept identical to live SUMMIT. Everything is
trailing-only (causal). We print a leaderboard; promising variants then get an
IS/OOS + robustness check separately.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data
import protocol
import strategy_dca as S

P = data.build_panel()
close, vol, high, low = P["close"], P["volume"], P["high"], P["low"]
ret = close.pct_change(fill_method=None)


def xr(df):
    return df.rank(axis=1, pct=True)


# ---- shared trailing features ----
SKIP = 21
mom_mh = sum(xr(close.shift(SKIP).pct_change(h - SKIP, fill_method=None))
             for h in (63, 126, 189, 252))          # base momentum (rank-sum)
mom_mh_r = xr(mom_mh)
size = xr((close * vol).rolling(63).mean())          # mega-cap tilt input
vol60 = ret.rolling(60).std()
vol120 = ret.rolling(120).std()
mom126 = close.shift(SKIP).pct_change(126 - SKIP, fill_method=None)
mom252 = close.shift(SKIP).pct_change(252 - SKIP, fill_method=None)
sharpe126 = ret.rolling(126).mean() / ret.rolling(126).std()
sharpe252 = ret.rolling(252).mean() / ret.rolling(252).std()
dist52 = close / close.rolling(252).max()
above50 = (close > close.rolling(50).mean())
frac50 = above50.rolling(126).mean()
maxret21 = ret.rolling(21).max()


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


rsi14 = rsi(close, 14)


def resid_z():
    """Fair-value residual z vs SPY (the user's example signal), trailing."""
    spy = data.load_benchmark("SPY")["Close"].reindex(close.index).ffill()
    lm = np.log(spy)
    ls = np.log(close)
    w = 756
    mx = lm.rolling(w).mean()
    my = ls.rolling(w).mean()
    cov = ls.mul(lm, axis=0).rolling(w).mean().sub(my.mul(mx, axis=0), axis=0)
    var = (lm * lm).rolling(w).mean() - mx * mx
    beta = cov.div(var + 1e-12, axis=0)
    fair = (my.sub(beta.mul(mx, axis=0))).add(beta.mul(lm, axis=0))
    z = (ls - fair)
    return z / (z.rolling(504).std() + 1e-12)


zfv = resid_z()

WS = S.W_SIZE_BULL  # 5.0


def compose(bull):
    bear = S.bear_scores(P)
    off = S.risk_off(P).to_numpy()[:, None]
    out = np.where(off, bear.to_numpy(float), bull.to_numpy(float))
    return pd.DataFrame(out, index=bull.index, columns=bull.columns)


base_bull = mom_mh_r + WS * size

VARIANTS = {
    "BASELINE": base_bull,
    # --- risk-adjusted selection ---
    "sharpe126_x_size": xr(sharpe126) + WS * size,
    "sharpe252_x_size": xr(sharpe252) + WS * size,
    "mom_blend_sharpe": xr(mom_mh_r + xr(sharpe252)) + WS * size,
    "vol_adj_mom": xr(mom126 / vol120) + WS * size,
    "mom_minus_vol": xr(mom_mh_r - 0.5 * xr(vol120)) + WS * size,
    # --- technical overlays (blended into bull score) ---
    "mom_x_trendqual": xr(mom_mh_r + 0.5 * xr(frac50)) + WS * size,
    "mom_x_dist52": xr(mom_mh_r + 0.5 * xr(dist52)) + WS * size,
    "mom_rsi_pullback": xr(mom_mh_r + 0.3 * xr(-rsi14)) + WS * size,
    "mom_antilottery": xr(mom_mh_r - 0.3 * xr(maxret21)) + WS * size,
    "mom_x_sharpe_trend": xr(mom_mh_r + 0.3 * xr(sharpe252) + 0.3 * xr(frac50)) + WS * size,
    # --- gates (filter eligible names, then mom+size) ---
    "gate_rsi_lt80": (base_bull).where(rsi14 < 80),
    "gate_above50": (base_bull).where(above50),
    "gate_uptrend": (base_bull).where(close > close.rolling(200).mean()),
    # --- mean-reversion (user's z) as bull score or blend ---
    "zfv_meanrev": xr(-zfv) + WS * size,
    "mom_blend_zdip": xr(mom_mh_r + 0.3 * xr(-zfv)) + WS * size,
    # --- size weight / horizon tweaks ---
    "size_w8": mom_mh_r + 8 * size,
    "size_w3": mom_mh_r + 3 * size,
    "mom_12_1_only": xr(mom252) + WS * size,
}


def main():
    base = protocol.evaluate_signal(compose(VARIANTS["BASELINE"]), "b", k=2,
                                    save=False, quiet=True)
    rows = []
    for name, bull in VARIANTS.items():
        c = protocol.evaluate_signal(compose(bull), name, k=2, save=False,
                                     quiet=True)
        rows.append((name, c["win_qqq"], c["win_spy"], c["med_vs_qqq"],
                     c["p10_vs_qqq"], c["worst_vs_qqq"], c["full_mult"]))
    rows.sort(key=lambda r: (-r[1], -r[3]))
    print(f"{'variant':22} {'winQQQ':>7} {'winSPY':>7} {'medQQQ':>8} "
          f"{'p10':>7} {'worst':>8} {'fullx':>7}   vs base")
    for n, wq, ws, md, p10, wr, fm in rows:
        flag = ""
        if wq > base["win_qqq"] + 0.005 and md >= base["med_vs_qqq"] - 0.01:
            flag = "  <== better win"
        if wr > base["worst_vs_qqq"] + 0.005:
            flag += "  <== better worst"
        print(f"{n:22} {wq*100:6.0f}% {ws*100:6.0f}% {md*100:+7.1f}% "
              f"{p10*100:+6.1f}% {wr*100:+7.1f}% {fm:6.1f}{flag}")


if __name__ == "__main__":
    main()
