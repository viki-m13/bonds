"""Research-backed enhancement tests, IS/OOS-gated so overfits are caught.

Ideas (each modifies only the risk-on score; regime + bear sleeve unchanged):
  * Frog-in-the-Pan (Da-Gulen-Hwang 2014): prefer momentum from smooth,
    continuous paths (low information-discreteness).
  * Volatility-managed momentum (Barroso-Santa-Clara 2015 / Daniel-Moskowitz
    2016): ONLY in high-vol regimes, tilt the leaders toward lower beta — crash
    protection that doesn't touch normal bull-market concentration.
  * Momentum x low-vol double sort.
  * 52-week-high / fresh-high refinements.
  * Acceleration (2nd-derivative momentum).
A real winner must beat baseline on the full grid AND hold up in BOTH the
2006-2014 and 2015-2023 start eras.
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
close, vol = P["close"], P["volume"]
ret = close.pct_change(fill_method=None)
spy = data.load_benchmark("SPY")["Close"].reindex(close.index).ffill()
spyret = spy.pct_change()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def xr(df):
    return df.rank(axis=1, pct=True)


SK = 21
mom_mh = sum(xr(close.shift(SK).pct_change(h - SK, fill_method=None))
             for h in (63, 126, 189, 252))
mom_mh_r = xr(mom_mh)
size = xr((close * vol).rolling(63).mean())
vol120 = ret.rolling(120).std()
dist52 = close / close.rolling(252).max()

# Frog-in-the-Pan information discreteness (low = smooth path)
ret252 = close.pct_change(252, fill_method=None)
upfrac = (ret > 0).rolling(252).mean()
ID = np.sign(ret252) * (1 - 2 * upfrac)        # negative ID = continuous winner

# rolling beta vs SPY (252d)
cov = ret.mul(spyret, axis=0).rolling(252).mean().sub(
    ret.rolling(252).mean().mul(spyret.rolling(252).mean(), axis=0), axis=0)
var = spyret.rolling(252).var()
beta = cov.div(var, axis=0)

# acceleration: recent 63d momentum minus prior 63d
accel = (close.pct_change(63, fill_method=None)
         - close.shift(63).pct_change(63, fill_method=None))

# high-vol regime (VIX percentile, causal)
def _vix():
    df = pd.read_csv(os.path.join(ROOT, "data", "fred", "VIXCLS.csv"))
    df.columns = ["date", "v"]
    df["date"] = pd.to_datetime(df["date"])
    s = pd.to_numeric(df.set_index("date")["v"], errors="coerce")
    return s.reindex(close.index).ffill()
vix = _vix()
vixpct = vix.rolling(756, min_periods=252).rank(pct=True)
highvol = (vixpct > 0.70).to_numpy()

WS = S.W_SIZE_BULL


def compose(bull):
    bear = S.bear_scores(P)
    off = S.risk_off(P).to_numpy()[:, None]
    return pd.DataFrame(np.where(off, bear.to_numpy(float), bull.to_numpy(float)),
                        index=bull.index, columns=bull.columns)


base = mom_mh_r + WS * size


def vol_managed(beta_pen=0.5):
    """In high-vol rows, tilt leaders toward lower beta; else baseline."""
    normal = base.to_numpy(float)
    defensive = (xr(mom_mh_r - beta_pen * xr(beta)) + WS * size).to_numpy(float)
    out = np.where(highvol[:, None], defensive, normal)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


VARIANTS = {
    "BASELINE": base,
    "FIP_blend": xr(mom_mh_r + 0.5 * xr(-ID)) + WS * size,
    "FIP_strong": xr(mom_mh_r + 1.0 * xr(-ID)) + WS * size,
    "FIP_gate_smooth": (base).where(ID < ID.median(axis=1).values[:, None] if False else xr(-ID) > 0.4),
    "double_lowvol_0.2": xr(mom_mh_r - 0.2 * xr(vol120)) + WS * size,
    "fresh_high_gate": (base).where(dist52 > 0.85),
    "mom_x_dist52_0.3": xr(mom_mh_r + 0.3 * xr(dist52)) + WS * size,
    "accel_blend": xr(mom_mh_r + 0.3 * xr(accel)) + WS * size,
    "volmanaged_0.5": vol_managed(0.5),
    "volmanaged_1.0": vol_managed(1.0),
    "FIP+volmanaged": None,   # filled below
}
VARIANTS["FIP+volmanaged"] = None


def gated(name, scores):
    import json
    c = protocol.evaluate_signal(scores, "tmpg", k=2, every=10, save=True,
                                 quiet=True)
    w = pd.DataFrame(json.load(open(os.path.join(
        protocol.RESULTS_DIR, "tmpg.json")))["windows"])
    g = w[~w["window"].isin(protocol.REGIMES)].copy()
    g["ys"] = g["start"].str[:4].astype(int)
    is_, oos = g[g.ys <= 2014], g[g.ys >= 2015]
    return (c["win_qqq"], c["med_vs_qqq"], c["worst_vs_qqq"], c["full_mult"],
            (is_["vs_qqq"] > 0).mean(), is_["vs_qqq"].median(), is_["vs_qqq"].min(),
            (oos["vs_qqq"] > 0).mean(), oos["vs_qqq"].median(), oos["vs_qqq"].min())


if __name__ == "__main__":
    # FIP + vol-managed combo
    fip_bull = xr(mom_mh_r + 0.5 * xr(-ID)) + WS * size
    defensive = xr(xr(mom_mh_r + 0.5 * xr(-ID)) - 0.5 * xr(beta)) + WS * size
    VARIANTS["FIP+volmanaged"] = pd.DataFrame(
        np.where(highvol[:, None], defensive.to_numpy(float),
                 fip_bull.to_numpy(float)), index=close.index, columns=close.columns)
    print(f"{'variant':18} {'win':>4} {'med':>6} {'worst':>7} {'full':>6} | "
          f"{'IS win/med/wst':>18} | {'OOS win/med/wst':>18}")
    for name, bull in VARIANTS.items():
        r = gated(name, compose(bull))
        print(f"{name:18} {r[0]*100:3.0f}% {r[1]*100:+5.1f}% {r[2]*100:+6.1f}% "
              f"{r[3]:5.1f}x | {r[4]*100:3.0f}% {r[5]*100:+5.1f}% {r[6]*100:+6.1f}% "
              f"| {r[7]*100:3.0f}% {r[8]*100:+5.1f}% {r[9]*100:+6.1f}%")
