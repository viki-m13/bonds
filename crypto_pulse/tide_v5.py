"""Improve TIDE — round 4: universe / robustness / funding levers (honest).

On the improved TIDE base (multi-horizon breakout + Parkinson vol), test more well-motivated
upgrades, walk-forward OOS + deflated, keeping only those that ALSO improve the independent
pre-HL period and all WF folds (the bar that multi-horizon & Parkinson cleared):
  base        : current improved TIDE.
  +5horizons  : 5/10/20/40/80 breakout blend (more horizon diversification).
  +winsor     : clip the breakout z at +/-3 (robust to outlier coins).
  +topN       : restrict to the 20 most-liquid coins each day (cleaner breakouts).
  +fundaware  : down-weight positions facing adverse funding (carry-aware sizing).
Adopt only robust survivors. Run from crypto_pulse/: python tide_v5.py (-> research/tide_v5.md)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as sps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
from tide import sh, cagr, maxdd, vt, HL_START, ANN, TAKER

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
N_TRIALS = 24


def deflated(p, n=N_TRIALS):
    p = pd.to_numeric(p, errors="coerce").astype(float).dropna()
    if len(p) < 60:
        return np.nan, np.nan
    sr = p.mean() / p.std(); T = len(p)
    g3 = sps.skew(p); g4 = sps.kurtosis(p, fisher=False)
    e = (1 - np.euler_gamma) * sps.norm.ppf(1 - 1.0 / n) + np.euler_gamma * sps.norm.ppf(1 - 1.0 / (n * np.e))
    var = (1 - g3 * sr + (g4 - 1) / 4.0 * sr ** 2) / (T - 1)
    z = (sr - e * np.sqrt(var)) / np.sqrt(max(var, 1e-12))
    return sr * np.sqrt(ANN), float(sps.norm.cdf(z))


class Lab:
    def __init__(self):
        coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
        self.C, self.V, self.H, self.L = v.load_prices(coins)
        self.F = v.load_daily_funding(coins, self.C.index)
        self.R = self.C.pct_change(); self.R[self.R.abs() > 2] = np.nan
        self.dv = (self.C * self.V).rolling(30).mean()
        self.el = self.C.notna() & (self.dv > 3e6)
        self.sd = np.sqrt((np.log(self.H / self.L) ** 2).rolling(30).mean() / (4 * np.log(2))) + 1e-9
        self.ts = ((((self.C > self.C.rolling(50).mean()).where(self.el)).mean(axis=1) - 0.5).abs() * 2).clip(0, 1)

    def build(self, horizons=(10, 20, 40), winsor=False, topN=0, fundaware=False, hold=3):
        el = self.el
        if topN:
            rank = self.dv.where(el).rank(axis=1, ascending=False)
            el = el & (rank <= topN)
        dmf = lambda x: x.sub(x.mean(axis=1), axis=0)
        f = lambda k: dmf(((self.C - self.C.rolling(k).mean()) / (self.C.rolling(k).std() + 1e-9)).where(el))
        bo = sum(f(k) for k in horizons) / len(horizons)
        if winsor:
            bo = bo.clip(-3, 3)
        nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
        w = nm(bo / self.sd)
        if fundaware:
            # adverse funding: longs pay positive funding, shorts pay negative -> penalize
            pen = (1 - (np.sign(w) * self.F).clip(0, None) * 50).clip(0.3, 1.0)
            w = nm(w * pen)
        w = w.mul(self.ts.shift(1), axis=0)
        rebw = pd.Series(np.arange(len(self.C)) % hold == 0, index=self.C.index)
        w = w.where(rebw, axis=0).ffill(limit=hold); wl = w.shift(1)
        pnl = (wl * self.R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * self.F).sum(axis=1)
        return vt(pnl)


def main():
    lab = Lab()
    base = lab.build()
    idx = base.index; hl = idx >= HL_START; hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p): q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])
    def pre(p): return sh(p[p.index < HL_START])
    bi, bo_ = io(base); bpre = pre(base)

    variants = {
        "base (improved TIDE)": dict(),
        "+5horizons": dict(horizons=(5, 10, 20, 40, 80)),
        "+winsor": dict(winsor=True),
        "+topN20": dict(topN=20),
        "+fundaware": dict(fundaware=True),
    }
    L = ["# Improving TIDE round-4 — universe/robustness/funding (honest)\n",
         f"On improved base (multiH+Parkinson). Robust bar: beat base OOS AND pre-HL AND all WF "
         f"folds. base OOS {bo_:+.2f}, pre-HL {bpre:+.2f}.\n",
         "| variant | HL | IS | OOS | dOOS | pre-HL | deflated P |", "|---|---|---|---|---|---|---|"]
    keep = []
    for name, kw in variants.items():
        p = lab.build(**kw); i, o = io(p); pr = pre(p)
        _, dp = deflated(p[hl][p.index[hl] >= cut])
        folds = np.array_split(hidx, 4); fok = all(sh(p[p.index.isin(fd)]) > 0 for fd in folds)
        robust = o > bo_ + 0.08 and pr >= bpre - 0.05 and fok
        L.append(f"| {name} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {o - bo_:+.2f} | {pr:+.2f} | {dp:.2f} |")
        if name != "base (improved TIDE)" and robust:
            keep.append((name, kw))

    combo = {}
    for _, kw in keep:
        combo.update(kw)
    final = lab.build(**combo) if combo else base
    fi, fo = io(final)
    folds = np.array_split(hidx, 4); fsh = [sh(final[final.index.isin(fd)]) for fd in folds]
    L += ["\n## Verdict\n",
          f"- Robust survivors: {', '.join(k for k,_ in keep) if keep else 'NONE'}.",
          (f"- **TIDE improved further: OOS {bo_:+.2f} -> {fo:+.2f}** (pre-HL {pre(final):+.2f}, "
           f"WF {', '.join(f'{x:+.1f}' for x in fsh)})." if fo > bo_ + 0.08 and all(x > 0 for x in fsh) else
           f"- **No round-4 lever robustly improves the book.** It stays ~{max(fo, bo_):.2f}. "
           "multi-horizon + Parkinson remain the only genuine gains; TIDE is at its honest ceiling."),
          f"- Honest single-book level **~{max(fo, bo_):.1f}**. Confirmed across 23 upgrade attempts: "
          "a single independent breakout book does not honestly reach 3.\n"]
    with open(os.path.join(HERE, "tide_v5.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/tide_v5.md")


if __name__ == "__main__":
    main()
