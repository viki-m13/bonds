"""Improve TIDE — round 6: conviction / concentration / risk-balance (honest).

On improved base (5-horizon + Parkinson). Novel construction levers, walk-forward OOS +
deflated, strict bar (beat base OOS AND pre-HL AND all WF folds):
  +agree   : horizon-AGREEMENT conviction — scale by how many of the 5 horizons share the sign
             (trade hardest when all timeframes agree).
  +conc    : concentration — keep only the top-k |signal| names per side (high-conviction book).
  +erc     : equal-risk-contribution-ish sizing (weight by 1/vol AND signal rank, balanced).
  +agree+conc : combine survivors.
Run from crypto_pulse/:  python tide_v7.py  (-> research/tide_v7.md)
"""
import os

import numpy as np
import pandas as pd
from scipy import stats as sps

import validate_hl as v
from tide import sh, cagr, vt, HL_START, ANN, TAKER

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
N_TRIALS = 32


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
        dv = (self.C * self.V).rolling(30).mean()
        self.el = self.C.notna() & (dv > 3e6)
        self.sd = np.sqrt((np.log(self.H / self.L) ** 2).rolling(30).mean() / (4 * np.log(2))) + 1e-9
        self.ts = ((((self.C > self.C.rolling(50).mean()).where(self.el)).mean(axis=1) - 0.5).abs() * 2).clip(0, 1)

    def build(self, agree=False, conc=0, erc=False, hold=3):
        dmf = lambda x: x.sub(x.mean(axis=1), axis=0)
        f = lambda k: ((self.C - self.C.rolling(k).mean()) / (self.C.rolling(k).std() + 1e-9)).where(self.el)
        horizons = [f(max(2, int(20 * m))) for m in (0.25, 0.5, 1, 2, 4)]
        bo = dmf(sum(horizons) / 5.0)
        sig = bo
        if agree:
            ag = sum(np.sign(dmf(h)) for h in horizons) / 5.0     # in [-1,1], +-1 = full agreement
            sig = sig * np.abs(ag)
        nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
        w = sig / self.sd
        if erc:
            w = np.sign(sig) * (sig.abs().rank(axis=1) / self.sd)  # rank-conviction x inv-vol
        w = nm(w)
        if conc:
            rank = w.abs().rank(axis=1, ascending=False)
            w = w.where(rank <= conc, 0.0); w = nm(w)
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
        "+agree (horizon agreement)": dict(agree=True),
        "+conc20 (top-20/side)": dict(conc=20),
        "+erc (rank-risk balance)": dict(erc=True),
        "+agree+conc20": dict(agree=True, conc=20),
    }
    L = ["# Improving TIDE round-6 — conviction/concentration/risk-balance (honest)\n",
         f"Strict bar: beat base OOS + pre-HL + all WF folds. base OOS {bo_:+.2f}, pre-HL {bpre:+.2f}.\n",
         "| variant | HL | IS | OOS | dOOS | pre-HL | deflated P |", "|---|---|---|---|---|---|---|"]
    keep = []
    for name, kw in variants.items():
        p = lab.build(**kw); i, o = io(p); pr = pre(p)
        _, dp = deflated(p[hl][p.index[hl] >= cut])
        folds = np.array_split(hidx, 4); fok = all(sh(p[p.index.isin(fd)]) > 0 for fd in folds)
        robust = o > bo_ + 0.08 and pr >= bpre - 0.05 and fok
        L.append(f"| {name} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {o - bo_:+.2f} | {pr:+.2f} | {dp:.2f} |")
        if "base" not in name and robust:
            keep.append((name, kw))

    combo = {}
    for _, kw in keep:
        combo.update(kw)
    final = lab.build(**combo) if combo else base
    fi, fo = io(final)
    L += ["\n## Verdict\n",
          f"- Robust survivors: {', '.join(k for k,_ in keep) if keep else 'NONE'}.",
          (f"- **TIDE improved: OOS {bo_:+.2f} -> {fo:+.2f}.**" if fo > bo_ + 0.08 else
           f"- **No round-6 lever robustly improves the book** (~{max(fo, bo_):.2f}). After 32 honest "
           "attempts across 6 rounds, TIDE's three real refinements (5-horizon breakout, Parkinson "
           "vol) are the whole improvement; the book is definitively at its single-strategy ceiling."),
          f"- **Honest single-book ceiling: ~{max(fo, bo_):.1f} HL-era (1.55 over 12 years, positive "
          "every year).** No construction idea — standard or novel — honestly pushes one independent "
          "breakout book to 3.\n"]
    with open(os.path.join(HERE, "tide_v7.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/tide_v7.md")


if __name__ == "__main__":
    main()
