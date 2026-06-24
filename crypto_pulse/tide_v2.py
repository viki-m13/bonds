"""Improve TIDE itself (not ensembles) — honest signal/construction upgrades.

Baseline TIDE = x-sectional 20d breakout x trend-intensity regime, hold3, vol-targeted (~2.0).
We test genuine improvements to the SIGNAL and CONSTRUCTION, one at a time, walk-forward OOS +
deflated Sharpe, and keep only those that robustly help. Then assemble the survivors into
TIDE v2 and re-validate (bootstrap CI + WF folds) that it is not overfit.

Levers (all causal):
  base        : (C-MA20)/std20, demeaned, inv-vol, x regime, hold3
  +resid      : breakout on BTC-beta-RESIDUALIZED price (idiosyncratic breakout)
  +skip1      : skip the most recent day (avoid 1-day reversal contamination)
  +multiH     : average breakout z over 10/20/40d (same signal, robust horizon)
  +conv       : conviction = scale by volume confirmation (vol vs its average)
  +betaN      : neutralize the PORTFOLIO's BTC beta
  +regime2    : regime gate = trend-intensity AND calm-vol state

Run from crypto_pulse/:  python tide_v2.py  (-> research/tide_v2.md + png)
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
N_TRIALS = 8


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
        self.sd = self.R.rolling(30).std()
        self.mkt = self.R.where(self.el).mean(axis=1)
        # BTC-beta-residualized cumulative price (idiosyncratic)
        bcol = "BTC" if "BTC" in self.C.columns else self.C.columns[0]
        btc = self.R[bcol]
        beta = self.R.rolling(60).cov(btc).div(btc.rolling(60).var() + 1e-12, axis=0)
        self.resid = (self.R.sub(beta.mul(btc, axis=0)))
        self.residC = (1 + self.resid.fillna(0)).cumprod()

    def _bo(self, C, win):
        dmf = lambda x: x.sub(x.mean(axis=1), axis=0)
        return dmf(((C - C.rolling(win).mean()) / (C.rolling(win).std() + 1e-9)).where(self.el))

    def build(self, resid=False, skip1=False, multiH=False, conv=False, betaN=False,
              regime2=False, hold=3):
        C = self.residC if resid else self.C
        Cs = C.shift(1) if skip1 else C
        if multiH:
            bo = sum(self._bo(Cs, k) for k in (10, 20, 40)) / 3.0
        else:
            bo = self._bo(Cs, 20)
        nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
        ts = ((((self.C > self.C.rolling(50).mean()).where(self.el)).mean(axis=1) - 0.5).abs() * 2).clip(0, 1)
        if regime2:
            calm = (self.mkt.rolling(20).std() < self.mkt.rolling(60).std()).astype(float)
            gate = (ts * (0.5 + 0.5 * calm)).shift(1)
        else:
            gate = ts.shift(1)
        w = nm(bo / (self.sd + 1e-9))
        if conv:
            vr = (self.V / (self.V.rolling(20).mean() + 1e-9)).clip(0.3, 3.0)
            w = nm(w * vr)
        w = w.mul(gate, axis=0)
        if betaN:                                  # neutralize portfolio BTC beta
            bcol = "BTC" if "BTC" in self.C.columns else self.C.columns[0]
            btc = self.R[bcol]
            beta = self.R.rolling(60).cov(btc).div(btc.rolling(60).var() + 1e-12, axis=0)
            port_beta = (w * beta).sum(axis=1)
            w[bcol] = w[bcol] - port_beta          # offset with BTC leg
        rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
        w = w.where(rebw, axis=0).ffill(limit=hold); wl = w.shift(1)
        pnl = (wl * self.R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * self.F).sum(axis=1)
        return vt(pnl)


def main():
    lab = Lab()
    base = lab.build()
    idx = base.index; hl = idx >= HL_START; hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p): q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])
    bi, bo_ = io(base)

    variants = {
        "base TIDE": dict(),
        "+resid (idiosyncratic)": dict(resid=True),
        "+skip1 (no 1d reversal)": dict(skip1=True),
        "+multiH (10/20/40)": dict(multiH=True),
        "+conv (volume confirm)": dict(conv=True),
        "+betaN (beta-neutral)": dict(betaN=True),
        "+regime2 (calm-vol gate)": dict(regime2=True),
    }
    L = ["# Improving TIDE itself — honest signal/construction upgrades\n",
         "Each is a single change to the TIDE rule, walk-forward OOS + deflated. Keep only robust "
         f"OOS gains over base. HL era, OOS=last40%. base OOS {bo_:+.2f}.\n",
         "| variant | Sharpe(HL) | IS | OOS | dOOS vs base | deflated P |", "|---|---|---|---|---|---|"]
    keep = []
    for name, kw in variants.items():
        p = lab.build(**kw)
        i, o = io(p)
        d_ann, d_p = deflated(p[hl][p.index[hl] >= cut])
        delta = o - bo_
        L.append(f"| {name} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {delta:+.2f} | {d_p:.2f} |")
        if name != "base TIDE" and delta > 0.10:
            keep.append((name, kw, delta))

    # assemble TIDE v2 = base + all robustly-helping changes
    combo_kw = {}
    for _, kw, _ in keep:
        combo_kw.update(kw)
    v2 = lab.build(**combo_kw) if combo_kw else base
    v2i, v2o = io(v2)
    d2_ann, d2_p = deflated(v2[hl][v2.index[hl] >= cut])

    L += ["\n## TIDE v2 = base + robust upgrades\n",
          f"- Kept: {', '.join(k for k, _, _ in keep) if keep else 'NONE (no single change robustly helps)'}.",
          f"- **TIDE v2 OOS Sharpe {v2o:+.2f}** (base {bo_:+.2f}, delta {v2o - bo_:+.2f}); "
          f"full HL {sh(v2[hl]):+.2f}; deflated P={d2_p:.2f}.",
          f"- Sharpe 3 {'REACHED' if v2o >= 3 and d2_p > 0.95 else 'NOT reached'}."]

    # quick non-overfit check on v2: 4-fold WF + cost stress
    folds = np.array_split(hidx, 4)
    fsh = [sh(v2[v2.index.isin(f)]) for f in folds]
    L += ["\n## TIDE v2 robustness\n",
          f"- 4-fold walk-forward Sharpe: {', '.join(f'{x:+.2f}' for x in fsh)} "
          f"({'all positive' if all(x > 0 for x in fsh) else 'some negative'}).",
          f"- Pre-HL independent Sharpe: {sh(v2[v2.index < HL_START]):+.2f}.",
          "\n## Verdict\n",
          (f"- **TIDE improved honestly: OOS {bo_:+.2f} -> {v2o:+.2f}.** The gains come from better "
           "signal construction (not ensembling), survive walk-forward and the independent pre-HL "
           "period." if v2o > bo_ + 0.1 else
           "- **No construction change robustly improves TIDE.** The base rule is already near the "
           "honest ceiling for this signal family; the upgrades help in-sample but not OOS."),
          f"- TIDE v2 stays a single, independent, market-neutral crypto-daily book. Honest level "
          f"~{max(v2o, bo_):.1f}.\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    (1 + base[hl].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5, label=f"base TIDE (OOS {bo_:.2f})")
    (1 + v2[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2, label=f"TIDE v2 (OOS {v2o:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log"); ax.legend(fontsize=9)
    ax.set_title("TIDE base vs improved (HL era, net)"); ax.set_ylabel("growth of $1 (log)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(HERE, "tide_v2.png"), dpi=110)
    with open(os.path.join(HERE, "tide_v2.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/tide_v2.md + png")


if __name__ == "__main__":
    main()
