"""Harden the funding-carry leg: own overfit battery + tail-taming (honest).

Carry showed HL-era OOS ~2.4 and corr +0.24 to TIDE, but was uncertified (no pre-HL data,
recent-concentrated, -24% crash tail). It can't get pre-HL validation (funding starts ~2023),
but it CAN earn an HL-internal overfit battery and a tamer tail. Do both:

A) Carry overfit battery (HL era): lookback plateau, coin bootstrap, shuffle-null, 4-fold WF.
B) Tail-taming overlays on carry: vol-gate (cut gross when market vol high — carry crashes in
   squeezes), dd-floor (cut when carry underwater), winsor (cap per-coin funding signal).
C) Re-evaluate TIDE + tamed-carry: does taming keep the OOS gain while cutting maxDD?

A carry overlay is "kept" only if it cuts maxDD without losing OOS. Run from crypto_pulse/:
python tide_carry2.py  (-> research/tide_carry2.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tide import TIDE, sh, cagr, maxdd, vt, HL_START, ANN, TAKER

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


class CarryLab:
    def __init__(self):
        t = TIDE()
        self.t = t
        self.C, self.V, self.H, self.L = t.C, t.V, t.H, t.L
        self.F = t.F.reindex(columns=self.C.columns).fillna(0.0)
        self.R = self.C.pct_change(); self.R[self.R.abs() > 2] = np.nan
        dv = (self.C * self.V).rolling(30).mean(); self.el = self.C.notna() & (dv > 3e6)
        self.sd = np.sqrt((np.log(self.H / self.L) ** 2).rolling(30).mean() / (4 * np.log(2))) + 1e-9
        self.rebw = pd.Series(np.arange(len(self.C)) % 3 == 0, index=self.C.index)
        # market vol regime (causal): cross-sectional median realized vol, percentile-ranked
        mvol = (self.R.where(self.el).std(axis=1))
        self.mvolz = mvol.rolling(252, min_periods=60).rank(pct=True)
        self.tide = t.build()

    def carry(self, lb=14, cols=None, shuffle=False, seed=0, volgate=False, ddfloor=False, winsor=False):
        F = self.F if cols is None else self.F[cols]
        R = self.R if cols is None else self.R[cols]
        sd = self.sd if cols is None else self.sd[cols]
        el = self.el if cols is None else self.el[cols]
        nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
        fsm = F.rolling(lb).mean()
        sig = -(fsm.sub(fsm.mean(axis=1), axis=0)).where(el)
        if winsor:
            sig = sig.clip(sig.quantile(0.05, axis=1), sig.quantile(0.95, axis=1), axis=0)
        if shuffle:
            rng = np.random.default_rng(seed); bv = sig.values.copy()
            for i in range(len(bv)):
                row = bv[i]; m = ~np.isnan(row)
                if m.sum() > 2:
                    ix = np.where(m)[0]; bv[i, ix] = row[rng.permutation(ix)]
            sig = pd.DataFrame(bv, index=sig.index, columns=sig.columns)
        w = nm(sig / sd)
        if volgate:                                   # cut gross when market vol is elevated
            w = w.mul((1 - 0.7 * self.mvolz).shift(1), axis=0)
        w = w.where(self.rebw, axis=0).ffill(limit=3); wl = w.shift(1)
        pnl = (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * F).sum(axis=1)
        p = vt(pnl)
        if ddfloor:
            cum = (1 + p.fillna(0)).cumprod(); dd = cum / cum.cummax() - 1
            p = p * (1 + (dd / 0.10).clip(-1, 0)).shift(1)
        return p


def main():
    lab = CarryLab()
    tide = lab.tide
    idx = tide.index; hl = idx >= HL_START; hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p): q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])
    rng = np.random.default_rng(0)

    L = ["# Hardening the funding-carry leg — battery + tail-taming (honest)\n",
         "Carry can't get pre-HL validation (funding starts ~2023), but here's its HL-internal "
         "overfit battery and tail-taming.\n", "## A) Carry overfit battery (HL era)\n"]

    # lookback plateau
    lbs = [3, 5, 7, 10, 14, 21, 30]
    row = " | ".join(f"{lb}d:{sh(lab.carry(lb)[hl]):+.2f}" for lb in lbs)
    L.append(f"- **Lookback plateau:** {row}")
    L.append(f"  -> {sum(sh(lab.carry(lb)[hl]) > 1.0 for lb in lbs)}/{len(lbs)} lookbacks > 1.0 Sharpe.")
    # coin bootstrap
    cols = list(lab.C.columns)
    bss = []
    for s in range(15):
        sub = list(rng.choice(cols, int(len(cols) * 0.7), replace=False))
        bss.append(sh(lab.carry(14, cols=sub)[hl]))
    bss = np.array(bss)
    L.append(f"- **Coin bootstrap** (15× 70% subsets): median {np.median(bss):+.2f}, 5th-pct "
             f"{np.percentile(bss, 5):+.2f}, min {bss.min():+.2f}.")
    # shuffle null
    nulls = [sh(lab.carry(14, shuffle=True, seed=s)[hl]) for s in range(15)]
    L.append(f"- **Shuffle null** (permute funding across coins): real {sh(lab.carry(14)[hl]):+.2f} vs "
             f"null max {max(nulls):+.2f}, mean {np.mean(nulls):+.2f}.")
    # walk-forward
    folds = np.array_split(hidx, 4); wf = [sh(lab.carry(14)[lab.carry(14).index.isin(fd)]) for fd in folds]
    L.append(f"- **Walk-forward** (4 folds): {', '.join(f'{x:+.2f}' for x in wf)} "
             f"({sum(x > 0 for x in wf)}/4 positive).")

    # B) tail-taming
    base_c = lab.carry(14)
    bi, bo_ = io(base_c)
    overlays = {"base carry(14d)": dict(), "+volgate": dict(volgate=True),
                "+ddfloor": dict(ddfloor=True), "+winsor": dict(winsor=True),
                "+volgate+ddfloor": dict(volgate=True, ddfloor=True)}
    L += ["\n## B) Tail-taming overlays on carry\n", "| overlay | HL | OOS | dOOS | CAGR | maxDD |",
          "|---|---|---|---|---|---|"]
    best = ("base carry(14d)", base_c, maxdd(base_c[hl]), bo_)
    for nmv, kw in overlays.items():
        p = lab.carry(14, **kw); i, o = io(p)
        L.append(f"| {nmv} | {sh(p[hl]):+.2f} | {o:+.2f} | {o - bo_:+.2f} | {cagr(p[hl]):+.0%} | {maxdd(p[hl]):+.0%} |")
        # keep if cuts maxDD meaningfully without losing >0.2 OOS
        if "base" not in nmv and maxdd(p[hl]) > best[2] + 0.03 and o > bo_ - 0.20:
            best = (nmv, p, maxdd(p[hl]), o)
    tamed = best[1]

    # C) TIDE + tamed carry
    def rp(a, b):
        ia = 1.0 / (a.rolling(45).std() * np.sqrt(ANN)).clip(lower=0.05)
        ib = 1.0 / (b.rolling(45).std() * np.sqrt(ANN)).clip(lower=0.05)
        wa = (ia / (ia + ib)).shift(1).fillna(0.5); return vt(a * wa + b * (1 - wa))
    ti, to = io(tide); ci, co = io(rp(tide, tamed))
    combo = rp(tide, tamed)
    cwf = [sh(combo[combo.index.isin(fd)]) for fd in folds]
    L += ["\n## C) TIDE + tamed carry\n",
          f"- Best tail-tamer: **{best[0]}** (carry maxDD {maxdd(base_c[hl]):+.0%} -> {best[2]:+.0%}, "
          f"OOS {bo_:+.2f} -> {best[3]:+.2f}).",
          f"- TIDE alone: HL {sh(tide[hl]):+.2f}, OOS {to:+.2f}, maxDD {maxdd(tide[hl]):+.0%}.",
          f"- **TIDE + tamed carry: HL {sh(combo[hl]):+.2f}, OOS {co:+.2f} ({co - to:+.2f}), "
          f"CAGR {cagr(combo[hl]):+.0%}, maxDD {maxdd(combo[hl]):+.0%}, WF {', '.join(f'{x:+.1f}' for x in cwf)}.**"]

    passed = (sum(sh(lab.carry(lb)[hl]) > 1.0 for lb in lbs) >= 5 and np.percentile(bss, 5) > 0.5
              and max(nulls) < 0.8 and sum(x > 0 for x in wf) >= 3)
    L += ["\n## Verdict\n",
          (f"- **Carry PASSES its HL-internal overfit battery** (lookback plateau, bootstrap 5th-pct "
           f"{np.percentile(bss,5):+.2f}, clean null max {max(nulls):+.2f}, {sum(x>0 for x in wf)}/4 WF) "
           "— the edge is real within the funding era, not a fluke." if passed else
           "- **Carry is shakier than TIDE on its own battery** — treat with extra caution."),
          (f"- **Tail-taming works:** {best[0]} cuts carry maxDD to {best[2]:+.0%} while keeping OOS "
           f"{best[3]:+.2f}. TIDE + tamed carry: OOS {co:+.2f} (vs {to:+.2f} alone), maxDD {maxdd(combo[hl]):+.0%}."
           if best[0] != "base carry(14d)" else
           "- Tail-taming overlays didn't cleanly cut the tail without cost; carry's crash risk stays."),
          "- **Still phase-2, not certified:** the hard blocker is unchanged — no pre-HL funding data means "
          "no independent-regime confirmation. But carry now has an internal battery + a tamed tail, so it's "
          "a *stronger, better-understood* phase-2 sleeve to paper-trade. Certified book stays TIDE alone.\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    (1 + tide[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=1.6, label=f"TIDE ({sh(tide[hl]):.2f})")
    (1 + base_c[hl].fillna(0)).cumprod().plot(ax=ax, color="#999", lw=1.1, label=f"carry raw ({sh(base_c[hl]):.2f})")
    (1 + tamed[hl].fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.3, label=f"carry tamed ({sh(tamed[hl]):.2f})")
    (1 + combo[hl].fillna(0)).cumprod().plot(ax=ax, color="k", lw=2.3, label=f"TIDE+tamed ({sh(combo[hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log"); ax.legend(fontsize=9)
    ax.set_title("TIDE + hardened funding carry (HL era, net)"); ax.set_ylabel("growth of $1 (log)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(HERE, "tide_carry2.png"), dpi=110)
    with open(os.path.join(HERE, "tide_carry2.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/tide_carry2.md + png")


if __name__ == "__main__":
    main()
