"""Orthogonal PRICE-based factor legs — the certifiable diversification path (honest).

Carry diversifies TIDE but can't be certified (no pre-HL funding data). Price-based factor legs
DO have the full 12-year history, so a leg that is (a) low-correlated to TIDE, (b) positive OOS,
AND (c) positive on the independent pre-HL period would be a CERTIFIED diversifier. Test the
classic non-momentum crypto anomalies, market-neutral, same plumbing as TIDE:
  LOTTERY : short high-MAX coins (Bali MAX effect: lottery demand -> underperformance).
  LOWBETA : long low-beta / short high-beta (betting-against-beta).
  STREV   : short-term reversal, long 3d losers (expected to LOSE in crypto — continuation).
  SKEW    : short positively-skewed coins.
Keep a leg only if corr<0.5 to TIDE AND OOS>0 AND pre-HL>0; combine survivors with TIDE and
check the combo beats TIDE on BOTH OOS and pre-HL.

Run from crypto_pulse/:  python tide_factors.py  (-> research/tide_factors.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tide import TIDE, sh, cagr, maxdd, vt, HL_START, ANN, TAKER

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def main():
    t = TIDE()
    C, V, H, L = t.C, t.V, t.H, t.L
    F = t.F.reindex(columns=C.columns).fillna(0.0)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); el = C.notna() & (dv > 3e6)
    sd = np.sqrt((np.log(H / L) ** 2).rolling(30).mean() / (4 * np.log(2))) + 1e-9
    nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
    dmf = lambda x: x.sub(x.mean(axis=1), axis=0)
    rebw = pd.Series(np.arange(len(C)) % 3 == 0, index=C.index)
    mkt = R.where(el).mean(axis=1)                      # equal-weight market

    def book(sig):
        w = nm((dmf(sig.where(el))) / sd).where(rebw, axis=0).ffill(limit=3); wl = w.shift(1)
        return vt((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER - (wl * F).sum(axis=1))

    # factor signals (all causal)
    lottery = -R.rolling(20).max()                                         # short high-MAX
    beta = R.rolling(60).cov(mkt).div(mkt.rolling(60).var(), axis=0)       # rolling beta to market
    lowbeta = -beta                                                        # long low beta
    strev = -R.rolling(3).sum()                                            # long 3d losers
    skew = -R.rolling(30).skew()                                           # short positive skew

    tide = t.build()
    legs = {"TIDE": tide, "LOTTERY": book(lottery), "LOWBETA": book(lowbeta),
            "STREV": book(strev), "SKEW": book(skew)}
    idx = tide.index; hl = idx >= HL_START; hidx = idx[hl]
    cut = hidx[int(len(hidx) * 0.6)]
    def io(p): q = p[p.index >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])
    def pre(p): return sh(p[p.index < HL_START])

    L_ = ["# Orthogonal price-factor legs — certifiable diversification (honest)\n",
          "Price legs have full history, so a low-corr leg positive on BOTH OOS and pre-HL is "
          "CERTIFIABLE (unlike carry).\n",
          "## Standalone factor legs\n",
          "| leg | HL | IS | OOS | pre-HL | corr→TIDE(HL) | maxDD |", "|---|---|---|---|---|---|---|"]
    ti, to = io(tide); tpre = pre(tide); keep = []
    for k, p in legs.items():
        i, o = io(p)
        cc = (pd.concat([tide[hl], p[hl]], axis=1).dropna().corr().iloc[0, 1] if k != "TIDE" else 1.0)
        L_.append(f"| {k} | {sh(p[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {pre(p):+.2f} | {cc:+.2f} | {maxdd(p[hl]):+.0%} |")
        if k != "TIDE" and abs(cc) < 0.5 and o > 0 and pre(p) > 0:
            keep.append(k)

    # risk-parity combine TIDE + survivors
    def rp(names):
        sub = {k: legs[k] for k in names}
        iv = {k: 1.0 / (p.rolling(45).std() * np.sqrt(ANN)).clip(lower=0.05) for k, p in sub.items()}
        ivs = sum(iv.values()); w = {k: (iv[k] / ivs).shift(1).fillna(1.0 / len(sub)) for k in sub}
        return vt(sum(sub[k] * w[k] for k in sub))

    L_ += ["\n## Verdict\n", f"- Certifiable diversifiers (corr<0.5, OOS>0, pre-HL>0): "
           f"**{', '.join(keep) if keep else 'NONE'}**."]
    if keep:
        combo = rp(["TIDE"] + keep); ci, co = io(combo); cpre = pre(combo)
        folds = np.array_split(hidx, 4); cwf = [sh(combo[combo.index.isin(fd)]) for fd in folds]
        win = co > to + 0.05 and cpre > tpre - 0.03 and all(x > 0 for x in cwf)
        L_.append(f"- TIDE alone: OOS {to:+.2f}, pre-HL {tpre:+.2f}. **TIDE+{'+'.join(keep)}: OOS {co:+.2f} "
                  f"({co - to:+.2f}), pre-HL {cpre:+.2f} ({cpre - tpre:+.2f}), HL {sh(combo[hl]):+.2f}, "
                  f"full {sh(combo.dropna()):+.2f}, maxDD {maxdd(combo[hl]):+.0%}, WF {', '.join(f'{x:+.1f}' for x in cwf)}.**")
        L_.append(f"- **{'CERTIFIED IMPROVEMENT: the combo beats TIDE on OOS AND the independent pre-HL period.' if win else 'Combo does NOT robustly beat TIDE on both OOS and pre-HL — diversification benefit is not certified.'}**")
        bp = combo
    else:
        L_.append("- No price factor is both orthogonal and positive across regimes — TIDE's momentum "
                  "structure already spans the price-based premia. Certified book stays TIDE alone.")
        bp = tide
    L_.append("- (Carry remains the most orthogonal leg but is HL-only; these price legs are the test of "
              "whether a *certifiable* second leg exists in price data. Beyond this: L4 order flow.)\n")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, p in legs.items():
        (1 + p[hl].fillna(0)).cumprod().plot(ax=ax, lw=1.2, alpha=0.65, label=f"{k} ({sh(p[hl]):.2f})")
    if keep:
        (1 + bp[hl].fillna(0)).cumprod().plot(ax=ax, color="k", lw=2.4, label=f"combo ({sh(bp[hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("TIDE + price-factor legs (HL era, net)"); ax.set_ylabel("growth of $1 (log)")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(HERE, "tide_factors.png"), dpi=110)
    with open(os.path.join(HERE, "tide_factors.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/tide_factors.md + png")


if __name__ == "__main__":
    main()
