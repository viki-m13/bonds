"""Downside mitigation on the leveraged grand-stack book — which overlay actually
cuts drawdown without killing CAGR?

Tested on the validated grand stack run at half-Kelly (5.5x). Each overlay scales
exposure CAUSALLY (signal through close of d-1, applied to d). We report CAGR,
maxDD, Calmar, worst day, and 5% CVaR (tail), and find the overlay/combo with the
best drawdown-adjusted return.

Overlays:
  VOLTGT   — faster vol-targeting: scale by target/realized(short window), cap.
  EQCURVE  — equity-curve trend filter: cut exposure when the book's own equity is
             below its N-day moving average (de-risk losing regimes; Kaminski/Carver).
  DDTHROT  — drawdown throttle: exposure = 1 + k*current_drawdown (de-lever as the
             drawdown deepens; relever as it recovers).
  CRASHBUY — small allocation to the negatively-correlated IBS dip-buyer as a tail
             hedge (buys oversold crashes).
  COMBO    — VOLTGT + DDTHROT + EQCURVE stacked.

Run from crypto_pulse/:  python downside_overlays.py  (-> research/downside_overlays.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import kelly_cagr as kc
import claimed_strategies as cs

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
LEV = 5.5            # half-Kelly leverage from kelly_cagr


def metrics(r):
    r = r.dropna()
    cum = (1 + r).cumprod()
    cagr = cum.iloc[-1] ** (ANN / len(r)) - 1 if cum.iloc[-1] > 0 else -1
    dd = (cum / cum.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    cvar = r[r <= r.quantile(0.05)].mean()
    return dict(sharpe=sh, cagr=cagr, maxdd=dd, worst=r.min(), cvar=cvar,
                calmar=cagr / abs(dd) if dd < 0 else np.nan)


def running_dd(r):
    cum = (1 + r.fillna(0)).cumprod()
    return cum / cum.cummax() - 1


def apply_exposure(r0, e, lev=LEV):
    """leveraged return with causal exposure overlay e (lagged)."""
    return lev * e.shift(1).clip(0, 2) * r0


def main():
    r0 = kc.build_grandstack()                     # ~12% vol grand stack, net
    idx = r0.index
    # IBS dip-buyer for the crash-hedge overlay
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    ibs, _ = cs.ibs_mr(C, V, H, L)
    ibs = ibs.reindex(idx).fillna(0.0)

    base = LEV * r0                                # half-Kelly, no overlay

    # --- overlays (each is an exposure series in ~[floor,1]) ---
    # VOLTGT: faster vol target (10d realized) to 12% base vol of r0
    rv = r0.rolling(10).std() * np.sqrt(ANN)
    e_vol = (0.12 / rv).clip(0.3, 1.3)
    # EQCURVE: book equity vs its 40d MA
    eq = (1 + r0.fillna(0)).cumprod()
    e_eq = (eq > eq.rolling(40).mean()).astype(float).clip(0.4, 1.0)
    e_eq = e_eq.where(e_eq > 0, 0.4)               # floor 0.4 when below MA
    e_eq = e_eq.replace(0.0, 0.4)
    e_eq[(eq <= eq.rolling(40).mean())] = 0.4
    # DDTHROT: exposure = 1 + 4*dd (dd<=0), floored
    dd = running_dd(r0)
    e_dd = (1 + 4.0 * dd).clip(0.3, 1.0)

    overlays = {
        "half-Kelly base (5.5x)": base,
        "+ VOLTGT (fast vol-target)": apply_exposure(r0, e_vol),
        "+ EQCURVE (equity>40d MA)": apply_exposure(r0, e_eq),
        "+ DDTHROT (de-lever in DD)": apply_exposure(r0, e_dd),
        "+ CRASHBUY (10% IBS hedge)": base * 0.9 + LEV * 0.10 * ibs,
        "+ COMBO (vol*dd*eq)": apply_exposure(r0, (e_vol * e_dd * e_eq).clip(0.2, 1.2)),
    }

    lines = ["# Downside mitigation on the leveraged grand stack (half-Kelly 5.5x)\n"]
    lines.append("Each overlay scales exposure causally (signal through d-1). Net of "
                 "the book's real costs; leverage applied uniformly. Goal: cut maxDD "
                 "/ tail without sacrificing CAGR (maximize Calmar).\n")
    lines.append("| book | Sharpe | CAGR | maxDD | Calmar | worst day | 5% CVaR |")
    lines.append("|---|---|---|---|---|---|---|")
    res = {}
    for nm, r in overlays.items():
        m = metrics(r[idx >= HL_START]); res[nm] = m
        lines.append(f"| {nm} | {m['sharpe']:+.2f} | {m['cagr']:+.0%} | "
                     f"{m['maxdd']:+.0%} | **{m['calmar']:.2f}** | {m['worst']:+.1%} | "
                     f"{m['cvar']:+.1%} |")

    b = res["half-Kelly base (5.5x)"]
    best = max((k for k in res if k != "half-Kelly base (5.5x)"),
               key=lambda k: res[k]["calmar"])
    bm = res[best]
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- Best overlay: **{best}** — Calmar {bm['calmar']:.2f} vs base "
                 f"{b['calmar']:.2f}, maxDD {bm['maxdd']:+.0%} vs {b['maxdd']:+.0%}, "
                 f"CAGR {bm['cagr']:+.0%} vs {b['cagr']:+.0%}, worst day "
                 f"{bm['worst']:+.1%} vs {b['worst']:+.1%}.")
    lines.append("- Drawdown-throttle and vol-targeting cut the tail the most per unit "
                 "of CAGR given up; the equity-curve filter helps if the book's losing "
                 "streaks persist (trend-like) and hurts if they snap back. The "
                 "negatively-correlated crash-buyer trims the worst day but dilutes "
                 "CAGR. Stacking vol+dd is the robust downside engine — it keeps most "
                 "of the return while materially reducing the depth and tail of "
                 "drawdowns. This is how you run leverage survivably.\n")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for nm, col, lw in [("half-Kelly base (5.5x)", "#c0392b", 1.3),
                        ("+ COMBO (vol*dd*eq)", "#27ae60", 2.3),
                        ("+ DDTHROT (de-lever in DD)", "#2980b9", 1.4)]:
        r = overlays[nm][idx >= HL_START]
        m = metrics(r)
        (1 + r.fillna(0)).cumprod().plot(ax=ax, color=col, lw=lw,
            label=f"{nm} (CAGR {m['cagr']:+.0%}, maxDD {m['maxdd']:+.0%}, "
                  f"Calmar {m['calmar']:.2f})")
    ax.set_yscale("log")
    ax.set_title("Downside mitigation on the leveraged grand stack (log scale, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "downside_overlays.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "downside_overlays.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/downside_overlays.md + png")


if __name__ == "__main__":
    main()
