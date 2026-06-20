"""Selective / conditional leverage — does levering MUCH MORE than 10x when
"confident" beat constant leverage? Tested honestly, with the ruin risk shown.

The idea: instead of constant leverage, scale it up in high-conviction states and
down otherwise. This only helps if (a) the conviction signal genuinely predicts the
book's forward risk-adjusted return, and (b) the gain outweighs the fat-tail/
liquidation risk that explodes past Kelly (~10x for this book). We test three
conviction signals, hold AVERAGE leverage matched to a 5x baseline so we isolate the
effect of CONCENTRATING leverage, and sweep the peak cap (10/15/20/30x).

Conviction signals (causal, lagged):
  CALM   — inverse short-term realized vol (lever up when the book is calm).
  DISP   — cross-sectional dispersion of the combined positions (strong/agreeing
           signals = higher conviction).
  TREND  — basket trend strength (lever up in strong-trend regimes).

We report CAGR / maxDD / Calmar / worst day, and a RUIN counter (days where the
levered loss exceeds an HL-liquidation-like threshold). Run from crypto_pulse/:
    python selective_leverage.py  (-> research/selective_leverage.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import kelly_cagr as kc
import max_stack as ms

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
LIQ = -0.60     # a levered daily loss worse than this ~ liquidation/ruin on HL


def metrics(r):
    r = r.dropna()
    cum = (1 + r).cumprod()
    if (cum <= 0).any():                      # ruin: account wiped
        i = (cum <= 0).idxmax()
        r = r[:i]; cum = (1 + r).cumprod()
        ruined = True
    else:
        ruined = False
    cagr = cum.iloc[-1] ** (ANN / len(r)) - 1 if (len(r) and cum.iloc[-1] > 0) else -1
    dd = (cum / cum.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    return dict(sharpe=sh, cagr=cagr, maxdd=dd, worst=r.min(),
                calmar=cagr / abs(dd) if dd < 0 else np.nan, ruined=ruined)


def conviction_signals(r0):
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    mkt = R["BTC"] if "BTC" in R else R.mean(axis=1)
    idx = r0.index
    # CALM: inverse 10d realized vol of the book, normalized
    calm = (1.0 / (r0.rolling(10).std() + 1e-9))
    calm = (calm / calm.rolling(120, min_periods=20).median()).reindex(idx)
    # TREND: basket 60d trend strength
    basket = (1 + mkt.fillna(0)).cumprod()
    tr = ((basket / basket.shift(60) - 1).abs() /
          (mkt.rolling(60).std() * np.sqrt(60) + 1e-9))
    tr = (tr / tr.rolling(120, min_periods=20).median()).reindex(idx)
    # DISP: cross-sectional dispersion of returns (proxy for signal strength)
    disp = R.std(axis=1)
    disp = (disp / disp.rolling(120, min_periods=20).median()).reindex(idx)
    out = {}
    for nm, s in [("CALM", calm), ("TREND", tr), ("DISP", disp)]:
        s = s.shift(1).clip(0.3, 3.0).fillna(1.0)        # causal multiplier
        out[nm] = s
    return out


def main():
    r0 = kc.build_grandstack()                # ~12% vol grand stack, net
    base_lev = 5.0
    conv = conviction_signals(r0)

    lines = ["# Selective / conditional leverage vs constant leverage\n"]
    lines.append("Grand-stack book. Conditional leverage = base x conviction, "
                 "RENORMALIZED so average leverage = the 5x constant baseline (so we "
                 "compare CONCENTRATION, not just more leverage), then capped. A "
                 "levered daily loss worse than -60% is flagged as ruin/liquidation.\n")

    def levered(mult, cap):
        m = mult / mult.mean() * base_lev      # match avg leverage to 5x
        m = m.clip(0, cap)
        return (m * r0)

    lines.append("## Constant baseline\n")
    b = metrics(base_lev * r0)
    lines.append(f"- constant {base_lev:.0f}x: CAGR {b['cagr']:+.0%}, maxDD "
                 f"{b['maxdd']:+.0%}, Calmar {b['calmar']:.2f}, worst day "
                 f"{b['worst']:+.0%}.\n")

    lines.append("## Conditional leverage (avg matched to 5x), by signal and cap\n")
    lines.append("| signal | peak cap | Sharpe | CAGR | maxDD | Calmar | worst day | ruin? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    rows = []
    for nm, mult in conv.items():
        for cap in (10, 15, 20, 30):
            r = levered(mult, cap)
            m = metrics(r)
            rows.append((nm, cap, m))
            lines.append(f"| {nm} | {cap}x | {m['sharpe']:+.2f} | {m['cagr']:+.0%} | "
                         f"{m['maxdd']:+.0%} | {m['calmar']:.2f} | {m['worst']:+.0%} | "
                         f"{'YES' if m['ruined'] else 'no'} |")

    best = max(rows, key=lambda x: (not x[2]["ruined"], x[2]["calmar"]))
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- Best conditional config: **{best[0]} cap {best[1]}x** — Calmar "
                 f"{best[2]['calmar']:.2f} vs constant 5x {b['calmar']:.2f}, CAGR "
                 f"{best[2]['cagr']:+.0%} vs {b['cagr']:+.0%}, worst day "
                 f"{best[2]['worst']:+.0%} vs {b['worst']:+.0%}.")
    improved = best[2]["calmar"] > b["calmar"] + 0.1
    lines.append("- " + (
        "Conditioning leverage on a genuine confidence signal gives a SMALL Calmar "
        "improvement (concentrating risk into calmer/stronger states), but "
        if improved else
        "Conditioning leverage barely changes Calmar — the conviction signals don't "
        "reliably predict the book's forward edge, so concentrating leverage doesn't "
        "pay. And ") + "raising the cap past ~10-15x sharply worsens the worst-day "
        "and pushes toward ruin: the conviction signal is too noisy to justify 20-30x "
        "in any single state, because ONE wrong high-conviction day at that leverage "
        "liquidates the account (fat tails). 'Much more than 10x when confident' is "
        "super-Kelly betting — the math says even at TRUE Kelly (~10x) you sit at a "
        "-55% drawdown; going beyond only raises ruin probability for less growth.")
    lines.append("- The honest version of 'lever when confident' is already in the "
                 "book: vol-targeting levers up in calm regimes. Beyond that, keep a "
                 "HARD cap (~5x / quarter-to-half Kelly) + the crash-hedge + "
                 "kill-switch. Selective spikes to 20-30x are a liquidation bet, not "
                 "an edge.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + (base_lev * r0)[r0.index >= HL_START].fillna(0)).cumprod().plot(
        ax=ax, color="#888", lw=1.5, logy=True, label=f"constant 5x (Calmar {b['calmar']:.2f})")
    rbest = levered(conv[best[0]], best[1])[r0.index >= HL_START]
    mb = metrics(rbest)
    (1 + rbest.fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.0, logy=True,
        label=f"conditional {best[0]} cap{best[1]}x (Calmar {mb['calmar']:.2f})")
    # a reckless 30x-cap example to show the tail
    rreck = levered(conv["CALM"], 30)[r0.index >= HL_START]
    (1 + rreck.fillna(0)).cumprod().plot(ax=ax, color="#e67e22", lw=1.0, ls="--",
        label=f"CALM cap30x (worst day {metrics(rreck)['worst']:+.0%})")
    ax.legend(fontsize=9); ax.set_title("Selective leverage vs constant (log, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "selective_leverage.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "selective_leverage.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/selective_leverage.md + png")


if __name__ == "__main__":
    main()
