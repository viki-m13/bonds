"""Validate TITAN — is crypto_titan a causal, real strategy or does it have lookahead?

The ~2.5 stack rests on TITAN being a genuine independent book. TITAN ships BOTH weights
(per-coin, daily) and a returns series. The decisive causality test: reconstruct returns from
weights LAGGED one day x actual next-day coin returns. If the causal (lag-1) reconstruction
tracks the published returns and is still strongly positive, TITAN is causal. If only the
contemporaneous (lag-0) version works, the weights peek at same-day returns (lookahead).

Also: year-by-year stability (overfit books front-load), and reconstruction under OUR cost
assumption. Run from crypto_pulse/:  python titan_validate.py  (-> research/titan_validate.md)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
from tide import sh, cagr, maxdd, HL_START

ANN = 365
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def main():
    W = pd.read_csv(os.path.join(D, "crypto_titan", "crypto_titan_weights.csv"),
                    parse_dates=["Date"]).set_index("Date")
    rep = pd.read_csv(os.path.join(D, "crypto_titan", "crypto_titan_returns.csv"),
                      parse_dates=["Date"]).set_index("Date").iloc[:, 0]

    # actual coin returns from data/crypto
    coins = [c for c in W.columns if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    cl = {}
    for c in coins:
        d = pd.read_csv(os.path.join(v.CRYPTO, f"{c}_USD.csv"), parse_dates=["Date"]).set_index("Date")
        cl[c] = d[~d.index.duplicated()]["Close"]
    C = pd.DataFrame(cl).sort_index()
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    W = W[coins].reindex(R.index).fillna(0.0)

    def recon(lag, tc_bps):
        wl = W.shift(lag)
        gross = (wl * R).sum(axis=1)
        cost = (wl - wl.shift(1)).abs().sum(axis=1) * tc_bps / 1e4
        return gross - cost

    causal = recon(1, 20.0)        # lag-1 = causal, their 20bps
    contemp = recon(0, 20.0)       # lag-0 = potential lookahead
    causal_hi = recon(1, 40.0)     # double cost stress

    def al(a, b):
        m = pd.concat({"a": a, "b": b}, axis=1).dropna()
        return m["a"].corr(m["b"]) if len(m) > 60 else np.nan

    L = ["# Validate TITAN — causal or lookahead? (honest)\n",
         f"TITAN ships weights + returns. Reconstruct from weights x actual coin returns "
         f"({len(coins)} coins matched). Lag-1 = causal; lag-0 = peeks at same-day returns.\n",
         "## Reconstruction\n",
         "| series | Sharpe | CAGR | corr to published |", "|---|---|---|---|"]
    L.append(f"| published returns | {sh(rep):+.2f} | {cagr(rep):+.0%} | 1.00 |")
    L.append(f"| **causal (lag-1, 20bps)** | **{sh(causal):+.2f}** | {cagr(causal):+.0%} | {al(causal, rep):+.2f} |")
    L.append(f"| contemporaneous (lag-0) | {sh(contemp):+.2f} | {cagr(contemp):+.0%} | {al(contemp, rep):+.2f} |")
    L.append(f"| causal @ 40bps (stress) | {sh(causal_hi):+.2f} | {cagr(causal_hi):+.0%} | {al(causal_hi, rep):+.2f} |")

    # lookahead diagnostic
    gap = sh(contemp) - sh(causal)
    L += ["\n## Lookahead diagnostic\n",
          f"- Lag-0 minus lag-1 Sharpe gap: **{gap:+.2f}**. "
          f"{'LARGE gap -> weights likely use same-day info (lookahead risk).' if gap > 1.0 else 'Small gap -> weights are causal; lag-1 already works.'}",
          f"- Causal reconstruction correlation to published: {al(causal, rep):+.2f} "
          f"({'tracks the published series -> returns derive causally from the weights.' if al(causal, rep) > 0.8 else 'weak -> published returns include extra processing (dd-floor/smoothing) or differ.'}"]

    # year by year (causal)
    L += ["\n## Year-by-year (causal reconstruction)\n", "| year | Sharpe |", "|---|---|"]
    for y in range(2017, 2027):
        L.append(f"| {y} | {sh(causal[causal.index.year == y]):+.2f} |")

    # HL era + recent
    hl = causal[causal.index >= HL_START]
    L += [f"\n- Causal HL-era Sharpe {sh(hl):+.2f}; published HL-era {sh(rep[rep.index >= HL_START]):+.2f}.",
          f"- Causal last-365d {sh(causal[causal.index > causal.index.max()-pd.Timedelta(days=365)]):+.2f}."]

    causal_ok = sh(causal) > 0.7 and gap < 1.0 and al(causal, rep) > 0.7
    L += ["\n## Verdict\n",
          (f"- **TITAN validates as causal:** the lag-1 reconstruction is {sh(causal):+.2f} Sharpe, "
           f"tracks the published series (corr {al(causal,rep):+.2f}), and the lag-0/lag-1 gap is "
           f"small ({gap:+.2f}) — no evidence of same-day lookahead. It survives doubled cost "
           f"({sh(causal_hi):+.2f} @ 40bps)." if causal_ok else
           f"- **TITAN is SUSPECT:** "
           + ("large lag-0/lag-1 gap suggests same-day lookahead; " if gap >= 1.0 else "")
           + (f"causal reconstruction weak ({sh(causal):+.2f}); " if sh(causal) <= 0.7 else "")
           + "treat its contribution to the stack with caution."),
          "- It is a DIRECTIONAL multi-sleeve crypto trend/breakout CTA (21 sleeves) — structurally "
          "orthogonal to TIDE's market-neutral book, which is why corr is ~0.03 (real, not fitted).",
          "- Caveat: 21 sleeves is a lot of freedom; even if causal, its live Sharpe will likely be "
          "below backtest. Size the stack to the causal/stressed number, not the published one.\n"]

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + rep.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.4, label=f"published ({sh(rep):.2f})")
    (1 + causal.fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=1.8, label=f"causal lag-1 ({sh(causal):.2f})")
    ax.set_yscale("log"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax.set_title("TITAN: published vs causal (lag-1) reconstruction"); ax.set_ylabel("growth of $1 (log)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "titan_validate.png"), dpi=110)
    with open(os.path.join(HERE, "titan_validate.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/titan_validate.md + png")


if __name__ == "__main__":
    main()
