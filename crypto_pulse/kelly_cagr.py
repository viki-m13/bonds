"""Highest-CAGR leverage (Kelly) on the grand-stack book, and its equity curve.

CAGR (geometric growth) is NOT maximized by infinite leverage: it rises, peaks at
the Kelly-optimal leverage f* ~ mu/sigma^2, then FALLS as volatility drag
(-0.5*L^2*sigma^2) overwhelms the linear return. We sweep constant leverage on the
validated grand-stack book (net of costs), find the leverage that maximizes realized
CAGR, and plot that equity curve — alongside an HONEST drawdown + liquidation check
(on Hyperliquid a single-day loss x leverage near the maintenance margin liquidates
you, and fat tails make full Kelly un-survivable; half-Kelly is the practical pick).

Run from crypto_pulse/:  python kelly_cagr.py  (-> research/kelly_cagr.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import max_stack as ms
import grand_stack as gs

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def metrics(r):
    r = r.dropna()
    cum = (1 + r).cumprod()
    n = len(r)
    cagr = cum.iloc[-1] ** (ANN / n) - 1 if cum.iloc[-1] > 0 else -1.0
    dd = (cum / cum.cummax() - 1).min()
    sh = r.mean() / r.std() * np.sqrt(ANN) if r.std() > 0 else np.nan
    return cagr, dd, sh, cum.iloc[-1], r.min()


def build_grandstack():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    raw = ms.build_sleeves(C, V, H, L, F)
    admitted = ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]
    sl = {k: ms.vt(raw[k]) for k in admitted}
    sl["FUNDFADE"] = ms.vt(gs.funding_fade(C, V, H, L, F, R, elig))
    hl = C.index >= HL_START
    P = pd.DataFrame({k: sl[k][hl] for k in sl}).dropna()
    cut = P.index[int(len(P) * 0.6)]
    Pis = P[P.index < cut]
    mu = Pis.mean().values * ANN
    S = Pis.cov().values * ANN
    d = np.diag(np.diag(S))
    Sshr = 0.6 * d + 0.4 * S
    w = np.clip(np.linalg.solve(Sshr + 1e-6 * np.eye(len(mu)), mu), 0, None)
    w = w / w.sum()
    comb = ms.vt((P * pd.Series(w, index=P.columns)).sum(axis=1).reindex(C.index))
    return comb[hl].dropna()


def main():
    r0 = build_grandstack()                       # ~12%-vol book, net
    base_cagr, base_dd, base_sh, base_term, base_worst = metrics(r0)
    years = len(r0) / ANN

    # sweep constant leverage (multiple of the 12%-vol book)
    Ls = np.arange(1, 20.1, 0.5)
    rows = []
    for L in Ls:
        rows.append((L,) + metrics(L * r0))
    arr = rows
    cagrs = [x[1] for x in arr]
    Lstar = arr[int(np.argmax(cagrs))][0]
    halfL = round(Lstar / 2 * 2) / 2

    L_ruin = 1.0 / abs(base_worst)                # leverage at which worst day ~ -100%

    lines = ["# Highest-CAGR leverage (Kelly) on the grand-stack book\n"]
    lines.append(f"Book = validated 6-sleeve grand stack, net of real funding + "
                 f"4.5bps taker, base vol-targeted to ~12% (Sharpe {base_sh:.2f}). "
                 f"Sample {r0.index.min().date()}->{r0.index.max().date()} "
                 f"({years:.1f}y). Leverage L = multiple of the 12% book; net P&L "
                 "scales ~linearly with L (gross, fees, funding all scale).\n")
    lines.append("## CAGR vs leverage (the Kelly curve)\n")
    lines.append("| leverage | ~ann vol | Sharpe | **CAGR** | maxDD | worst day | "
                 f"x in {years:.1f}y |")
    lines.append("|---|---|---|---|---|---|---|")
    for L, cagr, dd, sh, term, worst in arr:
        if L in (1, 2, 3, 5, Lstar, 8, 10, 14, 20) or abs(L - halfL) < 0.01:
            vol = (L * r0).std() * np.sqrt(ANN)
            mark = "  <- MAX CAGR" if abs(L - Lstar) < 0.01 else (
                "  <- half-Kelly" if abs(L - halfL) < 0.01 else "")
            lines.append(f"| {L:.1f}x | {vol:.0%} | {sh:+.2f} | **{cagr:+.0%}** | "
                         f"{dd:+.0%} | {worst:+.0%} | {term:.1f}x{mark} |")

    cagr_star, dd_star, sh_star, term_star, worst_star = metrics(Lstar * r0)
    cagr_half, dd_half, sh_half, term_half, worst_half = metrics(halfL * r0)
    lines.append(f"\n- **Max CAGR is at ~{Lstar:.1f}x leverage** (≈{(Lstar*r0).std()*np.sqrt(ANN):.0%} "
                 f"vol): CAGR **{cagr_star:+.0%}**, but maxDD **{dd_star:+.0%}** and "
                 f"worst day **{worst_star:+.0%}**. Beyond it, CAGR falls (vol drag).")
    lines.append(f"- **Liquidation reality:** the book's worst day is "
                 f"{base_worst:+.1%} at 1x; at ~**{L_ruin:.0f}x** a single day like "
                 f"that ≈ −100% = account wiped. Full-Kelly {Lstar:.0f}x sits "
                 f"dangerously close — on HL, margin/liquidation would trigger well "
                 "before −100%, and fat tails make realized Kelly LOWER than the "
                 "in-sample optimum. **Full Kelly is not survivable live.**")
    lines.append(f"- **Half-Kelly (~{halfL:.1f}x)** keeps most of the growth — CAGR "
                 f"**{cagr_half:+.0%}** — at far lower risk (maxDD {dd_half:+.0%}, "
                 f"worst day {worst_half:+.0%}). This is the practical aggressive pick.")
    lines.append(f"- Sharpe stays ~{base_sh:.1f} at every leverage (the ratio is "
                 "leverage-invariant); leverage buys CAGR up to Kelly, then destroys "
                 "it. The number that improved was never the Sharpe — only the risk "
                 "you take to convert it into return.\n")

    # equity curves: 1x, half-Kelly, max-CAGR
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for L, color, lw in [(1.0, "#888", 1.4), (halfL, "#2980b9", 1.8),
                         (Lstar, "#c0392b", 2.2)]:
        c, d_, s_, t_, w_ = metrics(L * r0)
        (1 + (L * r0)).cumprod().plot(ax=ax, color=color, lw=lw,
            label=f"{L:.1f}x  (CAGR {c:+.0%}, maxDD {d_:+.0%})")
    ax.set_yscale("log")
    ax.set_title("Grand-stack book at higher leverage — CAGR vs survivability "
                 "(log scale, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, title=f"Sharpe ~{base_sh:.1f} at all leverage")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "kelly_cagr.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "kelly_cagr.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/kelly_cagr.md + png")


if __name__ == "__main__":
    main()
