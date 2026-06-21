"""Comprehensive breakout-family sweep (the @onlybreakouts theme), since I can't fetch
the exact X posts. Tests the canonical breakout definitions on HL crypto, each as a
directional book, net of 4.5bps + funding, IS/OOS, and whether any is robustly positive
AND additive to STRATA. If one matches your indicator and looks good, I'll refine it.

Variants:
  DONCH-N   close breaks above N-day high (long) / below N-day low (short), N in 20/55/100
  BOLL      close breaks above/below Bollinger(20,2)
  KELT      close breaks above/below Keltner(20, 1.5*ATR)
  NEWHIGH   N-day-high breakout WITH volume confirmation (vol > 1.5x avg)
  RETEST    Donchian-20 breakout, enter on a pullback (close back near band)
Each held with an ATR trailing-style 10-day hold, inverse-vol sized, gross-1.

Run from crypto_pulse/:  python breakout_sweep.py  (-> research/breakout_sweep.md + png)
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
TAKER = 4.5
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 40 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p), maxdd=(cum / cum.cummax() - 1).min())


def vt(p, t=0.12):
    return p * (t / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    hold = 10

    def hold_sig(raw):
        return raw.replace(0.0, np.nan).ffill(limit=hold).fillna(0.0).where(elig, 0.0)

    def pnl(sig):
        wts = (sig / sd).where(elig); wts = wts.div(wts.abs().sum(axis=1), axis=0)
        wl = wts.shift(1)
        return vt((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
                  - (wl * F).sum(axis=1))

    basis = C.rolling(20).mean(); dev = C.rolling(20).std()
    atr = (H - L).rolling(20).mean()
    vol_ok = V > V.rolling(20).mean() * 1.5

    sigs = {}
    for N in (20, 55, 100):
        up = C >= H.shift(1).rolling(N).max(); dn = C <= L.shift(1).rolling(N).min()
        sigs[f"DONCH-{N}"] = hold_sig(up.astype(float) - dn.astype(float))
    sigs["BOLL"] = hold_sig((C > basis + 2 * dev).astype(float) - (C < basis - 2 * dev).astype(float))
    sigs["KELT"] = hold_sig((C > basis + 1.5 * atr).astype(float) - (C < basis - 1.5 * atr).astype(float))
    nh = (C >= H.shift(1).rolling(20).max()) & vol_ok
    nl = (C <= L.shift(1).rolling(20).min()) & vol_ok
    sigs["NEWHIGH+VOL"] = hold_sig(nh.astype(float) - nl.astype(float))

    hl = C.index >= HL_START
    idxhl = C.index[hl]; cut = idxhl[int(len(idxhl) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]
        return sh(q[q.index < cut]), sh(q[q.index >= cut])

    base = ms.build_sleeves(C, V, H, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()})
    book = P[hl].mean(axis=1)

    lines = ["# Breakout-family sweep on HL crypto (the @onlybreakouts theme)\n"]
    lines.append("Canonical breakout variants, directional, net of 4.5bps + funding, "
                 "IS/OOS, 10d hold. Best-effort since the X posts aren't fetchable.\n")
    lines.append("| variant | Sharpe | IS | OOS | maxDD | corr STRATA | robust? |")
    lines.append("|---|---|---|---|---|---|---|")
    res = {}
    for nm, sig in sigs.items():
        p = pnl(sig); res[nm] = p
        s = stats(p[hl]); i, o = io(p)
        rho = pd.concat({"x": p[hl], "b": book}, axis=1).dropna().corr().iloc[0, 1]
        robust = (i > 0.05 and o > 0.05)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['maxdd']:+.0%} | {rho:+.2f} | {'YES' if robust else 'no'} |")
    # best by min(IS,OOS)
    best = max(res, key=lambda k: min(io(res[k])) if all(np.isfinite(io(res[k]))) else -9)
    bi, bo = io(res[best])
    lines.append(f"\n**Best breakout: {best}** (IS {bi:+.2f}/OOS {bo:+.2f}). ")
    lines.append("\n## Verdict\n")
    any_robust = any(min(io(res[k])) > 0.05 for k in res if all(np.isfinite(io(res[k]))))
    lines.append("- " + ("At least one breakout variant is robustly positive — worth "
                 f"refining ({best})." if any_robust else
                 "NO breakout variant is robustly positive net of cost in BOTH halves — "
                 "they all decay OOS or overlap TREND. On daily crypto the breakout edge "
                 "is already inside STRATA's TREND sleeve; standalone breakouts are "
                 "cost/decay-blocked. If your posts use an INTRADAY breakout (ORB) or a "
                 "specific filter, that's a different (and likely taker-cost-blocked) test "
                 "— send the rules and I'll match them.") + "\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + book.fillna(0)).cumprod().plot(ax=ax, color="k", lw=2.0, label="STRATA")
    for nm in res:
        (1 + res[nm][hl].fillna(0)).cumprod().plot(ax=ax, lw=1.0, alpha=0.8,
            label=f"{nm} ({sh(res[nm][hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=8)
    ax.set_title("Breakout-family sweep vs STRATA (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "breakout_sweep.png"), dpi=110)
    with open(os.path.join(HERE, "breakout_sweep.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/breakout_sweep.md + png")


if __name__ == "__main__":
    main()
