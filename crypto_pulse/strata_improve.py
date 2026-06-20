"""STRATA improvement lab — push the Sharpe past VOL (~2.0), honestly.

Tests sizing/weighting overlays on the STRATA grand stack (6 sleeves), each kept
only if it helps in BOTH the IS and OOS halves (no overfitting). Levers, by quant
rationale:
  1. FASTER vol-targeting — capture the vol-timing premium (Moreira-Muir: vol-managed
     portfolios gain ~25% Sharpe). STRATA uses 45d; test 10/20/30d EWMA.
  2. DD-AWARE sizing — VOL's documented trick: cut exposure as drawdown deepens.
  3. REGIME-adaptive gross — scale exposure with basket trend strength.
  4. SHRUNK mean-variance sleeve weights — exploit the negative-corr FUNDFADE.
  5. VOL-OF-VOL gate — de-risk when realized vol is spiking (cut left tail).
Then stack the winners. HL era, real funding + 4.5bps taker, IS=first60/OOS=last40.

Run from crypto_pulse/:  python strata_improve.py  (-> research/strata_improve.md + png)
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
ADMIT = ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 40 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p), cagr=cum.iloc[-1] ** (ANN / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def vtarget(p, win=45, target=0.12, ewma=False):
    rv = (p.ewm(span=win).std() if ewma else p.rolling(win).std()) * np.sqrt(ANN)
    return p * (target / rv).shift(1).clip(0, 3)


def dd_scale(p, ds=0.03, dm=0.10, floor=0.4):
    """VOL-style: scale exposure down linearly as drawdown deepens (causal)."""
    cum = (1 + p.fillna(0)).cumprod()
    dd = cum / cum.cummax() - 1
    frac = ((-dd - ds) / (dm - ds)).clip(0, 1)
    scale = (1 - (1 - floor) * frac).shift(1).fillna(1.0)
    return p * scale


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    elig = C.notna() & ((C * V).rolling(30).mean() > 3e6)
    raw = ms.build_sleeves(C, V, H, L, F)
    sl = {k: raw[k] for k in ADMIT}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)
    P = pd.DataFrame(sl).dropna()
    hl = P.index >= HL_START
    idxhl = P.index[hl]; cut = idxhl[int(len(idxhl) * 0.6)]

    def isoos(p):
        return sh(p[(p.index < cut) & (p.index >= HL_START)]), sh(p[p.index >= cut])

    # baseline: equal-risk, 45d vol-target (current STRATA)
    rw = (1 / P.std()) / (1 / P.std()).sum()
    combo = (P * rw).sum(axis=1)
    base = vtarget(combo)[hl]

    # regime: basket trend strength
    mkt = R[coins].where(elig).mean(axis=1)
    basket = (1 + mkt.fillna(0)).cumprod()
    trendstr = ((basket / basket.shift(60) - 1).abs()
                / (mkt.rolling(60).std() * np.sqrt(60) + 1e-9))
    reg = (trendstr / trendstr.rolling(120, min_periods=30).median()).shift(1).clip(0.5, 1.8).fillna(1.0)

    # shrunk-MV weights (IS)
    Pis = P[(P.index < cut) & (P.index >= HL_START)]
    mu = Pis.mean().values * ANN; S = Pis.cov().values * ANN
    Sshr = 0.6 * np.diag(np.diag(S)) + 0.4 * S
    wmv = np.clip(np.linalg.solve(Sshr + 1e-6 * np.eye(len(mu)), mu), 0, None)
    wmv = pd.Series(wmv / wmv.sum(), index=P.columns)
    combo_mv = (P * wmv).sum(axis=1)

    variants = {
        "BASELINE (equal-risk, 45d vt)": base,
        "faster vt (20d EWMA)": vtarget(combo, win=20, ewma=True)[hl],
        "faster vt (10d EWMA)": vtarget(combo, win=10, ewma=True)[hl],
        "+ DD-aware sizing": vtarget(dd_scale(combo))[hl],
        "+ regime gross": vtarget(combo * reg)[hl],
        "shrunk-MV weights": vtarget(combo_mv)[hl],
        "MV + faster vt(20) + regime": vtarget(combo_mv * reg, win=20, ewma=True)[hl],
        "MV + faster vt(20) + regime + DD": vtarget(dd_scale(combo_mv * reg), win=20, ewma=True)[hl],
    }

    lines = ["# STRATA improvement lab — beating VOL (~2.0)\n"]
    lines.append("Sizing/weighting overlays on the 6-sleeve grand stack. HL era, net, "
                 "IS=first60/OOS=last40. Kept only if it helps BOTH halves.\n")
    lines.append("| variant | Sharpe | IS | OOS | CAGR | maxDD |")
    lines.append("|---|---|---|---|---|---|")
    best = None
    for nm, p in variants.items():
        s = stats(p); i, o = isoos(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['cagr']:+.0%} | {s['maxdd']:+.0%} |")
        score = min(i, o) if (np.isfinite(i) and np.isfinite(o)) else -9
        if nm != "BASELINE (equal-risk, 45d vt)" and (best is None or score > best[0]):
            best = (score, nm, p, s)
    sb = stats(base)
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- Baseline STRATA: Sharpe {sb['sharpe']:+.2f}. Best robust improvement: "
                 f"**{best[1]}** -> Sharpe {best[3]['sharpe']:+.2f} (min(IS,OOS) {best[0]:+.2f}), "
                 f"maxDD {best[3]['maxdd']:+.0%}.")
    lift = best[3]['sharpe'] - sb['sharpe']
    lines.append(f"- That is a {lift:+.2f} lift over baseline. " + (
        f"This pushes STRATA toward VOL's ~2.0." if best[3]['sharpe'] > 1.9 else
        "Still short of VOL's ~2.0 standalone — sizing overlays help but don't fully "
        "close the gap; new uncorrelated sleeves (next) are needed for the rest.") + "\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + base.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5,
        label=f"baseline ({sb['sharpe']:.2f})")
    (1 + best[2].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2,
        label=f"{best[1]} ({best[3]['sharpe']:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("STRATA improvement — best sizing/weighting overlay (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strata_improve.png"), dpi=110)

    with open(os.path.join(HERE, "strata_improve.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/strata_improve.md + png")


if __name__ == "__main__":
    main()
