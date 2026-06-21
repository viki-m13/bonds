"""Experiment: TTM Squeeze / Squeeze-Momentum breakout indicator (best-effort
interpretation of the shared @onlybreakouts / @polishquant posts — a volatility-
contraction breakout. Correct me if the tweet specifies different rules).

TTM Squeeze (Carter):
  * SQUEEZE ON  = Bollinger Bands (20, 2.0) INSIDE Keltner Channels (20, 1.5*ATR)
    — volatility contraction / coiling.
  * MOMENTUM    = linreg(close - avg( (donchian20 mid + sma20) / 2 ), 20)
  * SIGNAL      = when the squeeze RELEASES (BB exit KC), trade in the momentum
    direction (the breakout).

Tested 3 ways on the HL crypto universe, net of 4.5bps taker + funding, IS/OOS:
  A. TS-directional: per-coin long/short on squeeze-release momentum sign.
  B. XS sleeve: cross-sectional rank by squeeze-release momentum (market-neutral).
  C. as an addition to STRATA (does it lift the book / is it uncorrelated).

Run from crypto_pulse/:  python breakout_indicator.py  (-> research/breakout_indicator.md + png)
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
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p), cagr=cum.iloc[-1] ** (ANN / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def vt(p, t=0.12):
    return p * (t / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def linreg_slope(x, w):
    """rolling linear-regression endpoint value (TTM momentum histogram proxy)."""
    idx = np.arange(w)
    idx = idx - idx.mean()
    denom = (idx ** 2).sum()
    return x.rolling(w).apply(lambda y: (idx * (y - y.mean())).sum() / denom * (w - 1) / 2
                              + (y.mean()), raw=True)


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std(); n = len(C)

    # --- TTM Squeeze components (per coin, vectorized over the DataFrame) ---
    w = 20
    basis = C.rolling(w).mean()
    dev = C.rolling(w).std()
    bb_up, bb_lo = basis + 2.0 * dev, basis - 2.0 * dev
    atr = (H - L).rolling(w).mean()                      # simple range ATR proxy
    kc_up, kc_lo = basis + 1.5 * atr, basis - 1.5 * atr
    squeeze_on = (bb_up < kc_up) & (bb_lo > kc_lo)       # BB inside KC
    released = squeeze_on.shift(1).fillna(False) & ~squeeze_on  # squeeze just fired

    # momentum histogram: close - average(donchian-mid, sma), detrended
    dch_mid = (H.rolling(w).max() + L.rolling(w).min()) / 2
    mom_src = C - (dch_mid + basis) / 2
    mom = mom_src - mom_src.rolling(w).mean()            # TTM-style momentum

    # signal: hold the breakout direction from the last release for `hold` bars
    hold = 10
    sig = pd.DataFrame(np.nan, index=C.index, columns=C.columns)
    sig = sig.mask(released & (mom > 0), 1.0).mask(released & (mom < 0), -1.0)
    sig = sig.ffill(limit=hold).fillna(0.0).where(elig, 0.0)

    def pnl(wts):
        wl = wts.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
                - (wl * F).sum(axis=1))

    # A. TS-directional (each coin its own breakout, inverse-vol, gross-normalised)
    wA = (sig / sd).where(elig); wA = wA.div(wA.abs().sum(axis=1), axis=0)
    pA = vt(pnl(wA))
    # B. XS market-neutral: rank by signed momentum on release, demeaned
    msc = (mom * sig).where(elig)                        # momentum in breakout dir
    wB = (msc.sub(msc.mean(axis=1), axis=0) / sd)
    wB = wB.div(wB.abs().sum(axis=1), axis=0)
    pB = vt(pnl(wB))

    hl = C.index >= HL_START
    idxhl = C.index[hl]; cut = idxhl[int(len(idxhl) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]
        return sh(q[q.index < cut]), sh(q[q.index >= cut])

    # STRATA 6-sleeve book
    base = ms.build_sleeves(C, V, H, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()})
    book = P[hl].mean(axis=1)

    lines = ["# Breakout indicator experiment — TTM Squeeze (best-effort from the X posts)\n"]
    lines.append("Volatility-squeeze breakout (BB inside KC -> release -> trade momentum "
                 "direction). HL era, net of 4.5bps + funding, IS/OOS. I'm inferring this "
                 "is the indicator from @onlybreakouts/@polishquant — correct me if not.\n")
    lines.append(f"Squeeze fires on ~{released[hl].sum().sum()} coin-days "
                 f"({100*released[hl].mean().mean():.1f}% of coin-days).\n")
    lines.append("| variant | Sharpe | IS | OOS | maxDD | corr to STRATA |")
    lines.append("|---|---|---|---|---|---|")
    for nm, p in [("A. TS-directional breakout", pA), ("B. XS market-neutral", pB)]:
        s = stats(p[hl]); i, o = io(p)
        cc = pd.concat({"x": p[hl], "b": book}, axis=1).dropna()
        rho = cc["x"].corr(cc["b"])
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['maxdd']:+.0%} | {rho:+.2f} |")

    # C. add the better breakout variant to STRATA (shrunk-MV)
    cand = pB if sh(pB[hl]) >= sh(pA[hl]) else pA
    candnm = "XS" if cand is pB else "TS"
    iB, oB = io(cand)
    rhoB = pd.concat({"x": cand[hl], "b": book}, axis=1).dropna().corr().iloc[0, 1]
    lines.append(f"\n## Add breakout ({candnm}) to STRATA\n")
    if iB > 0.05 and oB > 0.05 and abs(rhoB) < 0.5:
        P2 = P.copy(); P2["BREAKOUT"] = cand
        Pi = P2[hl][P2[hl].index < cut]
        mu = Pi.mean().values * ANN; S = Pi.cov().values * ANN
        Ss = 0.6 * np.diag(np.diag(S)) + 0.4 * S
        wts = np.clip(np.linalg.solve(Ss + 1e-6 * np.eye(len(mu)), mu), 0, None)
        wts = pd.Series(wts / wts.sum(), index=P2.columns)
        exp = vt((P2 * wts).sum(axis=1))[hl]
        s7 = stats(exp); ie, oe = io(exp)
        s6 = stats(vt((P * ((1/P[hl].std())/(1/P[hl].std()).sum())).sum(axis=1))[hl])
        lines.append(f"- ADMITTED (corr {rhoB:+.2f}). STRATA+breakout = "
                     f"**{s7['sharpe']:+.2f}** (IS {ie:+.2f}/OOS {oe:+.2f}), maxDD "
                     f"{s7['maxdd']:+.0%}, vs STRATA {s6['sharpe']:+.2f}.")
    else:
        lines.append(f"- NOT admitted (IS {iB:+.2f}/OOS {oB:+.2f}, corr {rhoB:+.2f}). "
                     "The squeeze breakout overlaps existing trend/squeeze sleeves or "
                     "is cost-blocked.")
    lines.append("\n## Verdict\n")
    lines.append("- This is the canonical volatility-squeeze breakout. If the posts use "
                 "different parameters (ORB, Donchian-N, ATR-multiple, volume filter) or "
                 "a different indicator, paste the rules and I'll match them exactly.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + book.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5, label="STRATA")
    (1 + pA[hl].fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.3,
        label=f"breakout TS ({sh(pA[hl]):.2f})")
    (1 + pB[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=1.3,
        label=f"breakout XS ({sh(pB[hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("TTM Squeeze breakout on HL crypto (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "breakout_indicator.png"), dpi=110)
    with open(os.path.join(HERE, "breakout_indicator.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/breakout_indicator.md + png")


if __name__ == "__main__":
    main()
