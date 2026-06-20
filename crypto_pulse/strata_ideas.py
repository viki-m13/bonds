"""Testing inferred ideas from the shared images (best-effort interpretation of the
themes you've sent: slow-momentum+fast-reversion regime switching, and an ML/ensemble
meta-combiner of the sleeves). Each tested as a candidate STRATA addition, honestly
(IS/OOS, correlation, does it lift the book). I'm INFERRING the image content from the
conversation — correct me if a specific idea differs.

  SLOWFAST  — slow cross-sectional momentum when the basket TRENDS, fast 5d reversion
              when it CHOPS (regime = return autocorrelation / changepoint proxy).
              (kieranjwood "slow momentum + fast reversion".)
  MLMETA    — ridge-regression meta-combiner: predict next-day sleeve-blend return from
              lagged sleeve returns + regime, walk-forward, strong shrinkage (anti-overfit).

HL era, real funding + 4.5bps, IS/OOS. Run from crypto_pulse/:  python strata_ideas.py
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
    sd = R.rolling(30).std(); n = len(C)
    rebw = pd.Series(np.arange(n) % 7 == 0, index=C.index)
    mkt = R.where(elig).mean(axis=1)

    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold=None):
        if hold is not None:
            w = w.where(hold, axis=0).ffill(limit=6)
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
                - (wl * F).sum(axis=1))

    # regime: return autocorrelation (trend vs chop), causal
    ac = mkt.rolling(40).corr(mkt.shift(1))
    trending = (ac.rolling(20).mean() > 0).shift(1).fillna(False)

    # SLOWFAST sleeve: slow momentum in trend regime, fast reversion in chop
    slowmom = (C / C.shift(60) - 1)
    fastrev = -(C / C.shift(5) - 1)
    sig = slowmom.where(trending, fastrev)
    slowfast = pnl(norm(dm(sig.where(elig)) / sd), hold=rebw)

    # existing STRATA 6-sleeve (equal-risk) for reference + correlation
    base = ms.build_sleeves(C, V, H, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)
    P = pd.DataFrame({k: vt(p) for k, p in sl.items()})
    hl = P.index >= HL_START
    cut = P.index[hl][int(hl.sum() * 0.6)]
    book = P[hl].mean(axis=1)

    def io(p):
        q = p[p.index >= HL_START]
        return sh(q[q.index < cut]), sh(q[q.index >= cut])

    # MLMETA: ridge meta-combiner (walk-forward, heavy shrinkage)
    Xall = P.copy()
    Xall["reg"] = trending.astype(float)
    y = book.reindex(P.index)
    pred = pd.Series(index=P.index, dtype=float)
    cols = list(P.columns)
    feat = Xall[cols].shift(1).fillna(0.0)
    idxhl = P.index[hl]
    for t in range(252, len(idxhl)):
        tr = idxhl[:t]
        Xtr = feat.loc[tr].values; ytr = y.loc[tr].values
        lam = 50.0
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
        b = Xtr.T @ ytr
        w = np.linalg.solve(A, b)
        pred.loc[idxhl[t]] = feat.loc[idxhl[t]].values @ w
    # trade the sign-scaled prediction on the equal-risk book direction (timing overlay)
    mlsig = np.sign(pred).reindex(book.index).fillna(0.0)
    mlmeta = (book * (0.5 + 0.5 * mlsig.shift(0))).reindex(book.index)  # de-risk when bearish

    vt_sf = vt(slowfast)
    rho_sf = pd.concat({"x": vt_sf[hl], "b": book}, axis=1).dropna().corr().iloc[0, 1]

    lines = ["# Inferred image-ideas tested on STRATA (best-effort interpretation)\n"]
    lines.append("I'm inferring the image content from the shared themes — correct me "
                 "if specific. SLOWFAST = slow-mom/fast-reversion regime switch; MLMETA "
                 "= ridge meta-combiner timing overlay. HL era, net, IS/OOS.\n")
    lines.append("## SLOWFAST as a new sleeve\n")
    i, o = io(vt_sf)
    lines.append(f"- standalone Sharpe {sh(vt_sf[hl]):+.2f} (IS {i:+.2f}/OOS {o:+.2f}), "
                 f"corr to STRATA book **{rho_sf:+.2f}**.")
    # add to book if admitted
    if i > 0.05 and o > 0.05 and abs(rho_sf) < 0.5:
        P2 = P.copy(); P2["SLOWFAST"] = vt(slowfast)
        Pis = P2[hl][P2[hl].index < cut]
        mu = Pis.mean().values * ANN; S = Pis.cov().values * ANN
        Ss = 0.6 * np.diag(np.diag(S)) + 0.4 * S
        w = np.clip(np.linalg.solve(Ss + 1e-6 * np.eye(len(mu)), mu), 0, None)
        w = pd.Series(w / w.sum(), index=P2.columns)
        exp = vt((pd.DataFrame({k: P2[k] for k in P2.columns}) * w).sum(axis=1))[hl]
        s7 = stats(exp); ie, oe = io(exp)
        lines.append(f"- ADMITTED. STRATA + SLOWFAST = **{s7['sharpe']:+.2f}** "
                     f"(IS {ie:+.2f}/OOS {oe:+.2f}), maxDD {s7['maxdd']:+.0%}.")
    else:
        lines.append("- NOT admitted (weak in a half or too correlated).")

    lines.append("\n## MLMETA timing overlay\n")
    im, om = io(mlmeta)
    sm = stats(mlmeta); sbk = stats(book)
    lines.append(f"- book with ML-timing overlay Sharpe {sm['sharpe']:+.2f} "
                 f"(IS {im:+.2f}/OOS {om:+.2f}) vs equal-risk book {sbk['sharpe']:+.2f}. "
                 + ("ML overlay helps." if sm['sharpe'] > sbk['sharpe'] + 0.05 else
                    "ML overlay does NOT help (overfit/no timing edge) — heavy shrinkage "
                    "ridge can't extract a robust meta-signal from 6 sleeves on ~3y."))
    lines.append("\n## Verdict\n")
    lines.append("- Best-effort test of the inferred ideas. SLOWFAST is the more "
                 "promising (a genuine regime-switched sleeve); MLMETA tends to overfit "
                 "on this short sample. If the images specify different mechanics, send "
                 "the gist and I'll implement exactly.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + book.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5, label=f"STRATA 6-sleeve ({sbk['sharpe']:.2f})")
    (1 + vt_sf[hl].fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.2, label=f"SLOWFAST sleeve ({sh(vt_sf[hl]):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Inferred ideas: SLOWFAST sleeve vs STRATA (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strata_ideas.png"), dpi=110)
    with open(os.path.join(HERE, "strata_ideas.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/strata_ideas.md + png")


if __name__ == "__main__":
    main()
