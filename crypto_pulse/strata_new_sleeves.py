"""STRATA new-sleeve hunt — find genuinely-orthogonal positive sleeves to push the
Sharpe toward/above VOL (~2.0).

Existing 6: TREND, CARRY, BAB, SQUEEZE, ACCEL, FUNDFADE. We test NOVEL candidate
sleeves, each kept only if (a) positive in BOTH IS and OOS net of cost, AND (b)
low-correlated to the existing book (genuine diversification, not redundancy):

  FUNDMOM   — funding MOMENTUM (trend in funding, distinct from carry-level/extreme):
              coins whose funding has been RISING are building crowded positioning.
  RESIDMOM  — idiosyncratic momentum: momentum of the BTC-beta RESIDUAL (Liu-Tsyvinski
              residual factor), orthogonal to raw TREND.
  LOWMAX    — lottery/skew: short high-MAX (max daily return last month) coins, long
              low-MAX (Bali MAX effect; Grobys-Junttila in crypto).
  VOLSHOCK  — attention: coins with abnormal volume expansion, trend-confirmed.
  XREV5     — cross-sectional 5d reversal, market-neutral, weekly hold (low turnover).

Combined via shrunk-MV (exploits negative-corr sleeves). HL era, real funding +
4.5bps taker, IS=first60/OOS=last40. Run from crypto_pulse/:
    python strata_new_sleeves.py  (-> research/strata_new_sleeves.md + png)
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


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    mkt = R["BTC"] if "BTC" in R else R.where(elig).mean(axis=1)
    n = len(C); rebw = pd.Series(np.arange(n) % 7 == 0, index=C.index)

    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold=None):
        if hold is not None:
            w = w.where(hold, axis=0).ffill(limit=6)
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
                - (wl * F).sum(axis=1))

    # existing 6
    base = ms.build_sleeves(C, V, H, L, F)
    sl = {k: base[k] for k in ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]}
    sl["FUNDFADE"] = gs.funding_fade(C, V, H, L, F, R, elig)

    # ---- NEW candidate sleeves ----
    new = {}
    # FUNDMOM: funding change (5d) demeaned -> long rising-funding short falling? No:
    # rising funding = building crowded longs -> fade -> short. demeaned, weekly.
    fmom = (F.rolling(5).mean() - F.rolling(20).mean())
    wfm = norm(dm((-fmom).where(elig)))
    new["FUNDMOM"] = pnl(wfm, hold=rebw)
    # RESIDMOM: residual = R - beta*mkt; momentum of cumulative residual
    beta = R.rolling(60).cov(mkt).div(mkt.rolling(60).var(), axis=0)
    resid = R.sub(beta.mul(mkt, axis=0))
    rmom = resid.rolling(20).sum()
    new["RESIDMOM"] = pnl(norm(dm(rmom.where(elig)) / sd), hold=rebw)
    # LOWMAX: short high max-daily-return (lottery), long low; weekly
    mx = R.rolling(20).max()
    new["LOWMAX"] = pnl(norm(dm((-mx).where(elig)) / sd), hold=rebw)
    # VOLSHOCK: volume expansion x trend sign
    vsh = (V.rolling(5).mean() / V.rolling(60).mean())
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    new["VOLSHOCK"] = pnl(norm((dm(vsh.where(elig)) * np.sign(trend)) / sd), hold=rebw)
    # XREV5: cross-sectional 5d reversal, weekly
    rev = -(C / C.shift(5) - 1)
    new["XREV5"] = pnl(norm(dm(rev.where(elig)) / sd), hold=rebw)

    hl = C.index >= HL_START
    idxhl = C.index[hl]; cut = idxhl[int(len(idxhl) * 0.6)]
    def io(p):
        q = p[p.index >= HL_START]
        return sh(q[q.index < cut]), sh(q[q.index >= cut])

    # existing book (shrunk-MV) for correlation reference
    P6 = pd.DataFrame({k: vt(p) for k, p in sl.items()})[hl].dropna()
    book6 = P6.mean(axis=1)

    lines = ["# STRATA new-sleeve hunt (toward Sharpe 2)\n"]
    lines.append("Candidate sleeves, net of funding + 4.5bps, HL era, IS/OOS. ADMIT if "
                 "positive in BOTH halves AND |corr to 6-sleeve book| < 0.4.\n")
    lines.append("| candidate | Sharpe | IS | OOS | corr to book | admit? |")
    lines.append("|---|---|---|---|---|---|")
    admitted = []
    for nm, p in new.items():
        vp = vt(p)
        i, o = io(vp)
        cc = pd.concat({"x": vp[hl], "b": book6}, axis=1).dropna()
        rho = cc["x"].corr(cc["b"])
        ok = (i > 0.05 and o > 0.05 and abs(rho) < 0.4)
        if ok:
            admitted.append((nm, vp))
        lines.append(f"| {nm} | {sh(vp[hl]):+.2f} | {i:+.2f} | {o:+.2f} | {rho:+.2f} | "
                     f"{'YES' if ok else 'no'} |")

    # build expanded stack: 6 + admitted, shrunk-MV on IS
    allsl = dict(sl)
    for nm, vp in admitted:
        # store raw (un-vt) pnl for the combiner consistency
        allsl[nm] = new[nm]
    Pall = pd.DataFrame({k: vt(p) for k, p in allsl.items()})[hl].dropna()
    Pis = Pall[Pall.index < cut]
    mu = Pis.mean().values * ANN; S = Pis.cov().values * ANN
    Sshr = 0.6 * np.diag(np.diag(S)) + 0.4 * S
    w = np.clip(np.linalg.solve(Sshr + 1e-6 * np.eye(len(mu)), mu), 0, None)
    w = pd.Series(w / w.sum(), index=Pall.columns)
    expanded = vt((Pall * w).sum(axis=1).reindex(C.index))[hl]

    # 6-sleeve shrunk-MV baseline for comparison
    P6r = pd.DataFrame({k: vt(p) for k, p in sl.items()})[hl].dropna()
    Pis6 = P6r[P6r.index < cut]
    mu6 = Pis6.mean().values * ANN; S6 = Pis6.cov().values * ANN
    S6s = 0.6 * np.diag(np.diag(S6)) + 0.4 * S6
    w6 = np.clip(np.linalg.solve(S6s + 1e-6 * np.eye(len(mu6)), mu6), 0, None)
    w6 = pd.Series(w6 / w6.sum(), index=P6r.columns)
    base6 = vt((P6r * w6).sum(axis=1).reindex(C.index))[hl]

    lines.append(f"\nAdmitted: {', '.join(nm for nm, _ in admitted) or 'none'}\n")
    lines.append("## Expanded STRATA vs 6-sleeve (both shrunk-MV)\n")
    lines.append("| book | Sharpe | IS | OOS | CAGR | maxDD |")
    lines.append("|---|---|---|---|---|---|")
    for nm, p in [("STRATA 6-sleeve (shrunk-MV)", base6),
                  (f"STRATA {6+len(admitted)}-sleeve (+new)", expanded)]:
        s = stats(p); i, o = io(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['cagr']:+.0%} | {s['maxdd']:+.0%} |")
    s6, se = stats(base6), stats(expanded)
    lines.append("\n## Verdict\n")
    lines.append(f"- Adding {len(admitted)} new sleeve(s) takes STRATA from "
                 f"{s6['sharpe']:+.2f} to **{se['sharpe']:+.2f}** "
                 f"(IS {io(expanded)[0]:+.2f}/OOS {io(expanded)[1]:+.2f}), maxDD "
                 f"{se['maxdd']:+.0%}. " + (
                 "Genuine breadth gain toward VOL's ~2.0." if se['sharpe'] > s6['sharpe'] + 0.05
                 else "Marginal — the new sleeves are too correlated/weak to add much."))

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + base6.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5,
        label=f"STRATA 6-sleeve ({s6['sharpe']:.2f})")
    (1 + expanded.fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2,
        label=f"STRATA +new ({se['sharpe']:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("STRATA + new sleeves (HL era, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strata_new_sleeves.png"), dpi=110)
    with open(os.path.join(HERE, "strata_new_sleeves.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/strata_new_sleeves.md + png")


if __name__ == "__main__":
    main()
