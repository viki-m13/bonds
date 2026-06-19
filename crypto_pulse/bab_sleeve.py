"""BAB / LOW-BETA sleeve — the one cross-sectional crypto factor the research flags
as genuinely taker-viable and distinct from trend/carry/order-flow.

Frazzini-Pedersen "Betting Against Beta": leverage-/lottery-constrained investors
bid up high-beta assets, so low-beta earns higher risk-adjusted returns. Crypto is
a textbook setting — retail crowds into high-beta alts. The crypto low-vol/low-beta
anomaly is documented on LIQUID coins (Grobys-Junttila; FRL low-vol papers), and it
SURVIVES taker costs because the signal is SLOW (months-long beta/vol ranks => low
turnover), unlike size/liquidity/reversal (microcap artifacts that don't).

This builds a beta-neutral long-low-beta / short-high-beta book on the HL universe
(BTC as the market), long lookbacks, weekly hold, and tests honestly whether it
(a) is positive net of 4.5bps taker + funding and (b) ADDS to the validated
3-sleeve book (low correlation + higher combined Sharpe IS/OOS).

Run from crypto_pulse/:  python bab_sleeve.py  (-> research/bab_sleeve.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import three_sleeve as ts

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
TAKER = 4.5
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, ann=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(ANN), ann=p.mean() * ANN,
                maxdd=(cum / cum.cummax() - 1).min())


def rolling_beta(R, mkt, lb=90):
    """Causal rolling beta of each coin vs the market (BTC), window lb days."""
    cov = R.rolling(lb).cov(mkt)
    var = mkt.rolling(lb).var()
    return cov.div(var, axis=0)


def sleeve_pnl(w, R, F):
    wl = w.shift(1)
    return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
            - (wl * F).sum(axis=1))


def vt(p, target=0.12):
    return p * (target / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()

    mkt = R["BTC"] if "BTC" in R else R.mean(axis=1)

    # --- BAB: rank by trailing 90d beta, long low / short high, beta-demeaned ---
    beta = rolling_beta(R, mkt, lb=90)
    bz = (-beta).where(elig)                          # higher score = lower beta
    bz = bz.sub(bz.mean(axis=1), axis=0)
    wb = bz.div(bz.abs().sum(axis=1), axis=0)
    # weekly hold to keep turnover/cost down
    reb = pd.Series(np.arange(len(wb)) % 7 == 0, index=wb.index)
    wb = wb.where(reb, axis=0).ffill(limit=6)
    bab = sleeve_pnl(wb, R, F)

    # --- LOW-VOL: long-lookback (90d) idiosyncratic vol, long low / short high ---
    lvol = R.rolling(90).std()
    lz = (-lvol).where(elig); lz = lz.sub(lz.mean(axis=1), axis=0)
    wl = lz.div(lz.abs().sum(axis=1), axis=0)
    wl = wl.where(reb, axis=0).ffill(limit=6)
    lowvol = sleeve_pnl(wl, R, F)

    # --- the validated 3 sleeves ---
    base3 = ts.sleeves(C, V, H, L, F)
    P3 = pd.DataFrame(base3).dropna()
    rw3 = (1 / P3.std()) / (1 / P3.std()).sum()
    crypto3 = (P3 * rw3).sum(axis=1).reindex(C.index)

    hl = C.index >= HL_START
    idxhl = C.index[hl]
    cut = idxhl[int(len(idxhl) * 0.6)]

    def rep(p):
        s = stats(p[hl])
        return s, stats(p[(p.index < cut) & hl])["sharpe"], \
            stats(p[(p.index >= cut) & hl])["sharpe"]

    lines = ["# BAB / low-beta sleeve — does it add to the 3-sleeve book?\n"]
    lines.append(f"HL era, real funding + {TAKER}bps taker, IS=first60/OOS=last40, "
                 "weekly hold. BAB = long-low-beta/short-high-beta (BTC market, 90d "
                 "beta); low-vol = long-low/short-high 90d idio vol.\n")
    lines.append("## Standalone (net, vol-targeted)\n")
    lines.append("| sleeve | Sharpe | IS | OOS |")
    lines.append("|---|---|---|---|")
    for nm, p in [("BAB (low-beta)", vt(bab)), ("low-vol", vt(lowvol)),
                  ("crypto 3-sleeve (ref)", vt(crypto3))]:
        s, i, o = rep(p)
        lines.append(f"| {nm} | {s['sharpe']:+.2f} | {i:+.2f} | {o:+.2f} |")

    # correlation of BAB/lowvol to the 3-sleeve book
    Pall = pd.DataFrame({"crypto3": crypto3[hl], "BAB": bab[hl],
                         "lowvol": lowvol[hl]}).dropna()
    corr = Pall.corr()
    lines.append(f"\nCorrelation to 3-sleeve book: BAB {corr.loc['crypto3','BAB']:+.2f}, "
                 f"low-vol {corr.loc['crypto3','lowvol']:+.2f}\n")

    # does adding BAB help? Sharpe-optimal IS weights over {crypto3, BAB}
    B = pd.DataFrame({"crypto3": crypto3, "BAB": bab}).reindex(C.index)
    Bhl = B[hl].dropna()
    BIS = Bhl[Bhl.index < cut]
    isr = {c: stats(BIS[c])["sharpe"] for c in Bhl.columns}
    raw = {c: max(isr[c], 0.0) / BIS[c].std() for c in Bhl.columns}
    tot = sum(raw.values()) or 1.0
    ow = {c: raw[c] / tot for c in Bhl.columns}
    comb = vt(sum(ow[c] * B[c] for c in B.columns))

    lines.append("## Combination: 3-sleeve + BAB (Sharpe-optimal IS weights)\n")
    lines.append(f"IS weights: crypto3 {ow['crypto3']:.0%}, BAB {ow['BAB']:.0%}\n")
    lines.append("| book | Sharpe | IS | OOS | ann | maxDD |")
    lines.append("|---|---|---|---|---|---|")
    for nm, p in [("crypto 3-sleeve (base)", vt(crypto3)),
                  ("3-sleeve + BAB", comb)]:
        s, i, o = rep(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['ann']:+.1%} | {s['maxdd']:+.1%} |")

    sb = stats(vt(crypto3)[hl])["sharpe"]
    sc = stats(comb[hl])["sharpe"]
    lines.append("")
    lines.append("## Verdict\n")
    verdict = ("ADDS value" if sc > sb + 0.05 else "does NOT meaningfully add")
    lines.append(f"- BAB **{verdict}**: combined Sharpe {sc:+.2f} vs 3-sleeve {sb:+.2f}. "
                 f"BAB standalone Sharpe {stats(vt(bab)[hl])['sharpe']:+.2f}, "
                 f"correlation to the book {corr.loc['crypto3','BAB']:+.2f}. "
                 "The research flagged low-beta as the one cross-sectional crypto "
                 "factor that survives taker costs (slow signal, low turnover); this "
                 "is the honest in/out-of-sample test of that claim on our universe.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + vt(crypto3)[hl].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5,
        label=f"3-sleeve (Sharpe {sb:.2f})")
    (1 + comb[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2,
        label=f"+ BAB (Sharpe {sc:.2f})")
    (1 + vt(bab)[hl].fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=1.0,
        ls="--", label=f"BAB standalone (Sharpe {stats(vt(bab)[hl])['sharpe']:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("BAB / low-beta sleeve vs the crypto 3-sleeve book (HL, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "bab_sleeve.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "bab_sleeve.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/bab_sleeve.md + png")


if __name__ == "__main__":
    main()
