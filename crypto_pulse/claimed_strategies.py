"""Try the publicly-claimed high-Sharpe strategies on crypto, honestly.

Two archetypes from the shared X/Quantitativo writeups, implemented faithfully and
tested net of cost on the HL crypto universe — and checked for (a) standalone OOS
robustness and (b) whether they DIVERSIFY / mitigate the downside of the grand-stack
book (low-exposure long-only mean-reversion should buy crashes = natural hedge).

  IBS_MR  — Quantitativo index mean-reversion, per coin, long-only:
            HLmean = rolling mean of (High-Low), 25d
            IBS    = (Close-Low)/(High-Low)
            lower  = rolling max(High,10) - 2.5*HLmean
            ENTER long when Close < lower AND IBS < 0.3; EXIT when Close > prev High.
            Equal-weight active longs; in cash otherwise (low exposure).
  ROTMOM  — NDX-style long-only rotational momentum:
            trade only when market (BTC) > its 200d MA; monthly rebalance; among
            coins above their own 100d MA with positive 120d return, buy top-N by
            120d momentum, equal weight, hold 1 month.

Net of 4.5bps taker. IS=first60/OOS=last40 of HL era. Run from crypto_pulse/:
    python claimed_strategies.py   (-> research/claimed_strategies.md + png)
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


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan, calmar=np.nan, expo=np.nan)
    cum = (1 + p).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    sh = p.mean() / p.std() * np.sqrt(ANN) if p.std() > 0 else np.nan
    cagr = cum.iloc[-1] ** (ANN / len(p)) - 1 if cum.iloc[-1] > 0 else -1
    return dict(sharpe=sh, cagr=cagr, maxdd=dd,
                calmar=cagr / abs(dd) if dd < 0 else np.nan)


def vt(p, target=0.12):
    return p * (target / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def ibs_mr(C, V, H, L):
    """Per-coin IBS mean-reversion, long-only. Returns daily portfolio return +
    daily exposure (fraction of names active)."""
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    rng = (H - L).replace(0, np.nan)
    hlmean = rng.rolling(25).mean()
    ibs = (C - L) / rng
    lower = H.rolling(10).max() - 2.5 * hlmean
    enter = (C < lower) & (ibs < 0.3) & elig
    exit_ = C > H.shift(1)
    # build holding state per coin (vectorized state machine)
    pos = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for c in C.columns:
        e = enter[c].fillna(False).values
        x = exit_[c].fillna(False).values
        el = elig[c].fillna(False).values
        held = np.zeros(len(e)); cur = 0.0
        for i in range(len(e)):
            if cur == 0.0 and e[i]:
                cur = 1.0
            elif cur == 1.0 and (x[i] or not el[i]):
                cur = 0.0
            held[i] = cur
        pos[c] = held
    nactive = pos.sum(axis=1)
    w = pos.div(nactive.replace(0, np.nan), axis=0).fillna(0.0)   # equal-weight longs
    wl = w.shift(1)
    ret = (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
    expo = wl.abs().sum(axis=1)
    return ret, expo


def rotmom(C, V, topn=8):
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    mkt = C["BTC"] if "BTC" in C else C.mean(axis=1)
    bull = (mkt > mkt.rolling(200).mean())
    above = C > C.rolling(100).mean()
    mom = C / C.shift(120) - 1
    cand = above & (mom > 0) & elig
    score = mom.where(cand)
    # monthly rebalance: hold weights for ~30d
    n = len(C); reb = pd.Series(np.arange(n) % 30 == 0, index=C.index)
    w = pd.DataFrame(0.0, index=C.index, columns=C.columns)
    for d in C.index[reb]:
        if not bool(bull.get(d, False)):
            continue
        s = score.loc[d].dropna().sort_values(ascending=False)
        picks = s.head(topn).index
        if len(picks):
            w.loc[d, picks] = 1.0 / len(picks)
    w = w.replace(0.0, np.nan).where(reb).ffill(limit=29).fillna(0.0)
    # zero out when market not bull (carry the last bull weights only while bull)
    w = w.mul(bull.astype(float), axis=0)
    wl = w.shift(1)
    ret = (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
    return ret, wl.abs().sum(axis=1)


def grandstack_series():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    raw = ms.build_sleeves(C, V, H, L, F)
    adm = ["TREND", "CARRY", "BAB", "SQUEEZE", "ACCEL"]
    sl = {k: ms.vt(raw[k]) for k in adm}
    sl["FUNDFADE"] = ms.vt(gs.funding_fade(C, V, H, L, F, R, elig))
    P = pd.DataFrame(sl).dropna()
    rw = (1 / P.std()) / (1 / P.std()).sum()
    return vt((P * rw).sum(axis=1).reindex(C.index))


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    ibs_ret, ibs_expo = ibs_mr(C, V, H, L)
    rot_ret, rot_expo = rotmom(C, V)
    gstack = grandstack_series()

    hl = C.index >= HL_START
    idxhl = C.index[hl]
    cut = idxhl[int(len(idxhl) * 0.6)]

    def rep(p, vtgt=True):
        q = vt(p) if vtgt else p
        s = stats(q[hl])
        si = stats(q[(q.index < cut) & hl])["sharpe"]
        so = stats(q[(q.index >= cut) & hl])["sharpe"]
        return s, si, so

    lines = ["# Publicly-claimed strategies on crypto + downside diversification\n"]
    lines.append(f"Net of {TAKER}bps taker, IS=first60/OOS=last40 of HL era. "
                 "IBS_MR and ROTMOM are long-only; shown vol-targeted to 12% for "
                 "comparison, with raw exposure noted.\n")
    lines.append("## Standalone (vol-targeted to 12%)\n")
    lines.append("| strategy | Sharpe | IS | OOS | CAGR | maxDD | Calmar | avg expo |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for nm, p, expo in [("IBS mean-reversion", ibs_ret, ibs_expo),
                        ("rotational momentum", rot_ret, rot_expo),
                        ("grand stack (ref)", gstack, None)]:
        s, si, so = rep(p)
        e = f"{expo[hl].mean():.0%}" if expo is not None else "~100%"
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {si:+.2f} | {so:+.2f} | "
                     f"{s['cagr']:+.0%} | {s['maxdd']:+.0%} | {s['calmar']:.2f} | {e} |")

    # correlation + does adding MR to the grand stack reduce drawdown?
    P = pd.DataFrame({"grand": vt(gstack)[hl], "IBS": vt(ibs_ret)[hl],
                     "ROT": vt(rot_ret)[hl]}).dropna()
    corr = P.corr()
    lines.append(f"\nCorrelation to grand stack: IBS {corr.loc['grand','IBS']:+.2f}, "
                 f"ROT {corr.loc['grand','ROT']:+.2f}\n")

    lines.append("## Downside mitigation — blend grand stack + IBS dip-buyer\n")
    lines.append("Equal-risk blends, vol-targeted to 12%. The low-exposure long-only "
                 "MR sleeve buys oversold crashes, so it should cushion the directional "
                 "book's drawdowns.\n")
    lines.append("| book | Sharpe | CAGR | maxDD | Calmar | worst day |")
    lines.append("|---|---|---|---|---|---|")
    blends = {
        "grand stack alone": vt(gstack),
        "grand + IBS (50/50 risk)": vt(0.5 * gstack + 0.5 * ibs_ret),
        "grand + IBS + ROT (1/3 each)": vt((gstack + ibs_ret + rot_ret) / 3),
    }
    for nm, p in blends.items():
        s = stats(p[hl])
        wd = p[hl].min()
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {s['cagr']:+.0%} | "
                     f"{s['maxdd']:+.0%} | {s['calmar']:.2f} | {wd:+.1%} |")

    sg = stats(vt(gstack)[hl]); sb = stats(blends['grand + IBS (50/50 risk)'][hl])
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- IBS mean-reversion on crypto: standalone Sharpe "
                 f"{stats(vt(ibs_ret)[hl])['sharpe']:+.2f} (IS/OOS in table), "
                 f"correlation to the book {corr.loc['grand','IBS']:+.2f}, avg "
                 f"exposure {ibs_expo[hl].mean():.0%}. " +
                 ("It diversifies and the blend improves Calmar "
                  f"({sb['calmar']:.2f} vs {sg['calmar']:.2f}) / maxDD "
                  f"({sb['maxdd']:+.0%} vs {sg['maxdd']:+.0%}) — a genuine "
                  "downside-mitigating sleeve.\n" if sb['calmar'] > sg['calmar']
                  else "It does not improve the blended Calmar here.\n"))

    fig, ax = plt.subplots(figsize=(11, 5))
    for nm, p, col, lw in [("grand stack", vt(gstack), "#888", 1.5),
                           ("grand + IBS 50/50", blends['grand + IBS (50/50 risk)'],
                            "#c0392b", 2.2),
                           ("IBS standalone", vt(ibs_ret), "#16a085", 1.0)]:
        s = stats(p[hl])
        (1 + p[hl].fillna(0)).cumprod().plot(ax=ax, color=col, lw=lw,
            label=f"{nm} (Sharpe {s['sharpe']:.2f}, DD {s['maxdd']:.0%})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Claimed strategies on crypto + downside diversification (net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "claimed_strategies.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "claimed_strategies.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/claimed_strategies.md + png")


if __name__ == "__main__":
    main()
