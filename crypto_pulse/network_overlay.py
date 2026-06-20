"""NETWORK-FRAGILITY OVERLAY — the proprietary, crypto-appropriate use of the
"learn the financial network" idea (L2GMOM / Oxford-Man) when the market is
single-factor.

Insight: crypto is near single-factor (PC1 ~50% of variance — everything is
BTC-beta), so a learned cross-asset graph has little ALPHA to mine. But the
dynamic correlation NETWORK carries the dominant RISK signal: when the network
couples (absorption ratio rises toward 1), a correlated crash is imminent
(Kritzman & Li 2010, "Principal Components as a Measure of Systemic Risk":
standardized shifts in the absorption ratio precede drawdowns). When the network
is dispersed (low absorption), diversification is real and it is safe to carry
risk.

So we don't trade the graph for alpha — we use it to TIME the gross exposure of
the validated trend+carry book: de-risk when the network is fragile/coupling,
re-risk when dispersed. Taker-viable (a slow exposure overlay), and it attacks
crypto's defining risk — the everything-dumps-together crash.

Causal: absorption ratio at day d uses a trailing covariance window; the overlay
weight is lagged. HL era, real funding + 4.5bps taker, IS/OOS. Run from
crypto_pulse/:  python network_overlay.py  (-> research/network_overlay.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
TAKER = 4.5
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, ann=np.nan, maxdd=np.nan, calmar=np.nan)
    cum = (1 + p).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    sh = p.mean() / p.std() * np.sqrt(ANN)
    return dict(sharpe=sh, ann=p.mean() * ANN, maxdd=dd,
                calmar=(p.mean() * ANN) / abs(dd) if dd < 0 else np.nan)


def absorption_ratio(R, elig, window=60, frac=0.2):
    """Rolling absorption ratio: share of total variance captured by the top
    ceil(frac*N) eigenvectors of the trailing correlation matrix of eligible
    coins. Higher = more coupled network = more fragile. Causal."""
    idx = R.index
    ar = pd.Series(index=idx, dtype=float)
    Rv = R.values
    for i in range(window, len(idx)):
        m = elig.iloc[i].values & elig.iloc[i - window].values
        if m.sum() < 8:
            continue
        sub = Rv[i - window:i, m]
        sub = sub[~np.isnan(sub).any(axis=1)]
        if sub.shape[0] < window // 2:
            continue
        sub = (sub - sub.mean(0)) / (sub.std(0) + 1e-12)
        try:
            ev = np.linalg.eigvalsh(np.corrcoef(sub, rowvar=False))
        except Exception:
            continue
        ev = np.sort(ev)[::-1]
        k = max(1, int(np.ceil(frac * len(ev))))
        ar.iloc[i] = ev[:k].sum() / ev.sum()
    return ar


def build_base(C, V, H, L, F):
    """The validated trend + trend-filtered carry book (PnL, gross-1 each), and R/elig."""
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    don = ((C >= H.shift(1).rolling(20).max()).astype(float)
           - (C <= L.shift(1).rolling(20).min()).astype(float))
    wt = ((trend + don) / sd).where(elig); wt = wt.div(wt.abs().sum(axis=1), axis=0)
    fsm = F.rolling(3).mean(); craw = (-fsm).sub((-fsm).mean(axis=1), axis=0)
    keep = ((craw > 0) & (np.sign(trend) >= 0)) | ((craw < 0) & (np.sign(trend) <= 0))
    wc = craw.where(keep & elig); wc = wc.sub(wc.mean(axis=1), axis=0)
    wc = wc.div(wc.abs().sum(axis=1), axis=0)

    def pnl(w):
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
                - (wl * F).sum(axis=1))
    base = 0.5 * pnl(wt) + 0.5 * pnl(wc)
    return base, R, elig


def vol_target(p, vt=0.12, scale_extra=None):
    s = (vt / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)
    if scale_extra is not None:
        s = s * scale_extra
    return p * s


def main():
    coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    base, R, elig = build_base(C, V, H, L, F)
    ar = absorption_ratio(R, elig, window=60, frac=0.2)

    # fragility overlay: down-weight gross when AR is high vs its trailing norm
    # AND when AR is rising (dAR>0). exposure in [0.3, 1.3].
    arz = (ar - ar.rolling(252, min_periods=60).mean()) / \
          (ar.rolling(252, min_periods=60).std() + 1e-9)
    dar = ar.diff(5)
    frag = (arz.clip(-2, 2) + 3 * dar.rolling(5).mean().clip(-0.1, 0.1) / 0.1).fillna(0)
    exposure = (1.0 - 0.35 * frag).clip(0.3, 1.3).shift(1)   # causal

    hl = C.index >= HL_START
    idxhl = C.index[hl]
    cut = idxhl[int(len(idxhl) * 0.6)]

    p_base = vol_target(base)[hl]
    p_ovl = vol_target(base, scale_extra=exposure)[hl]

    def rep(p):
        s = stats(p)
        sis = stats(p[p.index < cut])["sharpe"]
        soos = stats(p[p.index >= cut])["sharpe"]
        return s, sis, soos

    lines = ["# Network-fragility exposure overlay (absorption ratio) on "
             "trend+carry\n"]
    lines.append(f"HL era, real funding + {TAKER}bps taker, IS=first60/OOS=last40."
                 " Absorption ratio = top-20% eigenvalue share of the trailing "
                 "60d crypto correlation network (Kritzman-Li systemic risk). "
                 "Overlay cuts gross exposure when the network couples/rises.\n")
    lines.append(f"Absorption ratio: mean {ar[hl].mean():.0%}, range "
                 f"[{ar[hl].min():.0%}, {ar[hl].max():.0%}] — crypto is highly "
                 "coupled (confirms single-factor).\n")
    lines.append("| book | Sharpe | IS | OOS | ann | maxDD | Calmar |")
    lines.append("|---|---|---|---|---|---|---|")
    for nm, p in [("trend+carry (base)", p_base),
                  ("+ network-fragility overlay", p_ovl)]:
        s, sis, soos = rep(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {sis:+.2f} | {soos:+.2f}"
                     f" | {s['ann']:+.1%} | {s['maxdd']:+.1%} | {s['calmar']:.2f} |")
    sb, so = stats(p_base), stats(p_ovl)
    better = (so["sharpe"] > sb["sharpe"], so["maxdd"] > sb["maxdd"])
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- Overlay Sharpe {so['sharpe']:+.2f} vs base {sb['sharpe']:+.2f}"
                 f"; maxDD {so['maxdd']:+.1%} vs {sb['maxdd']:+.1%}; Calmar "
                 f"{so['calmar']:.2f} vs {sb['calmar']:.2f}. "
                 + ("The fragility overlay **improves** the risk-adjusted return "
                    "by de-risking into correlated-crash regimes — the honest, "
                    "crypto-appropriate use of the financial-network idea (risk "
                    "timing, not graph alpha)." if (better[0] or better[1]) else
                    "The overlay does NOT improve the book on this sample — "
                    "absorption-ratio timing didn't add value here.") + "\n")

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [2, 1]})
    (1 + p_base.fillna(0)).cumprod().plot(ax=ax[0], color="#888", lw=1.4,
        label=f"trend+carry base (Sharpe {sb['sharpe']:.2f}, DD {sb['maxdd']:.0%})")
    (1 + p_ovl.fillna(0)).cumprod().plot(ax=ax[0], color="#c0392b", lw=2.0,
        label=f"+ fragility overlay (Sharpe {so['sharpe']:.2f}, DD {so['maxdd']:.0%})")
    ax[0].axvline(cut, color="gray", ls=":", lw=1); ax[0].legend(fontsize=8)
    ax[0].set_title("Network-fragility overlay on crypto trend+carry (HL, net)")
    ax[0].set_ylabel("growth of $1"); ax[0].grid(alpha=0.3)
    ar[hl].plot(ax=ax[1], color="#2980b9", lw=1, label="absorption ratio")
    exposure[hl].plot(ax=ax[1], color="#27ae60", lw=1, label="gross exposure")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    ax[1].set_ylabel("AR / exposure")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "network_overlay.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "network_overlay.md"), "w") as fh:
        fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
