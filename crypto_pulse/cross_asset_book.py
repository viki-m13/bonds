"""CROSS-ASSET DIVERSIFICATION — does adding an uncorrelated traditional-futures
trend book raise the PORTFOLIO Sharpe of the validated crypto book?

The honest math behind every high-Sharpe program: combined Sharpe of K books each
at Sharpe S with average pairwise correlation rho is

    S_combined = S * sqrt(K) / sqrt(1 + (K-1)*rho)

so the lever isn't a better single signal — it's adding GENUINELY uncorrelated
return streams. Crypto is ~single-factor, so extra crypto signals are redundant
(ensemble.py showed this). Traditional asset classes (equity indices, bonds,
commodities, FX) trend too, and their trend books are ~uncorrelated to crypto —
the classic cross-asset CTA diversification (Moskowitz-Ooi-Pedersen time-series
momentum; AQR/Man-AHL managed futures).

This builds a cross-asset trend book on liquid ETFs (per-class TS-momentum +
Donchian, inverse-vol, vol-targeted per class, risk-parity across classes), takes
the validated crypto 3-sleeve book, and measures — on the OVERLAPPING window,
weekly, net of costs — each standalone Sharpe, their correlation, and the
combined diversified Sharpe (IS/OOS). Costs: ETFs 2 bps/side, crypto via
three_sleeve (real HL funding + 4.5 bps).

Run from crypto_pulse/:  python cross_asset_book.py
    (-> research/cross_asset_book.md + png)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import three_sleeve as ts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ETF = os.path.join(ROOT, "data", "etfs")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
HL_START = pd.Timestamp("2023-05-12")
ETF_COST = 2.0   # bps/side

# Liquid cross-asset sleeves (uncorrelated to crypto). Curated, not exhaustive.
CLASSES = {
    "EQUITY": ["SPY", "DIA", "QQQ", "IWM", "EFA", "EEM", "ACWI", "VEA", "VWO"],
    "BONDS":  ["AGG", "BND", "TLT", "IEF", "SHY", "EDV", "LQD", "HYG", "EMB",
               "BNDX", "TIP"],
    "COMMOD": ["DBC", "DBA", "GLD", "SLV", "USO", "CORN", "CPER", "UNG",
               "WEAT", "PALL", "PPLT"],
    "FX":     ["UUP", "FXE", "FXY", "BZF", "CEW", "CYB", "FXB", "FXF"],
}


def load_etf(tickers):
    cl, hi, lo = {}, {}, {}
    for t in tickers:
        f = os.path.join(ETF, f"{t}.csv")
        if not os.path.exists(f):
            continue
        d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        d = d[~d.index.duplicated()].sort_index()
        cl[t], hi[t], lo[t] = d["Close"], d["High"], d["Low"]
    C = pd.DataFrame(cl).sort_index()
    return C, pd.DataFrame(hi).reindex_like(C), pd.DataFrame(lo).reindex_like(C)


def trend_book(C, H, L, cost_bps=ETF_COST, vt=0.10, ann=252):
    """Directional cross-sectional trend book within ONE asset class: slow
    TS-momentum sign-ensemble + Donchian breakout, inverse-vol, gross-normalised,
    vol-targeted. Slower lookbacks than crypto (trad markets trend slower)."""
    R = C.pct_change(); R[R.abs() > 0.5] = np.nan
    elig = C.notna() & C.shift(160).notna()
    sd = R.rolling(40).std()
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (40, 80, 160)) / 3.0
    don = ((C >= H.shift(1).rolling(60).max()).astype(float)
           - (C <= L.shift(1).rolling(60).min()).astype(float))
    w = ((trend + don) / sd).where(elig)
    w = w.div(w.abs().sum(axis=1), axis=0)
    wl = w.shift(1)
    pre = ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * cost_bps / 1e4)
    scale = (vt / (pre.rolling(63).std() * np.sqrt(ann))).shift(1).clip(0, 3)
    return pre * scale


def crypto_book():
    """Validated crypto 3-sleeve risk-weighted combined daily PnL (the +1.12
    book), full-sample inverse-vol risk weights."""
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    pnls = ts.sleeves(C, V, H, L, F)
    P = pd.DataFrame(pnls).dropna()
    rw = (1 / P.std()) / (1 / P.std()).sum()
    combined = (P * rw).sum(axis=1).reindex(C.index)
    scale = (0.12 / (combined.rolling(45).std() * np.sqrt(365))).shift(1).clip(0, 3)
    return (combined * scale)[C.index >= HL_START]


def wk(p):
    """Weekly (W-FRI) PnL — aligns the 7-day crypto and 5-day ETF calendars."""
    return p.fillna(0.0).resample("W-FRI").sum()


def sharpe(p, ann=52):
    p = p.dropna()
    if len(p) < 26:
        return np.nan
    return p.mean() / p.std() * np.sqrt(ann) if p.std() > 0 else np.nan


def stats_wk(p):
    p = p.dropna()
    if len(p) < 26:
        return dict(sharpe=np.nan, ann=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(52), ann=p.mean() * 52,
                maxdd=(cum / cum.cummax() - 1).min())


def main():
    # cross-asset class trend books (full ETF history)
    class_pnl = {}
    for cls, tk in CLASSES.items():
        C, H, L = load_etf(tk)
        class_pnl[cls] = trend_book(C, H, L)
    CA = pd.DataFrame(class_pnl)
    # risk-parity across classes (equal vol), then vol-target the blend to 10%
    rp_w = (1 / CA.std()) / (1 / CA.std()).sum()
    ca_daily = (CA * rp_w).sum(axis=1)
    ca_daily = ca_daily * (0.10 / (ca_daily.rolling(63).std() * np.sqrt(252))
                           ).shift(1).clip(0, 3)

    cr_daily = crypto_book()

    # weekly, overlapping window
    cr_w, ca_w = wk(cr_daily), wk(ca_daily)
    both = pd.concat({"crypto": cr_w, "xasset": ca_w}, axis=1).dropna()
    both = both[both.index >= HL_START]
    corr = both["crypto"].corr(both["xasset"])
    cut = both.index[int(len(both) * 0.6)]

    # combined: SHARPE-OPTIMAL weights set on IS only (Markowitz w/ ~diagonal cov
    # since corr~0 => weight ∝ max(IS Sharpe,0)/vol); applied causally to OOS.
    # A naive inverse-vol blend would over-weight a weak book and DRAG the strong
    # one — diversification helps only if the added book has positive expectancy.
    bIS = both[both.index < cut]
    isr = {c: sharpe(bIS[c]) for c in both.columns}
    raw = {c: max(isr[c], 0.0) / bIS[c].std() for c in both.columns}
    tot = sum(raw.values()) or 1.0
    ow = {c: raw[c] / tot for c in both.columns}
    comb = sum(ow[c] * both[c] for c in both.columns)
    comb = comb * (0.12 / np.sqrt(52)) / comb.std()        # scale to ~12% ann vol

    def rep(p):
        s = stats_wk(p)
        return s, sharpe(p[p.index < cut]), sharpe(p[p.index >= cut])

    lines = ["# Cross-asset diversification — does an uncorrelated CTA book lift "
             "the crypto Sharpe?\n"]
    lines.append("Weekly (W-FRI) returns over the crypto-tradeable overlap "
                 f"({both.index.min().date()} -> {both.index.max().date()}, "
                 f"{len(both)} weeks). Crypto = validated 3-sleeve book (real HL "
                 "funding + 4.5bps). Cross-asset = risk-parity trend book over "
                 "EQUITY/BONDS/COMMOD/FX ETF sleeves (2bps/side). IS=first60/"
                 "OOS=last40.\n")

    # cross-asset standalone over its FULL history (context) and overlap
    ca_full = wk(ca_daily).dropna()
    lines.append(f"Cross-asset trend book standalone over its FULL ETF history "
                 f"({ca_full.index.min().date()}->{ca_full.index.max().date()}): "
                 f"Sharpe **{sharpe(ca_full):+.2f}**, {len(ca_full)} weeks.\n")
    lines.append("Per-class trend Sharpe (full history): " + ", ".join(
        f"{c} {sharpe(wk(class_pnl[c])):+.2f}" for c in CLASSES) + "\n")

    lines.append("## The combination (overlap window, weekly)\n")
    lines.append(f"**Crypto–cross-asset correlation: {corr:+.2f}**"
                 + ("  (genuinely uncorrelated — diversification is real)\n"
                    if abs(corr) < 0.3 else "  (correlated — limited benefit)\n"))
    lines.append(f"Sharpe-optimal weights (set on IS): crypto {ow['crypto']:.0%}, "
                 f"cross-asset {ow['xasset']:.0%}  (a negative-IS book gets 0).\n")
    lines.append("| book | Sharpe | IS | OOS | ann | maxDD |")
    lines.append("|---|---|---|---|---|---|")
    for nm, p in [("crypto 3-sleeve (alone)", both["crypto"]),
                  ("cross-asset trend (alone)", both["xasset"]),
                  ("COMBINED (Sharpe-optimal)", comb)]:
        s, i, o = rep(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['ann']:+.1%} | {s['maxdd']:+.1%} |")

    sc = stats_wk(both["crypto"])["sharpe"]
    so = stats_wk(comb)["sharpe"]
    # theoretical 2-book combination check
    s1, s2 = stats_wk(both["crypto"])["sharpe"], stats_wk(both["xasset"])["sharpe"]
    theo = (s1 + s2) / np.sqrt(2 * (1 + corr)) if (1 + corr) > 0 else np.nan
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- Combined Sharpe **{so:+.2f}** vs crypto-alone {sc:+.2f} "
                 f"(theoretical 2-book optimum ~{theo:+.2f} at corr {corr:+.2f}). "
                 + ("The uncorrelated cross-asset trend book **lifts** the "
                    "portfolio Sharpe — the honest diversification lever.\n"
                    if so > sc + 0.05 else
                    "The lift is marginal on this short overlap window.\n"))
    lines.append("- This is the mechanism behind high program-level Sharpes: "
                 "stack uncorrelated books, not more correlated crypto signals. "
                 "Reaching 3 needs MANY such uncorrelated streams (S*sqrt(K)/"
                 "sqrt(1+(K-1)rho)); two books move the needle but don't get there "
                 "alone. The cross-asset book also adds crisis-alpha (bonds/trend "
                 "rally when crypto crashes).\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + both["crypto"]).cumprod().plot(ax=ax, color="#8e44ad", lw=1.5,
        label=f"crypto 3-sleeve (Sharpe {s1:.2f})")
    (1 + both["xasset"]).cumprod().plot(ax=ax, color="#16a085", lw=1.5,
        label=f"cross-asset trend (Sharpe {s2:.2f})")
    (1 + comb).cumprod().plot(ax=ax, color="#c0392b", lw=2.3,
        label=f"COMBINED (Sharpe {so:.2f}, corr {corr:+.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Cross-asset diversification: crypto + uncorrelated CTA book "
                 "(weekly, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "cross_asset_book.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "cross_asset_book.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/cross_asset_book.md + png")


if __name__ == "__main__":
    main()
