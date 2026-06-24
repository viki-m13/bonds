"""EBB — equity short-term reversal (TIDE's mirror), and the TIDE+EBB cross-asset portfolio.

TIDE (crypto momentum/breakout) INVERTS on equities (-0.8..-1.3): short-horizon x-sectional
moves continue in crypto but REVERSE in stocks. So the honest mirror book on equities is
REVERSAL: long the oversold/breakdown names, short the overbought/breakout names. We name it
EBB (the tide going out). Because crypto-momentum and equity-reversal are different asset
classes AND opposite signals, they should be ~uncorrelated -> a real diversifier, deployable on
HL via crypto perps + HIP-3 equity perps.

We (1) build & validate EBB on US equities (OOS, year-by-year, cost sensitivity — not just a
sign flip), (2) measure TIDE-EBB correlation, (3) risk-parity combine and report the honest
combined Sharpe. Equity cost 2bps base (reversal is high-turnover, so cost sensitivity matters).

Run from crypto_pulse/:  python tide_ebb.py  (-> research/tide_ebb.md + png)
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
from tide import TIDE

ANN = 365
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p, ann=ANN):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ann) if len(p) > 30 and p.std() > 0 else np.nan


def cagr(p, ann=ANN):
    p = p.dropna()
    return (1 + p).prod() ** (ann / len(p)) - 1 if len(p) > 30 else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vtf(p, t=0.12, win=45, ann=ANN):
    return p * (t / (p.rolling(win).std() * np.sqrt(ann))).shift(1).clip(0, 3)


def load_dir(path):
    cl, vo, hi, lo = {}, {}, {}, {}
    for f in glob.glob(os.path.join(path, "*.csv")):
        name = os.path.basename(f)[:-4]
        try:
            d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        except Exception:
            continue
        if "Close" not in d.columns:
            continue
        d = d[~d.index.duplicated()].sort_index()
        cl[name], vo[name] = d["Close"], d.get("Volume", pd.Series(np.nan, index=d.index))
        hi[name], lo[name] = d.get("High", d["Close"]), d.get("Low", d["Close"])
    C = pd.DataFrame(cl).sort_index()
    return C, pd.DataFrame(vo).reindex_like(C), pd.DataFrame(hi).reindex_like(C), pd.DataFrame(lo).reindex_like(C)


def ebb(C, V, H, L, cost_bps=2.0, win=20, reg=50, hold=3, ann=252, regime=True):
    """Equity short-term REVERSAL, x-sectional market-neutral, gated to trade more in
    balanced/choppy regimes (1 - trend_intensity). Mirror of TIDE."""
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    has_vol = (V.fillna(0) > 1).any().any()
    dv = (C * V).rolling(30).mean()
    el = (C.notna() & (dv > 3e6)) if has_vol else C.notna()
    sd = R.rolling(30).std()
    nm = lambda x: x.div(x.abs().sum(axis=1) + 1e-9, axis=0)
    dmf = lambda x: x.sub(x.mean(axis=1), axis=0)
    rev = -dmf(((C - C.rolling(win).mean()) / (C.rolling(win).std() + 1e-9)).where(el))  # REVERSAL
    if regime:
        chop = 1 - ((((C > C.rolling(reg).mean()).where(el)).mean(axis=1) - 0.5).abs() * 2).clip(0, 1)
        sig = rev.mul(chop.shift(1), axis=0)
    else:
        sig = rev
    w = nm(sig / (sd + 1e-9))
    rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
    w = w.where(rebw, axis=0).ffill(limit=hold); wl = w.shift(1)
    pnl = (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * cost_bps / 1e4
    return vtf(pnl, ann=ann)


def main():
    # EBB on equities
    Cs, Vs, Hs, Ls = load_dir(os.path.join(DATA, "stocks"))
    Cx, Vx, Hx, Lx = load_dir(os.path.join(DATA, "stocks_extended"))
    ebb96 = ebb(Cs, Vs, Hs, Ls); ebb430 = ebb(Cx, Vx, Hx, Lx)
    ebb96_plain = ebb(Cs, Vs, Hs, Ls, regime=False)

    cut96 = ebb96.dropna().index[int(ebb96.dropna().shape[0] * 0.6)]
    def io(p, ann=252):
        q = p.dropna(); c = q.index[int(len(q) * 0.6)]
        return sh(q[q.index < c], ann), sh(q[q.index >= c], ann)

    L_ = ["# EBB (equity reversal) + TIDE+EBB cross-asset portfolio (honest)\n",
          "TIDE inverts on equities, so EBB = x-sectional REVERSAL on stocks (long oversold/short "
          "overbought), gated to choppy regimes. Validated, not just sign-flipped. Equity cost 2bps.\n",
          "| book | Sharpe | IS | OOS | CAGR | maxDD |", "|---|---|---|---|---|---|"]
    for k, p in [("EBB stocks-96 (regime)", ebb96), ("EBB stocks-96 (plain)", ebb96_plain),
                 ("EBB stocks-430 (regime)", ebb430)]:
        i, o = io(p)
        L_.append(f"| {k} | **{sh(p,252):+.2f}** | {i:+.2f} | {o:+.2f} | {cagr(p,252):+.0%} | {maxdd(p):+.0%} |")

    # year by year + cost sensitivity for EBB-96
    L_ += ["\n## EBB robustness (stocks-96)\n", "| year | Sharpe |", "|---|---|"]
    for y in range(2010, 2027):
        L_.append(f"| {y} | {sh(ebb96[ebb96.index.year == y], 252):+.2f} |")
    L_ += ["\n| equity cost | Sharpe |", "|---|---|"]
    for cb in [2, 5, 10, 20]:
        L_.append(f"| {cb}bps | {sh(ebb(Cs, Vs, Hs, Ls, cost_bps=cb), 252):+.2f} |")

    # ---- TIDE + EBB cross-asset portfolio ----
    t = TIDE(); tide = t.build()                         # crypto daily, ann 365
    # align EBB onto the daily (crypto) calendar; equity has 0 pnl on non-trading days
    common = tide.dropna().index
    ebb_d = ebb96.reindex(common).fillna(0.0)
    tide_d = tide.reindex(common)
    both = pd.DataFrame({"TIDE": tide_d, "EBB": ebb_d}).dropna()
    both = both[both.index >= pd.Timestamp("2023-05-12")]    # HL era (deployable window)
    rho = both["TIDE"].corr(both["EBB"])
    # risk-parity combine, re-vol-target the combined book
    wv = 1 / (both["TIDE"].std() + 1e-9); we = 1 / (both["EBB"].std() + 1e-9)
    wv, we = wv / (wv + we), we / (wv + we)
    combo = vtf(wv * both["TIDE"] + we * both["EBB"], ann=ANN)

    st, se, sc = sh(both["TIDE"]), sh(both["EBB"]), sh(combo)
    L_ += ["\n## TIDE (crypto) + EBB (equity) cross-asset portfolio — HL era\n",
           f"- Correlation TIDE vs EBB: **{rho:+.2f}** "
           f"({'genuinely uncorrelated -> real diversification' if abs(rho) < 0.3 else 'some overlap'}).",
           f"- TIDE {st:+.2f}, EBB {se:+.2f}, **risk-parity combo {sc:+.2f}** "
           f"({'+' if sc > max(st, se) else ''}{sc - max(st, se):+.2f} vs the better leg).",
           f"- Sharpe 3 {'REACHED' if sc >= 3 else 'NOT reached'}; combined ~{sc:.1f}. "
           "Deployable on HL: TIDE on crypto perps, EBB on HIP-3 equity perps.\n"]

    s96i, s96o = io(ebb96)
    L_ += ["## Verdict (honest — EBB does NOT validate)\n",
           f"- **The reversal SIGN is right but EBB is not a tradeable book.** Equity reversal is "
           f"only {sh(ebb96,252):+.2f} Sharpe on large-caps (IS {s96i:+.2f} / **OOS {s96o:+.2f}**), "
           "regime-unstable year-to-year, and **dies above ~2bps cost** (5bps -> -0.20, 10bps -> "
           "-0.92). Short-term equity reversal is real academically but arbitraged away net of "
           "realistic costs — flipping TIDE's sign does NOT recover a clean book.",
           f"- **So the cross-asset combo does NOT help.** TIDE-EBB correlation is genuinely "
           f"{rho:+.2f} (the diversification premise was correct!), but EBB's ~0 Sharpe means "
           f"adding it DILUTES rather than diversifies: combo {sc:+.2f} < TIDE-alone {st:+.2f}. "
           "Cross-asset diversification only lifts Sharpe when BOTH legs are individually strong; "
           "EBB isn't.",
           f"- **Net: TIDE alone ({st:+.2f}) stays the answer.** The honest equity-perp takeaway: "
           "neither TIDE (momentum, loses) nor EBB (reversal, too weak net of costs) is deployable "
           "on HL HIP-3 equity perps. Sharpe 3 unreached; the price/equity routes are exhausted, "
           "L4 order flow remains the only orthogonal lever.\n"]

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    (1 + both["TIDE"].fillna(0)).cumprod().plot(ax=ax[0], color="#2980b9", lw=1.6, label=f"TIDE crypto ({st:.2f})")
    (1 + both["EBB"].fillna(0)).cumprod().plot(ax=ax[0], color="#27ae60", lw=1.6, label=f"EBB equity ({se:.2f})")
    (1 + combo.fillna(0)).cumprod().plot(ax=ax[0], color="#c0392b", lw=2.4, label=f"COMBO ({sc:.2f})")
    ax[0].set_yscale("log"); ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[0].set_title(f"TIDE + EBB cross-asset (HL era, corr {rho:+.2f})")
    (1 + ebb96.fillna(0)).cumprod().plot(ax=ax[1], color="#27ae60", lw=1.6)
    ax[1].axvline(cut96, color="gray", ls=":", lw=1); ax[1].set_yscale("log"); ax[1].grid(alpha=0.3)
    ax[1].set_title(f"EBB equity reversal full history ({sh(ebb96,252):.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "tide_ebb.png"), dpi=110)
    with open(os.path.join(HERE, "tide_ebb.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/tide_ebb.md + png")


if __name__ == "__main__":
    main()
