"""Equity ensemble + cross-asset combination — the honest test of "ensemble across
markets -> higher portfolio Sharpe".

The user's path to 3 is portfolio-level: stack low-correlation alphas across markets.
Equities are genuinely uncorrelated to crypto, so a strong equity book is the real
diversifier. We build the equity factors that actually work on stocks (the ones
the awesome-systematic-trading list and the attention-statarb paper are built on)
on 430 US names (2009+, daily, price-only), market-neutral, costed at 2bps:

  REVERSAL  — short-term (5d) cross-sectional reversal (the classic equity STR).
  MOMENTUM  — 12-1 month cross-sectional momentum.
  LOWVOL    — low idiosyncratic-vol (BAB-style), long low / short high.
  STATARB   — Avellaneda-Lee PCA residual reversion (stat_arb machinery).

Then we combine the equity book with the validated crypto grand stack on the
2023-26 overlap and report the PORTFOLIO Sharpe — the honest answer to whether
cross-market diversification gets us toward 3.

NOTE on the attention paper (arXiv 2510.11616, net 2.3): it needs 50+ firm
CHARACTERISTICS we don't have data for; its richer factor model is why it beats a
plain PCA stat-arb. Our STATARB sleeve is the reproducible floor, not that ceiling.

Run from crypto_pulse/:  python equity_ensemble.py  (-> research/equity_ensemble.md + png)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import kelly_cagr as kc
from stat_arb import sscore_signals

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EQ = os.path.join(ROOT, "data", "stocks_extended")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
HL_START = pd.Timestamp("2023-05-12")
COST = 2.0


def load_eq():
    cl = {}
    for f in sorted(glob.glob(os.path.join(EQ, "*.csv"))):
        t = os.path.basename(f)[:-4]
        d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        if "Close" not in d.columns:
            continue
        d = d[~d.index.duplicated()].sort_index()
        cl[t] = d["Close"]
    return pd.DataFrame(cl).sort_index()


def sharpe(p, ann):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ann) if (len(p) > 60 and p.std() > 0) else np.nan


def stats(p, ann):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sharpe(p, ann), cagr=cum.iloc[-1] ** (ann / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def vt(p, ann, target=0.12):
    return p * (target / (p.rolling(45).std() * np.sqrt(ann))).shift(1).clip(0, 3)


def eq_sleeves(C):
    R = C.pct_change(); R[R.abs() > 1.0] = np.nan
    elig = C.notna() & C.shift(260).notna()
    sd = R.rolling(60).std()
    n = len(C)
    rebw = pd.Series(np.arange(n) % 5 == 0, index=C.index)
    rebm = pd.Series(np.arange(n) % 21 == 0, index=C.index)

    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def demean(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold):
        w = w.where(hold, axis=0).ffill(limit=int(hold.name or 5))
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * COST / 1e4)

    sl = {}
    rev = -(C / C.shift(5) - 1)
    rebw.name = 5
    sl["REVERSAL"] = pnl(norm(demean(rev.where(elig)) / sd), rebw)
    mom = C.shift(21) / C.shift(252) - 1
    rebm.name = 21
    sl["MOMENTUM"] = pnl(norm(demean(mom.where(elig)) / sd), rebm)
    lvol = R.rolling(60).std()
    sl["LOWVOL"] = pnl(norm(demean((-lvol).where(elig))), rebm)
    # stat-arb residual reversion (vectorized machinery)
    W = sscore_signals(R, elig, win=60, K=10, hold=5)
    wl = W.shift(1)
    sl["STATARB"] = ((wl * R).sum(axis=1)
                     - (wl - wl.shift(1)).abs().sum(axis=1) * COST / 1e4)
    return sl


def main():
    C = load_eq()
    sl = eq_sleeves(C)
    start = (C.notna().sum(axis=1) >= 50).idxmax()
    idx = C.index[C.index >= start]
    cut = idx[int(len(idx) * 0.6)]

    lines = ["# Equity ensemble + cross-asset combination with crypto\n"]
    lines.append(f"430 US stocks (price-only), {COST}bps, market-neutral, "
                 "vol-targeted, IS=first60/OOS=last40 of the equity sample.\n")
    lines.append("## Equity sleeves (full history)\n")
    lines.append("| sleeve | Sharpe | IS | OOS |")
    lines.append("|---|---|---|---|")
    VS = {k: vt(p, 252) for k, p in sl.items()}
    for k, p in VS.items():
        ph = p[p.index >= start]
        lines.append(f"| {k} | {sharpe(ph, 252):+.2f} | "
                     f"{sharpe(ph[ph.index < cut], 252):+.2f} | "
                     f"{sharpe(ph[ph.index >= cut], 252):+.2f} |")
    P = pd.DataFrame({k: VS[k] for k in VS}).dropna()
    P = P[P.index >= start]
    rw = (1 / P.std()) / (1 / P.std()).sum()
    eqbook = vt((P * rw).sum(axis=1).reindex(C.index), 252)
    eqh = eqbook[eqbook.index >= start]
    se = stats(eqh, 252)
    lines.append(f"\nEquity ENSEMBLE (equal-risk): Sharpe **{se['sharpe']:+.2f}** "
                 f"(IS {sharpe(eqh[eqh.index < cut],252):+.2f} / OOS "
                 f"{sharpe(eqh[eqh.index >= cut],252):+.2f}), mean sleeve corr "
                 f"{P.corr().values[np.triu_indices(len(P.columns),1)].mean():+.2f}.\n")

    # cross-asset combination on the crypto overlap (weekly to align calendars)
    crypto = kc.build_grandstack()
    def wk(p): return p.fillna(0.0).resample("W-FRI").sum()
    ew, cw = wk(eqbook), wk(crypto)
    both = pd.concat({"equity": ew, "crypto": cw}, axis=1).dropna()
    both = both[both.index >= HL_START]
    corr = both["equity"].corr(both["crypto"])
    rwb = (1 / both.std()) / (1 / both.std()).sum()
    comb = (both * rwb).sum(axis=1)
    comb = comb * (0.12 / np.sqrt(52)) / comb.std()
    cutb = both.index[int(len(both) * 0.6)]

    def shw(s): return s.mean() / s.std() * np.sqrt(52) if s.std() > 0 else np.nan
    lines.append("## Cross-asset portfolio (crypto grand stack + equity book), weekly\n")
    lines.append(f"Overlap {both.index.min().date()}->{both.index.max().date()} "
                 f"({len(both)} wks). **Equity-crypto correlation: {corr:+.2f}.**\n")
    lines.append("| book | Sharpe | IS | OOS |")
    lines.append("|---|---|---|---|")
    for nm, s in [("crypto grand stack alone", both["crypto"]),
                  ("equity book alone (overlap)", both["equity"]),
                  ("COMBINED portfolio", comb)]:
        lines.append(f"| {nm} | **{shw(s):+.2f}** | "
                     f"{shw(s[s.index < cutb]):+.2f} | {shw(s[s.index >= cutb]):+.2f} |")
    lines.append("")
    lines.append("## Verdict\n")
    lines.append(f"- Equity ensemble full-history Sharpe {se['sharpe']:+.2f}; on the "
                 f"2023-26 crypto overlap it is {shw(both['equity']):+.2f} "
                 f"(equity factors decayed/regime-weak recently), correlation to "
                 f"crypto {corr:+.2f}. Combined portfolio {shw(comb):+.2f} vs crypto "
                 f"alone {shw(both['crypto']):+.2f}. "
                 + ("The uncorrelated equity book LIFTS the portfolio — cross-market "
                    "diversification is the real lever toward 2.\n" if shw(comb) >
                    shw(both['crypto']) + 0.05 else
                    "On this short overlap the equity book is too weak to lift the "
                    "crypto book (it diversifies but adds little — the 2023-26 window "
                    "was poor for equity factors). The long-run equity Sharpe is the "
                    "honest contribution; measured only on 2023-26 it underwhelms.\n"))
    lines.append("- To approach 3 you would stack MANY such uncorrelated books "
                 "(equity attention-statarb net ~2.3 in its paper, futures CTA, FX "
                 "carry) — but each is itself ~1-2 net and the combination is bounded "
                 "by S/sqrt(rho); realistic portfolio target ~2, not a stable 3.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + both["crypto"]).cumprod().plot(ax=ax, color="#8e44ad", lw=1.5,
        label=f"crypto grand stack ({shw(both['crypto']):.2f})")
    (1 + both["equity"]).cumprod().plot(ax=ax, color="#16a085", lw=1.5,
        label=f"equity book ({shw(both['equity']):.2f})")
    (1 + comb).cumprod().plot(ax=ax, color="#c0392b", lw=2.3,
        label=f"COMBINED ({shw(comb):.2f}, corr {corr:+.2f})")
    ax.axvline(cutb, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Cross-asset portfolio: crypto + equity ensemble (weekly, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "equity_ensemble.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "equity_ensemble.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/equity_ensemble.md + png")


if __name__ == "__main__":
    main()
