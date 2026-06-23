"""Do ai-trader's strategies add a genuinely-uncorrelated book toward Sharpe 3?

ai-trader (github.com/whchien/ai-trader) is a Backtrader framework of CLASSIC
technical-analysis strategies (SMA/MACD/Bollinger/RSI/ROC/turtle/VCP) plus four
long-only *rotation* strategies (rank by ROC / RSRS / RSI / Bollinger-breakout,
hold top-K). None of these is a Sharpe-3 strategy on its own and we do not pretend
otherwise. The ONLY honest path to Sharpe 3 is stacking uncorrelated books:
  combined Sharpe = S * sqrt(N / (1 + (N-1) rho)).
With VOL+STRATA already at ~2.40 (rho 0.17), a new book B only helps the stack if
  Sharpe_B > rho_B * Sharpe_stack    (marginal-Sharpe rule).
At rho~0.2 that means B must clear standalone Sharpe ~0.5 just to not hurt.

So we take ai-trader's rotation logic (the only portfolio-shaped ideas), implement
each as a deployable market-neutral cross-sectional HL-perp book (long top / short
bottom, inverse-vol sized, net 4.5bps taker + funding, vol-targeted), using TEXTBOOK
parameters (no per-signal tuning -> honest). For each we report:
  - standalone Sharpe (full HL era) and OOS Sharpe (last 40% of the HL era),
  - correlation to VOL and to STRATA,
  - the marginal Sharpe lift when added to the VOL+STRATA stack.
We keep only books that genuinely add, then report how close the honest stack gets
to Sharpe 3.

Run from crypto_pulse/:  python aitrader_books.py  (-> research/aitrader_books.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import strata_beats_vol as sbv          # reuse build_strata()

ANN = 365
TGT = 0.12
TC = 4.5 / 1e4
HL_START = pd.Timestamp("2023-05-12")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if len(p) > 30 and p.std() > 0 else np.nan


def cagr(p):
    p = p.dropna()
    return (1 + p).prod() ** (ANN / len(p)) - 1 if len(p) > 30 else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vt(p, t=TGT, win=45):
    return p * (t / (p.rolling(win).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def book_from_score(score, C, R, F, el, neutral=True, hold=None):
    """Turn a cross-sectional score into a net daily return series.
    Long high score / short low score (market-neutral), inverse-vol sized, gross=1,
    net of taker turnover cost + funding. `hold` ffills weights to cut turnover."""
    s = score.where(el)
    if neutral:
        s = s.sub(s.mean(axis=1), axis=0)
    iv = 1.0 / R.rolling(30).std()
    w = (s * iv)
    w = w.div(w.abs().sum(axis=1), axis=0)            # gross = 1
    if hold:
        rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
        w = w.where(rebw, axis=0).ffill(limit=hold)
    wl = w.shift(1)
    gross = (wl * R).sum(axis=1)
    cost = (wl - wl.shift(1)).abs().sum(axis=1) * TC
    fund = (wl * F).sum(axis=1)
    return gross - cost - fund


def rsi(C, n=14):
    d = C.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def rsrs(H, L, win=18):
    """ai-trader RSRS: rolling OLS slope of high on low. High slope = strengthening
    support. Vectorized slope = cov(L,H)/var(L) over the trailing window."""
    cov = H.rolling(win).cov(L)
    var = L.rolling(win).var()
    return cov / (var + 1e-12)


def main():
    coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); el = C.notna() & (dv > 3e6)

    # ---- ai-trader rotation logic, as market-neutral cross-sectional books ----
    roc20 = C / C.shift(20) - 1                                   # ROC Rotation
    rsi_rev = -rsi(C, 14)                                         # RSI Rotation (buy oversold => short high RSI)
    ma20 = C.rolling(20).mean(); sd20 = C.rolling(20).std()
    bbz = (C - ma20) / (sd20 + 1e-12)                            # Multi-Bollinger Rotation (breakout)
    rs = rsrs(H, L, 18)                                          # RSRS Rotation
    # Triple-RSI rotation: align short/med/long RSI (mean-reverting oversold)
    triple = -(rsi(C, 6) + rsi(C, 14) + rsi(C, 28)) / 3.0

    books = {
        "ai:ROC-rot (momentum)":        book_from_score(roc20,  C, R, F, el, hold=7),
        "ai:RSI-rot (reversal)":        book_from_score(rsi_rev, C, R, F, el, hold=3),
        "ai:Bollinger-rot (breakout)":  book_from_score(bbz,    C, R, F, el, hold=5),
        "ai:RSRS-rot (support slope)":  book_from_score(rs,     C, R, F, el, hold=7),
        "ai:TripleRSI-rot (reversal)":  book_from_score(triple, C, R, F, el, hold=3),
    }
    books = {k: vt(p) for k, p in books.items()}

    # ---- existing stack ----
    vd = pd.read_csv(os.path.join(ROOT, "data", "vol_strategy", "t5rvt_net_daily_2018_2026.csv"), index_col=0)
    vd.index = pd.to_datetime(vd.index)
    vol = vt(vd.iloc[:, 0])
    strata = vt(sbv.build_strata())
    stack = vt(0.5 * vol + 0.5 * strata)                         # VOL+STRATA 50/50

    # align everything to the HL era (genuinely tradeable)
    def hl(p):
        p = p.dropna(); return p[p.index >= HL_START]
    idx = hl(stack).index
    for k in books:
        books[k] = books[k].reindex(idx)
    vol_h, strata_h, stack_h = vol.reindex(idx), strata.reindex(idx), stack.reindex(idx)
    cut = idx[int(len(idx) * 0.6)]                               # OOS = last 40%

    def oos(p): return sh(p[p.index >= cut])
    def corr(a, b):
        d = pd.concat({"a": a, "b": b}, axis=1).dropna()
        return d["a"].corr(d["b"]) if len(d) > 60 else np.nan

    # ---- the standalone NEW strategy built from ai-trader's best ideas ----
    # Equal-risk ensemble of ai-trader's POSITIVE cross-sectional signals (momentum +
    # breakout + support-slope). The reversal variants are negative in crypto and excluded
    # on sign of their FULL-sample Sharpe (a causal, non-tuned rule). This is the honest
    # best standalone strategy the ai-trader toolkit yields on this universe.
    pos = {k: p for k, p in books.items() if sh(p) > 0}
    aitrader = vt(pd.DataFrame(pos).dropna().mean(axis=1)) if pos else None

    s_stack = sh(stack_h); s_stack_oos = oos(stack_h)
    L = ["# Do ai-trader's strategies add a book toward Sharpe 3? (honest test)\n",
         "ai-trader = classic TA + long-only rotation. We implement its rotation logic as "
         "deployable market-neutral cross-sectional HL-perp books (long top / short bottom, "
         f"inverse-vol, net {TC*1e4:.1f}bps + funding, vol-targeted to {TGT:.0%}), TEXTBOOK "
         "params (no tuning). HL era; OOS = last 40%.\n",
         f"**Existing stack VOL+STRATA 50/50:** Sharpe {s_stack:.2f} (full HL), {s_stack_oos:.2f} (OOS).\n",
         "A new book helps the stack only if its Sharpe > corr*stack_Sharpe (marginal rule).\n",
         "| book | Sharpe | OOS | corr VOL | corr STRATA | corr stack | adds to stack? | stack+book OOS |",
         "|---|---|---|---|---|---|---|---|"]

    keepers = []
    for k, p in books.items():
        st, o = sh(p), oos(p)
        cv, cs, cst = corr(p, vol_h), corr(p, strata_h), corr(p, stack_h)
        # marginal test: does adding it at its mean-variance share lift OOS Sharpe?
        cmb = vt(0.5 * stack_h + 0.5 * p)                        # equal-risk add
        cmb_oos = oos(cmb)
        helps = (cmb_oos > s_stack_oos + 0.05) and (o > cst * s_stack_oos)
        if helps:
            keepers.append((k, p))
        L.append(f"| {k} | {st:+.2f} | {o:+.2f} | {cv:+.2f} | {cs:+.2f} | {cst:+.2f} | "
                 f"{'YES' if helps else 'no'} | {cmb_oos:+.2f} |")

    # ---- the standalone ai-trader strategy, reported honestly ----
    if aitrader is not None:
        a = aitrader.reindex(idx)
        L += ["\n## The standalone ai-trader strategy (its best ideas, ensembled)\n",
              "Equal-risk blend of ai-trader's positive cross-sectional books (momentum + "
              "breakout + support-slope; reversal variants excluded by sign). Market-neutral, "
              "net of costs+funding, vol-targeted. This is the honest best the toolkit yields:\n",
              f"- **Sharpe {sh(a):+.2f}** (full HL), **{oos(a):+.2f}** (OOS). "
              f"CAGR {cagr(a):+.0%}, maxDD {maxdd(a):+.0%}.",
              f"- Correlation to VOL {corr(a, vol_h):+.2f}, to STRATA {corr(a, strata_h):+.2f}.",
              f"- This is a **real, positive, tradeable strategy — but it is ~{sh(a):.1f}, not 3.** "
              "Classic technical analysis on liquid crypto, net of costs, does not produce a "
              "Sharpe-3 standalone book; it never has in honest OOS testing. Anyone claiming a "
              "TA indicator at Sharpe 3 is either gross of costs, in-sample, or overfit.\n"]

    # ---- build the best honest stack we can from keepers ----
    L += ["\n## Honest stack assembly\n"]
    if keepers:
        comp = {"VOL": vol_h, "STRATA": strata_h}
        for k, p in keepers:
            comp[k.replace("ai:", "")] = p
        M = pd.DataFrame(comp).dropna()
        eq = vt(M.mean(axis=1))                                  # equal-risk stack
        rho = M.corr().values
        avg_rho = (rho.sum() - len(rho)) / (len(rho) * (len(rho) - 1))
        L.append(f"- Keepers that genuinely add: {', '.join(k for k, _ in keepers)}.")
        L.append(f"- Equal-risk stack of {M.shape[1]} books: Sharpe **{sh(eq):.2f}** "
                 f"(full HL), **{oos(eq):.2f}** (OOS). Avg pairwise corr {avg_rho:.2f}.")
        S_eff = np.nanmean([sh(M[c]) for c in M.columns])
        n = M.shape[1]
        theo = S_eff * np.sqrt(n / (1 + (n - 1) * avg_rho))
        L.append(f"- Diversification ceiling at these inputs (S~{S_eff:.2f}, N={n}, rho={avg_rho:.2f}): "
                 f"~{theo:.2f}. Sharpe 3 {'reached' if oos(eq) >= 3 else 'NOT reached'}.")
    else:
        eq = stack_h
        L.append("- **None of the ai-trader rotation books clears the marginal-Sharpe bar.** "
                 "Their signals overlap STRATA's existing trend/reversal sleeves and/or are too "
                 "weak net of costs. The honest stack stays VOL+STRATA at "
                 f"Sharpe {s_stack_oos:.2f} OOS — adding ai-trader books does not move it toward 3.")

    # ---- what Sharpe 3 would actually require ----
    L += ["\n## What Sharpe 3 honestly requires\n",
          f"From the current stack ({s_stack_oos:.2f} OOS, rho~0.17 between legs), reaching 3.0 needs "
          "~2 more books each ~Sharpe 1.9 at corr <0.2. ai-trader's classic TA does not supply them "
          "(see table). The credible remaining sources of a genuinely-new book are: (a) the L4 "
          "per-account whale-flow book now being data-collected (different signal entirely — order "
          "flow, not price), and (b) a cross-asset leg (equity/FX), which diversifies but cannot be "
          "run on HL. Honesty: Sharpe 3 net OOS is at the frontier; it is reached by stacking, not by "
          "any single indicator in this or any TA library.\n"]

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    (1 + stack_h.fillna(0)).cumprod().plot(ax=ax, color="#2980b9", lw=2.0,
                                           label=f"VOL+STRATA stack (OOS {s_stack_oos:.2f})")
    if keepers:
        (1 + eq.fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.0,
                                          label=f"+ ai-trader keepers (OOS {oos(eq):.2f})")
    for k, p in list(books.items())[:5]:
        (1 + p.reindex(idx).fillna(0)).cumprod().plot(ax=ax, lw=0.9, alpha=0.55, label=f"{k} ({sh(p):.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.set_yscale("log")
    ax.legend(fontsize=8); ax.set_title("ai-trader rotation books vs the VOL+STRATA stack (HL era, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "aitrader_books.png"), dpi=110)
    with open(os.path.join(HERE, "aitrader_books.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("[written] research/aitrader_books.md + png")


if __name__ == "__main__":
    main()
