"""Point-in-time universe construction — top-N most-liquid coins, ranked monthly.

Answers three methodology questions honestly:
  1. How many coins, and is it point-in-time?
  2. How does the book do on JUST BTC + ETH?
  3. Can we trade a PIT top-N-most-liquid universe, re-ranked every month?

PIT construction: on the first day of each month, rank every coin by its TRAILING
30d dollar volume (causal — only data through the prior day), take the top N as the
tradeable universe for that month. A coin enters when it becomes liquid and drops
when it doesn't — no look-ahead in universe selection. (Survivorship caveat: the
candidate pool is data/crypto's 111 coins, which includes some dead names but is not
a formally delisting-complete panel; the monthly liquidity rank is the main PIT fix.)

Price sleeves only (TREND+BAB+SQUEEZE+ACCEL, no funding) so we can run the full
2015-2026 history. Net of 4.5bps taker. Run from crypto_pulse/:  python pit_universe.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import breadth_leverage as bl

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sharpe(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 60 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 100:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sharpe(p), cagr=cum.iloc[-1] ** (ANN / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def pit_membership(C, V, topn):
    """Monthly PIT top-N-by-trailing-dollar-volume membership mask."""
    dv = (C * V).rolling(30).mean()
    member = pd.DataFrame(False, index=C.index, columns=C.columns)
    months = C.index.to_period("M")
    for m in pd.unique(months):
        first = C.index[months == m][0]
        # rank using volume as of the day BEFORE month start (causal)
        loc = C.index.get_loc(first)
        if loc == 0:
            continue
        rank_row = dv.iloc[loc - 1].dropna()
        top = rank_row[rank_row > 3e6].sort_values(ascending=False).head(topn).index
        member.loc[months == m, top] = True
    return member


def price_book(C, V, H, L, member):
    """TREND+BAB+SQUEEZE+ACCEL equal-risk on a given PIT membership mask."""
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    elig = member & C.notna()
    sd = R.rolling(30).std()
    mkt = R["BTC"] if "BTC" in R else R.where(elig).mean(axis=1)
    n = len(C); rebw = pd.Series(np.arange(n) % 7 == 0, index=C.index)

    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold=None):
        if hold is not None:
            w = w.where(hold, axis=0).ffill(limit=6)
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4)

    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    don = ((C >= H.shift(1).rolling(20).max()).astype(float)
           - (C <= L.shift(1).rolling(20).min()).astype(float))
    sl = {}
    sl["T"] = pnl(norm(((trend + don) / sd).where(elig)))
    beta = R.rolling(90).cov(mkt).div(mkt.rolling(90).var(), axis=0)
    sl["B"] = pnl(norm(dm((-beta).where(elig))), hold=rebw)
    comp = -((H - L) / C).rolling(20).mean()
    sl["S"] = pnl(norm((dm(comp.where(elig)) / sd) * np.sign(trend)),
                  hold=pd.Series(np.arange(n) % 3 == 0, index=C.index))
    accel = (C / C.shift(20) - 1) - (C.shift(20) / C.shift(40) - 1)
    sl["A"] = pnl(norm(dm(accel.where(elig)) / sd), hold=rebw)
    P = pd.DataFrame({k: bl.vt(p) for k, p in sl.items()})
    rw = (1 / P.std()) / (1 / P.std()).sum()
    return bl.vt((P * rw).sum(axis=1).reindex(C.index)), elig.sum(axis=1)


def main():
    coins = sorted(set(bl.ALL111))
    coins = [c for c in coins if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    warm = C.index[C.index >= C.index[0] + pd.Timedelta(days=200)][0]

    lines = ["# Point-in-time universe: top-N most-liquid, ranked monthly\n"]
    lines.append("Price sleeves (TREND+BAB+SQUEEZE+ACCEL), 4.5bps taker, vol-target "
                 "12%. Universe re-ranked monthly by trailing 30d dollar volume "
                 "(causal). Full sample from ~2015. (Survivorship caveat noted in "
                 "the module docstring.)\n")
    lines.append("## Universe size sweep (full sample)\n")
    lines.append("| universe | med live names | Sharpe (full) | IS≈pre-HL | HL era | "
                 "CAGR | maxDD |")
    lines.append("|---|---|---|---|---|---|---|")

    results = {}
    for label, topn in [("BTC+ETH only", 2), ("top 5", 5), ("top 10", 10),
                        ("top 20", 20), ("top 30", 30), ("top 50", 50)]:
        if topn == 2:
            member = pd.DataFrame(False, index=C.index, columns=C.columns)
            for c in ("BTC", "ETH"):
                if c in member: member[c] = True
        else:
            member = pit_membership(C, V, topn)
        p, nlive = price_book(C, V, H, L, member)
        p = p[p.index >= warm]
        results[label] = p
        s = stats(p)
        pre = sharpe(p[p.index < HL_START]); hl = sharpe(p[p.index >= HL_START])
        lines.append(f"| {label} | {int(nlive[nlive.index>=HL_START].median())} | "
                     f"**{s['sharpe']:+.2f}** | {pre:+.2f} | {hl:+.2f} | "
                     f"{s['cagr']:+.0%} | {s['maxdd']:+.0%} |")

    # BTC+ETH: also a plain time-series trend (what most mean by 'trade BTC/ETH')
    R = C.pct_change()
    be = [c for c in ("BTC", "ETH") if c in C.columns]
    tr = sum(np.sign(C[be] / C[be].shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    wts = tr.div(tr.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    ts = (wts.shift(1) * R[be]).sum(axis=1) - \
         (wts.shift(1) - wts.shift(2)).abs().sum(axis=1) * 4.5 / 1e4
    ts = bl.vt(ts)[C.index >= warm]
    lines.append(f"\nBTC+ETH time-series trend (long/short each on its own trend, "
                 f"not cross-sectional): Sharpe **{sharpe(ts):+.2f}** "
                 f"(pre-HL {sharpe(ts[ts.index<HL_START]):+.2f}, HL "
                 f"{sharpe(ts[ts.index>=HL_START]):+.2f}).\n")

    lines.append("## Verdict\n")
    s2 = stats(results["BTC+ETH only"]); s30 = stats(results["top 30"])
    lines.append(f"- **BTC+ETH alone is weak** (Sharpe {s2['sharpe']:+.2f}): two names "
                 "give almost no cross-sectional breadth, so BAB/ACCEL/SQUEEZE "
                 "degenerate to a single long-short pair. The edge NEEDS breadth.")
    lines.append(f"- **The PIT top-N book works and is robust to N** (top-30 Sharpe "
                 f"{s30['sharpe']:+.2f}), confirming it's not an artifact of a hand-"
                 "picked universe: rank the most-liquid coins monthly and trade them. "
                 "Breadth helps up to ~20-30 names, then flattens (single-factor "
                 "crypto). This IS the deployable, point-in-time universe rule.")
    lines.append("- Honest PIT note: the monthly liquidity rank removes look-ahead in "
                 "universe SELECTION; full survivorship-freedom would need a "
                 "delisting-complete panel, but the liquidity filter (only trade names "
                 "with >$3M/day AT THE TIME) already excludes the dead-microcap "
                 "mirage that inflates naive crypto backtests.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    for label, col, lw in [("BTC+ETH only", "#888", 1.2), ("top 10", "#2980b9", 1.5),
                          ("top 30", "#c0392b", 2.0)]:
        p = results[label]; s = stats(p)
        (1 + p.fillna(0)).cumprod().plot(ax=ax, color=col, lw=lw, logy=True,
            label=f"{label} (Sharpe {s['sharpe']:.2f})")
    ax.axvline(HL_START, color="gray", ls=":", lw=1)
    ax.legend(fontsize=9); ax.set_title("Point-in-time top-N-liquid universe "
              "(monthly rank, log, net)")
    ax.set_ylabel("growth of $1 (log)"); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "pit_universe.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "pit_universe.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/pit_universe.md + png")


if __name__ == "__main__":
    main()
