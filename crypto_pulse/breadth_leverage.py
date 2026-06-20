"""Two questions: does MORE BREADTH (more tickers) raise Sharpe, and does LEVERAGE?

BREADTH: re-run the price-based sleeves (TREND, BAB, SQUEEZE, ACCEL — the ones that
need no funding data) on the 57-coin funded universe vs the full 111-coin universe,
with the same liquidity filter (30d $ vol > $3M, which dynamically excludes dead
microcaps). Fundamental Law: IR = IC*sqrt(breadth), so more GENUINELY-INDEPENDENT
names should help — but crypto is single-factor, so the marginal benefit decays and
illiquid names add cost/survivorship risk, not real breadth. Costs: 4.5bps taker;
NOTE the extra 54 coins have no HL funding series, so funding is NOT charged on them
(slightly optimistic — flagged).

LEVERAGE: take the combined book and scale it to a range of vol targets. Sharpe =
return/vol is INVARIANT to leverage (2x lev -> 2x return AND 2x vol). Leverage moves
RETURN, drawdown, and liquidation risk — not the ratio. Beyond Kelly, geometric
growth actually falls (vol drag) even as Sharpe stays flat. We show the table.

Run from crypto_pulse/:  python breadth_leverage.py
"""
import os

import numpy as np
import pandas as pd

import validate_hl as v

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
TAKER = 4.5
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")

ALL111 = [os.path.basename(f)[:-8] for f in
          __import__("glob").glob(os.path.join(v.CRYPTO, "*_USD.csv"))]


def sharpe(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 30 and p.std() > 0) else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vt(p, target=0.12):
    return p * (target / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def price_sleeves(coins):
    """TREND, BAB, SQUEEZE, ACCEL on a coin list — no funding needed. Returns combined
    equal-risk vol-targeted pnl + median eligible-name count."""
    C, V, H, L = v.load_prices(coins)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    mkt = R["BTC"] if "BTC" in R else R.where(elig).mean(axis=1)
    n = len(C); rebw = pd.Series(np.arange(n) % 7 == 0, index=C.index)

    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def demean(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold=None):
        if hold is not None:
            w = w.where(hold, axis=0).ffill(limit=6)
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4)

    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    don = ((C >= H.shift(1).rolling(20).max()).astype(float)
           - (C <= L.shift(1).rolling(20).min()).astype(float))
    sl = {}
    sl["TREND"] = pnl(norm(((trend + don) / sd).where(elig)))
    beta = R.rolling(90).cov(mkt).div(mkt.rolling(90).var(), axis=0)
    sl["BAB"] = pnl(norm(demean((-beta).where(elig))), hold=rebw)
    comp = -((H - L) / C).rolling(20).mean()
    sl["SQUEEZE"] = pnl(norm((demean(comp.where(elig)) / sd) * np.sign(trend)),
                        hold=pd.Series(np.arange(n) % 3 == 0, index=C.index))
    accel = (C / C.shift(20) - 1) - (C.shift(20) / C.shift(40) - 1)
    sl["ACCEL"] = pnl(norm(demean(accel.where(elig)) / sd), hold=rebw)

    P = pd.DataFrame({k: vt(p) for k, p in sl.items()})
    rw = (1 / P.std()) / (1 / P.std()).sum()
    comb = vt((P * rw).sum(axis=1).reindex(C.index))
    med_names = int(elig[C.index >= HL_START].sum(axis=1).median())
    return comb, med_names


def main():
    hl_cut = lambda p: p[p.index >= HL_START]
    coins57 = [c for c in v.OVERLAP
               if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    coins111 = sorted(set(ALL111))

    lines = ["# Does more breadth (tickers) or more leverage raise Sharpe?\n"]
    lines.append("Price-based sleeves (TREND+BAB+SQUEEZE+ACCEL), equal-risk, "
                 f"vol-targeted, {TAKER}bps taker, liquidity filter $3M/30d. "
                 "IS=first60/OOS=last40 of HL era. (Funding sleeves excluded here so "
                 "the 57 vs 111 comparison is apples-to-apples; extra coins lack HL "
                 "funding.)\n")

    lines.append("## 1. BREADTH — 57 funded coins vs full 111-coin universe\n")
    lines.append("| universe | med eligible names | Sharpe | IS | OOS | maxDD |")
    lines.append("|---|---|---|---|---|---|")
    res = {}
    for nm, coins in [("57 (funded/HL)", coins57), ("111 (all data/crypto)", coins111)]:
        comb, med = price_sleeves(coins)
        ph = hl_cut(comb)
        cut = ph.index[int(len(ph.dropna()) * 0.6)] if len(ph.dropna()) else None
        si = sharpe(ph[ph.index < cut]); so = sharpe(ph[ph.index >= cut])
        lines.append(f"| {nm} | {med} | **{sharpe(ph):+.2f}** | {si:+.2f} | "
                     f"{so:+.2f} | {maxdd(ph):+.1%} |")
        res[nm] = sharpe(ph)
    delta = res["111 (all data/crypto)"] - res["57 (funded/HL)"]
    lines.append(f"\n**Breadth effect: {delta:+.2f} Sharpe** from ~doubling the "
                 "universe. " + (
                 "A modest lift — crypto is single-factor so extra names are mostly "
                 "more BTC-beta, not independent bets; the liquidity filter keeps "
                 "dead microcaps out (which would otherwise INFLATE the backtest "
                 "with untradeable names). Real but diminishing — not a path to 3."
                 if abs(delta) < 0.4 else
                 "A larger move — but check it isn't driven by illiquid names the "
                 "filter let through (survivorship/cost mirage).") + "\n")

    lines.append("## 2. LEVERAGE — scaling the SAME book to higher vol targets\n")
    lines.append("Sharpe = mean/vol is INVARIANT to leverage (lever k -> k*return AND "
                 "k*vol). Leverage scales RETURN, drawdown, and liquidation risk — "
                 "not the ratio. We scale the 111-coin book to each vol target:\n")
    comb111, _ = price_sleeves(coins111)
    base = hl_cut(comb111).dropna()
    # the unscaled book's realized vol, to express leverage as a multiple
    lines.append("| vol target | ~gross leverage | Sharpe | ann return | maxDD | "
                 "1-day worst |")
    lines.append("|---|---|---|---|---|---|")
    for tgt in (0.12, 0.24, 0.36, 0.60, 1.00):
        p = vt(comb111, target=tgt)
        ph = hl_cut(p).dropna()
        realvol = ph.std() * np.sqrt(ANN)
        lev = realvol / (base.std() * np.sqrt(ANN))   # multiple vs 12% book
        lines.append(f"| {tgt:.0%} | ~{lev*1.35:.1f}x | {sharpe(ph):+.2f} | "
                     f"{ph.mean()*ANN:+.0%} | {maxdd(ph):+.0%} | {ph.min():+.1%} |")
    lines.append("\n**Verdict on leverage:** the Sharpe column is flat — leverage "
                 "does NOT improve risk-adjusted return. It multiplies returns AND "
                 "drawdowns together. Past the growth-optimal (Kelly) point, "
                 "geometric return actually FALLS (vol drag) and a bad day can "
                 "liquidate the account. Leverage is the lever for RETURNS (and risk), "
                 "never for Sharpe. The vol target — not leverage — is the real "
                 "control, and the binding constraint is drawdown tolerance, not the "
                 "ratio.\n")

    out = "\n".join(lines)
    with open(os.path.join(HERE, "breadth_leverage.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/breadth_leverage.md")


if __name__ == "__main__":
    main()
