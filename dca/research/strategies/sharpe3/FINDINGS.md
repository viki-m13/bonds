# SHARPE3 — Findings of an 8-hour honest attempt to invent a net-Sharpe-3 stock-picking strategy

Campaign: 2026-08-08 04:03–12:03 UTC. Data: daily adjusted close + volume,
~24,000 US tickers (4,590 liquid-ever after PIT screens), 1990–2026,
delisting-inclusive. Honesty contract in README.md; every experiment in
WORKLOG.md (~90 configurations across 16 experiment families).

## Bottom line

**No configuration reached net Sharpe 3. The campaign instead measured *why*,
precisely — and found the ghost of a strategy that once did.**

Best honest results (net, per unit gross exposure, execution lag respected):

| era | best net Sharpe | configuration | fee |
|-----|-----------------|---------------|-----|
| DEV 1995–2014 | **+0.51** | patient residual-reversal ladder, regime-scaled | 10 bps |
| DEV 1995–2014 | **+1.09** | same family, faster | 2 bps |
| DEV, crisis-only days (22%) | **+1.83 active-period** | daily ladder, vol-pct>0.8 | 2 bps |
| VAL 2015–2019 (first look) | **−1.05 to −1.46** | both surviving configs, all fees | 2–10 bps |
| TEST 2020–2026 | **never opened** | nothing survived VAL | — |

## The four structural walls (each measured, not assumed)

**Wall 1 — the tradability gap.** The dominant price-only alpha is short-term
reversal, and its payoff sits in the one day you cannot trade. Event study
(72,017 events, z<−2 idiosyncratic 1-day crashes, entry at t+2 as honesty
requires): +4 bps by day 1, +13 bps by day 5, +20 bps by day 10 — versus a
20 bps round-trip cost. Deeper crashes pay no more (z<−5: +9 bps by d5).

**Wall 2 — alpha-per-trade ≈ cost-per-trade everywhere.** The full frontier on
the best construction found (residual-reversal ladder z1+z5+z21, weights frozen
on 1996–2004, all-names proportional book):

| execution | turnover/day | gross S | 0 bps | 2 bps | 5 bps | 10 bps | 20 bps |
|-----------|-------------|---------|-------|-------|-------|--------|--------|
| daily, unsmoothed | 0.99 | 2.03 | +2.00 | +0.91 | −0.72 | −3.44 | −8.88 |
| EMA3, λ=0.3, regime | 0.124 | 1.29 | +1.26 | +1.09 | +0.83 | +0.39 | −0.49 |
| EMA5, λ=0.15, regime | 0.065 | 1.01 | +0.99 | +0.89 | +0.75 | +0.51 | +0.03 |

Slowing the book always buys cost reduction at the price of gross decay; the
envelope of the whole frontier never exceeds ≈1.1 net at realistic fees —
**even at zero cost it never exceeds ≈2.0.**

**Wall 3 — the alpha is extinct.** Gross Sharpe of the 1-day residual-reversal
book by era: 1996–98: 0.7 · **1999–2001: 4.6** · 2002–04: 2.1 · 2005–07: 0.6 ·
2008–10: 0.0 · 2011–13: 0.9 · 2014–16: −0.2 · 2017–19 (ladder): −1.4.
A genuine Sharpe-3+ stock-picking strategy **did exist** in this data — daily
residual reversal, i.e. statistical liquidity provision, in 1999–2004 — and it
was arbitraged to zero by the industry that industrialized exactly that trade
(cf. Khandani–Lo 2007, the quant-crash post-mortem of this very strategy
family). Both configs that survived DEV failed validation 2015–19 at every fee
level **including 2 bps** (−1.05 to −1.46). The locked TEST period was never
opened: nothing earned the look.

**Wall 4 — nothing else in price/volume space clears the bar.** Everything
else invented or reproduced, with its measured DEV gross (net worse in every
case): raw 5d reversal 1.06 · ML ranker, 14 features, walk-forward (rank IC
+0.021, t=8.6 — statistically real!) books 0.67 · lead-lag catch-up 0.86 ·
pairs-lite 0.62 · monthly reversal 0.35 · path-conditioned crash book 0.71 ·
peer-gap 0.84 · quiet-drift 0.44 · turn-of-month 0.31 · momentum 12-1 ≈0 ·
low-vol, high-volume premium, 52w-high, seasonality, long-term reversal,
lottery-short, spike-fade, rank-crossers: ≤0.3 or negative. The strongest
finding among these — path-conditioning (crashes that keep falling on t+1 pay
+44 bps by d5 vs −6 for already-bounced ones, a tripled per-trade edge) — is
real science that fails as business: the filter shrinks breadth faster than it
grows the edge.

## What net Sharpe 3 would require (the arithmetic)

For a market-neutral book with σ ≈ 3%/yr per unit gross and cost c per side on
turnover T/day: required gross annual alpha ≈ 3·σ + 2·c·T·252. At T=0.065/day
and 10 bps: ≈ 12.6%/yr on 3% vol — gross Sharpe ≈ 4 sustained across decades.
The best sustained gross this dataset offers in the modern era is ≈ 0.
The binding constraint is not cleverness and (past ~2 bps) not even cost — it
is that daily-close prices and volume no longer contain a large exploitable
signal after the people with faster data have traded.

## What survived honestly (the usable residue)

- The patient regime-scaled ladder is a real, mildly profitable construction
  in eras when reversal exists at all: net +0.5 (10 bps) to +1.1 (2 bps) over
  1995–2014, positive in most years, best in crises. It is a *crisis liquidity
  provider*, not an always-on business — and it has been flat-to-negative
  since 2015.
- The event-study, path-conditioning, and frontier methodology transfer to any
  richer dataset (intraday, order-book, fundamentals) where the walls sit
  further out.

*(exp16 slow-ML results and final experiment count to be appended before
close of campaign.)*
