# SHARPE3 — an 8-hour honest attempt to invent a net-Sharpe-3 stock-picking strategy

Campaign: 2026-08-08, 8 hours of compute. Data: daily adjusted prices + volume
for ~24,000 US tickers (4,590 passing liquidity screens), 1990–2026,
delisting-inclusive; plus adjusted OHLC for 105 mega-liquid names.
Honesty contract: `README.md`. Full ledger of ~100 configurations across 19
experiment families: `WORKLOG.md`. All code in `scripts/`.

---

## 1. The result

**No strategy reached net Sharpe 3. The best honest configuration reached
+0.51, and it failed validation.** The locked TEST period (2020–2026) was
never opened, because nothing earned the look.

| period | best net Sharpe | configuration | fee |
|--------|-----------------|---------------|-----|
| DEV 1995–2014 | **+0.51** | patient residual-reversal ladder, regime-scaled | 10 bps |
| DEV 1995–2014 | **+1.09** | same family, faster (EMA3/λ0.3) | 2 bps |
| DEV, crisis days only (22%) | **+1.83** (active-period) | daily ladder, vol-pct > 0.8 | 2 bps |
| **VAL 2015–2019** | **−1.05 to −1.46** | both DEV survivors, every fee level | 2–10 bps |
| TEST 2020–2026 | *never opened* | — | — |

This is a negative result, but not an empty one: the campaign produced a
**measurement of exactly where and why the ceiling sits**, and it found the
fossil of a strategy that genuinely did clear Sharpe 3 — before it was
arbitraged away.

---

## 2. The fossil: a real Sharpe-4.6 signal that died

Gross Sharpe of the 1-day idiosyncratic-reversal book, by era:

| era | 96–98 | **99–01** | 02–04 | 05–07 | 08–10 | 11–13 | 14–16 | 17–19 |
|-----|-------|-----------|-------|-------|-------|-------|-------|-------|
| gross Sharpe | 0.74 | **4.60** | 2.06 | 0.59 | 0.03 | 0.92 | −0.24 | −1.44 |

A genuine Sharpe-3+ signal existed in this data — daily residual reversal,
which is statistical liquidity provision — and it decayed monotonically to
extinction. This is the well-documented life cycle of that exact strategy
family (Khandani–Lo's post-mortem of the 2007 quant crash describes the same
trade and the same crowding). The 9.2σ matched-null test (below) confirms the
DEV-era edge was real rather than luck; the era table shows it is gone.

**The campaign did not fail to find Sharpe 3 because it looked in the wrong
place. It found the right place, arrived 20 years late, and measured the
corpse.**

---

## 3. Four walls, each measured

**Wall 1 — the payoff lives in the one session you cannot trade.**
This is the campaign's sharpest finding, and the OHLC data proved it
mechanically. After a −2.5σ idiosyncratic crash at close *t*:

| session | market-adjusted payoff |
|---------|------------------------|
| overnight *t* → open *t+1* (untradeable — you're already positioned or you're not) | **+4 bps** |
| **intraday open *t+1* → close *t+1*** (the first session you can actually trade) | **−12 bps** |
| overnight *t+1* → open *t+2* | **+16 bps** |
| intraday *t+2* | **−6 bps** |

The entire bounce is in the gap. Buy at the earliest honest entry — the open
after the signal — and you systematically buy *after* the rebound and hold
through a negative intraday drift. Every crash-bounce book with open entry
loses money before costs (−8 to −35 bps per trade). The close-to-close event
study said the same thing with less resolution: +13 bps by day 5 against a
20 bps round trip.

**Wall 2 — alpha-per-trade ≈ cost-per-trade across the entire frontier.**
The best construction found (residual-reversal ladder, weights frozen on
1996–2004, proportional all-names book):

| execution | turnover/day | gross | 0 bps | 2 bps | 5 bps | 10 bps | 20 bps |
|-----------|-------------|-------|-------|-------|-------|--------|--------|
| daily, unsmoothed | 0.99 | 2.03 | +2.00 | +0.91 | −0.72 | −3.44 | −8.88 |
| EMA3, λ=0.3, regime | 0.124 | 1.29 | +1.26 | +1.09 | +0.83 | +0.39 | −0.49 |
| EMA5, λ=0.15, regime | 0.065 | 1.01 | +0.99 | +0.89 | +0.75 | +0.51 | +0.03 |

Slowing the book buys cost relief at exactly the price of gross decay. The
envelope never exceeds ≈1.1 net at realistic fees — and **never exceeds ≈2.0
even with free execution.** Sharpe 3 is not on this surface anywhere.

**Wall 3 — the modern era offers nothing in this space.** Every idea tested on
2005–2019 (with the TEST period still locked) failed, several at deliberately
optimistic same-close execution: spike-fade −0.02 gross · spike-momentum 0.02 ·
quiet-drift 0.44 · turn-of-month 0.31 · beta turn-of-month 0.29 · rank-crossers
0.04 · peer-gap 0.84 gross but −1.15 at 5 bps · gap-fade −0.17 · Friday-losers
−0.13. The max-assembly of every positive sleeve, netted and regime-scaled,
scored **−0.38 gross in 2015–2019**.

**Wall 4 — real statistical skill exists, and is an order of magnitude too
small.** The walk-forward ML rankers are the honest version of "AI finds the
winners." Weekly model: rank IC +0.021 (t = 8.6). Monthly model: rank IC
+0.028 (t = 5.0), and it **still works in 2015–2019** — the only modern-era
positive signal the campaign found. Both convert to gross Sharpe ≈0.5 and
negative net at any fee ≥ 5 bps. The skill is statistically undeniable and
economically irrelevant: predicting ranks slightly better than chance is
roughly 10× short of what a tradeable book requires.

---

## 4. What net Sharpe 3 would actually require

For a market-neutral book at σ ≈ 3%/yr per unit gross, turnover *T*/day, cost
*c* per side: required gross alpha ≈ 3σ + 2·c·T·252. At the surviving
turnover (0.065/day) and 10 bps, that is ≈12.6%/yr of gross alpha on 3% vol —
**a sustained gross Sharpe near 4, for decades.** The best sustained modern-era
gross this dataset contains is ≈0.

The binding constraint is not cleverness, and past ~2 bps it is not even cost.
It is that daily closing prices and volume no longer contain a large
exploitable signal once faster participants have traded — Wall 1 states that
in its most physical form: *the money is in the gap, and the gap is not for
sale at the close.*

---

## 5. Honest caveats (things that could make this verdict wrong)

- **Data ceiling.** Daily bars only: no intraday bars, no order book, no
  quotes, no fundamentals, no earnings dates, no short-interest, no options.
  The strategies that plausibly clear Sharpe 3 today (market-making, latency
  arbitrage, capacity-limited stat-arb on richer data) require exactly what
  this dataset lacks. **This campaign bounds what daily price-and-volume data
  can do — not what is possible.**
- **The OHLC test used 105 names chosen by full-period liquidity**, a
  survivorship-flavored universe. Note the direction: that bias would
  *inflate* results, and the results were still negative — which strengthens
  rather than weakens the conclusion.
- **~100 configurations were tried.** Any single DEV "win" must be discounted
  for that search. The matched-null test (30 runs of identical machinery on
  pure noise: mean −1.24, sd 0.19, max −0.88) puts the +0.51 at **9.2σ**, so
  the DEV edge survives even a harsh multiple-testing haircut. Its VAL failure
  is a decay result, not a data-mining artifact.
- **Costs are modeled, not incurred.** 10 bps/side is realistic-to-generous for
  liquid US equities at modest size; a fund at 2 bps sees the 2 bps column,
  where the crisis-only book reaches +1.83 in its active periods. Still not 3.
- **The overnight anomaly is real and unexploited here**: +8.9 bps per night
  versus −0.5 bps intraday per name. Capturing it costs a full round trip
  daily, so it needs sub-4.5 bps round-trip execution to survive — a
  market-structure business, not a stock-picking one.

---

## 6. What survived, honestly

1. **A method, fully reusable.** The engine (`scripts/engine.py`) enforces
   PIT signals, t+1 execution, delisting exits, costs, and turnover-matched
   nulls. The event-study → path-conditioning → cost-frontier sequence is the
   right way to interrogate any candidate signal, and it transfers directly to
   richer data where the walls sit further out.
2. **One genuine construction.** The patient regime-scaled ladder is a real
   crisis-liquidity-provision book: net +0.5 (10 bps) to +1.1 (2 bps) over
   1995–2014, best when volatility is high — and flat-to-negative since 2015.
   Deployable only by someone who believes the regime returns.
3. **Two findings worth publishing as science.** Path-conditioning: crashes
   still falling at *t+1* pay +44 bps by day 5 versus −6 bps for those that
   already bounced — a tripled per-trade edge that fails as a business because
   the filter destroys breadth faster than it grows edge. And the session
   decomposition of Wall 1, which explains *why* the daily reversal literature
   does not survive contact with honest execution.
4. **A data bug caught before it became a discovery.** The first battery
   produced annualized returns in the thousands of percent from spike-reversal
   bad prints. Cleaning them (and voiding those results) was the difference
   between this report and a fake one.

---

## 7. Verdict

Asked to invent an honest stock-picking strategy with Sharpe 3+ and forbidden
from assuming it impossible, the campaign searched ~100 configurations across
19 families — reversal, momentum, lead-lag, pairs, volume, seasonality,
flow/calendar, tail events, path-conditioning, two walk-forward ML systems,
regime scaling, and a max-assembly of everything positive — and reached
**+0.51 net at realistic costs, which then failed out-of-sample.**

The honest answer is not "Sharpe 3 is impossible." It is: **Sharpe 3 is not
available from daily price and volume data in the modern era.** It once was,
in 1999–2004, from statistical liquidity provision — and this campaign
measured both that it was real (9.2σ over its matched null) and that it is
gone (−1.4 gross by 2017–19). Anyone claiming a Sharpe-3 daily-data stock
picker today should be asked for three things this study produced and their
claim probably cannot: the cost frontier, the era decomposition, and the
turnover-matched null.
