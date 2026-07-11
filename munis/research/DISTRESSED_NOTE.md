# Can we systematically do the "buy at 60, sell at 90" trade?

**Short answer: not with a price signal, and not with this dataset — the
apparent easy money is survivorship bias. Those clients are exercising
credit skill (or have dealer/private information), which is a different
business than the price-action strategy in FINDINGS.md.**

## What the data naively shows

Among our 1,416 bonds, the ones that ever printed at a deep discount
appear to recover almost every time:

| entry level | bonds | median Δ 6–18mo later | win rate | went <−15pts |
|---|--:|--:|--:|--:|
| < 70 | 52 | +9 pts | 86% | 0% |
| < 60 | 22 | +11 pts | 90% | 0% |
| < 50 | 10 | +11 pts | 90% | 0% |

Looks like free money: buy distressed, ~90% recover, none crater.

## Why it is an illusion

**The universe is survivorship-selected.** It was built from EMMA's
discovery endpoint, which only lists securities that *traded in
2025–2026*. So every bond in the sample — including every "distressed"
one — is alive and trading today by construction. We confirmed it: all 67
distressed-touching bonds have their **last trade in 2026**. A muni that
went to 60, then defaulted to 20 and stopped trading in 2019 is
**physically absent** from our data.

The "0% went below −15pts" and "0 stopped trading" rows are not a finding;
they are the bias itself. We only kept the winners.

## What the clients are actually doing

Buying at 60 and selling at 90 in a few months is **distressed /
special-situations investing**. The edge is *selection*: knowing which
distressed credit recovers versus which is a zero. That comes from
fundamentals a price series doesn't contain —

- essential-purpose revenue (water/sewer/toll) vs speculative
  (land-development, senior-living, industrial-development);
- bond **insurance** wraps (AGM/BAM/Assured) that backstop recovery;
- the **security structure** and lien priority in the official statement;
- coverage ratios and continuing-disclosure trends;
- restructuring / workout terms and recovery precedent.

That is a credit desk's job, not a momentum signal. The ones who "seem to
know which to buy" *do* know — from analysis or from seeing dealer axes
and holder lists we don't have.

## Could we build it honestly? Yes — but it needs different data

To backtest distressed muni selection without fooling ourselves we would
need:

1. **The full CUSIP universe including dead bonds** — a point-in-time list
   that contains bonds which later defaulted and stopped trading (not
   EMMA's survivor-only discovery list).
2. **Default / impairment history** — EMMA material-event notices
   (payment default, rating withdrawal, Chapter 9) give real default
   flags; MMA/Bloomberg have curated default+recovery data.
3. **Credit attributes at entry** — sector, insurance, rating, lien —
   most available from official statements / EMMA, but per-CUSIP scraping.

With those, the question becomes: does a *rules-based* distressed screen
(deep discount + still-insured + essential-purpose + coverage stable) beat
the base rate once the zeros are included? That is a real, fundable
research question — but it is a credit strategy, not the price-dislocation
strategy, and it is a multi-week data build, not a filter on what we have.

## Bottom line

- The price-dislocation strategy in FINDINGS.md is what our data *can*
  honestly support: mild dislocations on healthy bonds, +3.1% excess,
  survivorship-free in its OOS year.
- The 60→90 trade is real and larger, but it is credit selection on a
  survivorship-complete universe with default data — none of which we
  have yet. Anyone backtesting it on a survivor-only tape will "discover"
  a 90% win rate that does not exist out of sample.
