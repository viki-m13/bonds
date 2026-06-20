# Do we add to existing positions each period? What if we don't?

**Current behavior: YES.** Every biweekly/monthly contribution buys the current
top-2 by score, whether or not those names are already held. When a name stays
top-ranked (NVDA, AAPL did for years), the contributions keep piling in — which
is exactly why the book is ~36% NVDA / 27% AAPL.

## Experiment: never add — each contribution buys only names not already held

244-window grid, biweekly k=2, 5 bps.

| | beat QQQ | beat SPY | median vs QQQ | p10 | worst | full mult | positions | top holding |
|---|---|---|---|---|---|---|---|---|
| **Add to existing (current)** | 93% | 98% | +28.8% | +3.0% | −10.6% | 20.0× | 91 | 36% |
| **New names only (no adds)** | 7% | 85% | −10.9% | −30.2% | −46.0% | 4.9× | 589 | 9% |

## Why "add to existing" is the engine of the whole strategy

Momentum **persists**: the names that are top-ranked tend to stay top-ranked.
So "buy the current top-2 even if held" is really *keep pouring money into the
proven winners*. That compounding into NVDA/AAPL/etc. is where the 20× comes
from.

Forcing every contribution into **new** names breaks this three ways:
1. You buy each winner only once (a tiny lot) and never add — you ride the
   price but starve the position of fresh capital.
2. Once you own the leaders, contributions get pushed into progressively
   lower-ranked names.
3. You end up owning ~589 names — essentially an equal-weight momentum index of
   the whole universe (top position just 9%). That diffuse book beats SPY about
   85% of the time (equal-weight has periods over cap-weight SPY) but **loses to
   QQQ 93% of the time**, because Nasdaq mega-cap tech crushed equal-weight over
   2009-2026.

## Takeaway

This is the most decisive result in the whole study, and the mirror image of
every diversification test: SUMMIT's edge **is** concentration into persistent
winners via continued adds. Don't-add maximally diversifies and is the worst
outcome tested (7% win, 4.9×). Adding to existing positions is not a side
effect — it is the strategy. (Investors who want *less* concentration should use
the optional loose trims/caps, which dial it back a little at low cost — not
switch off adding entirely, which dismantles the edge.)
