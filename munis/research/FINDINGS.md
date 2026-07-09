<!-- FINDINGS: filled from results/is_grid.csv and results/oos_results.csv -->
# Can you trade individual munis like stocks? — honest verdict

**Short answer: no — not profitably, with a systematic price-action approach,
after the real cost of trading as a customer.** Across 1,416 individual
municipal bonds and ~2.0 million real trade prints, every strategy family we
tested fails to beat a matched random-entry control in the same bonds, both
in-sample and out-of-sample. The binding constraint is the customer
bid–ask spread: you buy at the dealer's ask and sell at the dealer's bid,
and that round-trip cost (~0.4–0.9 points of par) swamps any timing edge a
price/flow signal produces.

This is the answer the data gives, not the answer we hoped for. The value of
the project is that the harness is honest enough to produce it.

## What "honest" means here (rules fixed before any result)

- **Fills only at prices real customers actually got that day.** Buys fill at
  the day's par-weighted customer-buy print (dealer-sell, retail size band
  $5k–$250k); sells at the customer-sell print. Never at mid or inter-dealer
  prices — a customer can't trade there.
- **No intraday assumptions** (EMMA timestamps are date-granular): a signal on
  day *t* enters at day *t+1*'s prints at the earliest.
- **Trailing-liquidity gate with no lookahead** (≥8 active trading days in the
  trailing 90) decides tradability at trade time.
- **Coupon accrual credited** (muni prices are clean).
- **Matched random-entry control**: every strategy is scored against random
  entries in the *same bonds* over the *same window* with the *same* exit
  logic and cost model. This nets out carry, spread and panel composition,
  leaving only timing skill. Excess-vs-control is the metric that matters;
  raw return is dominated by the spread everyone pays.
- **IS/OOS split with a mechanical config lock**: families and small
  parameter grids pre-specified; entries ≤2022-12-31 in-sample; one config
  per family locked by a fixed rule (`lock_configs.py`); each locked config
  run once on 2023-01-01→2026-07-07, with the survivorship-free final year
  (2025-07-08→) reported separately.

## Data

- **1,416 securities, 2,024,432 trade prints**, first trades back to 2005
  (dense from 2016). 1,312 bonds have complete-from-issuance histories
  (uncapped by EMMA's 5000-trade endpoint limit).
- Discovery universe: **201,616 securities across 42 states/territories.**
- Validation (`data/VALIDATION.md`): 100% agreement with an independent EMMA
  summary endpoint on price ranges and trade counts; Δprice/Δyield
  correlation −0.98; 96.9% of two-sided days have customer-buy ≥
  customer-sell. The data is what it claims to be.

### The cost that decides everything

Median same-day customer round-trip (customer-buy − customer-sell price):
**~0.44 points**; mean ~0.91 (illiquid names have far wider spreads). On a
par-100 bond held a few weeks, a signal must reliably capture more than this
*plus* the adverse selection of trading when the market wants to trade with
you. None tested does.

## In-sample grid (entries ≤ 2022-12-31)

<!--IS_TABLE-->

Reading it: mean per-trade returns are strongly negative for the reversion
("firesale") and momentum families — that is the spread. What matters is
**excess vs control**, which is ~0 (bootstrap p-values far from significance).
The signals identify *when a bond is trading*, not *when it is mispriced*.

## Locked configs (selected mechanically from IS)

<!--LOCKED_TABLE-->

## Out-of-sample results (locked configs, run once)

<!--OOS_TABLE-->

## Conclusion

<!--CONCLUSION-->

## Why this is the expected result (and what would change it)

Municipal bonds are a **dealer market with ~50,000+ issuers and a million+
CUSIPs**; any single bond trades thinly and the retail spread is wide
relative to the drift you can forecast from its own trade history. Treating a
CUSIP "like a stock" ignores that a stock has a continuous two-sided
lit order book with cent-wide spreads, while a muni has an intermittent
dealer quote with point-wide spreads. The edge, where it exists in this
market, is captured by dealers (the round-trip we measured) and by
relative-value desks with credit models and inventory — not by price-action
timing on public trade prints.

What could plausibly beat the control (untested here, honest next steps):
holding to **maturity/call** to convert the spread from a cost into an
amortized entry discount; **primary-market** allocation (buying the new-issue
concession at par, which retail can't reliably get); or a **cross-sectional
credit/richness signal** with fundamentals rather than own-price momentum.
Those are different mandates than "trade it like a stock."
