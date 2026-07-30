# MUNIS — trading individual municipal bonds like stocks

Goal: treat *individual* muni CUSIPs as tradable line items — download real
per-bond trade data, validate it, and honestly backtest whether a
stock-style trading approach (enter, hold days–weeks, exit) clears the
market's true frictions out-of-sample.

## Data

Source: **MSRB EMMA** (emma.msrb.org), the regulatory trade tape for US
municipal securities. Every dealer must report every muni trade to the
MSRB; EMMA republishes each print with price, yield-to-worst, par size and
side:

| side | meaning |
|------|---------|
| `S`  | dealer **sold to a customer** (a customer *bought* at this price) |
| `P`  | dealer **purchased from a customer** (a customer *sold* at this price) |
| `D`  | inter-dealer trade |

This is tick-level truth, not a matrix price: for each bond we get the
actual prices retail/institutional customers paid and received, per day,
going back ~15 years.

Pipeline (all in `scripts/`):

1. `emma_client.py` — session handling (EMMA's disclaimer gate + WAF
   quirks; python-requests is TLS-fingerprint-blocked, so HTTP goes
   through curl) and the three JSON endpoints used.
2. `build_universe.py` — for every state, every security that traded in
   the trailing year, with trade counts and volumes
   (`data/universe/universe.csv.gz`). States over EMMA's result cap are
   partitioned by maturity tiles.
3. `download_trades.py` — full trade-by-trade history for the most liquid
   securities (`data/trades/{securityId}.csv.gz`).
4. `validate_data.py` — structural, economic and cross-endpoint checks
   (`data/VALIDATION.md`).

EMMA renders CUSIP strings as images (licensing), so securities are keyed
by EMMA's opaque `securityId` everywhere; descriptions, state, coupon and
maturity are stored alongside. `emma_client.EmmaClient.validate_cusip`
maps a known CUSIP-9 to a securityId when needed.

## Honest-backtest ground rules

Fixed before results were produced (see `research/backtest.py` docstring):

- **Fills only at real customer prints.** Buys fill at the day's
  par-weighted customer-buy (S) price in a retail size band
  ($5k–$250k); sells at the customer-sell (P) price. No fills at
  inter-dealer or mid prices — those aren't available to a customer.
  If the needed side didn't print, you don't trade.
- **No intraday assumptions.** EMMA chart timestamps are date-granular;
  signals on day *t* trade at day *t+1*'s prints at the earliest.
- **Trailing-liquidity gate only.** A bond is tradable on day *t* only if
  it printed on ≥ 8 distinct days in the trailing 90 — no future
  information in universe selection at trade time.
- **Coupon accrual credited**, since muni prices are clean.
- **Matched random control.** Every strategy is scored against random
  entries in the same bonds with the same exit logic and cost model —
  this nets out carry, spread and composition, isolating timing skill.
- **IS/OOS split.** Strategy families and small parameter grids were
  pre-specified; entries through 2022-12-31 are in-sample; one locked
  config per family is evaluated once on 2023-01-01 → 2026-07-07.

### Known biases (disclosed, not hidden)

- **Universe survivorship.** EMMA's discovery endpoint only lists
  securities that traded in the trailing year (scan window
  2025-07-08 → 2026-07-08). Earlier backtest years therefore
  over-represent bonds that stayed liquid — bonds that defaulted, were
  called or went quiet before mid-2025 are missing. The final year of the
  OOS window is survivorship-free *by construction* and is reported
  separately.
- **Chart-endpoint granularity.** Per-trade timestamps are dates;
  same-day ordering is unknown (handled by the next-day-fill rule).

## Results — headline

Short-term price-action timing **fails** (the dealer spread eats every
round trip). The strategy that **works**, validated honestly OOS:

> **Deep-dislocation reversion** — buy when a bond prints ≥3 points below
> its own trailing 60-day median (a forced-seller dislocation), hold ~1
> year, sell into a customer bid. Full-sample 2013–2025: 3,089 trades,
> **74% win, +4.6% mean/trade, +3.25% excess vs a matched random-entry
> control (p<0.001)**. It loses in sustained rate selloffs (2022), so it
> is a mean-reversion strategy, not all-weather.

Full path, per-era robustness, OOS lock, and caveats in
[`research/FINDINGS.md`](research/FINDINGS.md). Today's live screen:
`research/current_picks.py`.

## Reproduce

```bash
pip install -r ../requirements.txt pyarrow
python scripts/build_universe.py      # ~1h, polite rate limits
python scripts/download_trades.py     # top-1200 by liquidity, ~30min
python scripts/validate_data.py
python research/run_backtest.py is    # in-sample grid
# lock configs into research/locked_configs.json, then:
python research/run_backtest.py oos
```
