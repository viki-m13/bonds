# Findings: can you get an honest OOS Sharpe of 3+ trading individual Treasuries?

**Short answer: no — not from end-of-day price data, and this folder proves
why rather than papering over it.** Every configuration that shows a Sharpe
anywhere near 3 is harvesting *bid-ask bounce* — mean-reversion in the
end-of-day marks themselves — which is not tradeable. Once you require
realistic execution (trade a day after you see the signal) and charge
realistic costs, the edge is gone.

This document records the evidence so the negative result is auditable.

## What was tried

All design and tuning used the in-sample window (2010–2019) only.

1. **Curve relative value** — yield residual vs a daily Nelson-Siegel-Svensson
   fit (cheap bonds long, rich bonds short, duration-neutral).
2. **Local relative value** — each bond's yield minus the duration-weighted
   mean yield of its `k` nearest-maturity coupon neighbours. No curve model;
   pure local cheapness. This is the sharpest, most "arbitrage-like" signal.
3. **Carry / rolldown** per unit duration from the fitted curve.
4. **Idiosyncratic momentum** in the residual.

Each was run across fractions traded, rebalance frequencies (daily → weekly),
turnover-control (hysteresis), holding horizons (1–21 days) and — decisively —
**execution lags**.

## The decisive test: execution lag

`src/explore_rv.py` and `src/rv_backtest.py` sweep an *execution lag* `L`: the
book is formed from the signal, then actually put on `L` trading days later.

- `L = 0` means you trade at the very close whose marks generated the signal.
  You cannot really do this (you'd need to observe the close and simultaneously
  trade at it), and a large part of any next-tick reversion is just the mark
  bouncing inside its own bid-ask. So `L = 0` is the *upper bound of an
  illusion*, not a tradeable number.
- `L ≥ 1` means observe at close `t`, trade at a later close. Genuine
  multi-day convergence survives; pure bounce does not.

### Result 1 — idealized tranche sim (gross, no costs), IS

`local` signal, top/bottom 20%, 1-day holding:

| execution lag | gross Sharpe |
|---:|---:|
| 1 | 6.25 |
| 2 | 4.08 |
| 3 | **−2.27** |

A Sharpe of 6 that **flips negative by lag 3** is the textbook fingerprint of
bid-ask bounce, not a risk premium. Real edges decay smoothly; they don't
invert two days later.

### Result 2 — real cost-charging engine, IS

Daily rebalance, top/bottom 20%, FedInvest half-spread charged per side:

| execution lag | gross Sharpe | net Sharpe | ann. cost |
|---:|---:|---:|---:|
| 0 | **7.8** | −1.33 | 4.3%/yr |
| 1 | 0.68 | −8.2 | 4.3%/yr |
| 2 | 0.60 | −8.3 | 4.3%/yr |

Two independent nails in the coffin:

1. **Gross Sharpe collapses 7.8 → 0.68 the moment you lag execution by one
   day.** The 7.8 was bounce.
2. **Net Sharpe is negative everywhere.** The tradeable convergence is worth
   ~15 bp/yr gross; turning the book over to capture it costs multiples of
   that. The best net Sharpe found in the entire real-engine sweep was ≈ +0.36,
   and only with an optimistic lag-0 fill plus heavy turnover throttling —
   i.e. still partly bounce, and still nowhere near 3.

### Result 3 — bounce decomposition figure (IS **and** OOS)

![bounce](results/bounce_decomposition.png)

Gross and net annualized Sharpe of the daily local-RV book vs execution lag:

| lag | IS gross | IS net | OOS gross | OOS net |
|---:|---:|---:|---:|---:|
| 0 | **7.80** | −1.33 | **7.74** | −4.23 |
| 1 | 0.68 | −8.21 | 1.19 | −10.97 |
| 2 | 0.60 | −8.28 | 1.42 | −10.99 |
| 3 | 0.47 | −8.35 | 1.18 | −11.51 |
| 4 | 0.29 | −8.76 | 1.07 | −11.09 |

The pattern is identical in both windows: gross Sharpe ≈ 7.8 at lag 0 — right
through the Sharpe-3 line — then falls off a cliff to ~1 the instant execution
is delayed a single day, while net Sharpe sits far below zero at every lag.
That the lag-0 spike is present out-of-sample too confirms it is a *structural
microstructure artifact* of EOD marks, not a fragile in-sample fluke — and
equally, that it is uncapturable: no honest execution touches Sharpe 3.

## Why this is the *right* answer

A Sharpe of 3+ in cash Treasuries would be extraordinary. The genuine
high-Sharpe trades in this market — on-the-run/off-the-run specialness,
auction-cycle financing, cheapest-to-deliver — are **repo- and
futures-financing** phenomena that need securities-lending/repo data this
folder does not have; they are not visible in outright end-of-day prices.
Everything that *is* visible in EOD prices and looks like a 3+ is the bounce
demonstrated above.

The value delivered here is therefore the **honest machinery and the proof**:
a validated CUSIP-level dataset (YTMs matching FRED to ~1 bp), a no-look-ahead
cost-aware engine, and a reproducible demonstration that separates a real (but
small, sub-1-Sharpe, net-negative) convergence signal from an
attractive-looking artifact. Fabricating a 3 would have required either
lag-0 fills, ignoring costs, or both — exactly the moves this folder exists to
catch.

## Reproduce

```bash
python3 src/explore_rv.py        # idealized lag/horizon grid (IS)
python3 src/rv_backtest.py       # real cost-charging IS sweep
python3 src/prove_bounce.py      # the decomposition figure (IS + OOS)
```
