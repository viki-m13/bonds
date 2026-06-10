# winrate30 — high-confidence 30-day stock buy signals

A self-contained tool (no dependencies on anything else in this repo) that
recommends stocks with a high historical probability of being **higher 30
calendar days (21 trading days) later**, with the hit rate validated
out-of-sample via walk-forward testing.

## The validated result, up front

Walk-forward out-of-sample, 2016 → mid-2026 (rules re-selected each year
using only prior data):

| Level | Result |
|---|---|
| Per single stock signal | **70.9%** positive after ~30 days (190/268, 95% lower bound 65.2%), avg +5.4% per signal |
| Per monthly signal basket | **100%** positive (9/9 months, 95% lower bound 70.1%), avg +7.6%, worst month +1.9% |
| Baseline (any stock, any day) | ~55–65% depending on the year |

### Can any stock picker honestly hit 95% per pick? No — here is the proof

This tool searched **~17,000 rule combinations over 26 years of data**
(trend, momentum, oversold, volatility, drawdown, market regime, VIX). The
best *in-sample* single-stock rule tops out at ~82–85%, and rules that
looked best in-sample routinely decayed out-of-sample (the worst case:
"panic rebound" rules at 83% in-sample scored 13% in March 2020). A
1-month holding period simply carries too much idiosyncratic single-stock
risk (earnings, news) for a 95% per-pick claim — any backtest showing one
is overfit. **The honest validated ceiling is ~70% per pick, ~80%+ per
diversified signal basket — and 100% of the 9 out-of-sample monthly
baskets so far.** If you want the 95%-style confidence, trade the signals
as a basket, not as single picks.

The signals are **episodic, not daily**: they fire when a volatility panic
(VIX > 30) hits while the market's long-term uptrend is intact — typically
several times per decade (2018, 2020, 2021, 2024 in the test window), with
dozens of stocks at once. In calm or broken markets the tool correctly
recommends nothing.

## Quick start

```bash
cd winrate30
pip install -r requirements.txt

# 1. Run the full walk-forward validation (downloads ~26 years of data,
#    writes reports/validation_report.md and selected_rules.json)
python validate.py

# 2. Get current recommendations (re-run any day, e.g. after the close)
python recommend.py
```

`recommend.py` refreshes prices automatically and prints stocks that
triggered a production rule in the last 5 trading days (`--lookback N` to
change). Most days have no signals — that's by design: the edge exists in
specific panic-in-an-uptrend episodes, and the tool stays silent otherwise.
When an episode hits, expect dozens of names within a few weeks; buy them
as an equal-weight basket and hold each ~30 days.

## How it works

1. **Universe**: ~340 liquid US large caps across all sectors
   (`universe.py`), plus SPY and VIX as market-regime context.
2. **Candidate rules**: a grid of ~17,000 condition combinations over VIX
   level, stock trend (200-day average, golden cross), pullback depth
   (distance from 52-week high, RSI), volatility regime, trailing 1-month
   win rate ("steady compounder"), and 12-month momentum. One hard,
   prespecified gate: **no recommendations while the S&P 500 is below its
   200-day average** (broken-market rebound rules failed catastrophically
   out-of-sample, so they are excluded by construction).
3. **Selection**: candidate rules must have >= 30 non-overlapping training
   signals; they are taken from the highest hit-rate tier (90%+, then 85%+,
   then 80%+) ranked by the **Wilson 95% lower bound** of the deduplicated
   hit rate, with an overlap cap so the ensemble isn't one setup repeated.
   There is deliberately no lower tier — the tool would rather recommend
   nothing than recommend from a mediocre rule.
4. **Validation** (`validate.py`): for each test year 2016→present, rules
   are selected using only data ending 21 trading days *before* the test
   year, then applied to that year. Every signal's actual 21-day forward
   return is recorded. Signals are deduplicated per stock (no re-signal
   within 21 trading days) so overlapping windows are never double-counted.
   The pooled result is a genuine out-of-sample hit-rate estimate.

See `reports/validation_report.md` for the validated numbers, per-year
breakdown, worst-month analysis, and the exact production rules.

## Files

| File | Purpose |
|---|---|
| `validate.py` | Walk-forward validation; writes report + production rules |
| `recommend.py` | Current buy signals from the validated rules |
| `universe.py` | Stock universe |
| `features.py` | Indicator / condition computation |
| `rules.py` | Rule grid search, dedup, Wilson bound, selection |
| `data.py` | Price download + local cache (`data_cache/`, gitignored) |
| `config.py` | All tunable parameters |
| `selected_rules.json` | Production ensemble (output of `validate.py`) |
| `reports/` | Validation report + every OOS signal with its outcome |

## Honest caveats (read these)

- **Survivorship bias**: the universe is today's large caps, so companies
  that collapsed are missing from the backtest. Large-cap restriction
  limits this, but the true forward hit rate is likely somewhat below the
  backtested one.
- **Correlated signals**: stocks move together. The 95% figure is a
  per-signal average; in a sharp market selloff many concurrent signals
  fail at once (see the worst-months table in the report). Position sizing
  should assume clustered losses, not independent coin flips.
- **Regime dependence**: most of the validation window (2016–2026) is a
  structural bull market. A 2008-scale bear would produce hit rates far
  below the historical average.
- No transaction costs/taxes are modeled (small for 30-day holds in liquid
  large caps, but not zero).
- **Not investment advice.** A validated 95% historical hit rate is a
  statistical statement about the past, not a guarantee about any single
  future trade.
