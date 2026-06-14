# PULSE — price-action crypto trend, and the honest hunt for Sharpe 3

This folder is the record of a specific brief: **invent a price-action technical
strategy with Sharpe > 3, however necessary — honestly.**

The honest answer, fully evidenced in
[`research/SHARPE_INVESTIGATION.md`](research/SHARPE_INVESTIGATION.md): a
genuine, causal, cost-aware, out-of-sample Sharpe > 3 is **not attainable on any
OHLCV data available in this repo**. Every strategy that prints a Sharpe above ~2
is a **bid-ask-bounce / stale-price artifact** that collapses (and flips sign)
once you stop trading at the exact bar used to form the signal.

![PULSE equity curve](research/pulse_equity.png)

## What you actually get

- **PULSE** ([`strategy_daily.py`](strategy_daily.py)) — the best *honest* result:
  a vol-targeted, dollar-neutral daily-crypto **trend + 20-day Donchian breakout**
  book over 111 coins (2014–2026). **Sharpe ≈ 1.20 net of 10 bps/side, −16% max
  drawdown**, ~18%/yr at 15% vol. A real, tradeable, market-neutral edge — just
  not a 3.
- **The mirage, demonstrated** ([`mirage_demo.py`](mirage_demo.py)) — the same
  hourly cross-sectional reversal scores **Sharpe +17** traded at the formation
  close and **−7** when you skip one bar. That sign flip *is* the artifact behind
  most published "Sharpe 3+" price-action claims.
- **The full search** — daily US equities, daily crypto, and live-fetched hourly
  crypto, with every number and why it does or doesn't survive honest execution.

## Files
| file | role |
|---|---|
| `strategy_daily.py` | PULSE: the honest daily-crypto strategy + equity curve |
| `mirage_demo.py` | reproducible +17 → −7 hourly bounce artifact |
| `data_hourly.py` | hourly crypto panel loader |
| `backtest.py` | vectorized hourly long/short engine (causal lag + turnover costs) |
| `fetch_hourly.py` | re-downloads hourly OHLCV (binance.us klines); data is git-ignored |
| `research/SHARPE_INVESTIGATION.md` | the full honest record + the Sharpe-3 arithmetic |

## Reproduce
```bash
pip install -r ../requirements.txt
python strategy_daily.py        # PULSE (uses data/crypto, already in repo)
python fetch_hourly.py          # re-fetch hourly data (~45 MB, git-ignored)
python mirage_demo.py           # the bounce artifact
```

*Research code, not investment advice. The point of this folder is the honesty:
a real Sharpe-1.2 strategy and a demonstration of why the bigger number isn't
real here — claiming a 3 would require trading at the formation bar, hiding
costs, or cherry-picking a regime.*
