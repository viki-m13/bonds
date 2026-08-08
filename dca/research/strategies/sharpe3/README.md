# SHARPE3 — an honest attempt to invent a stock-picking strategy with net Sharpe ≥ 3

Started 2026-08-08 04:03 UTC. Budget: 8 hours of continuous research.
Mandate: no prior bias that this is impossible; invent freely, validate ruthlessly.

## Data (all committed in this repo)
- Daily adjusted close + daily share volume, ~24,000 US tickers including ~8,900
  delisted, 1990–2026 (`dca/research/data/tiingo/prices/`). No OHLC, no
  fundamentals — strategies must live on prices, volume, and PIT universe
  metadata (exchange, assetType, listing dates).
- Delisting-inclusive: dead tickers carry their final traded prices.

## The honesty contract (fixed BEFORE any experiment; violations = result void)
1. **Point-in-time everywhere.** Universe membership, liquidity filters, and all
   signals use only data available at the close before the trade.
2. **Execution lag.** Signals computed at close t trade at close t+1 (no
   same-close execution).
3. **Costs.** 10 bps per side base case on all turnover; results also reported
   at 5 and 20 bps. Shorts charged an extra 25 bps/yr borrow on short notional
   (hard-to-borrow names excluded via liquidity floor).
4. **Liquidity floor.** Tradable = price ≥ $5 and trailing-63d median dollar
   volume ≥ $10M at signal time. Positions sized so the portfolio is a
   realistic retail/small-fund size (capacity reported).
5. **Delisting handling.** A held name that stops trading exits at its final
   recorded price (which embeds the delisting collapse or the deal price).
6. **Sample splits.** DEV = 1995–2014 (invention playground).
   VAL = 2015–2019 (touched only to confirm survivors).
   TEST = 2020–2026-07 (locked; opened once, for the single final candidate).
7. **Multiple-testing ledger.** Every configuration ever evaluated is logged in
   WORKLOG.md with its DEV Sharpe. The final claim is judged against the count
   of everything tried, not just the survivors (Bonferroni-style honesty).
8. **Null battery.** Matched random strategies (same universe, turnover,
   position count, rebalance grid) define the luck distribution; a candidate
   must beat the 99.9th percentile of its own null.
9. **Sharpe definition.** Annualized mean/std of daily NET strategy returns,
   no cash yield credited on short proceeds or idle cash (conservative).
   Market exposure hedged strategies report hedged returns.
10. **No survivorship of experiments.** Failures stay in the log and in the
    final write-up.

## Structure
- `scripts/build_daily.py` — daily PIT panel builder (prices, volume, universe)
- `scripts/engine.py` — shared backtest engine + null generator + reporting
- `scripts/expNN_*.py` — numbered experiments, one idea each
- `WORKLOG.md` — timestamped ledger of every experiment and result
- `FINDINGS.md` — the final, honest conclusion (written at hour 8)
