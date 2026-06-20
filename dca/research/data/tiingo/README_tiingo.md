# Tiingo PIT (point-in-time / delisting-inclusive) data — SOURCE: Tiingo

Downloaded via the Tiingo API (free tier) on 2026-06-20. Labeled `tiingo_*`.

## Files
- **`tiingo_universe_pit.parquet`** — the survivorship-key: every US ticker
  (16,026 stocks + 7,509 ETFs) with `startDate`/`endDate`. `endDate` < today ⇒
  delisted/acquired. **8,911 delisted names have price data** — these are exactly
  the survivors-AND-losers that yfinance lacks. Use this to build point-in-time
  universes (a stock is tradeable on date d iff startDate ≤ d ≤ endDate).
- **`tiingo_price_downloader.py`** — resumable daily-price downloader (adjClose =
  total return + adjVolume, float32, chunked parquet to `tiingo_chunks/`). Reads
  `TIINGO_KEY` from env (key NOT stored in repo). Skips already-downloaded tickers
  so it resumes across runs.
- `tiingo_chunks/ac_*.parquet`, `vol_*.parquet` — daily prices/volume (added as
  the download progresses).

## CRITICAL: free-tier rate limit
Tiingo free = **~500 requests/hour**. Full 22,704-ticker pull ≈ **45 hours** of
hourly-batched running. Options: (1) run the downloader in hourly batches over
1–2 days (it resumes); (2) upgrade Tiingo (paid tiers lift the cap) for a one-shot
bulk pull; (3) prioritize a subset (the downloader front-loads key ETFs + the
S&P500/S&P400/NDX universe, then delisted names).

## Honest coverage caveat
Tiingo has ~7–8k delisted names (huge upgrade over yfinance's ~0) BUT is missing
the OTC-Q catastrophic bankruptcies (LEHMQ, ENRNQ → no data). So backtests on it
are *far less* survivorship-biased than yfinance, but still **mildly optimistic**
(the worst bankruptcies are absent). For fully clean: CRSP/Sharadar (paid).
