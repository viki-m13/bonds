# Hyperliquid L4 / per-account flow data — how to get it, and the pipeline

**Question:** can we get HL L4 trade data and build a higher-Sharpe / higher-CAGR
strategy with it? **Yes, the data is obtainable** — HL is one of the only venues that
exposes order-by-order data *with the counterparty wallet addresses*, because the book
is on-chain. That per-account attribution is a genuinely different information axis from
price-based factors (STRATA) and the vol channel (VOL), so it is the most plausible
source of a *low-correlation* sleeve we have not yet tapped.

## What exists, and what each path costs

| source | granularity | history? | cost / blocker |
|---|---|---|---|
| WS `trades` | every fill + `users:[buyer,seller]`, side, px, sz, tid | forward only | **free**; needs a persistent box; weeks to accumulate |
| WS `l2Book` | depth aggregated by price level | forward only | free (already recorded by `record_l2.py`) |
| S3 `node_trades` archive | every fill + buyer/seller + `start_pos` + `oid` | **yes** | **AWS creds** (requester-pays: you pay egress) |
| S3 `node_raw_book_diffs` / `node_order_statuses` (by_block) | full L4 book: every add/cancel/modify | yes | AWS creds; heavy to replay |
| Hosted (QuickNode L4 gRPC / Dwellir) | full L4 stream | realtime only (some archival) | paid subscription |
| node `--write-fills/-order-statuses/-raw-book-diffs` | authoritative realtime L4 | forward only | capable always-on server |

**Verified in-session:** the public WS `trades` message really does carry
`users:[buyer, seller]` (sampled live), and the public REST/WS is reachable from here.
The S3 archive returns HTTP 403 without credentials (it is requester-pays), so the
*historical* path cannot run in this sandbox until AWS creds are provided.

## node_trades schema (the dataset we target)

`coin, side('B'=taker buy/'A'=taker sell), time, px, sz, hash, trade_dir_override,
side_info:[{user,start_pos,oid,...}(buyer), {...}(seller)]`. The WS tape has the same
fields minus `start_pos` (so the whale-by-position feature is S3-only).

## The pipeline (built + smoke-tested this session)

- **`record_trades_l4.py`** — lean forward recorder of the per-account tape → one
  Parquet shard per UTC hour (cheap to run for weeks). *Smoke test: recorded a live
  sample, full buyer/seller attribution captured.*
- **`fetch_hl_l4.py`** — S3 requester-pays puller for the historical `node_trades`
  archive → the **same tidy schema**, so the backtest is source-agnostic. Creds-gated.
- **`flow_l4.py`** — builds per-(date,coin) flow features and (with ≥120 days) a
  cross-sectional market-neutral sleeve, reporting Sharpe/CAGR IS/OOS and correlation
  to STRATA and VOL.
  - `cvd` net taker $ flow, `imb` signed imbalance, `big` large-trade tilt,
    `conc` per-account flow concentration (Herfindahl over aggressor addresses),
    `whale` net flow of large-position accounts (S3 `start_pos`, history only).
  - *Smoke test on a live sample: features are economically sane* (e.g. BTC imbalance
    −0.93 = heavy taker selling that window; LINK +0.83 with 0.78 concentration = one
    address accumulating).

## What is needed to produce a *validated* Sharpe > VOL flow strategy

History. Two honest routes, both needing one input only the user can supply:
1. **AWS credentials** (read-only, requester-pays) → `fetch_hl_l4.py` pulls months/years
   of `node_trades`, `flow_l4.py` backtests it here, net of costs, vs VOL. Fastest to a
   real number.
2. **Run `record_trades_l4.py` forward** on an always-on box for a few weeks, then
   backtest the accumulated tape. Free, but no number until the data accrues.

Either way the feature + backtest code is identical and ready. Expectation, stated
honestly: per-account flow is a real, different signal that should be *low-correlation*
to STRATA/VOL (the diversification the blend math actually rewards), but most pure-L4
edge is sub-second/HFT and below our 4.5bps taker floor — the realistic win is a
medium-frequency (daily) flow sleeve that *diversifies* the blend, not a standalone 3.
