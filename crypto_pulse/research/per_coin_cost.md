# Top-30 book with PER-COIN realistic cost (higher on thin names)

Each coin pays fee + its own sqrt-impact slippage (from its 30d ADV and realized vol). Combined per-coin position, vol-targeted. Equity curves for short (HL era) and long (2015-2026).

## Net Sharpe by account size & execution

| account | exec | avg all-in bps | LONG 2015-26 | SHORT (HL era) |
|---|---|---|---|---|
| $1M | taker | 7.6 | **+1.16** | +1.01 |
| $1M | maker | 4.6 | **+1.23** | +1.11 |
| $10M | taker | 12.1 | **+1.03** | +0.96 |
| $10M | maker | 9.1 | **+1.10** | +1.07 |
| $100M | taker | 23.6 | **+0.72** | +0.83 |
| $100M | maker | 20.6 | **+0.79** | +0.93 |

## Verdict

- At **$10M taker with per-coin slippage** (BTC ~1 bp, thin names several bp): LONG Sharpe **+1.03** (CAGR +16%, maxDD -36%), SHORT/HL **+0.96**. The less-liquid names paying more does NOT break the book — the daily rebalance keeps clips small vs ADV.
- Scales cleanly to ~$10M; at $100M the thin-name slippage starts to bite (see table) and a liquidity-weighted sizer would help. Maker execution adds ~0.1-0.2 throughout. With the HL-era funding sleeves the full book is ~1.5 on top of this price-only curve.
