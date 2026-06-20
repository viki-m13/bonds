# Skip-megacaps vs liquidity — each universe at its OWN realistic cost

'skip K' = exclude the K most-liquid coins, trade the next 30. Each universe priced at its OWN slippage (from its held ADV) at $1M and $10M accounts; fee 4.5bps taker + sqrt-impact slippage.

| universe | median held ADV | slip $1M | slip $10M | net Sharpe $1M (LONG/HL) | net Sharpe $10M (LONG/HL) |
|---|---|---|---|---|---|
| top-30 | $2,283M | 1.7bps | 3.1bps | +1.30/+1.15 | +1.26/+1.09 |
| skip 2/next-30 | $1,427M | 1.8bps | 3.7bps | +0.95/+0.97 | +0.92/+0.90 |
| skip 5/next-30 | $794M | 2.1bps | 4.6bps | +0.96/+1.01 | +0.92/+0.92 |
| skip 9/next-30 | $404M | 2.6bps | 6.0bps | +1.30/+1.11 | +1.21/+0.98 |
| skip 14/next-30 | $187M | 3.3bps | 8.4bps | +1.11/+0.97 | +0.99/+0.80 |

## Verdict

- At **$1M**, best net is **skip 0** (HL Sharpe +1.15); slippage is tiny (1.7bps) so the mid-cap alpha wins and skipping megacaps is worth it.
- At **$10M**, best net is **skip 0** (HL Sharpe +1.09); the thinner mid-cap names now cost 3.1bps, so the optimum shifts toward keeping more liquidity (lower skip).
- **The honest resolution:** skipping the top megacaps helps gross alpha, but each skipped rank trades a thinner coin. At small size (<$1-5M) slippage is negligible and skip-9 is best; as size grows the slippage of the thinner names eats the edge and the optimum drifts to skip-2 to skip-5 (drop just BTC/ETH/a few, keep the rest liquid). A liquidity-WEIGHTED or ADV-capped position sizer would let you keep more of the mid-cap alpha at size — the clean next step.
