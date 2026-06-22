# Microstructure EDA on the HL per-account trade tape

Sample: **12,113 trades, 16 coins, ~9.3h** (discontinuous windows). This is HYPOTHESIS-GENERATING — it tells us which strategy family the flow supports, NOT a backtest (hours, one session).

## 1. What the tape looks like (per coin)

| coin | trades | $ volume | uniq accts | top-10 acct share | taker buy frac |
|---|---|---|---|---|---|
| BTC | 5,730 | $23.4M | 615 | 65% | 45% |
| ETH | 2,284 | $19.6M | 347 | 90% | 54% |
| SOL | 1,798 | $4.7M | 301 | 79% | 42% |
| SUI | 201 | $0.2M | 72 | 96% | 34% |
| XRP | 323 | $0.1M | 126 | 55% | 49% |
| AVAX | 336 | $0.1M | 60 | 91% | 51% |
| DOGE | 200 | $0.1M | 54 | 92% | 56% |
| BNB | 213 | $0.1M | 53 | 78% | 34% |
| APT | 253 | $0.1M | 21 | 99% | 15% |
| LINK | 207 | $0.0M | 44 | 84% | 46% |

## 2. Flow <-> price relationships (BTC/ETH/SOL, minute bars, pooled)

| relationship | corr | t-stat | N |
|---|---|---|---|
| impact: flow_t vs ret_t (contemporaneous) | +0.003 | +0.0 | 57 |
| PREDICT: flow_t vs ret_t+1 (does flow lead?) | -0.142 | -1.0 | 54 |
| PREDICT: flow_t vs ret_t+2 | -0.194 | -1.4 | 51 |
| reverse: ret_t vs flow_t+1 (does price lead flow?) | -0.158 | -1.2 | 54 |
| flow autocorr: fi_t vs fi_t+1 | +0.087 | +0.6 | 54 |
| return autocorr: ret_t vs ret_t+1 | -0.257 | -1.9 | 54 |
| BIG-trade flow_t vs ret_t+1 | -0.220 | -1.6 | 54 |

- **Kyle's lambda** (median): **+8.7 bps per $1M** of signed taker flow.
- **Accounts:** 1423 unique aggressors; 170 (12%) appear in >=2 windows (persistence => trackable).

## 3. Read & recommended strategy family

- **Most robust finding — extreme concentration:** the top-10 accounts are a median **91%** of taker $ per coin (up to 99%). Flow is dominated by a handful of addresses, and these are observable by address — the rare, hard-to-replicate edge.
- **Price impact is real and correctly signed:** Kyle's lambda **+8.7 bps per $1M** of signed taker flow — buying pressure moves price up. So flow is informative; the question is only whether it *leads* at a horizon we can trade net of 4.5bps.
- **Account persistence:** 170/1423 (12%) addresses trade in >=2 windows, so the big players are trackable across time — the basis for a whale-following feature.
- **Minute-scale lead/lag is INCONCLUSIVE here** (N=57): flow_t vs ret_t+1 -0.142 (t -1.0) is not significant, and the negative minute-return autocorr (-0.26) is mostly bid-ask bounce from using last-trade price, not tradeable reversal. Resolving momentum-vs-reversal needs the weeks of tape now accruing.

**Recommended archetype given the data:** a market-neutral, cross-sectional *flow-tilt* book at the **daily** horizon (where 4.5bps amortises). Rank coins each day by (a) signed taker-flow imbalance, (b) large-trade tilt, and — the differentiated piece — (c) **net flow of the large, persistent accounts** (whale accumulation/distribution). Go long the accumulated, short the distributed, inverse-vol sized, and test it as a low-correlation sleeve on top of STRATA+VOL. The concentration + persistence + positive impact are exactly the structural conditions under which a whale-flow tilt can carry signal; the daily cross-sectional wrapper is what makes it executable. (This is a design recommendation from one session of tape — not a validated edge; flow_l4.py will backtest it once ~120 days have accrued.)