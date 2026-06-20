# What's working on Hyperliquid now (June 2026) — intelligence synthesis

Researched live: HL docs/vaults, DefiLlama, on-chain post-mortems, BitMEX Q1-2026
report, top-trader teardowns, HIP-3 dexs. Filtered vault-marketing from verifiable.

## Bottom line for STRATA (a daily taker book)
The easy retail edges are **crowded or decayed** (75-85% of HL addresses lose money).
A pure-taker daily overlay can realistically add **only ~0.2-0.5 incremental Sharpe**,
and the genuinely fresh edges are **capacity-tiny, maker-dependent, and tail-risky**
(they come from the same fragilities that liquidate you). This CONFIRMS the
STRATA+VOL blend (~2.14) is near the accessible ceiling — there is no third *strong,
scalable, taker-daily* uncorrelated book sitting on HL right now.

## Ranked additive edges for a daily taker
| edge | realistic add | maker/taker | capacity | verdict |
|---|---|---|---|---|
| HIP-3 equity/commodity basis vs CEX/CME | +0.3-0.5 Sh, 10-25% APR | taker-feasible (growth-mode ~0.45bp) | low-med, one-sided books | best NEW edge, capacity-capped |
| long-tail / new-listing funding capture | +0.2-0.4 Sh, 8-15% APR | taker in 6-48h windows | $10k-500k/pair | real residual; reversal risk |
| weekend/after-hours equity+commodity gap basis | episodic, high Sh when on | taker | low | event overlay, not steady |
| HIP-3 oracle-lag arb (1%-tick cap) | high when fires | latency bot really | very low | edge = the fragility that liquidates you |
| HLP vault deposit (passive benchmark) | ~12-22% APR, Sharpe ~1.65 | maker pool | very high | yield w/ short-vol tail, NOT a taker edge; decaying |

## Decayed — do NOT chase (matches our prior findings)
- Major-pair funding arb (maker-only economics; bots take 80%+ of spread; ~2-4% APR on
  majors eaten by taker cost). Copy-trading (structurally losing). HYPE farming (over).
  Liquidation sniping & basic MM rebates (HLP/infra capture it). Passive HLP "edge"
  (decaying 42%→22%→~12% with TVL).

## The two things genuinely worth FORWARD-RECORDING (not backtestable now)
1. **HIP-3 cross-venue basis** — equities/commodities/FX as perps (Trade[XYZ] = S&P 500
   licensed, gold/oil/copper, FX) vs CEX/CME. Cash-and-carry + weekend gap basis. Needs
   HIP-3 funding/mark history (we have record_hip3.py) + a CEX/CME reference feed.
2. **Long-tail funding cross-section** — fat funding lives in new-listing alts (250-1100%
   APR on the paying side), not majors. Our CARRY/FUNDFADE sleeves already capture the
   funded-universe version; the long-tail extension needs forward funding recording
   (record_orderflow.py / a funding recorder) on the newest HL listings.

## Tail risks to model explicitly (the research's hardest warnings)
- **HIP-3 oracle tail risk:** Ventuals' SpaceX perp crashed 45% in 30 min (May 2026) on
  a split-handling oracle error, liquidating 405 users; the dex is winding down. Pre-IPO
  HIP-3 failed outright. Don't size HIP-3 until 90+ days of live oracle/slippage data.
- **HL funding mean-reversion** flips within an hour (4%/hr cap → extreme prints); a +20%
  week can become -30%. Only enter deltas ≥0.02%/8h that have held 5-7 days.

## Verdict
HL's current live edges reinforce our conclusion: the deployable best is the STRATA+VOL
blend (~2.1), and the only honest path *higher* is forward-recording the capacity-tiny
HIP-3 basis / long-tail funding edges (+0.2-0.5 Sharpe, tail-risky) — not a backtest we
can run today. Recommend standing up the HIP-3 + funding recorders to harvest forward
data before sizing any of it.
