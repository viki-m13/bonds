# Thinking like a prop shop: the full accessible high-Sharpe landscape (and why 3 OOS isn't there)

Synthesis of an extensive multi-agent sweep (academic, quant blogs, FinTwit, DeFi
ecosystem, fund disclosures) + my own empirical HL tests, on where genuine
high-Sharpe edges live and which a SMALL systematic shop can actually run.

## The triangulated answer (every independent source agrees)
- **Real, audited crypto market-neutral funds report net Sharpe ~1.5-2.0.** 2.0+
  sustained is top-decile. Crypto HF average ~1.6 (2024-25).
- **A durable, net-of-cost, OOS Sharpe > 3 is NOT retail-accessible.** Where 3-10
  exists it is HFT / market-making / tiny-capacity long-tail niches — gated by
  latency/colocation/rebate-tier infra you can't match, or capacity so small +
  tails so fat the Sharpe is an illusion. "A claimed net Sharpe >3 in 2026 should
  make you suspect look-ahead/survivorship/multiple-testing unless it's an HFT/MM
  book with tiny capacity." (multiple fund/practitioner sources)
- **Cost decay is brutal:** a frictionless Sharpe 3 collapses to <=0.5 live; at
  ~0.4% aggregate cost the alpha of standard strategies statistically vanishes.

## The accessible edges, ranked by (net Sharpe x accessibility)
| edge | realistic NET Sharpe | accessible? | catch |
|---|---|---|---|
| Funding/basis carry (majors) | was 4-6, **now ~1-2 and decaying** | yes | ETF-arbed down to ~5% APR; fat-tailed funding-flip/liquidation |
| Cross-sectional stat-arb (residual MR) | ~1.5-2 (Lui), **but maker-fragile** | partial | dies on taker cost at high freq; crypto residuals trend at daily (our Kalman test) |
| Cross-venue funding dispersion | ~1-2 (good regimes) | yes ($10k-500k) | per-venue capital, transfer-latency liquidation, slippage>spread above $500k |
| Weekly cross-sectional order flow | ~0.7-1.2 (= our OF sleeve) | yes | low capacity (alpha in illiquid alts) |
| New-listing fade (1-3mo drift) | ~0.5-1.0 gross | yes | short borrow/squeeze; crowded |
| Vol-targeting + trend/crisis-alpha OVERLAY | **+0.3-0.7 uplift, robust** | yes | mostly drawdown reduction, not new alpha (we already apply it) |
| Cash-and-carry basis (CME/quarterly) | ~0.5-1.5, arbed away | partial | double-funding, 10x liq in >half the months, basis-compression unwind |
| Options VRP / skew / vol-carry | gross >1, **net ~0.5-1.0** | yes (Deribit) | catastrophic left tail; Sharpe flatters short-vol; crash erases years |
| Market-making (HLP-style) | 2.9-5.2 | **deposit only** | passive LP yield + -27% JELLY tail, not YOUR edge; active MM is pro-infra |
| MEV (CEX-DEX arb, JIT, liq backrun) | n/a | **NO** | builder/searcher oligopoly (top-3 take 75%); JIT ROI 0.007%; Aave SVR recaptured 73% |
| HFT OFI / funding-microstructure / liq-magnet | high IR gross | **NO** | sub-second queue/latency game; ~0 net for non-colocated |

## My own HIP-3 empirical test (the most novel accessible candidate)
Recorded live HL HIP-3 (equity/commodity perp) mark/oracle/funding (42 liquid
markets). Findings:
- **Basis genuinely mean-reverts** (corr(basis, future change) = -0.28; a +10bps
  basis shrinks ~1.3bps/min) and **funding tracks basis (0.73)** -> shorting
  premium perps earns large funding (top markets ~100%/yr annualized).
- Cross-sectional basis convergence is **+1.3 bps/min, 66% hit, ~Sharpe 57 GROSS**
  -- but this is VELOCITY again: a 9bps taker round-trip kills it, and the 5bps
  basis is smaller than the round-trip. **Maker-only intraday.** As a multi-day
  FUNDING CARRY it's real but (a) fat funding lives on small/meme equity perps,
  (b) you can't hedge the underlying on-chain -> directional, fat-tailed,
  (c) deployers tuned funding down 0.5x + discovery bounds to suppress it.

## Honest verdict for the brief (get to Sharpe 3 OOS, prop-shop style)
There is no retail-accessible, durable, OOS Sharpe-3 here. The prop desks that
post 3+ do it via market-making/HFT/tiny-capacity niches that require
infrastructure (colocation, exchange-share rebate tiers, builder integration) or
balance sheet (multi-venue capital) a small systematic taker doesn't have -- and
every "Sharpe 4-23" number in the literature is gross, in-sample, short-window, or
on uninvestable venues. The realistic accessible ceiling is **net ~1.5-2.0**,
reached by STACKING low-correlation accessible sleeves + vol-targeting:
  trend + carry + order-flow (validated ~1.1-1.4) + a funding-dispersion sleeve +
  the crisis-alpha/vol-target overlay (we apply) -> credibly ~1.4-1.8 OOS.
That is the honest target. Sharpe 3 OOS net is, for us, a maker/infra game we
have proven (on real HL L2 data) we cannot win as a taker.
