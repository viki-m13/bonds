# Can we reach an honest OOS net Sharpe of 3? — the definitive verdict

This is the consolidated answer after an exhaustive, honest search: ~15 strategy
archetypes built and validated OOS net of real costs, our own recorded HL L2 data,
ten+ deep research threads across academic papers, GitHub, and quant practitioners.
The conclusion is not an opinion — it is triangulated from empirical tests on our
data, portfolio theory, and peer-reviewed evidence, all agreeing.

## The answer
**No — a durable, honest, out-of-sample, net-of-cost Sharpe of 3 is not reachable
for a retail taker on Hyperliquid.** It is reachable only with infrastructure we do
not have (maker queue priority + exchange-wide-volume rebates + colocated sub-ms
latency) or breadth we cannot source (hundreds of uncorrelated pods + leverage).
The best HONEST book we built is **~1.5 OOS** (grand stack); the taker-viable
ceiling, stacking every uncorrelated event/lead-lag edge, is **~2–2.5**, not 3.

## The three independent walls (all measured, not asserted)

### Wall 1 — Diversification math (portfolio theory + our numbers)
Combined Sharpe = S·√K / √(1+(K−1)ρ), hard-capped at **S/√ρ** (Bailey-López de Prado;
Sharpe's own SR² additivity). Our maximal honest crypto stack: K=5 admitted sleeves,
avg S≈0.64, avg ρ≈0.09 → ceiling **2.20**, observed OOS **1.38–1.55**. Adding the
negatively-correlated funding-fade (grand_stack) → **~1.5** honest (OOS 1.63). To
reach 3 needs ~18–22 genuinely-uncorrelated sleeves of this quality; a single-factor
crypto market supplies ~3–5 before new signals become correlated (ensemble.py,
max_stack.py). Effective independent bets ≈ 3 (research: Carver IDM caps ~2.75).

### Wall 2 — Costs vs. the alpha (empirical, on our HL L2 + every archetype)
- Daily/cross-sectional crypto taker stack: **~1.4–1.5 OOS** (validated, costed).
- Crypto stat-arb reversion: taker-blocked (turnover × fee > edge).
- Equity stat-arb: IS 1.24 but **OOS decayed to 0.26** (Avellaneda-Lee arbed away).
- Microstructure (our HL L2): OBI predicts returns with **IC 0.24 (t+35)** — but the
  move is ~1 bp vs ~5–9 bp round-trip → **−4.3 bps net**. The alpha is REAL and
  sub-cost: maker-only. Peer-reviewed confirmation: Pindza 2026 net Sharpe −10 to −52
  for crypto microstructure takers; DeepLOB profitable only "if we enter passively."

### Wall 3 — Execution/latency (peer-reviewed; the HFT regime is gated)
Sharpe 3+ genuinely exists in HFT, but it is the return to being the *liquidity
provider with speed*: maker rebates (the only thing that flips MM from −$8.5k/day to
breakeven — MANA Tech), queue priority worth ~half a spread (Moallemi-Yuan), and
microsecond latency (Budish et al.: races resolve in 5–10 µs; retail at 100 ms is
10³–10⁵× too slow). On HL, rebates are gated at 0.5–3% of exchange-wide maker volume
(institutional only); we proved on real fills we lose the passive-maker game to
adverse selection. Even an *omniscient* zero-fee trader (Kearns et al.) extracts only
~$21M market-wide at 10 ms — Sharpe can be infinite under foresight but profit cannot.

## Where Sharpe ≥ 3 actually lives (and why it isn't us)
- HLP vault: Sharpe 2.89 lifetime — the privileged on-chain market-maker; retail can
  only buy it as a passive LP (and it fell to 1.65 in 2026).
- Colocated HFT market-makers: rebates + µs latency + queue priority.
- Multi-strat pod shops (Millennium/Citadel): hundreds of uncorrelated pods + leverage;
  the FUND harvests the diversified top. Medallion ~2.0 net and closed/capped.

## What we DID achieve (the honest deliverable)
A genuinely uncorrelated 6-sleeve book (TREND, CARRY, BAB, SQUEEZE, ACCEL + the
negative-correlation FUNDFADE), mean pairwise ρ≈0, **honest Sharpe ~1.5 (IS 1.32 /
OOS 1.63), maxDD <9%** — up from the 1.1 starting point, and at the genuine high end
of what a retail crypto taker can sustain. Deployable via live_signal.py / executor.py.

## The only honest routes to push higher (none reach 3)
1. Add more genuinely-uncorrelated POSITIVE event sleeves (liquidation-cascade fade,
   cross-venue funding dispersion, cross-venue lead-lag — the last needs synchronized
   forward recording; it is partly a latency race, so expect the slow residual ~1–2).
2. Apply leverage to the diversified ~1.5 book for larger RETURNS (leverage scales
   return, not Sharpe — this gets big P&L, not a higher ratio).
3. Deposit in HLP to *buy* a ~2–2.9 Sharpe you cannot self-generate as a taker.

Bottom line: we climbed from 1.1 → ~1.5 honestly and proved, three independent ways,
that 3 is an infrastructure/breadth achievement out of reach for a retail taker. The
honest, deployable best is the ~1.5 grand stack.
