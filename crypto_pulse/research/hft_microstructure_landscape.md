# HFT / microstructure landscape — GitHub + academic survey, and our empirical match

Research mandate: find an honest OOS Sharpe of 3 via HFT methods. This documents
the survey (GitHub repos + academic) and how it lines up — exactly — with our own
test on recorded HL L2 (`microstructure_alpha.py`).

## The fee regime inverts the equities intuition (the crux)
On Hyperliquid retail pays **4.5 bps taker / 1.5 bps maker per side** (round-trip
~9 bps taker, ~3 bps maker). Maker *rebates* (−0.1 to −0.3 bps) require being a top
market-maker by share of maker volume — unreachable for us. BTC/ETH perp spreads
are **~0.1–1.5 bps — far smaller than the fee.** So the fee, not the spread, is the
dominant cost, and remote retail order-to-fill latency is **~0.9–1.0 s** (vs ~0.2 s
colocated). A taker signal must predict a **net move > ~9 bps** to be viable.

## Our empirical result (real HL L2, microstructure_alpha.py)
- Order-book imbalance predicts short-horizon mid-returns with **IC +0.24 (t+35)**
  at 1 s, +0.15 at 10 s; microprice +0.20. The predictability is real and strong.
- But the top-decile predicted move is **~1 bp** vs ~5–9 bps round-trip → **−4.3 bps
  net**. Taker-blocked. Monetizable only as a maker (earn the spread), which needs
  queue priority/low latency we falsified (`maker_sim`, selective-fast-maker both
  net-negative from adverse selection).

## GitHub survey — independent confirmation, repo by repo
- **`nkaz001/hftbacktest`** — the only honest queue+latency backtester. Its OBI
  market-making example shows Sharpe **10.8 (2023) → 5.4 → 3.0 (2025)** — but these
  are **maker** Sharpes assuming Binance's **0.5 bps rebate + ideal latency**. On HL
  (+1.5 bps maker, no rebate, ~1 s latency) the per-trade edge flips negative.
- **OFI (Cont-Kukanov-Stoikov)** — Dean Markwick's BTC study: headline R² 0.4–0.8 is
  **contemporaneous** (not tradeable); **predictive** next-second R² ≈ **3%**, hit
  rate 53%, resulting **Sharpe 0.12 before costs**, his verdict "trading costs will
  eat you alive." `grantreed1/Crypto-Order-Flow-Imbalance`: crypto taker version
  "not practically tradeable."
- **OBI / queue-imbalance crypto repos** — `Benson0914/Crypto_Microstructure_Research`:
  gross **+6.18% → net −94.26%** after fees, bootstrapped P(profit)=0%, their words
  "statistical edge does not imply executable alpha after fees." `YV0007`: signal
  evaporates at **2 bps** cost (we face 9).
- **DeepLOB / transformer LOB** (`zcakhaa/DeepLOB`, `LeonardoBerti00/TLOB`): F1
  75–85% on FI-2010 but **no/poor costed backtest**; `chirath-st/lob-deep-learning-
  trading`: best classifier (79%) → **net PnL −7.9** over 37k trades, "the best
  classifier is the worst trader"; DeepLOB at 1 bp cost → **−82% terminal**. The
  accuracy is dominated by the no-move majority class on zero-cost labels.
- **Avellaneda-Stoikov / microprice** (`fedecaccia`, `sstoikov/microprice`,
  `djienne/AVELLANEDA_MARKET_MAKING_FREQTRADE` on HL): all **maker**; reported
  Sharpes assume no-queue/no-latency on rebate-bearing venues. Microprice/OFI are
  **defensive quote-skew** tools, not taker alpha.

## The honest Sharpe verdict (cross-validated)
- **Pure IC signals (OFI/microprice/OBI/DeepLOB) as a taker:** net Sharpe ≤ 0 after
  9 bps round-trip. IC is real (52–54% hit, R² 1–3%) and still loses money. THE TRAP.
- **Maker strategies:** where micro-alpha lives, but need queue priority + rebate
  tiers (−0.3 not +1.5 bps) and sub-second cancel latency. Realistic retail maker
  Sharpe compresses toward <1 / negative from adverse selection (filled mainly when
  wrong). The Sharpe-3–10 figures all assume rebates + colocation.
- **Genuinely taker-viable edge (move > ~9 bps):** NOT LOB classification — it's
  rarer LARGE-move events: **liquidation cascades** (our FUNDFADE sleeve), **funding
  dislocations**, and **cross-venue lead-lag** (another venue moves >9 bps before HL).
  Low turnover, so ~1 s latency is a small fraction of the hold. Plausible net Sharpe
  **1–2.5** — not 3.

## Peer-reviewed anchors (the theory + the costed crypto evidence)
- **Kearns, Kulesza & Nevmyvaka (2010), "Empirical Limitations on HFT
  Profitability"** (arXiv:1007.2593): an *omniscient* trader (sees the future) is a
  hard upper bound on any strategy. Even with **zero fees**, total aggressive-HFT
  profit for ALL US equities in 2008 is **≤ ~$21B at 10 s holding, collapsing to
  ≤ ~$21M at 10 ms**. "Sharpe ratios can be driven arbitrarily high under omniscience
  but real *profits* cannot." Add retail fees + latency → realizable share → ~0.
- **Pindza (2026), Frontiers in Blockchain** (peer-reviewed): crypto microstructure
  TAKER strategies on Binance, leakage-controlled, OOS R² **1.23%**, net Sharpe
  **−10.7 to −52.0** after fees; daily turnover 124–204×; "costs overwhelm any
  statistical edge by orders of magnitude." Directly refutes the wider-spread thesis.
- **Gašperov et al. (Entropy/MDPI, peer-reviewed):** Avellaneda-Stoikov + RL on
  BTC-USD → **all net Sharpes negative** (−0.11 to −0.24), and that's the maker side.
- **MANA Tech decomposition (MSFT, ~2y):** spread capture **+$44.8k/day**, adverse
  selection **−$53.3k/day**, rebate **+$12.3k/day** → naive MM loses ~$8.5k/day and
  the **rebate is the only thing that flips it to breakeven.** Retail gets no rebate.
- **Moallemi & Yuan (2016):** queue position is worth ~half a spread; retail at
  ~100 ms is at the BACK of the FIFO queue → maximal adverse selection ("filled only
  when wrong"), ~0 queue value.
- **Hyperliquid specifics:** maker rebate (−0.001 to −0.003%) is gated on being
  **0.5–3.0% of the entire exchange's maker volume** — institutional only. HLP vault
  Sharpe **2.89 lifetime** (briefly 5.2, **1.65 in early 2026**) is the *privileged
  market-maker entity's* return — touchable by retail only as a passive LP depositor,
  and not durable.

## The latency wall (why the surviving alpha is the part already traded)
- **Budish-Cramton-Shim (2015, QJE):** latency-arb opportunities resolve in single-
  digit ms (ES-SPY median 7 ms by 2011); price correlation at 1 ms is ~0.008. Retail
  at 50–200 ms is **5,000–40,000× too slow** to place in the race.
- **Aquilina-Budish-O'Neill (2022, QJE):** modal race = **5–10 microseconds**; top 6
  firms win >80%. The race is a ~0.5 bp "tax" retail *pays*, not earns.
- **Cont-Kukanov-Stoikov (2014):** OFI explains ~65% of 10 s price variance but its
  autocorrelation **vanishes by ~10 s** — the fast firms capture the 0–1 s predictive
  content; what reaches a 100 ms-late retail order is the already-traded residue.
- **DeepLOB authors' own admission:** their profit sim is **mid-to-mid, zero-cost**,
  and they state aggressive (taker) strategies are "difficult to design profitably"
  and only work "**if we enter passively**" (maker). Prata et al. (2025): "high
  forecasting power does not necessarily correspond to actionable trading signals" —
  predicted moves are sub-tick; a taker pays the spread twice.

## Conclusion for the Sharpe-3 mandate
HFT is where Sharpe 3+ genuinely exists, but it is **execution-gated, not alpha-
gated**: it requires maker queue priority + volume-tier rebates + colocated sub-ms
latency. We proved on real HL fills we lose that game as retail. The taker-viable
residue (event/lead-lag) tops out ~2.5. So the honest ceiling holds: stack the
taker-viable sleeves (our grand stack ~1.5, plus a real liquidation/lead-lag event
sleeve) toward ~2, and recognize 3 OOS net is an infrastructure achievement we
cannot reach. The one remaining untested taker-viable idea — **cross-venue lead-lag**
— is worth a forward synchronized recording; it is itself partly a latency race, so
expect the slow residual (~1–2), not 3.
