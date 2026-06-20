# Network-fragility exposure overlay (absorption ratio) on trend+carry

HL era, real funding + 4.5bps taker, IS=first60/OOS=last40. Absorption ratio = top-20% eigenvalue share of the trailing 60d crypto correlation network (Kritzman-Li systemic risk). Overlay cuts gross exposure when the network couples/rises.

Absorption ratio: mean 86%, range [72%, 93%] — crypto is highly coupled (confirms single-factor).

| book | Sharpe | IS | OOS | ann | maxDD | Calmar |
|---|---|---|---|---|---|---|
| trend+carry (base) | **+0.86** | +0.91 | +0.79 | +11.3% | -12.4% | 0.91 |
| + network-fragility overlay | **+0.69** | +0.89 | +0.34 | +9.0% | -12.7% | 0.71 |

## Verdict

- Overlay Sharpe +0.69 vs base +0.86; maxDD -12.7% vs -12.4%; Calmar 0.71 vs 0.91. The overlay does NOT improve the book on this sample — absorption-ratio timing didn't add value here.

## All three network constructions tested (HL era)

| construction | Sharpe | maxDD |
|---|---|---|
| trend+carry base (50/50) | +0.86 | -12.4% |
| + fragility EXPOSURE overlay (cut gross when AR rises) | +0.69 | -12.7% |
| + AR REGIME-ROTATION (trend when coupled / carry when dispersed) | +0.87 | -12.8% |
| + DISPERSION rotation | +0.81 | -11.5% |

None beats the simple base. **Why (the structural reason):** crypto's correlation
network is *degenerate* — one factor (BTC) explains ~50% of variance, coupling sits
at 72–93% (AR mean 86%) at all times. So there is (a) no cross-sectional GRAPH
ALPHA to mine (single factor) and (b) too little variation in network state to TIME
risk. The "learn the financial network" edge (L2GMOM, Sharpe 1.74) fundamentally
needs a multi-factor, structurally-diverse universe (the paper's ~50 futures across
commodities/bonds/FX/equities). Crypto isn't that, and HL can't assemble it (the
non-crypto HIP-3 markets are efficient/trendless). The honest deployable book stays
trend+carry ~1.1; the network idea adds nothing here.
