# Order flow (signed taker volume) and MFIN — honest assessment

Two Oxford-Man / Guelph crypto papers tested against real data + costs.

## MFIN (Multi-Factor Inception, deep net fusing price+volume+Google-search factors)
The posted figure is captioned **"at 0 transaction costs."** That caveat is fatal
for cross-sectional crypto — the most cost-fragile, overfit-prone category. Tested
the gross-vs-net gap directly with an order-flow proxy: continuation 1d is **Sharpe
0.61 GROSS -> ~0 NET** at 4.5 bps (turnover 1.17). At zero cost everything looks
tradeable; it isn't. Plus MFIN needs Google-Trends data (alignment/lookahead
hazard) and deep learning on ~50-80 features (the parent repo already found
ML-on-OHLCV adds nothing here). Not credibly net-tradeable as shown.

## Order flow (signed taker volume): real but modest and cost-fragile
Order flow is a genuine, peer-reviewed cross-sectional predictor — and the SAME
information class as VELOCITY (which is huge gross but maker-only at taker). Tests:
- OHLC order-flow PROXY (close-location-value x volume) on the liquid 111-coin
  panel: predicts CONTINUATION (fading it loses -1.3); at a 5d hold survives taker
  at **Sharpe ~0.3-0.4, positive IS and OOS** (a weak directional sleeve,
  uncorrelated with trend, corr ~0.02).
- REAL signed taker volume (Binance taker-buy-volume field): the accessible liquid
  venue (Binance.com) is geo-blocked; binance.us is too thin (1-14 coins eligible),
  giving a noisy, IS/OOS-inconsistent read (5d: IS -0.14 / OOS +0.68). Can't
  cleanly validate the paper with accessible data.
- At 0 cost the proxy is ~0.6 (matching the paper's gross-figure look); net ~0
  (1d) to ~0.3-0.4 (5d).

## Synthesis across L2GMOM / MFIN / order-flow
These are serious papers reporting Sharpe ~1-2, but on the accessible-data,
net-of-real-cost reality they (a) lean on 0/low cost figures, (b) are
cross-sectional (short-side infeasibility, 500-coin universes), (c) decay post-2022
(visible in their own figures), and (d) need data I can't cleanly access (Google
Trends, liquid-venue taker volume). Order flow is the most real of them and MIGHT
add a modest (~0.3-0.4) third sleeve, but it's too small/noisy on accessible data
to be a confident win. The net-of-cost deployable book stays trend+carry ~1.1-1.3.
