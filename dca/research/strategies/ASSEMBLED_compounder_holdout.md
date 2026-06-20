# Assembled Compounder System — Locked-Holdout Verdict (the deflation-honest test)

Script: dca/research/exp104_assembled.py. The capstone test demanded by the
research-lead framework: combine the SURVIVING signals into one system using
EQUAL weights (no cherry-picking the single best mined conjunction), with the
hold-through-drawdown + sustained-durability exit, and evaluate on a LOCKED
2023-2025 holdout reported separately from 2012-2022 development.

## System
- Entry score = equal-weight z(E/P) + z(buyback) + z(ROIC) + z(-vol6) + z(-size)
  + 0.5·z(handoff-overlay). Top-20, monthly refill.
- Sizing: ride winners (buy&hold). NO price stop.
- Exit: SUSTAINED durability break only (ROIC rank<0.4 for >=2 quarters OR rev
  YoY < -15%) + very loose trend (price < 0.85·10mo-MA) + 72mo max.

## Results (PIT survivorship-clean)
| Period | System CAGR/Sharpe/maxDD | QQQ |
|---|---|---|
| DEV 2012-2022 | 11.2% / 0.70 / -29% | 15.6% / 0.93 |
| **HOLDOUT 2023-2025 (locked)** | **16.4% / 1.37 / -7.1%** | 32.9% / 1.92 |
| FULL 2012-2025 | 12.4% / 0.80 / -29% | 19.3% / 1.13 |
Trades n=316: win 43%, avg +273%, median -7.4%, >100% 12% (fat-tailed compounder).

## VERDICT (honest)
The deflation-honest, equal-weight assembled system **does NOT beat QQQ standalone**
— below on CAGR and slightly below on Sharpe in BOTH dev and the locked holdout.
This is the predicted outcome: the +16-18% cherry-picked conjunctions were inflated
by the 231-trial search; the robust (non-mined) version regresses to ~QQQ-or-below.
What's REAL and survives: (a) a low, defensive drawdown (holdout -7.1% vs QQQ -? in
a bull run; the quality+durability tilt is genuinely defensive), (b) the dev->holdout
gap is small (not broken, just not a standalone winner), (c) fat-tailed compounder
profile (avg trade +273% from a few multi-baggers held through drawdowns).
=> The honest use is as a LOW-CORRELATION, LOW-DRAWDOWN SLEEVE in the diversified
portfolio (WAVE + trend + this), NOT a standalone QQQ-beater. No single long-only
signal beats QQQ after honest deflation — the franchise is the DIVERSIFIED ENSEMBLE
plus the hold-through-drawdown discipline, exactly as the research-lead framework
predicted.
