# Leveraged Risk-Parity improvement to VOLT — findings

Script: `scripts/improve_riskparity.py` (imports `strategy.py` harness). Sleeves are the
reconstructed leveraged ETFs in `_etf_panel.pkl`: TQQQ (3x NASDAQ), TMF (3x 20y Treasury),
UGL (2x gold), UPRO (3x S&P), SOXL (3x semis). TLT/GLD underlyings start 2005 → book runs
2006–2026. Each sleeve sized by inverse-vol (naive risk parity) or equal-risk-budget, scaled
to a portfolio-vol target with an ex-ante trailing-covariance estimate, rest in CASH (BIL/SHY).
Monthly rebalance, 10bps/side, all weights read at PRIOR month-end (shift 1, no look-ahead).

Sleeve monthly-return correlations 2006–26 confirm the thesis' premise:
TQQQ–TMF −0.07, TQQQ–UGL +0.05, TMF–UGL +0.24 (genuinely uncorrelated);
UPRO/SOXL are +0.92/+0.84 to TQQQ (redundant equity beta).

## Era table — lump-sum $1 risk (full 2006–2026)

| strategy                    | CAGR | RealVol | Sharpe | maxDD | worst12 | mult |
|-----------------------------|-----:|--------:|-------:|------:|--------:|-----:|
| **BASE VOLT vt30 TQQQ\|GT** | 22.7% | 30% | **0.84** | −47% | −47% | 67x |
| RP3 TQQQ+TMF+UGL 12%        | 10.7% | 14% | 0.82 | −27% | −24% | 8x  |
| RP3 TQQQ+TMF+UGL 15%        | 12.8% | 17% | 0.80 | −33% | −30% | 12x |
| RP3 TQQQ+TMF+UGL 20%        | 15.0% | 21% | 0.78 | −42% | −38% | 18x |
| RP3 TQQQ+TMF+UGL 25%        | 16.3% | 23% | 0.77 | −48% | −44% | 22x |
| RP3 TQQQ+TMF+UGL 30% (matched) | 21.0% | 31% | 0.77 | **−57%** | −52% | 51x |
| ERB3 (equal-risk-budget) 15% | 12.6% | 17% | 0.80 | −33% | −31% | 12x |
| RP2 TQQQ+TMF (Hedgefundie) 15% | 11.6% | 16% | 0.75 | −35% | −34% | 9x |
| RP2 TQQQ+TMF 20%           | 14.3% | 21% | 0.74 | −45% | −43% | 16x |
| RP4 +UPRO 15%              | 13.0% | 16% | 0.83 | −33% | −33% | 12x |
| RP5 +UPRO+SOXL 15%         | 13.3% | 16% | 0.86 | −32% | −32% | 13x |
| QQQ buy&hold               | 15.9% | 18% | 0.90 | −50% | −43% | 21x |
| TQQQ buy&hold              | 25.8% | 57% | 0.70 | −94% | −89% | 112x |

## DCA final-wealth ratio vs BASE VOLT (>1 beats base), era-sliced

| construction              | 06-09 | 10-14 | 15-19 | 20-26 | 10-26 | full |
|---------------------------|------:|------:|------:|------:|------:|-----:|
| RP3 TQQQ+TMF+UGL 15%      | 1.04 | 0.50 | 0.76 | 0.62 | 0.25 | 0.20 |
| RP3 TQQQ+TMF+UGL 20%      | 1.12 | 0.53 | 0.79 | 0.69 | 0.30 | 0.26 |
| RP3 TQQQ+TMF+UGL 30%      | 1.29 | 0.61 | 0.95 | 0.87 | 0.51 | 0.57 |
| RP2 TQQQ+TMF 20%          | 0.92 | 0.82 | 0.96 | 0.45 | 0.28 | 0.26 |
| RP5 +UPRO+SOXL 15%        | 0.83 | 0.59 | 0.84 | 0.71 | 0.37 | 0.27 |

(vs QQQ-DCA, base VOLT = 1.20 / 1.50 / 1.28 / 1.33 / 1.89 / 2.41.)

## Critical tests

**(a) Lift Sharpe + cut DD + keep return edge?** NO on Sharpe, NO on return.
At MATCHED risk (RP3 @30% vol = base's 30%): CAGR 21.0% vs 22.7%, Sharpe 0.77 vs 0.84,
maxDD **−57% vs −47%** — strictly dominated. The diversified book *looks* safer only because
it is de-levered; equalize the vol and it is worse on all three. Best diversified Sharpe is
~0.80 (3-sleeve) — below base 0.84. RP5's 0.86 comes only from re-adding equity leverage
(UPRO+SOXL), i.e. re-concentrating, not from diversification.

**(b) 2022 — does gold save it?** YES, at low vol targets. 2022 returns:
BASE VOLT −45.9%, RP2 TQQQ+TMF −33.7%, **RP3 (+gold) −27.6%**. Sleeves: TQQQ −79%, TMF −73%
(bonds AND stocks crashed together), **UGL only −7.5%**. Gold is the one genuine diversifier
when the stock–bond correlation breaks. BUT the vol-matched 30% version still printed −57%
because leverage chasing the vol target amplified the correlation break faster than gold cushioned it.

**(c) Phase robustness** — STRONG. Rebalance-day offsets ±8d: CAGR 12.7–13.7%, Sharpe 0.77–0.87,
maxDD −32 to −36%. Not a phase artifact.

**(d) Just the 2010s bond bull?** NO — the OPPOSITE. RP beats base only in **2006-09** (GFC:
+4 to +29% over base) and cushions 2022. Through the actual falling-rate bond bull 2010–2021 it
badly LAGGED base (0.50x in 2010-14, 0.76x in 2015-19) because 3x NASDAQ compounded far faster
than TMF's bond gains. The diversification's value is CRISIS insurance (2008 flight-to-quality,
2022 gold), not a rates tailwind. TMF's bull did not produce the edge.

## Verdict

Leveraged risk parity does NOT improve VOLT on a risk-adjusted basis. It is a lower-octane
alternative: ~13–15% CAGR at −33% maxDD (Sharpe ~0.80) versus base VOLT's 22.7% at −47%
(Sharpe 0.84). At MATCHED risk it loses on CAGR, Sharpe and drawdown. Reason: TQQQ (3x NASDAQ)
was the single highest-Sharpe sleeve in-sample; diluting it with lower-return diversifiers
(TMF especially) cut return more than risk, and vol-targeting an already-diversified book levers
INTO correlation breaks (2022) rather than de-risking the way concentrated single-asset
vol-targeting does.

The one real, robust finding worth keeping: the **UGL (gold) sleeve** is genuine stock–bond-crash
insurance (−7.5% in 2022 vs −73%/−79% for bonds/stocks). If VOLT's defense sleeve is to be
improved, add gold — not full leveraged risk parity.

**Single best diversified construction (if required):** RP3 TQQQ+TMF+UGL, ~15–20% vol target,
inverse-vol (ERB ≈ RP, equally robust). Gold is essential; SOXL/UPRO add only redundant equity
beta. Honest risk at 15%: CAGR 12.8%, Sharpe 0.80, maxDD −33%, worst-12m −30%, phase-robust,
2022 −27.6%. It is a de-risking, not a return or Sharpe upgrade over base VOLT.
