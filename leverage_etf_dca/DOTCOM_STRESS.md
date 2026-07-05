# VOLT through the dot-com bubble (1999-2026) — the real worst-case

Scripts: `scripts/dotcom.py` (dot-com extension), `scripts/ddgate.py` (drawdown-gate test).
QQQ history reaches 1999-03, so TQQQ is reconstructed (3x daily QQQ - fees) through the
2000-2002 crash. Defensive sleeve = CASH (1-mo T-bill, FRED) since bond ETFs start 2005 —
the conservative choice (no bond tailwind).

## The catastrophe vol-targeting avoids
- Dot-com underlying: **QQQ -83%**, and **buy&hold 3x TQQQ -99.9%** (total wipeout; 2% CAGR
  over the entire 1999-2026 — never recovers).
- VOLT de-levered automatically: as TQQQ vol hit 108-208% in 2000-2002, the vol-target
  weight fell to **14-27% TQQQ / 73-86% cash**. It sat out most of the bust by design.

## Results including dot-com
| period | VOLT CAGR | VOLT maxDD | VOLT/QQQ-DCA | vs buy&hold TQQQ |
|---|--:|--:|--:|--:|
| dot-com bust 2000-02 | -23% | -65% | 1.11x | TQQQ -77%/yr |
| dotcom+GFC 2000-09 | -3% | -65% | 1.04x | |
| 2010s bull | 24.5% | -31% | 1.59x | |
| 2020s | 21.8% | -37% | 1.30x | |
| **full 1999-2026** | **15.5%** | **-65%** | **1.89x** | B&H TQQQ 2%/-100% |
| (2006-2026 factsheet) | 22.7% | -47% | 2.41x | *starts after dot-com* |

## The honest correction
- **True worst-case drawdown is -65%, not the -47% on the 2006+ factsheet.** The -47% was an
  artifact of the backtest window starting after the dot-com crash. The -65% came mostly from
  the FIRST leg down (Mar-Sep 2000), before trailing 63-day vol had spiked enough to de-lever
  — the "fast gap-down the vol window can't dodge" tail, now quantified.
- **VOLT still beats QQQ-DCA over the full 1999-2026 span (1.89x) and survived the dot-com
  bust** (1.11x via cash defense + DCA), while buy&hold TQQQ was wiped out. Vol-targeting is
  the reason it survived.
- Risk-off gates (SPY-200MA, drawdown-based) were tested (`riskoff.py`, `ddgate.py`) and all
  HURT full-period return; even the dot-com crash didn't justify one — vol-targeting cut
  exposure to ~20% on its own.

## Bottom line
A vol-targeted leveraged-NASDAQ DCA is survivable through the worst tech crash on record —
but only because the vol scaling de-levers hard. Plan for a ~-65% drawdown, not -47%.
Leverage's worst case is brutal; this strategy bounds it (vs -100% naive) but does not
eliminate it.
