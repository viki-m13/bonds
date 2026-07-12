# Implied-vol dial for VOLT — tested (cross-pollination from SUMMIT), verdict: log, don't ship

Motivation: the SUMMIT market-neutral project found inverse-**implied**-vol (VIX3M)
sizing beat realized-vol sizing. Does the same swap improve VOLT's dial?
Script: `scripts/impliedvol.py` (fetches CBOE VIX/VIX3M; PIT: month-end values,
trailing-252d QQQ/SPY ratio to map SPX implied -> QQQ; ^2 mirrors shipped accel_k).

| variant (all keep rev-dial) | 2006-09 | 2010-14 | 2015-19 | 2020-26 | full 2006-26 | d10 phase | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| BASE (shipped: realized 63d + 20d accel) | 1.28 | 1.34 | 1.27 | 1.18 | **2.67** | 1.67 | 24.1% | 0.94 | −45% |
| IV (pure implied level) | 1.16 | 1.22 | 1.26 | 1.14 | 1.85 | 1.89 | 20.6% | 0.89 | −46% |
| MAXRV_IV (worse-of-both) | 1.23 | 1.16 | 1.24 | 1.11 | 1.79 | 1.35 | 21.2% | 0.95 | −43% |
| TERM (VIX/VIX3M backwardation as the accel trigger) | 1.27 | **1.54** | **1.33** | **1.27** | **2.82** | **2.00** | 24.2% | 0.87 | −47% |

- **Implied *level* does NOT transfer** — unlike SUMMIT. VOLT's asset (TQQQ) has a
  directly observable realized vol; SPX implied + a ratio proxy is a noisier estimate
  of the same thing, and the risk premium embedded in VIX makes the dial systematically
  too small (IV CAGR 20.6% vs 24.1%). MAXRV_IV is over-conservative — reject both.
- **Term-structure trigger (TERM) is promising but NOT Pareto:** improves the full-period
  ratio at both tested phases (2.82/2.00 vs 2.67/1.67) and every 2010s era, but costs a
  little Sharpe (0.87 vs 0.94) and drawdown (−47% vs −45%), and it cannot be tested
  through dot-com (VIX3M exists only 2009+; pre-2009 it falls back to the shipped accel).
  By this project's shipping bar (the accel overlay was a strict Pareto win), TERM stays
  a **logged candidate**: revisit if someone reconstructs a pre-2009 term-structure proxy
  or accepts the risk trade for the extra return.
