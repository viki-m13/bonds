# Does a SPY risk-off cash gate help VOLT? — tested, mostly no

Question: add a market-regime gate (SPY below its 200-day MA -> go to cash) to cut
VOLT's drawdown. Script: `scripts/riskoff.py`. Causal (gate read at prior month-end).

| variant | 2006-09 | 2010-14 | 2015-19 | 2020-26 | full | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| VOLT base (no gate) | 1.20 | 1.50 | 1.28 | 1.33 | **2.41x** | 22.7% | 0.84 | -47% |
| + SPY<200MA -> all cash (BIL) | 1.47 | 1.44 | 0.99 | 0.84 | 1.13x | 19.0% | 0.81 | -40% |
| + SPY<200MA -> half TQQQ | 1.39 | 1.49 | 1.18 | 1.07 | 2.00x | 22.5% | **0.89** | -46% |
| + SPY<200MA & mom<0 -> cash | 1.60 | 1.38 | 0.97 | 0.84 | 1.20x | 19.8% | 0.83 | -45% |

## Findings
- **Full risk-off-to-cash is a net negative.** It trims drawdown modestly (-47%->-40%)
  but costs large return (CAGR 22.7%->19.0%) and makes VOLT LOSE to QQQ-DCA in the modern
  eras (0.99x 2015-19, 0.84x 2020-26). It whipsaws: SPY crosses below its 200dMA *after*
  the drop, you sell into weakness, then the sharp *leveraged* recovery (2020 V, 2018/2022
  chop) fires while you sit in cash. A 3x strategy punishes bad timing 3x.
- **Two reasons it underwhelms:** (1) vol-targeting already de-levers on vol spikes
  (most crashes), so a slower SPY gate is largely redundant; (2) SPY-200MA lags — by the
  time it triggers, most of the drawdown is already taken, so it locks in losses.
- **Only useful variant:** *partial* de-risk (halve TQQQ into GLD/TLT, not cash) when SPY
  is risk-off -> Sharpe 0.84->0.89, still beats QQQ every era at ~same CAGR, DD ~1pt better.

## Takeaway
With leverage you cannot have QQQ-beating return AND small drawdown — the drawdown *is*
the cost of the leverage, and vol-targeting already holds it at QQQ's own level (-47% vs
-50%). The clean drawdown lever is not a trend gate (whipsaw) but turning the dial down:
lower vol target (vt20 vs vt30) or 2x QLD vs 3x TQQQ — both cut drawdown and return
smoothly with no timing risk.
