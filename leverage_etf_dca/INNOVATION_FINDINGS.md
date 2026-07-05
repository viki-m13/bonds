# Can we beat QQQ-DCA in *every* regime, including dot-com? — the full experimental arc

Response to: *"dynamically buy ETFs/gold/sectors/leveraged when undersold so we
outperform DCA-into-QQQ substantially in all periods, even dot-com — VOLT is just
recency bias."* The critique is fair, so this was tested exhaustively and honestly.
Scripts: `oversold.py`, `regime.py`, `innovate.py`, `bondrecon.py`, `riskparity.py`.

## A harness bug I found and fixed first (integrity note)
The first cut of `innovate.py` **double-lagged the weight** (a `.shift(1)` weight
applied inside a forward-buy loop that already earns next-month), which de-levered a
month late into every crash and inflated a "reversal" result to a fake 2.48x. Caught
by reconciling against the trusted `strategy.py` harness (they now match to the penny,
corr 1.0 at zero cost). All numbers below are post-fix.

## What was tested (six independent angles) and what each showed

| # | idea | wins the busts? | full-period vs QQQ-DCA | verdict |
|--|--|--|--|--|
| 1 | Oversold rotation (buy the dip, base menu) | yes (dot-com 1.11x, ½ drawdown) | **0.06–0.45×** | loses tech decade |
| 2 | Trend-gated regime switch (lev-tech ON / oversold OFF) | per-era yes | 0.99–1.08×, **fails phase (0.44×@d14)** | timing-luck |
| 3 | Reversal-augmented leverage dial, aggressive (rev10) | — | 2.48× ME but **0.24@d4** | phase-luck, killed |
| 4 | Reversal-augmented leverage dial, moderate (rev6-MA) | neutral | **improves every trade day, no DD cost** | **REAL, modest win** |
| 5 | Leveraged risk parity TQQQ+TMF (+gold) | **yes, strongly** | **0.15–0.49×** | loses tech decade |
| 6 | Momentum-tilted risk parity (+gold) | **yes** (dot-com 1.19, GFC 1.18, 2022 1.04) | **0.49×** | loses tech decade |

### Bond reconstruction (enabling the dot-com test)
Long-Treasury total return reconstructed from FRED DGS20/DGS10 (duration D=15,
convexity C=100, calibrated to real TLT 2005–2018: **0.95 daily corr, 4.2% tracking
error**, conservative on cumulative). Over the dot-com bust (Nasdaq peak→trough,
2000-03→2002-10) reconstructed long Treasury returned **+46%**, and **TMF (3x) +127%**
— while leveraged tech collapsed. This is why risk parity *can* win dot-com.

## The structural theorem (why it keeps failing)
Every construction that **wins the tech crash** (dot-com, GFC, 2022) does so by
**holding non-tech** (bonds/gold/cash/diversified dips). That exact positioning is
**pure drag in the 2010–2021 & 2023–2025 tech bull that dominates the sample.**
QQQ-DCA "wins every period" only because the sample *is* a tech era — the benchmark
is itself the winning recency bet. To beat it everywhere you'd need to know each
period's regime in advance; every attempt to infer the regime from data either
(a) loses the full period (static diversification), or (b) fails phase-robustness
(regime switching = timing luck). **No static blend, and no robust switch, wins both.**

## The honest fork (you must choose a regime bet — you cannot have both)
- **Bet tech keeps leading → VOLT (+ the rev6-MA reversal dial).** ~2× QQQ-DCA over
  the sample, QQQ-magnitude drawdown, **protects** busts (½ the drawdown) but does
  **not** out-return them. The rev6-MA dial is a genuine, phase-robust ~+8% refinement.
- **Want to win the crises and don't want to bet everything on tech → all-weather
  leveraged risk parity (momtilt TQQQ+TMF+gold).** Substantially beats QQQ-DCA in
  dot-com (1.19×), GFC (1.18×), and 2020–26/2022 (1.04×) with far lower drawdowns —
  but gives up the tech bull, ending ~0.49× over the full sample.

## Macro/credit gating of the leverage dial (follow-up)
Tested whether a macro regime signal times the dial better than trailing vol (`macrogate.py`).
- **Yield-curve gates (T10Y3M / T10Y2Y inversion): FAILED** — same structural wall. The
  curve inverted in 2022–23 and stayed inverted through the 2023–25 tech bull, so the gate
  chronically de-levers into the biggest up-move (2020–26 collapses to 0.78–0.99×).
- **Credit-spread gate (HY OAS above its 252d MA, from FRED BAMLH0A0HYM2, data from 2000):
  a REAL, phase-robust DRAWDOWN reducer.** Credit spikes *during* stress and normalizes
  fast, so it de-levers 2008/2020/2022 without dragging the bull. GFC maxDD **−45% → −20%**,
  improving on *every* rebalance day (1.36–1.51× vs baseline 1.10–1.27×); full-history maxDD
  **−62% → −52%**; dot-com-era maxDD −62% → −51%. BUT it does **not** robustly improve
  *return* — the full-period worst-rebalance-day ratio slips (06–26: 1.76→1.51; 99–26:
  2.01→1.88). Verdict: a genuine **defensive overlay** for a drawdown-averse investor
  (optional "VOLT-Defensive" mode), not a free upgrade to VOLT's return.

## Honesty correction for the factsheet
VOLT's shipped headline (2.57× full-period, 2006–26) is the **month-end-flattered
best case.** Across rebalance days it is **1.17×–1.77× (2006–26) / 1.24×–2.06×
(1999–26)** — it beats QQQ-DCA on *every* trade day (so it is not phase-luck), but a
fair central estimate is ~1.3–1.5×, not 2.6×. The factsheet should state the range.
