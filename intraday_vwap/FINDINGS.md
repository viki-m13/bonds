# VWAP Trend Trading (SSRN 4631351) — replication on repo intraday data + honest improvement

Rules replicated: at the first candle close after the open, go long if close > session
VWAP else short; flip whenever a candle CLOSES on the other side of VWAP; flat at 16:00;
100% of capital; $0.0005/share commission. Paper: 1-min QQQ 2018–2023, no slippage,
claims 671% / Sharpe 2.1 / DD −9.4%.

**Our data:** 5-minute RTH bars (the repo's finest), 7 ETFs (QQQ SPY DIA IWM XLF GLD TLT),
2016-01 → 2026-04, vendor per-bar VWAP (session VWAP = cum(vwap·vol)/cum(vol)).
Single stocks in the repo have only *daily* bars — the intraday strategy cannot be
tested on them with current data. Engine: `scripts/engine.py`.

## 1) Replication (paper window, paper costs)
QQQ 2018–23: **3.13× vs B&H 2.70×**, Sharpe 1.13 vs 0.79, maxDD −21% vs −35%, 11,204
trades, 24% win rate, W/L 3.5 (paper: 17%, 5.7). Directionally confirmed; magnitude far
below the paper — 5-min closes catch trend turns later than 1-min. The character matches:
low win rate, big winners, crisis-alpha profile.

## 2) The four honest problems (before improvement)
- **Slippage kills the base rules.** ~7.5 trades/day. At 0.5c/share (half the QQQ penny
  spread): full-sample 4.01× → **1.63×**. At 1c: **0.66×**. The paper's zero-slippage
  assumption does enormous work (their 1-min version trades even more).
- **Cross-section (mandatory leave-one-out): QQQ ranks 1 of 7.** Base rules lose money
  outright on IWM (0.44×), XLF (0.22×), TLT (0.79×), GLD (0.96×). The "mechanism" is not
  general — same selection-bias signature that killed PULSE, this time in the paper's
  own ticker choice.
- **Window dependence.** 2016–17: negative. 2024–26 (true OOS past the paper): positive
  (Sharpe 0.85) but **below B&H** in a calm bull.
- **Where the money comes from:** 2018 +39.5% (B&H −0.1%), 2020 +10.9%, 2022 **+45.5%
  (B&H −32.4%)** — vs ~0% in 2017/2019/2023. It is a **crisis-alpha engine**, ~zero
  correlation to B&H (−0.04), not a B&H replacement.

## 3) The improvement that survives: a no-flip band + the complement framing
**Band filter**: only flip when the close is > k bps away from VWAP (inside the band,
hold the current side). Directly attacks whipsaw = the cost problem.

QQQ full 2016–26, commission + 0.5c/share slippage:
| variant | mult | CAGR | Sharpe | maxDD | trades |
|---|--:|--:|--:|--:|--:|
| base rules | 1.63× | 4.9% | 0.37 | −33% | 19,393 |
| **band 20bps** | **4.11×** | **14.8%** | **0.97** | **−16.5%** | 5,145 |
| band 25bps | 3.90× | 14.2% | 0.96 | −12.1% | 4,257 |
| B&H QQQ | 6.39× | ~20% | 0.92 | −35.0% | — |

- **Band shape is a smooth plateau** (15→40bps: 2.73/4.11/3.90/3.13/2.07) — not a
  knife-edge parameter. 20–25bps ≈ several spreads, economically sensible.
- **Cost-robust:** at a punitive 1c/share slippage band-20 still 3.28× (Sharpe 0.83).
- Long/short split: longs carry the full sample (1.75× vs shorts 0.93×) but shorts
  supply the 2018/2020/2022 crisis alpha. Vol-gating did NOT help (tested, rejected).
- Paper window with band-20: 3.06× (Sharpe 1.17); 2024–26 OOS: 1.28× (Sharpe 0.85,
  still below B&H's 1.15 in that calm stretch).

**The right way to hold it — 50/50 with B&H QQQ (daily rebalance), corr −0.04:**
**5.66× · CAGR 18.4% · Sharpe 1.34 · maxDD −17.6%** — vs B&H 6.39× / 0.92 / −35.0%.
Nearly the B&H return at half the drawdown, because the strategy pays out exactly in
B&H's worst years.

## 4) Honest verdict
- The paper's headline (43%/yr, Sharpe 2.1, −9% DD) does **not** survive realistic
  granularity + costs. Base rules at 0.5c slippage are barely mid-single-digit CAGR.
- What DOES survive: a **band-filtered QQQ version** with genuine crisis-alpha character
  and ~zero correlation to buy-and-hold; best used as a **complement** (50/50 → Sharpe
  1.34, DD halved), not a replacement.
- **QQQ-specificity is the open risk** (rank 1/7 cross-sectionally; SPY weakly positive,
  everything else negative). Partial mitigation: the paper fixed QQQ ex ante (2023) and
  2024–26 remains profitable OOS — but by this repo's own standards, treat part of the
  QQQ edge as selection until it survives more assets or more OOS time.
- Execution reality: ~2 trades/day at band-20, must trade at bar closes with limit-ish
  execution; taxes are all short-term; do not run this in a taxable account and expect
  the backtest.
