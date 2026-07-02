# PHOENIX v3+ invention log

Standing goal (user directive): keep inventing toward honest OOS Sharpe ≥ 2.0
and CAGR ≥ 25% using ETFs (leveraged/inverse allowed), next-day-open
execution, gross ≤ 100%. Protocol unchanged from PHOENIX_V3.md: all design
decisions on IS (2010-2018) only; walk-forward allocation; OOS looked at only
for locked release candidates; every experiment logged here including dead
ends (the dead ends are the honesty budget).

Baseline to beat (v3, walk-forward, full 2014-2026): SR 1.17, CAGR 20.0%,
MDD -23.3% (2019+: SR 1.09, CAGR 19.0%).

---

## Iteration 1 — 2026-07-02

Tested four idea families (scratchpad `invent_lab1.py`, IS prints only):

| Idea | Result (IS 2010-2018) | Verdict |
|---|---|---|
| Leverage-tier rotation (QQQ trend, vehicle = TQQQ/QLD/QQQ by 21d vol tier; weekly + daily variants; 3 threshold sets) | SR 0.44-0.49, MDD -54..-57% | **Dead end** standalone — single-asset trend too weak in IS; tiering doesn't rescue it. May retry as vehicle-selection inside the cross-sectional book. |
| **Breadth + credit gate** on the ORION-style top-4 momentum book (sector breadth = fraction of 9 XL* sectors above 200dma > 0.5, AND HYG−IEF 63d relative momentum > −2%) | **SR 0.85, CAGR 24.4%, MDD −32.8%** vs VIX-gate reference SR 0.62, MDD −47.1% | **HIT.** +0.23 IS SR and 14pts less drawdown on the biggest risk book. Robust to breadth threshold 0.4-0.5 (0.6 worse). Credit leg carries most of the lift (no-credit variant 0.68). |
| Reversal entries scaled by dip depth (z-magnitude) | SR 0.72 vs 0.70 base, vol −2.6pts | Marginal. Park; maybe fold in with the next REV touch. |
| 5 staggered weekday tranches on the momentum book | SR 0.56 vs 0.62 single-tranche | **Dead end** (costs + diluted Wednesday effect). |

Next iteration plan:
1. Integrate the breadth+credit gate into ORION (replace VIX<30 & HY<7 level
   gate), keeping everything else fixed; check ORION IS.
2. Try the same gate on VANGUARD's participation ladder and HELIOS's macro
   gate (one change at a time).
3. Rebuild the walk-forward blend with the upgraded sleeve(s); if IS blend
   improves ≥ +0.1 SR, lock as v3.1 candidate and take the one-shot OOS look.
4. If v3.1 locks: regenerate production, bump frozen reference, commit.

Open idea backlog: vehicle-tiering inside cross-sectional picks; BTC weekend
momentum → Monday IBIT (blocked: only ~2.5y of IBIT history); options-income
ETF sleeves (JEPI/QYLD 2020+, too short for IS protocol); quarter-turn
seasonality variants; HYG/LQD credit-rotation sleeve at low vol (dilutes CAGR
at gross ≤ 1 — only worth it if SR gain > CAGR loss).
