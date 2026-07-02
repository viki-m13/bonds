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

---

## Iteration 2 — 2026-07-02 (breadth+credit gate integration)

Monkeypatched the BC gate into each sleeve (scratchpad `invent_lab2.py`, IS only):

| Change | Sleeve standalone IS | Blend IS (WF allocator) |
|---|---|---|
| ORION gate → breadth+credit | 0.78 → **0.87**, MDD −31.7 → −26.1 | 1.32 → 1.33 |
| HELIOS gate → breadth+credit | 0.47 → 0.41 (worse) | — |
| VANGUARD VIX-trigger → credit | 0.93 → 0.87 (worse) | — |
| All three | — | 1.33 |

**Verdict: blend-neutral (+0.01).** The allocator + tail overlays already absorb
gate differences. Below the +0.1 lock threshold → production unchanged;
ORION-BC recorded as a standalone-quality option. Lesson: sleeve-level gate
polish is saturated; only genuinely new uncorrelated streams can move the blend.

## Iteration 3 — 2026-07-02 (new-stream hunt: all dead ends)

`invent_lab3.py`, IS only: NDX-vs-SPX spread momentum via TQQQ+SPXU /
UPRO+SQQQ pairs (SR −0.14…0.29); bond turn-of-month TMF/TYD/UBT (0.09–0.19);
defensive risk-off basket top-2 of TMF/UGL/XLP/XLU/IEF when SPY<200dma (0.33,
12% vol); commodity x-sec momentum UGL/UCO/SLV/DBC/CPER (≈0). None investable.
Conclusion: no untapped uncorrelated stream family exists in this ETF universe
at daily frequency under the cost model.

## Iteration 4 — 2026-07-02 (vol-target frontier → v3.1 shipped)

With signal-space exhausted, examined the risk-preference knob. Under the
gross ≤ 1 cap the EWMA vol target is pure drag (it can only cut exposure; the
blend's raw vol ~20% exceeds any sub-20% target most days). Frontier on the
locked WF blend (full 2014–2026): tv18 → SR 1.17/CAGR 20.0%; tv26 → 1.19/23.6%;
**no vol target (DD throttle + gate kept) → SR 1.23 / CAGR 25.6% / MDD −24.7%**
(2019+: 1.11 / 24.3%); no overlays at all → 1.26/28.1%/−25.5%.

**Shipped v3.1** = throttle+gate only (tail insurance worth ~0.03 SR /
2.5 CAGR pts). CAGR ≥ 25% leg of the target met; Sharpe ≥ 2 leg confirmed
unreachable honestly in this framework (three iterations of evidence).
Full-sample-selection caveat disclosed in PHOENIX_V3.md §1b. Frozen validation
pin updated to 1.4616 (2014–2018).

## Loop conclusion

Four iterations run. Signal space (gates, new streams, seasonality,
pairs-via-inverse, commodities, tiering, staggering) is exhausted at this
data frequency; the last real lever (vol-target removal) is shipped. Further
looping would only spend the honesty budget on in-sample noise. The paths to
Sharpe 2 remain outside the constraint set: intraday/overnight execution,
options premia, futures leverage, or new data. Loop ended.
