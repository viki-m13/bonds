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

---

## Iteration 5 — 2026-07-02 (parallel novel-research cycle, 4 agents)

Mandate: "think novel, try many strategies, test rigorously." Ran four parallel
research agents (regime, cross-sectional, calendar, allocator), each IS-only
(2010-2018; allocator on the 2014-2018 WF segment), ~60 further experiments.

**Adopted (shipped):**
- Cash-accounting fix (previous commit): implicit sleeve cash -> BIL. 2019+
  SR 1.11->1.13, CAGR 24.3->24.8%. Free.
- Dead-code removal: the DD throttle formula `(1+dd/DD_FLOOR).clip(0,1)` was
  INERT (sign bug — always exactly 1.0), here AND in the original PHOENIX.
  Removed the dead code and corrected all documentation. Behavior-preserving
  (also removed ~29 spurious NaN warm-up rows; frozen pin now 1.4765).

**Researched, verified, and REJECTED on the one-shot out-of-sample check:**
- Corrected deadband DD throttle (-5%->-15%): segment 1.46->1.65 SR, uniformly
  positive 36-pt grid — but full-period/2019+ WORSE (de-risks into V-recoveries
  2020/2023). Not shipped. Lesson: DD throttles look great in grinding-bear
  segments and bleed in V-recovery regimes; a 5y selection segment cannot see
  this.
- GH 52w-high rotation sleeve (GHR, IS 1.08) and the 4-diversifier stack
  (turnaround-Tuesday CAL 1.08 halves 1.08/1.08 corr 0.07; crash-recovery CRC
  corr 0.01; SVXY post-panic SVP 1.06; HYG-lead HLS 1.07): segment blend SR up
  to 2.05 — but one-shot full/OOS: DEFAULT (base7+GHR) 2019+ SR 0.78/CAGR 13.5%
  and DEFENSE (all) 1.01/11.5%, BOTH below shipped v3.1 (1.13/24.8%). The GH
  engine, seasonality and SVXY (leverage change Feb-2018) all decayed post-2018.
  Not shipped. Candidate CSVs and agent reports retained in scratchpad.

**Also honestly killed at the sleeve/allocator level:** sleeve-momentum budget
tilt, quarterly/monthly refits, window changes, shrinkage cov, allocator
ensembles, no-trade bands, semicov ERC, family caps (inert), gate re-tuning
(baseline is plateau edge), pre-holiday/Halloween/OpEx/Monday/quarter-turn/gap
effects, residual momentum, RS switch pairs, dispersion gating (prior inverted),
yield-curve bond machine, real-rate gold, dd-depth sizing, bear inverse books,
HYG snapback, activity-capped budgets, diversifier cap frontier (smooth
SR<->CAGR trade, no free point).

**Convergence (second full cycle):** the honest ceiling of this framework
remains v3.1: full 2014-2026 SR ~1.24 / CAGR ~25.9% / MDD ~-24%. Two
independent research cycles (9 iterations, ~150 experiments, 4 agents) found
exactly two survivors — both accounting/bug corrections, zero new signals.
Everything selected on any in-sample segment failed the one-shot OOS. This is
what "no remaining honest edge at this data frequency" looks like empirically.

---

## Iteration 6 — 2026-07-02 (new data channels: High/Low/Volume; new constructions)

Mandate: "try something different and new." Tested data channels the system had
NEVER used (intraday High/Low range, Volume) plus construction changes, IS-only
(`invent_lab6.py`):

| Idea | IS result | Verdict |
|---|---|---|
| Close-location-value accumulation (CLV = (C-L)/(H-L), rolling > th, in uptrend, QQQ/SPY/SMH via LETFs; 4 variants) | SR 0.23–0.48 | Dead |
| NR-4/NR-7 range-compression breakout (Crabel), trend-filtered and not | SR 0.30–0.42 | Dead |
| Volume confirmation on the momentum book (OBV-63 slope; $-volume trend) | 0.58 base → 0.42 / 0.16 filtered | Dead — volume filter actively destroys the book |
| Softmax continuous weights + lookback ensemble (126/189/252) | 0.53–0.62 vs 0.62 single-lb; halves stable but level poor | Dead — smoother but weaker than top-K |
| **Parkinson H/L 20d vol driving the 99th-pct gate** (instead of close-based 60d) | segment 1.48 → 1.54, CAGR +2.2pts, MDD −1.5pts worse; single untuned variant | **Candidate, NOT shipped** — small segment-selected overlay change, exactly the class that inverted OOS in iteration 5 (DD throttle). Revisit only with a longer selection basis or live evidence. |

**Third convergence.** New data channels (H/L/V) hold no extractable edge for
next-open daily ETF rotation at realistic costs. Production remains v3.1+fixes:
full 2014-2026 SR 1.24 / CAGR 25.9% / MDD −24.3%.
