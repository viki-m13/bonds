# HANDOFF-GAP — Ownership-Rotation Compounder Capture (novel franchise)

> Status: VALIDATED ENTRY-ALPHA + HOLD-THROUGH-DRAWDOWN; durability-exit needs
> refinement. The most novel, economically-grounded edge in the program.
> Script: dca/research/exp100_handoff.py. PIT survivorship-clean, 2012-2024.

## Thesis (the survival mechanism — why it isn't arbitraged)
Companies PAST peak growth-RATE but with stable high ROIC fall into an ownership
"handoff gap": growth funds MUST sell (mandate: decelerating growth), quality funds
haven't arrived yet (not-yet-re-rated). The natural arbitrageurs are STRUCTURALLY
ABSENT during the window, so the multi-year compounding is mispriced cheap. This is
an investor-base-constraint edge (persists post-publication: funds can't change
mandates) — NOT a competed-away factor. NVDA/LLY/long compounders all spent years
"too expensive for value, too slow for growth" here.

## Signals (computable from price + SEC fundamentals)
- Decelerating growth: rev YoY > 0 AND YoY < YoY a year ago (past peak rate).
- Stable high ROIC: operating-income/(assets−cash−STinv) top-40% AND min over
  trailing 6q > 5% (durable, not a one-off).
- Ownership rotation (growth leaving): 12-mo momentum was strong 6m ago (>10%) but
  recent 6-mo momentum now flat/soft (<5%) — relative-strength decay.
- Not-yet-re-rated: P/S ≤ its own trailing-18mo median (multiple hasn't expanded).
- Liquidity ≥ $3.

## Validated results (PIT survivorship-clean, dead names included)
ENTRY-ALPHA (cross-sectional, ~22 names/mo):
| Cohort | fwd-12m mean | median | hit | >200% | fwd-24m |
|---|--:|--:|--:|--:|--:|
| **HANDOFF-GAP** | **+14.4%** | **+11.2%** | **64%** | 0.3% | **+30.9%** |
| accel-growth+quality (mainstream) | +11.8% | +6.9% | 59% | 0.5% | +24.5% |
| universe | +8.7% | +4.5% | 57% | 1.2% | +18.3% |
Beats the universe AND the mainstream growth cohort; high hit/median, low tail =
"mispriced length of compounding," not lottery.

EXIT DECOMPOSITION (same entries, different exits) — THE EXIT IS THE ALPHA:
| Exit | win% | avg trade | >100% | port maxDD |
|---|--:|--:|--:|--:|
| price-stop −20% (naive) | 45% | +22.6% | 10.9% | −37.5% |
| **fixed / hold-through** | **80%** | **+107.5%** | **27.5%** | −25.8% |
| durability-break v1 (too twitchy) | 59% | +27.3% | 9.6% | −21.4% |
Not cutting on price ⇒ win 45→80%, avg trade +23→+108%, multibaggers 11→27%, AND
lower drawdown. Winners rode a MEDIAN −42% drawdown. "Everyone has the exit backwards."

## Honest caveats / next iterations
- Portfolio CAGR (~12%) trails QQQ (19.9%) in ABSOLUTE over 2012-24 (mega-cap-tech
  regime; equal-weight ~20 mid-cap compounders, monthly dilution). The EDGE is the
  entry-alpha vs universe (+5.7pp) and the trade quality, not raw beta.
- **Durability-exit v1 HURT (8.8%)** — exiting on a one-quarter ROIC-rank dip
  amputates compounders in temporary softness. FIX: exit only on SUSTAINED (3+ qtr)
  severe ROIC/margin deterioration. This is the highest-value next build.
- Next: concentrate + let-winners-ride (WAVE mechanics) on handoff entries; refine
  durability exit; add the 13F breadth ownership-rotation signal explicitly.

## Trajectory-shape search (exp102) — Tier-1 #1, mostly null (as predicted)
Tested per-firm trajectory transforms of 5 fundamentals (ROIC, op-margin, gross-
margin, asset-turnover, rev-growth) × {level, slope, accel(2nd-deriv), persistence}:
- LEVELS strong but known (ROIC IC 0.110, op-margin 0.097 = quality factor).
- SLOPE & ACCELERATION ≈ DEAD: ROIC-slope +0.001, accel +0.011, rev-growth-slope
  −0.003 — all |IC|<0.03 (noise after 20-trial deflation). The per-firm derivative
  of fundamentals adds ~nothing cross-sectionally.
- PERSISTENCE (consecutive rising quarters) = modest survivor: IC 0.02-0.03 but
  L/S Sharpe ~0.8 consistently across all 5 fundamentals — a real weak signal.
Confirms the research-lead prior: most trajectory marginals die; persistence
survives weakly; the genuine novelty is in INTERACTIONS (the handoff-gap, a
4-way conjunction, gave the best entry-alpha +14.4% of anything tested).

## Interaction generator (exp103) — Tier-1 #2, the seam confirmed
Searched 231 conjunctions (11 base signals: handoff parts + value/quality/low-vol/
buyback/small-cap/persistence; singles + all 2-way + all 3-way), ranked by entry-
alpha (fwd-12m mean vs universe +8.7%):
- BEST 3-way: hi_value+buyback+small_cap +18.2% (hit 63%, 76/mo);
  stable_ROIC+low_vol+small_cap +17.3% (hit 67%); low_vol+buyback+hi_quality
  +15.9% (hit 72%); stable_ROIC+low_vol+buyback +16.2% (hit 72%).
- Conjunctions BEAT marginals AND the handoff 4-way (+14.4%). 3-way > 2-way > single.
- Recurring ingredients in the top-18: BUYBACK (almost all), SMALL-CAP, VALUE,
  QUALITY/stable-ROIC, LOW-VOL. Economically coherent (not random feature salad)
  => the deflation defense vs 231 trials: signal clusters around the durable
  survivors combined, not lone flukes.
CAVEATS: entry-alpha only (single as-of cross-sectional test, not full portfolio
w/ exit); small-cap-heavy (capacity-limited; absolute portfolio CAGR capped by
small-cap beta ~12% like handoff). The ENTRY edge is strong; concentration +
hold-through-drawdown + durability-exit convert it (per exp100). Confirms the
research-lead prediction: the survivors are INTERACTIONS, and exit>entry.
