# Cadence study — SUMMIT vs ROTATOR at daily / weekly / biweekly / monthly

Same score & sell matrices, sampled at different contribution cadences
(`every` = 1 / 5 / 10 / 21 trading days). Each strategy keeps its native cost
(SUMMIT 5 bps, ROTATOR 10 bps) and concentration (SUMMIT k=2, ROTATOR k=3).
Benchmarks use the SAME cadence, so win-rates are apples-to-apples within each
cadence. 244-window grid. Run: `python research/cadence_study.py`.

## Headline (% of 244 windows beating QQQ-DCA, etc.)

| strategy | cadence | beat QQQ | beat SPY | median vs QQQ | p10 vs QQQ | worst vs QQQ | full mult |
|---|---|---|---|---|---|---|---|
| SUMMIT | daily | 94% | 98% | +28.8% | +2.5% | −11.1% | 19.9× |
| SUMMIT | weekly | 94% | 98% | +29.7% | +2.7% | −11.8% | 19.6× |
| SUMMIT | **biweekly** | 93% | 98% | +28.8% | +3.0% | −10.6% | 20.0× |
| SUMMIT | monthly | 93% | 98% | +30.9% | +1.3% | −11.2% | 18.2× |
| ROTATOR | daily | 58% | 66% | +15.4% | −40.1% | −59.2% | 19.4× |
| ROTATOR | weekly | 64% | 79% | +18.4% | −30.5% | −50.9% | 20.9× |
| ROTATOR | **biweekly** | 65% | 77% | +21.3% | −23.4% | −57.5% | 30.0× |
| ROTATOR | monthly | 60% | 73% | +13.9% | −34.4% | −58.6% | 19.6× |

## Findings

**SUMMIT is cadence-robust.** Win-rate 93-94%, median ~+29%, worst ~−11%,
full multiple ~18-20× — essentially identical at every cadence. Because it
never sells, cadence only changes how finely contributions are spread, so
there is nothing to whipsaw. Run it daily, weekly, biweekly or monthly and you
get the same strategy. The only soft spot is monthly in the 2022 bear (+3% vs
+18% biweekly) — a coarser cadence reacts a touch slower. This robustness is
itself evidence the edge is structural, not tuned.

**ROTATOR's peak is cadence-specific; biweekly is its tuned sweet spot.**
Its headline 30× full multiple is a *biweekly artifact* — at daily / weekly /
monthly it collapses to ~19-21×, i.e. the same terminal wealth as SUMMIT, while
keeping all of ROTATOR's bad tails. Two specifics:

* **The GFC "protection" only exists at biweekly.** ROTATOR's 2007-09 result
  is +13% vs QQQ at biweekly but −16% / −20% / −13% at daily / weekly /
  monthly. The cash switch's benefit in 2008 depends on hitting the right
  rebalance dates — change the cadence and it whipsaws into a loss.
* **Edge concentration is unchanged.** The AI-bull window dominates at every
  cadence (+128% to +253% vs QQQ); strip 2023-26 and little is left.

**ROTATOR's downside is cadence-immune (and bad).** Worst window −51% to −59%
and p10 −23% to −40% at *every* cadence. You cannot fix ROTATOR's tail by
changing how often you rebalance.

**Daily is actively bad for ROTATOR** (58% win, p10 −40%): daily re-ranking
churns positions (sell-below-rank-8 fires constantly) → cost drag + whipsaw.
**Monthly is too slow** (60% win, +13.9% median): it misses rotations. Biweekly
genuinely is the rotation goldilocks — responsive enough to catch leadership,
slow enough to let winners compound.

## Takeaway for the SUMMIT-vs-ROTATOR debate

This sharpens the earlier conclusion. SUMMIT's advantage in *consistency* is
robust to a parameter (cadence) it was never tuned on; ROTATOR's advantage in
*terminal wealth* is fragile to that same parameter. A strategy whose headline
number halves when you change the rebalance frequency from biweekly to weekly
is showing its tuning. SUMMIT's doesn't move.

## Regime vs_qqq by cadence (detail)

ROTATOR:

| regime | daily | weekly | biweekly | monthly |
|---|---|---|---|---|
| GFC 2007-09 | −16% | −20% | **+13%** | −13% |
| recovery 2009-12 | −12% | −9% | −20% | +7% |
| bull 2013-17 | +30% | +46% | +40% | +21% |
| COVID 2020 | −0% | +5% | −4% | −9% |
| bear 2022 | +6% | −3% | −0% | −15% |
| AI bull 2023-26 | +253% | +209% | +206% | +128% |

SUMMIT (for contrast — stable everywhere):

| regime | daily | weekly | biweekly | monthly |
|---|---|---|---|---|
| GFC 2007-09 | +7% | +8% | +9% | +11% |
| recovery 2009-12 | +7% | +6% | +7% | +7% |
| bull 2013-17 | +25% | +24% | +27% | +24% |
| COVID 2020 | +8% | +10% | +14% | +10% |
| bear 2022 | +17% | +17% | +18% | +3% |
| AI bull 2023-26 | +50% | +52% | +48% | +56% |
