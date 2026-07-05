# The all-era loop — can a monthly-DCA stock strategy beat QQQ-DCA in EVERY period?

*A multi-agent search (10× cron loop, ~14 subagents) targeting the strongest possible
version of the mandate: "buy stocks every month, significantly outperform DCA-into-QQQ
in **all** periods." Each candidate was graded on the same bar: era-sliced ratio vs
QQQ-DCA (2000-02, 2003-09, 2010-14, 2015-19, 2020-26), beating a random-in-pond null's
MAX (not just mean), 20 bps/side, min-30-day hold, survivorship-clean PIT data. Scripts:
`scripts/loop*.py`. Running log: this file.*

## Headline

**No signal tested clears the all-era bar, and the literal bar is partly ill-posed.**
Two structural walls make "significantly beat QQQ-DCA in every era" unachievable with
retail-accessible data:

1. **The 2000-02 era is not a real test.** QQQ (Nasdaq-100) fell ~80% in the dot-com
   bust, so a *random* pick from the broad stock pond beats QQQ-DCA there (null mean
   **1.70×**). "Beating QQQ" in 2000-02 measures nothing.
2. **The 2003-09 era is a genuine wall, and it is provably unbeatable by any signal we
   can build.** Every robust price-only rule loses to QQQ-DCA there (median 0.56×, best
   0.98×); SEC fundamentals don't start until 2012 so no fundamental signal is even
   *testable* pre-2012; and the two non-price signals available back to 2005 (8-K events,
   10-K text) are dead / sub-threshold.

So the honest deliverable is the **modern-era achievable frontier** (2010-14, 2015-19,
2020-26 — where beating QQQ is hard but well-posed), documented in §6/§7 of `FINDINGS.md`.

## Candidate-by-candidate (all graded on the era bar)

| # | Candidate | Result | Why it fails the all-era bar |
|---|---|---|---|
| 1 | Opportunistic vs routine insider (CMP) | real but small | opp>routine confirmed pre-2020; decays post-2020; ~QQQ-level, never >1.1 |
| 2 | Twin-momentum (price∧fundamental agree) | keeper component | agreement beats either leg, but composite dies 2003-09 (0.24×, −89% DD) |
| 3 | Expectancy/fat-tail (max payoff asymmetry) | **falsified** | picking high-vol names makes the loss-cut whipsaw → 0.01× full-period |
| 4 | 2003-09 feasibility probe (16 price rules) | **decisive** | no robust price rule beats QQQ in 2003-09; only a low-price junk artifact does |
| 5 | Defended composite (keepers + Faber regime) | tracks QQQ | defense cushions 2003-09 (−54% DD) but converges to QQQ (0.8-1.0×), no edge |
| 6 | 8-K negative-event veto (2005+) | **dead** | veto is a net drag in 3/4 eras; below null everywhere (confirms loop7) |
| 7 | Lazy-Prices 10-K text change (2005+) | **sub-threshold, sign-unstable** | IC ~0.012 flat, 2010-14 sign flips *negative*; tail precision 0.074 < 0.10 base rate |

Candidate 7 was the crux: the *only* signal both **factor-orthogonal** (passes the §6c
spec — |corr|<0.06 vs momentum/size/vol/quality) **and** available across the 2003-09
wall. Built a live EDGAR pipeline (`scripts/loop_lazyprices_build.py`, streaming
consecutive-10-K cosine similarity, 209→437 firms). Verdict at 209 firms: detectable but
weak, and the 2010-14 sign reversal is disqualifying — more coverage sharpens the same
weak, sign-unstable estimate rather than revealing an edge.

## The two components worth keeping (not QQQ-beaters, but real)

* **Twin-momentum agreement** — where price momentum and fundamental momentum confirm
  each other, drift is more durable than either alone (3-5× price-mom-only in-era). Not
  all-era, but a legitimate selection ingredient in the modern eras.
* **Opportunistic-insider tilt** — the Cohen-Malloy-Pomorski routine/opportunistic split
  is real pre-2020; a small diversifying tilt, not a core.

## Why this keeps happening (the mechanism, restated)

QQQ is a cap-weighted machine that already holds the winners in proportion to how much
they've won. Any equal-weight selection holds *less* of the winning beta; any
concentration within a pond beats QQQ only while leadership persists and gives it back in
rotation. Genuine cross-sectional skill (the honest ML's IC ≈ 0.06-0.075) predicts
*absolute* returns in the small/mid pond that structurally lagged QQQ. The required-skill
curve (loop4, exp): a signal with rank-IC ≈ 0.05 **and idiosyncratic (factor-orthogonal)
errors** would beat QQQ in every era — but no signal in reach has both. Price transforms
have the IC but factor-structured (tail-blind) errors; the one orthogonal signal
(Lazy-Prices) is sub-threshold and sign-unstable.

## Standing conclusion

Consistent with the repo's 119 prior experiments and the external methods survey
(`METHODS_SURVEY.md`): **there is no monthly-DCA stock strategy that significantly and
durably beats QQQ-DCA in all eras with retail-accessible data.** The deployable answers
remain QQQ-DCA (expected-value optimum) and ASCENT (the priced leadership bet, modern-era
outperformer). The modern-era achievable frontier is quantified in the next section.
