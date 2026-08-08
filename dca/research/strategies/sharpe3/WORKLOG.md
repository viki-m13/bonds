# SHARPE3 worklog — every experiment, honest ledger

Start 2026-08-08 04:03 UTC. Deadline 12:03 UTC.
Sharpe basis: annualized daily NET returns per unit GROSS exposure, costs
10 bps/side + 25 bps/yr borrow, execution lag close t -> trade close t+1.
DEV = 1995–2014. VAL = 2015–2019 (survivors only). TEST = 2020+ (final only).

| # | time | experiment | DEV Sharpe (net) | notes |
|---|------|-----------|------------------|-------|
| 0 | 04:05 | panel built: 9,230 days × 4,590 liquid-ever names, ~900 eligible/mo | — | |
| 1 | 04:15 | battery1 on RAW returns — **VOID: data glitches found** (annualized returns in the 1000s of %; spike-reversal bad prints dominate) | void | led to the cleaning layer |
| 2 | 04:25 | engine: glitch cleaner added (|r|>500% or spike-reversal pairs → NaN, next day too); ELIG rebuilt DV-only ($10M, PIT-pure — dropped adjusted-price filter) | — | all signals now from cleaned returns |
| 3 | 04:45 | battery1 CLEAN: rev5 wk gross 1.06 net -0.48; rev1 daily gross 1.28 net -1.86; rev21 -0.77; mom12-1 -0.57; lowvol -1.36; short-lottery -0.76; volspike -1.78 | best net -0.48 | reversal is real GROSS, dies to turnover. Problem restated: alpha per unit turnover |
| 4 | 04:50 | exp03: leadlag gross 0.86 net -1.05; high-volume premium dead (-2.28 net, -0.27 gross) | — | leadlag also gross-real / net-dead |
