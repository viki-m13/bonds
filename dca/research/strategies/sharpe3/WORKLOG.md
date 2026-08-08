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
| 5 | 04:55 | exp02 residual reversal (7 configs): best GROSS 1.59 (h5 skip0 f10 prop, all-names book) but net deeply negative — hyper-diversified residual book has tiny vol so fixed cost drag dominates | gross 1.59 | metric that matters = gross alpha per $ traded; need concentration + patience (exp06) |
| 6 | 05:05 | exp05 volume-conditioned reversal: quiet-vol gross 0.56 (WORSE than unconditional), loud continuation -0.43 gross, interaction 0.73 | dead end | LMSW conditioning does not help at weekly horizon on this universe |
| 7 | 05:05 | exp03 pairs-lite: gross 0.62, net -2.23 | dead end as-is | classic Gatev decay confirmed |
| 8 | 05:10 | exp06 crashed (dtype bug), fixed + residual cache added, rerunning; exp07 slow sleeves launched | — | |
| 9 | 05:20 | exp06 patient books: best net +0.13 (zin1.0/zout0.25/lam0.25, gross 1.37, tno 0.131/d). Hysteresis works but edge/trade ~ cost/trade | +0.13 | first non-negative net |
| 10 | 05:20 | exp07 slow sleeves: hi52 -0.29 gross, seasonal -0.08 gross, ltrev 0.28 gross, peerspill 0.20 gross | all ~dead | slow price-only sleeves too weak to matter |
| 11 | 05:30 | exp08 EVENT STUDY (the ceiling measurement): after 1d resid crash z<-2, entry t+2: +4bp d1, +13bp d5, +20bp d10; deeper crashes pay NO more (z<-5: +9bp d5); up-spike short side ~-5bp. Round-trip cost 20bp. **The bounce is in the day we cannot trade (t+1). Per-trade ceiling ~= costs for all lag-2 tail reversal.** | ceiling | family capped at net ~0-0.5; escalate to ML + regime scaling |
