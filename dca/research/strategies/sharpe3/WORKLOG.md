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
| 12 | 05:50 | exp09 netted composite: EMA+patience push net to ~0 but concentration (K40-80/side) cuts gross to 0.4-0.7 — concentration hurts gross more than it saves costs on this alpha | +0.08 best | wide book confirmed superior |
| 13 | 05:55 | exp10 walk-forward ML (14 features, HistGB, yearly retrain, lag-honest labels): rank IC +0.021 (t=8.6) — REAL skill, but decile book gross 0.67, net -3.6 (weekly churn). ML rediscovers reversal, noisier | IC real, book dead | ML at this horizon adds no net value over direct residual reversal |
| 14 | 06:20 | exp11 THE FRONTIER: daily ladder gross 2.03 (tno 0.99/d): 0bp +2.00, 2bp +0.91, 5bp -0.72, 10bp -3.44. Patient (ema5 lam0.15 REG, tno 0.065/d): 0bp 0.99, 5bp 0.75, 10bp +0.51. Fit-window (96-04) gross Sharpes: z1=3.43(!), z5=2.56, z21=1.54 | best 10bp net +0.51 | a Sharpe-3 GROSS signal existed pre-2005 (1d residual reversal = stat market-making); even FREE close-execution today caps at ~2.0 |
| 15 | 06:45 | exp12 path-conditioning: crashes still falling at t+1 pay +44bp d5 vs -6bp for bounced (3x per-trade edge, hypothesis CONFIRMED as science) — but filtered breadth caps book gross at 0.71, net ~0 at 10bp | gross 0.71 | edge concentrates exactly where breadth vanishes |
| 16 | 06:45 | exp13 DECAY + VALIDATION: z1 gross by era: 96-98 0.74, 99-01 **4.60**, 02-04 2.06, 05-07 0.59, 08-10 0.03, 11-13 0.92, 14-16 -0.24, 17-19 -1.44. Survivor configs in VAL 2015-19: **-1.05..-1.46 at ALL fee levels incl. 2bp**. NOTHING SURVIVES TO TEST | VAL failed | the DEV alpha was 1999-2004 vintage; family extinct in modern era |
| 17 | 06:50 | contract amendment logged: DEV2 = 2005-2019 for modern-regime probes; TEST 2020+ stays locked | — | exp14 launched: null battery, fixed crisis-only, spike-fade, quiet-drift, friday-losers |
| 18 | 07:15 | exp14: null battery not turnover-matched (mean -5.85 = pure cost drag of random churn; rules out pick-luck only — noted). Crisis-only ladder: active-only gross 2.57, 2bp 1.83, 5bp 0.73 (22% of days) — same extinct alpha, diluted full-period 0.91@2bp. Modern probes ALL dead: spike-fade -0.02 gross, spike-momo 0.02, quiet-drift 0.44 gross/-0.02@5bp, friday-losers -0.13 gross | all dead | modern regime offers nothing in this family space |
| 19 | 07:20 | exp15 launched: turn-of-month (univ + beta-spread), rank-crossers (inclusion proxy), peer-gap reversion — flow/calendar mechanics on DEV2 | — | |
