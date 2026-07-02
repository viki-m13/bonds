# Agent: calendar findings (IS-only, sub-period consistency enforced)

## Usable candidates (scratchpad/candidates/cal_*.csv):
| file | rule | IS SR | halves | corr blend |
|---|---|---|---|---|
| cal_twdn_ph | TQQQ Tue+Wed when QQQ<200sma (lag1) + session after NYSE holidays; else BIL | 1.08 | 1.08/1.08 | 0.07 |
| cal_twdn_ph_tmf | + TMF month-sessions -3,-2 | 1.32 | 1.49/1.10 | 0.08 (TMF leg halves inconsistent — prefer purity) |
| cal_tue_dn_posthol | Tue-downtrend + post-holiday | 1.06 | 0.89/1.28 | 0.07 |
| cal_tuewed_dn | Tue+Wed downtrend only | 0.95 | 0.95/0.96 | 0.05 |

Core effect: Turnaround Tuesday in downtrends (+146bp/day t=3.2, halves 121/195,
robust across SPY/QQQ/QLD, survives ex-TOM; SMA 100/150/252 variants all positive).
Post-holiday +84bp t=2.6 consistent. Counter-cyclical: active mostly in downtrends
=> corr ~0 to trend books. TQQQ B&H IS baseline 0.88.

## Letter-passes to REJECT: cal_postopex (ex-TOM collapses, halves 1.02/-0.05),
cal_jul (single-month mining), cal_skipaugsep (beta in disguise).

## Dead priors (honestly killed): pre-holiday (sign flips), OpEx week long,
Monday, Tue-Thu-in-UPtrend (alpha lives in DOWNtrends), Halloween/Sep, quarter-turn
(Q-ends WEAKER than month-ends), gap follow/fade (lagged properly), TMF month-end
standalone.
