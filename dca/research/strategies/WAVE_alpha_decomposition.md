# WAVE — Alpha Decomposition (detailed trade analysis)

Source: `exp88_decomp.py` on the WAVE champion (ML + runner-gate + signal-accel,
N12), 151 closed trades, 2015–2025, PIT survivorship-clean.

## The edge is asymmetric holding + fat tails
- Win rate **53%** — barely above a coin flip. The edge is NOT hit-rate.
- **Avg WIN +44.6%  vs  avg LOSS −7.9%** → ~5.6:1 payoff asymmetry. Cutting
  losers (trailing −30% + trend exit) keeps losses small.
- **Winners held 15.7 months; losers held 3.1 months.** Letting winners run while
  culling losers fast is the core mechanic.
- **Top 5% of trades capture 42% of total return** (15% are >50% gainers, 4% are
  >100% multibaggers). A handful of runners drive everything — *missing or
  prematurely cutting a runner is the dominant risk*, which is why aggressive
  exits/gates hurt (they amputate runners).

## Where the alpha concentrates
| Market cap | avg ret | win% | note |
|---|--:|--:|---|
| micro | +33.5% | 53% | highest return (fat-tail upside), capacity-limited |
| small | +9.5% | 53% | weakest |
| mid | +21.7% | 47% | |
| large | +25.7% | **63%** | most reliable (highest win rate) |

All buckets positive — no "trade only X" refinement. Entry-volatility buckets all
similar (no clean vol edge at entry).

## Winners vs losers at ENTRY (what to tilt toward)
| feature | winners | losers |
|---|--:|--:|
| 6-mo momentum | **+0.170** | +0.113 |
| ROA | +0.072 | +0.055 |
| 6-mo vol | 0.059 | 0.064 |
| log mktcap | ~21.9 | ~21.6 |

Winners enter with **stronger momentum, higher quality (ROA), slightly lower
vol** — already partly captured by the runner+accel gates.

## Implication (tested in exp89) — gentle exit is LOAD-BEARING
Hypothesis: cut losers faster, let proven winners run longer (two-stage trailing
stop). RESULT: it BACKFIRED — every two-stage variant (tight early stop −15/−20%
+ loose proven trail −35/−50%) scored Sharpe 0.63–0.76 vs the champion's 1.34.
Reason: future runners often DIP early before taking off; a tight early stop
amputates them. The champion's single gentle trailing-30%-from-peak + trend exit
is already optimal — it gives nascent runners room. **Do not tighten the exit.**

## Bottom line
WAVE champion (single ML + runner-gate + signal-accel, N12, gentle trail−30%+trend)
≈ CAGR 21%/Sharpe 1.41/maxDD −18% is the validated long-only frontier on this data.
The edge is structural (asymmetric holding + fat-tail runner capture), not
refinable by more entry filters, score blends, sleeves, ensembles, or tighter
exits — all tested, all neutral-or-worse. Further alpha needs NEW data
(8-K catalysts, options, intraday), not recombination of existing signals.
