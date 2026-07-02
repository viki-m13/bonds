# Agent: allocator findings (selection on 2014-2018 segment only)

## CRITICAL BUG: DD throttle inert (sign error), also in original PHOENIX
phoenix_production.py:195: dd_mult = (1 + dd/DD_FLOOR).clip(0,1) — with dd<=0 and
DD_FLOOR=-0.10, dd/DD_FLOOR>=0 => mult>=1 => clipped to 1.0 ALWAYS. Verified:
state["dd_mult"] has single unique value 1.0 over entire path. Correct linear is
(1 - dd/DD_FLOOR). The v2/original code had the same formula => the advertised
"DD throttle" never throttled anything there either (review addendum needed).

## Fix adopted (plateau center of uniformly-positive 36-pt grid):
Deadband throttle: dd_mult = ((-0.15 - dd)/(-0.15 - (-0.05))).clip(0,1)
(full exposure until -5% DD, linear to 0 at -15%), same shift(2), on 252d HWM
of the (now un-vol-targeted) raw series.
2014-2018 segment: SR 1.456 -> 1.654, CAGR 27.5 -> 26.7%, vol 17.8 -> 15.0%,
MDD -23.2 -> -18.0%. Improves EVERY allocator variant (+0.09..+0.23 SR),
halves-consistent (calm half neutral, drawdown-heavy half big win).
Candidate series: scratchpad/candidates/alloc_deadband_ddthrottle.csv

## Weight layer: baseline annual-4y budget-ERC WINS vs everything:
quarterly/monthly cadence (worse + turnover), 3y/5y/expanding windows (worse),
sleeve-momentum tilt k=0.1/0.15/0.3 (monotonically worse — prior didn't survive),
Ledoit shrink (small SR loss for small MDD gain), ensembles (worse), no-trade
bands (never recover), semicov ERC (higher CAGR 28.1%, lower SR), family cap
(never binds — inert). NO weight-layer change adopted.

## Vol gate: keep exactly (0.99, 0.5) — monotone degradation toward more
aggressive gating; baseline is plateau edge.
