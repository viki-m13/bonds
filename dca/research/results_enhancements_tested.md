# Can we improve SUMMIT further? Top-quant ideas, tested honestly

Drew on the enhancement literature (`literature_enhancements.md`) and tested the
six ranked ideas plus combinations, each on the full 244-window grid with an
IS (2006-14 starts) / OOS (2015-23 starts) split so overfits are caught.
Baseline: win QQQ 93%, median +28.8%, worst −10.6%, OOS 99%/+48.8%, 20.0×.

## What each idea did

| idea (research) | win | median | worst | OOS med | verdict |
|---|---|---|---|---|---|
| **Baseline** | 93% | +28.8% | −10.6% | +48.8% | — |
| Frog-in-the-Pan blend (Da-Gurun-Warachka) | 86% | +22.3% | −13.4% | +44.1% | **hurts** — mega-cap earnings-gaps score "discrete" |
| FIP smooth-path gate | 89% | +24.7% | −16.4% | +48.3% | hurts |
| Momentum × low-vol double-sort | 91% | +18.6% | −10.6% | +33.9% | hurts (re-imports low-vol tilt) |
| 52-week-high gate (George-Hwang) | 93-94% | +28.0% | −9.8%* | +48.8% | **neutral** (*tail gain was offset-0 only; phase-median worst unchanged) |
| Acceleration blend | 92% | +24.8% | −10.8% | +47.8% | neutral-to-worse |
| Vol-managed beta tilt (Barroso/Daniel-Moskowitz) | 93% | +26.2% | **−8.0%** | +43.1% | better IS tail, **−5pp OOS median** |
| **Panic-defer gate (DM triple condition)** | 93% | +28.2% | **−9.0%** | +48.9% | **mild tail win, no OOS cost, −1.1× full** |
| Turn-of-month timing (Ariel/Etula) | 92-95% | +28-32% | — | — | no robust effect (biweekly already spans the month) |
| Conviction/score-gap sizing | (≈ k-sweep) | — | — | — | = higher return/higher risk, already known via k=1 |

## The one defensible enhancement: a targeted panic-defer gate

Daniel-Moskowitz-style triple condition — **SPX < 200dma AND 20-day vol > 80th
pctile AND 2-year SPX return < 0** (only 178 days in 22 years) — defers that
period's contribution to cash and redeploys into the rebound. Effect: worst
window −10.6% → **−9.0%** with **identical OOS** (99%/+48.9%) and unchanged
win-rate, costing ~1.1× of terminal multiple (18.9× vs 20.0×) and a touch of
GFC. It is a genuine, robust, *targeted* crash buffer — it fires only in true
panics, so unlike my earlier high-VIX version it does **not** sit out the
2020-21 / 2023-26 high-vol bulls (OOS is untouched).

## Honest conclusion

**The core edge cannot be improved without a trade-off.** Every
research-backed selection idea that touches the *picking* — Sharpe, vol-adjust,
mean-reversion, FIP, low-vol, 52w-high-as-selector, residual momentum — either
hurts or is neutral, because each dilutes the concentrated mega-cap momentum
that *is* the strategy. This is now confirmed from two independent directions
(my sweeps and the literature).

The only honest gains are **mild, optional tail buffers that cost a little
terminal return**:
* **panic-defer gate** (−10.6→−9.0 worst, no OOS cost, −1.1× full) — the best of them;
* vol-managed beta tilt (−8.0 worst, but −5pp OOS median);
* trend-quality tilt (mild tail help, −1× full);
* loose concentration caps (single-name 33% / sector 50%) for sizing risk.

And a key realization about the worst window itself: SUMMIT's −10.6% worst is a
**2010-2013 3-year window**, where QQQ's own AAPL concentration simply beat the
strategy — that is not a risk event, so no crash protection can fix it; only
being *more* concentrated in Apple (hindsight) would have.

**Recommendation:** keep the live default exactly as is (it's at a robust
optimum), and optionally offer a "conservative mode" = panic-defer gate +
loose caps for investors who will trade ~1× of long-run multiple for a shallower
worst case. There is no free-lunch improvement to the headline numbers.
