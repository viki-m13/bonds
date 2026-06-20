# Cadence follow-up: phase robustness + native universe

Two checks on the cadence study. Run: `python research/cadence_phase_universe.py`.

## 1. Phase robustness — is ROTATOR's 30× biweekly a lucky offset?

Every schedule phase run at each cadence (daily 1 phase, weekly 5, biweekly 10,
monthly 21); median and [min,max] of win-rate and full multiple.

| strategy | cadence | win QQQ med [min,max] | full mult med [min,max] |
|---|---|---|---|
| SUMMIT | daily | 94% [94,94] | 19.9× [19.9,19.9] |
| SUMMIT | weekly | 94% [93,95] | 19.9× [19.3,20.4] |
| SUMMIT | biweekly | 93% [93,95] | 19.6× [**18.9,21.3**] |
| SUMMIT | monthly | 93% [92,95] | 19.6× [18.2,21.9] |
| ROTATOR | daily | 58% [58,58] | 19.4× [19.4,19.4] |
| ROTATOR | weekly | 68% [64,69] | 29.5× [20.7,34.8] |
| ROTATOR | biweekly | 67% [65,71] | 27.3× [**12.8,44.2**] |
| ROTATOR | monthly | 64% [60,71] | 21.7× [11.7,47.8] |

**Finding — the reported 30× wasn't cherry-picked, but it is dangerously
phase-dependent.** Offset-0 (30.0×) sits just above the biweekly median
(27.3×), so it wasn't a lucky pick. The problem is the *spread*: across the 10
biweekly phases ROTATOR's full multiple ranges **12.8× to 44.2×** — a 3.5×
swing purely from which calendar days you happen to rebalance on. Two investors
running "the same ROTATOR" but starting their biweekly schedule a week apart
could finish at 13× or 44×. SUMMIT's biweekly range is **18.9-21.3×** (a 1.1×
swing); its outcome barely depends on phase. Win-rate is steadier than full
multiple for both, but ROTATOR's (65-71%) never approaches SUMMIT's (93-95%).
This phase-chaos is a real fragility SUMMIT does not have.

## 2. Native universe — S&P 500 + Nasdaq-100 (758 tickers, +38 vs S&P-only)

I had been handicapping ROTATOR by running it on the S&P-500-only panel; its
published universe adds the Nasdaq-100. Re-run on the union (N100 PIT data is
2015+, so the extra names are eligible from 2015 — which is exactly ROTATOR's
key window):

| strategy | cadence | beat QQQ | beat SPY | median vs QQQ | p10 vs QQQ | worst vs QQQ | full mult |
|---|---|---|---|---|---|---|---|
| SUMMIT | daily | 94% | 98% | +35.3% | +2.1% | −11.1% | 19.5× |
| SUMMIT | weekly | 93% | 98% | +34.8% | +2.5% | −11.8% | 19.2× |
| SUMMIT | biweekly | 93% | 98% | +35.7% | +2.6% | −10.6% | 19.6× |
| SUMMIT | monthly | 93% | 98% | +35.4% | +1.6% | −11.2% | 17.9× |
| ROTATOR | daily | 77% | 84% | +42.9% | −26.1% | −48.6% | 51.4× |
| ROTATOR | weekly | 79% | 84% | +41.5% | −20.8% | −50.9% | 51.7× |
| ROTATOR | biweekly | 76% | 84% | +34.0% | −16.8% | −43.1% | 34.1× |
| ROTATOR | monthly | 73% | 81% | +22.7% | −21.6% | −48.5% | 42.2× |

**Finding — the native universe genuinely helps ROTATOR, and it partly
retracts my "biweekly artifact" critique.**

* ROTATOR is **meaningfully stronger on its native universe**: win-rate rises
  to 73-79% (from 58-65% on S&P-only) and full multiple to 34-51× (from
  20-30×). The extra Nasdaq names (high-beta tech — PDD, MELI, etc.) suit its
  momentum rotation. Running it on S&P-only was an unfair handicap; credit
  where due.
* The **"biweekly is uniquely special" pattern was S&P-specific.** On the
  native universe ROTATOR's best cadences are daily/weekly (77-79%, ~51×), not
  biweekly. So its cadence optimum is *universe-dependent* — itself a milder
  form of instability, but the earlier "biweekly artifact" framing was too
  strong.
* **What does NOT change:** the tails. Even at its best (weekly, native
  universe) ROTATOR's worst window is −50.9% and its 10th-percentile is −20.8%,
  versus SUMMIT's −11.8% / +2.5%. And its win-rate (max 79%) never reaches
  SUMMIT's (93-94%). The broader universe raises ROTATOR's ceiling; it does not
  fix its floor.
* **SUMMIT also improves and stays robust.** Median excess rises to ~+35%
  (from +29%) on the broader universe, while win-rate (93-94%), worst window
  (−11%) and cadence/phase-insensitivity are unchanged. SUMMIT banks the extra
  Nasdaq winners without importing any tail risk.

## Net update to the SUMMIT vs ROTATOR picture

Being fair to ROTATOR on its native universe **narrows the gap on returns**
(its ceiling, 51×, is now clearly above SUMMIT's ~19×) but **leaves the risk
gap intact**: −20 to −51% downside windows and 65-79% win-rates vs SUMMIT's
+2.5%/−11% and 93-94%, plus a terminal-wealth outcome that swings 3.5× on
rebalance phase. The two strategies' identities are unchanged by these checks —
ROTATOR is the high-ceiling / high-variance / phase-and-universe-sensitive bet;
SUMMIT is the consistent, robust-to-everything one — but the honest correction
is that ROTATOR's return advantage is real and larger than the S&P-only test
implied, once it's allowed to fish in its full pond.
