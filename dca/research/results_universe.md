# Broader universe (Russell-1000-style large+mid) vs S&P 500 PIT

Question: does widening the universe beyond the S&P 500 — to a Russell-1000-style
large+mid-cap pool, mid-caps, mid-cap tech — improve SUMMIT?

## Data note (important)

I could **not** pull the iShares IWB/IVV holdings CSV from this cloud
environment: iShares serves the HTML interstitial (not the file) to the
container's datacenter IP regardless of User-Agent / referer / region cookie.
The provided `requests`+Chrome-UA code is correct and works from a normal IP;
it's an environment/IP block here. So the "broad" universe is built from what I
could download cleanly:

* S&P 500 **point-in-time** (730 tickers, clean, real historical membership)
* **+285 non-S&P** large/mid-cap and Nasdaq names (OHLCV from Yahoo) →
  **1,005 tickers total**.

The 285 add-ons are **survivorship-biased** (only names that survived to today
are in the file), so every broad-universe number is checked against a
random-pick control on the *same* universe, which carries the identical bias.
Drop in the exact IWB ticker list and I'll rerun with it.

## Result — widening the universe does not change SUMMIT

| universe | beat QQQ | beat SPY | median vs QQQ | p10 | worst | full mult | random-pick beats QQQ |
|---|---|---|---|---|---|---|---|
| **S&P 500 PIT (clean)** | 93% | 98% | +28.8% | +3.0% | −10.6% | 20.0× | 8% |
| Broad large+mid (1,005, survivorship) | 93% | 97% | +26.9% | +1.6% | −9.8% | 20.1× | 9% |

Essentially identical — the broad universe's median is even slightly *lower*.
SUMMIT's edge over its own random control is +86pp (S&P) and +84pp (broad):
the same skill, and the survivorship bias in the wider pool didn't inflate it.

## Why: the mega-cap tilt keeps SUMMIT in the biggest names

SUMMIT's risk-on score is momentum **× a strong dollar-volume (mega-cap) tilt**,
so adding hundreds of mid-caps changes almost nothing about what it buys. On the
1,005-name universe the final portfolio is still NVDA 40%, AAPL 24%, MU, AMZN,
NFLX, AMD — and only **17 of 100** positions are non-S&P names, all tiny tail
positions (and mostly large Nasdaq/ADR tech like TSM, BABA, SHOP, PDD — not true
mid-caps). The mid-caps sit in the pool and rarely get picked.

## Could we force mid-caps in? (lower the size tilt)

On the broad universe, dialing the mega-cap tilt down to admit smaller names:

| size tilt | beat QQQ | median | worst | full mult |
|---|---|---|---|---|
| 5.0 (default) | 93% | +26.9% | −9.8% | 20.1× |
| 2.0 | 88% | +26.4% | −16.2% | 22.8× |
| 1.0 | 89% | +23.1% | −18.1% | 23.0× |
| 0.5 | 88% | +20.6% | −19.9% | 23.0× |
| 0.0 (pure momentum) | 73% | +9.0% | −21.0% | 14.3× |

Letting mid-caps in buys a slightly higher terminal multiple (20→23×) at the
cost of a lower win-rate (93→88%) and roughly **doubled** worst-case drawdown
(−10% → −16 to −20%) — and those extra gains are survivorship-inflated, so the
real net is worse, not better. Removing the size tilt entirely is clearly bad.

## Verdict

Widening to a Russell-1000-style or mid-cap universe **does not improve
SUMMIT**, because its mega-cap tilt deliberately concentrates in the largest,
most-liquid names — which is also where survivorship bias is smallest and
drawdowns are most contained. The S&P 500 point-in-time universe is the right,
clean home for this strategy. "Mid-cap tech" exposure specifically would require
a different, lower-tilt strategy run on point-in-time mid-cap data we don't
have — and the quick test above suggests it would be higher-variance, not
better.
