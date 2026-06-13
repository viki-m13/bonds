# What if SUMMIT periodically sold everything and rebalanced?

Test: keep biweekly contributions into the top-2, but on each rebalance
boundary liquidate the ENTIRE portfolio at the next open and redeploy the
proceeds + new contribution into the current top-2. 244-window grid, 5 bps.
`python research/rebalance_study.py`.

## Headline (same grid as the live SUMMIT page)

| rebalance | beat QQQ | beat SPY | median vs QQQ | p10 vs QQQ | worst vs QQQ | full multiple (since 2006) |
|---|---|---|---|---|---|---|
| **never sell (SUMMIT)** | 93% | 98% | +28.8% | +3.0% | **−10.6%** | 20.0× |
| annual | 92% | 97% | +28.3% | +2.3% | −40.5% | 34.3× |
| quarterly | 93% | 98% | +32.1% | +3.0% | −14.2% | 130.3× |
| monthly | 93% | 98% | +30.2% | +2.7% | −15.7% | 58.3× |
| every buy (biweekly) | 92% | 97% | +28.3% | +2.2% | −35.4% | 23.7× |

At first glance quarterly looks amazing (130× vs 20×). It is a mirage.

## Why the big multiples are a timing mirage, not an edge

**The full multiple swings 15× → 130× on the rebalance offset.** Quarterly
rebalance run at each of the 10 biweekly schedule phases:

| offset | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | median | range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full mult | 130 | 63 | 19 | 18 | 29 | 15 | 111 | 40 | 49 | 60 | 44.9× | **15.2–130.3×** |

Never-sell over the same offsets: **18.9–21.3×** (a 1.1× swing). So the 130×
was offset-0 luck; some phases (15-18×) are *worse* than never-sell. Selling
everything and concentrating into the current top-2 momentum names is a bet on
the recent leaders continuing — when the timing lines up with the 2023-26 AI
run it explodes, when it doesn't it lags. This is the exact fragility we found
in ROTATOR: high-turnover concentration gives eye-popping but unrepeatable
single-path numbers.

**The robust, repeatable metrics barely move.** Across all 244 start dates the
win-rate (92-93%) and median lead (+28 to +32%) are essentially the same as
never-sell — you do **not** reliably win more by rebalancing. And the
worst-case window gets *worse* (annual −40.5%, every-buy −35.4% vs never-sell
−10.6%), because a fully-concentrated 2-name book has fatter tails than the
accumulated 91-name book.

## The decisive real-world cost the backtest ignores: taxes

Quarterly rebalance fires **417 full liquidations** over the run; monthly far
more. Every one realizes (mostly short-term) capital gains. The engine charges
trading cost but **no tax** — so all of these numbers are pre-tax. In a taxable
account, short-term gains taxed at ~35-50% on every liquidation would erase
most or all of the paper advantage and compound *against* you. Never-sell pays
**zero** tax until you withdraw, letting the full pre-tax balance keep
compounding for decades. That is a large, structural edge for never-sell that
doesn't even show up in the table above.

## Verdict

Periodic full-liquidation rebalancing does not robustly beat never-sell. It:

1. produces headline multiples that are **timing artifacts** (15×-130× on
   phase alone), not a repeatable edge;
2. leaves the **reliable** metrics (win-rate, median) essentially unchanged;
3. **worsens the worst-case** drawdown vs QQQ; and
4. triggers hundreds of taxable liquidations whose drag the backtest omits —
   the one effect that matters most in practice, and it favors never-sell.

This is exactly why SUMMIT never sells: letting winners ride is both the
higher-robustness and the higher-after-tax choice. The flashy rebalanced
multiples are the same mirage as ROTATOR's — concentration that happened to
pay on one path.
