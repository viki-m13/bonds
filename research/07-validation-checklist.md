# Validation Checklist — Vetting Any Claimed Sharpe > 5

A practical rubric distilled from [`06-validation-methodology.md`](06-validation-methodology.md). Ask every
question below. A claim that cannot answer them, or answers them the "wrong" way, should be treated as
overfit/inflated/non-scalable until proven otherwise.

---

## A. Definition & accounting
1. **Gross or net?** Net of *all* costs — commissions, spread, market impact, borrow/short fees, financing,
   management/incentive fees? Gross Sharpe is nearly meaningless for an allocator. *(Most >5 stat-arb and
   crypto figures fail here — see [`01`](01-statistical-arbitrage.md), [`05`](05-crypto-niche.md).)*
2. **What's the risk-free rate and base currency?** Mis-set rf inflates excess return.

## B. Sample & out-of-sample integrity
3. **In-sample or live?** Is there an audited, third-party **live track record**, measured in years (not months)?
4. **How many configurations/strategies were tested (N)?** Without N you cannot deflate the Sharpe
   (Deflated Sharpe / Harvey-Liu). Demand it.
5. **Is the backtest long enough for that N?** Apply Minimum Backtest Length: with ~5 years, >~45 independent
   configs ≈ guaranteed in-sample Sharpe 1 / OOS 0.
6. **Was significance tested against the full search universe?** (White Reality Check / SPA;
   Bonferroni/Holm/BHY.) Require **t-stat ≈ 3+**, not 2. (t ≈ SR·√(years).)

## C. Annualization & time-series properties
7. **How was it annualized?** √T is valid only if returns are IID. Intraday Sharpe scaled by √252 or
   √(periods) is almost always overstated (Lo: up to ~65%). *Exception that proves the rule:* genuine
   market-making P&L is near-IID, so its √252 annualization is legitimate — which is why real HFT Sharpes are
   high *and* valid (see [`02`](02-hft-market-making.md)).
8. **What is the P&L autocorrelation (ρ₁…)?** Positive autocorrelation ⇒ true annualized Sharpe is lower;
   recompute with Lo's η(q) adjustment.

## D. Return distribution
9. **Skew and kurtosis?** Negative skew / fat tails (option-selling, carry, merger arb, basis trade, crypto
   carry) make a high Sharpe fragile; the Deflated Sharpe penalizes them. A high Sharpe from a short-volatility
   profile is "picking up pennies in front of a steamroller" (XIV: −96% in one day).
10. **Is the equity curve suspiciously smooth in illiquid assets?** Smoothing/mark-to-model inflates Sharpe
    (Getmansky-Lo-Makarov). De-smooth and recompute. *(The convertible-arb and crypto-carry trap.)*

## E. Data quality
11. **Point-in-time data?** Any look-ahead, survivorship, or restatement bias in the inputs? Backfill /
    instant-history if it's a track-record database?
12. **Are costs realistic at the claimed size?** Slippage and impact scale super-linearly with size.

## F. Capacity & decay (the decisive test)
13. **What is the capacity (max AUM)?** A high Sharpe with *high* capacity is a contradiction — the best
    documented fund (Medallion ~2 net) is **capped at ~$10–15B and closed**. Extreme Sharpe ⇒ small capacity.
14. **Is the signal public/known?** Expect 26–58% decay post-publication (McLean-Pontiff); strongest signals
    decay most. *(Pairs, index effect, crypto carry all decayed — documented.)*
15. **Does it survive a realistic haircut?** Apply the Harvey-Liu nonlinear haircut for data mining plus the
    post-publication discount, then re-ask whether the residual edge is economically meaningful net of costs.

## G. Infrastructure & accessibility
16. **What infrastructure does capturing it require?** If the edge is real but needs colocation, FPGA,
    microwave links, exchange-AP status, or builder/validator relationships, it is a *business*, not an
    allocatable strategy (latency arb, MEV, ETF creation/redemption).
17. **Could *you* actually execute the fills assumed?** Many gross HF Sharpes are liquidity-provision returns
    capturable only by the fastest, lowest-cost executor at trivial AUM.

---

## Decision rule of thumb

A claimed Sharpe > 5 is credible **only if** it is **net of all costs, on a multi-year audited live record,
with disclosed N and a t-stat clearing ~3 after multiple-testing correction, on liquid point-in-time data,
with non-negative skew, low P&L autocorrelation, and an explicitly small/capped capacity.**

Failing any of these — especially **"gross," "no N disclosed," "intraday √T-annualized," "smooth illiquid
curve," "leverage-manufactured low vol," "short-volatility/carry tail," or "scalable/open to capital"** — the
number is almost certainly inflated, mismeasured, or non-scalable.

**The single sharpest test:** *Extreme Sharpe and open/scalable capacity cannot coexist.* The best investable
fund on earth is ~2 net and closed. A pitch of Sharpe > 5, scalable, open to your capital, with no audited
live record, contradicts everything publicly documented.
