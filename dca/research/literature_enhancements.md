# Literature-Backed Enhancements for a Concentrated, Long-Only, Mega-Cap Momentum DCA Book

**Mandate.** Improve a concentrated (top 1-2 names/period), never-sell, buy-and-hold DCA strategy on S&P 500 PIT data. Core edge: multi-horizon cross-sectional momentum with a strong mega-cap dollar-volume tilt, switching to a "quality-at-a-discount" sleeve below the 200dma. Beats QQQ-DCA in ~93% of rolling windows. Known weakness: worst 3y window ~-10% vs QQQ in transition/momentum-crash regimes (2010-2013, 2009-style rebounds).

**Hard constraint.** The book is *always invested* (DCA stream), *long-only*, and *never sells*. So any "vol-managed" or "dynamic-weight" idea from the long-short literature must be re-expressed as a deploy-timing or name-selection-conditioning rule, NOT a leverage/short rule. The empirically-established negatives (Sharpe selection, vol-adjusted momentum *as a selector*, mean-reversion z-scores, MAX filter, RSI pullback, sector diversification, universe broadening, "don't add to winners") are treated as fixed walls — none of the ranked ideas below re-introduces a low-vol/anti-lottery *selection* tilt.

---

## TL;DR — Ranked top 6

| # | Idea | Targets | Where it acts | Dilution risk to momentum edge |
|---|------|---------|---------------|-------------------------------|
| 1 | **Frog-in-the-Pan information discreteness (continuous-path filter)** | Returns + persistence | Tie-breaker WITHIN momentum winners | Low — keeps highest-momentum names, just prefers smooth ones |
| 2 | **Deploy-timing via Daniel-Moskowitz panic state / Barroso forecast-vol gate** | Tail/crash | WHEN to deploy each DCA lot (cash buffer in panic states) | Low-moderate — can lag rebounds |
| 3 | **Turn-of-the-month / Dash-for-Cash deploy scheduling** | Returns (free) | WHICH DAYS to deploy the lot | None — pure execution timing |
| 4 | **Residual / idiosyncratic momentum as a conditioning overlay** | Tail/crash | Selection in/near regime transitions only | Moderate — anti-beta tilt fights QQQ if used always |
| 5 | **52-week-high proximity as a persistence gate (not a selector)** | Returns + persistence | Tie-breaker / veto on "stale" momentum | Low if used as gate, NOT as primary score |
| 6 | **Conviction (score-gap) position sizing, fractional-Kelly capped** | Returns | k=1 vs k=2 sizing when top name dominates | Moderate — concentration already high; cap hard |

Time-series-momentum "smart adds" and momentum×low-vol double-sort are covered as **secondary/with-caveats** ideas (sections 8 and 4) — both are at high risk of re-importing a low-vol tilt you already proved hurts.

---

## 1. Frog-in-the-Pan / Information Discreteness — **TOP PICK**

**Paper.** Da, Gurun & Warachka, "Frog in the Pan: Continuous Information and Momentum," *Review of Financial Studies* 27(7), 2014, 2171-2218. ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2370931), [PDF](https://www3.nd.edu/~zda/Frog.pdf), [CFA digest](https://rpc.cfainstitute.org/research/cfa-digest/2015/03/frog-in-the-pan-continuous-information-and-momentum-digest-summary))

**Thesis.** Limited attention: information arriving *continuously* in small increments is under-reacted to (a frog boiled slowly), producing momentum that **persists and does not reverse**. Information arriving in *discrete* jumps grabs attention, is priced quickly, and its momentum reverses.

**Exact formula.** Over the 12-month formation window (skip the most recent month as usual):

```
ID = sign(PRET) × (%neg − %pos)
```
- `PRET` = cumulative return over the formation window (the momentum signal itself).
- `%pos`, `%neg` = fraction of *daily* returns in the window that were positive / negative.
- For a **winner** (PRET > 0): a name that rose via many small up-days has %pos high, so `%neg − %pos < 0` → **ID strongly negative = continuous information = high-quality momentum**. A winner that jumped on a few big days has ID positive = discrete = low-quality.

**Effect size.** Among same-cumulative-return winners, momentum runs **monotonically from +5.94%/yr (continuous, low ID) down to −2.07%/yr (discrete, high ID)** in the double sort. Continuous-information momentum **does not reverse long-run**; discrete does. The continuous leg roughly doubles raw momentum profitability.

**How to implement from daily OHLCV (no extra data).** Compute ID for each of your momentum top-N candidates. Use it as a **tie-breaker / multiplier inside the winners you already pick**, e.g. final score = momentum_rank with a penalty on positive-ID names, or simply prefer the lower-ID name when choosing among 2-3 near-tied mega-cap winners. This is the *least dilutive* enhancement: you still buy mega-cap momentum leaders, you just prefer the ones that climbed smoothly — which is exactly the "persistent winner that won't reverse" profile your edge depends on.

**Skeptical note / decay.** (a) Mega-caps trend smoothly anyway, so the cross-sectional ID dispersion among your tiny candidate set may be small — test whether ID actually separates your top-10 names. (b) The discrete-momentum *reversal* is largely a short-leg phenomenon; long-only you mostly capture the "avoid the jumpy winner" benefit, which is weaker. (c) Earnings-gap names (legitimately great mega-caps like NVDA post-print) score as "discrete" and you'd under-weight them — a real cost in your universe. Consider excluding scheduled-earnings gaps from the %pos/%neg count.

---

## 2. Volatility-Managed / Dynamic-Weight Momentum → re-expressed as **DEPLOY-TIMING gate** — TOP TIER for your tail problem

**Papers.**
- Barroso & Santa-Clara, "Momentum has its moments," *JFE* 116, 2015. ([RG](https://www.researchgate.net/publication/256017573_Momentum_Has_Its_Moments))
- Daniel & Moskowitz, "Momentum Crashes," *JFE* 122, 2016, 221-247. ([NBER w20439](https://www.nber.org/system/files/working_papers/w20439/w20439.pdf), [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Momentum-Crashes))
- Moreira & Muir, "Volatility-Managed Portfolios," *JF* 72(4), 2017. ([PDF](https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf), [NBER w22208](https://www.nber.org/system/files/working_papers/w22208/w22208.pdf))

**Exact scaling rules (long-short originals).**
- **Barroso–Santa-Clara:** scale the momentum portfolio by `σ_target / σ̂_t`, where `σ̂_t²` = realized variance of the *daily WML strategy* return over the trailing **6 months (126 days)**, annualized, and `σ_target = 12%` annual. Lifts Sharpe ~**0.53 → 0.97** and kills the fat left tail.
- **Moreira–Muir:** weight `w_t = c / σ̂²_{t−1}`, inverse of *last month's realized variance*. Market alpha 4.9%, appraisal ratio 0.33, ~25% Sharpe lift; works because vol rises without expected return rising proportionally.
- **Daniel–Moskowitz dynamic weight:** `w_t = (1/2λ) · μ_t / σ̂_t²`, where `μ_t` is the conditionally-forecast momentum mean (lower in "panic" states = after big market declines + high vol + during rebounds) and `σ̂_t²` the forecast variance. Roughly **doubles** static momentum Sharpe (to ~1.18 in US equities). Panic states = bear-market indicator (cumulative 2y market return < 0) × high realized vol.

**Why you can't use them verbatim.** They all *de-lever / short* in high-vol states. You're long-only, never-sell, always-DCA-ing. You've already proven that **using vol as a selector hurts** (low-vol tilt loses to QQQ). So apply the *forecast*, not the *scaling*, to **deployment**:

**Implementable rule (the one to test).** Maintain a small **cash buffer that fills, not sells**. Define a "panic" flag from daily OHLCV only:
```
panic_t = (SPX < SPX_200dma) AND (realized_vol_20d > 80th pctile of its own 2y history)
          AND (trailing 2y SPX return < 0)
```
When `panic_t` is on, **route the period's DCA lot into a deferred bucket** (e.g. split the contribution: deploy 40% now, hold 60% to deploy over the *next* 2-3 periods or trigger-release when SPX reclaims its 20dma). This deliberately *avoids buying momentum names into the teeth of the crash/rebound whipsaw* — the exact 2010-2013-style transition that costs you — and redeploys into the recovery. Crucially it never sells and never pauses contributions long-term; it only *re-times* lots. This is the long-only, DCA-native translation of "manage exposure when momentum's forecast variance spikes."

**Effect / target.** Pure **tail/crash** play. Won't add unconditional return (may cost a little in V-shaped rebounds), but directly attacks the worst-3y-window weakness. Your existing bear→quality-sleeve switch already does part of this on the *selection* side; this adds the *timing* side.

**Skeptical note.** (a) Deferring lots in panic can *miss the cheapest lots* — your own review (§3) warns a DCA stream that pauses in bears misses cheap lots. Mitigate by *redeploying into the rebound*, not sitting in cash. (b) Whipsaw: the 200dma/vol gate flips on/off and you can buy the top of a failed rally. Use a confirmation lag (reclaim 20dma for N days). (c) Moreira-Muir's monthly-variance version has been shown to be fragile out-of-sample (Cederburg et al., "Do volatility-managed portfolios work?") — prefer the *slower* 6-month realized-vol / 200dma version, which is sturdier and matches your existing regime gate.

---

## 3. Turn-of-the-Month / Dash-for-Cash — **DEPLOY-DAY scheduling, free alpha**

**Papers.** Ariel, "A monthly effect in stock returns," *JFE* 1987 ([effect: returns concentrated in window [−1,+8]; second half of month ≈ 0]). Lakonishok–Smidt (1988), Ogden (1990) narrow it to **[−1,+3]** (last trading day of month + first 3). Etula, Rinne, Suominen & Vaittinen, "Dash for Cash: Monthly Market Impact of Institutional Liquidity Needs," *RFS* 33(1), 2020, 75-111 ([Oxford](https://academic.oup.com/rfs/article/33/1/75/5494694), [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2528692), [Aalto](https://research.aalto.fi/en/publications/dash-for-cash-monthly-market-impact-of-institutional-liquidity-ne)).

**Mechanism.** The monthly payment/redemption cycle forces mutual funds and pensions to raise cash near month-end → temporary **price pressure (depressed prices / elevated cost of equity capital) in the days *leading up to* month-end (~T−5 to T−1)**, which **reverses (high returns) over the turn-of-month window [−1,+3]**. Ariel: essentially *all* of the average monthly equity return historically accrued in the turn-of-month window; mid-month ≈ 0.

**Implementable rule.** Your cadence is biweekly. You can't change *which names* you buy, but you can choose *which day* to execute the lot. **Deploy each lot 1-2 trading days BEFORE month-end** (into the dash-for-cash dip) rather than mid-month or right at the turn. You buy at the liquidity-driven low and let the turn-of-month bounce work for you. Pure execution — zero selection-edge dilution.

**Effect / target.** Small, reliable **return** improvement on entry price; also a tiny **tail** benefit (buying weakness). Historically tens of bps per lot on average. Targets *price-of-entry*, orthogonal to your momentum selection.

**Skeptical note.** (a) Heavily studied → partially arbitraged; the magnitude has shrunk and the exact high/low days drift (some studies now find pre-month-end strength too). (b) With biweekly DCA only ~half your lots land near month-end; benefit is diluted. (c) It's an *average* — in any single month the dip may not come. Treat as a costless default execution-day heuristic, not a timing bet to size up on. Validate the sign on your own PIT panel before committing (your `cadence_study` infra already exists for this).

---

## 4. Momentum × Low-Vol Double Sort — **SECONDARY, high dilution risk**

**Papers.** Rajalin, "Combining idiosyncratic volatility and momentum" (Vaasa); "Combining low-volatility and momentum: Nordic evidence," *Applied Economics* 2024 ([Taylor&Francis](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2337806)). Cross-market evidence that a **momentum-first, then low-vol** screen gives the best long-only Sharpe and reduces drawdown.

**Method.** Sort to momentum winners FIRST, then within winners drop/penalize the highest-(idiosyncratic-)vol names. Reported: best long-only Sharpe of the variants, lower max drawdown.

**Why this is risky FOR YOU.** Your own panel found vol-adjusted momentum, low-vol, and MAX/anti-lottery filters all **hurt vs QQQ**, because top-1% forward winners sit at the **86th vol / 78th beta percentile** — your parabolic mega-cap winners *are* the high-vol names. A low-vol screen within winners would systematically demote exactly the names that drive your edge. **Recommendation: do NOT adopt as a selector.** The only defensible use is a *crash-window-conditional* version: apply a mild high-idio-vol penalty *only when `panic_t` is on* (link to §2), never in bull regimes. Even then, test against the wall you already hit.

**Effect / target.** Tail, but at material **return cost** in your specific high-beta-benchmark setting. Lowest-conviction idea here; included for completeness because you asked.

---

## 5. 52-Week-High Proximity — as a **persistence GATE**, not a selector

**Paper.** George & Hwang, "The 52-Week High and Momentum Investing," *JF* 59, 2004, 2145-2176 ([Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2004.00695.x)).

**Formula.** `FH_i = P_i,t / high52w_i,t` (current price ÷ trailing 52-week high). Higher = nearer the high. Nearness to the 52w high forecasts returns *better than* past return, and crucially **does not reverse long-run**.

**Why a gate, not a score.** Your review already found 52wh proximity *as a primary selector* underperforms plain momentum (it tilts low-vol/quiet names). But as a **veto/gate it's still useful**: a momentum winner trading *well below* its own 52w high is a *fading* winner (peaked and rolling over) — precisely the kind of stale momentum that reverses in transitions. Rule: among your momentum top-N, **require `FH ≥ ~0.90`** (within 10% of its 52w high) to be eligible for a *new* lot; otherwise prefer the next name. This keeps you in *fresh-high* leaders and out of post-peak names, addressing transition-regime drawdowns without changing the momentum ranking itself.

**Effect / target.** Persistence/**returns** + mild tail benefit (avoids buying topped-out winners). Da-FIP (§1) and fresh-high (§5) are complementary: both push toward "smooth, still-leading" winners.

**Skeptical note.** Anchoring/52wh effect has weakened post-2000s and overlaps mechanically with momentum (a 12-1 winner is usually near its high anyway), so marginal info may be thin in mega-caps. Use as a loose gate (0.85-0.90), not a tight one, or you'll reject legit consolidating leaders.

---

## 6. Conviction / Score-Gap Position Sizing — fractional-Kelly, **hard-capped**

**Basis.** Kelly (1956) `f* = edge/odds`; practitioner consensus = use **fractional (½ or ¼) Kelly** as a ceiling, never full Kelly (variance/ruin). Kelly logic: concentrate where edge is largest, penalize uncertainty. Your own evidence already says concentration-by-momentum helps and "not adding to winners" hurts — so conviction sizing is directionally aligned with your edge.

**Implementable rule.** You pick top 1-2. Make the **1-vs-2 split conviction-driven by the score gap**:
```
gap = (score_1 − score_2) / cross_sectional_std(score)
w_1 = clip(0.5 + κ·gap, 0.5, 1.0)   # κ small, e.g. 0.15; cap at 100%
w_2 = 1 − w_1
```
When the top mega-cap momentum name *dominates* (large gap), lean harder into it (toward k=1); when names are near-tied, split 50/50. This is fractional-Kelly-flavored: size by *relative edge*, capped so you never bet the farm on noise.

**Effect / target.** **Returns** (compounds your best-name edge) with a slight concentration-risk increase. Your book is already very concentrated, so the *incremental* benefit is modest and the *incremental tail* is real — hence the hard cap and small κ.

**Skeptical note.** (a) Score-gap ≠ true edge-gap; momentum scores are noisy, and over-sizing on a marginally-higher score is classic over-fitting. (b) Single-name concentration raises idiosyncratic blow-up risk (one mega-cap accounting fraud / antitrust event). (c) Full Kelly on equities massively over-bets because return estimates are noisy — stay fractional. Keep κ tiny and validate that conviction-sizing beats 50/50 *out of sample* before believing it.

---

## Secondary ideas (asked-for, lower priority)

### 7. Time-Series Momentum "smart adds" (Moskowitz, Ooi & Pedersen 2012)
**Paper.** *JFE* 104, 2012 ([NYU PDF](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf)). Signal: `sign(past 12-month excess return)`; size to **constant ex-ante vol**: `position = (σ_target / σ̂_t)·sign(r_{t−12,t})`, `σ_target ≈ 40%` annual, `σ̂_t²` = **EWMA** of daily returns, `σ̂_t² = 261·Σ(1−δ)δ^i (r_{t−i}−r̄)²` with decay δ giving ~**60-day** center of mass (com ≈ (1−δ)/δ). TSM Sharpe ≈ 0.7+ diversified; famous **"smile"** — *positive* in extreme up AND extreme down markets, which is exactly the crash protection you want.

**For your book (long-only, never-sell).** You can't apply TSM as sizing (no shorting/de-levering, never sell). The usable kernel: **only deploy a new lot into a name while that name's OWN time-series trend is intact** (its price > its own 200dma / 12-1 own-return > 0). This is "add to winners only while the winner is still winning" — aligned with your "adding to winners helps" finding and a direct guard against adding to a name that has quietly rolled over. Combine with §5 (fresh-high gate): both are per-name trend-intact gates.

**Skeptical note.** TSM's crash-protection comes mostly from the *short/de-lever* legs you can't use; long-only you keep only the "don't add to a broken name" piece, which is weaker. Also the absolute-trend gate rarely binds for mega-cap leaders in bulls (they're always above their 200dma) — so it only acts near transitions, which is fine (that's your weak spot) but means low frequency.

### 8. Accelerating momentum (recent, niche)
**Papers.** Ardila/Sornette ETH theses & "Acceleration effect and Gamma factor in asset pricing," *Physica A* 2021 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378437120307196)); Chen-Yu "Buying on Impulse" ([CXO summary](https://www.cxoadvisory.com/momentum-investing/buying-on-impulse-change-in-momentum/)). Acceleration = first difference of successive returns (2nd derivative of price / "Gamma"). High-acceleration minus low ≈ **+6.15% (6m) controlling for momentum alone, ~+4.5%** with other controls; adds ~**3%** over plain momentum.

**For your book.** Test acceleration as an *additional* horizon in your multi-horizon momentum blend: `accel = r_{recent 3m} − r_{prior 3m}` (or slope-of-slope). Aligns with your edge (concentrate into *strengthening* winners) and your EDA ("parabolic moves come from high-energy names"). Use as a *tilt within winners*, not a standalone.

**Skeptical note.** Acceleration is high-turnover and noisy (2nd derivative amplifies noise); it overlaps your short-horizon momentum already; and "accelerating" names are the most crash-prone (blow-off tops). Could *worsen* tail even as it helps return — net-test against your worst-window metric, not just mean.

### 9. What AQR / top quants actually do for long-only momentum crash protection
([AQR Momentum Crashes](https://www.aqr.com/Insights/Research/Journal-Article/Momentum-Crashes), [Factor Momentum Everywhere](https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Factor-Momentum-Everywhere-JPM-Quant-19.pdf), Daniel-Moskowitz.) In practice:
- **Dynamic exposure to forecast mean/variance** (panic-state down-weighting) — the §2 idea; "approximately doubles Sharpe."
- **Combine momentum with value/quality** (negatively-correlated sleeves) so the *blend* never crashes — your bull-momentum / bear-quality regime switch is a retail-grade version of exactly this.
- **Multi-signal momentum** (price + residual + fundamental/operating momentum) to dilute the price-momentum crash beta.
- **Hedging the time-varying market beta** of the momentum book — not available to you long-only, but the *spirit* (cut exposure when momentum beta to the market goes negative in panic) maps to §2's deploy-gate.
Takeaway: nobody runs naked long-only price-momentum into crashes; they all either time exposure on forecast variance or blend with a counter-cyclical sleeve. You already do the second; §2 adds the first.

### 10. Residual / Idiosyncratic Momentum (Blitz, Huij & Martens 2011)
**Paper.** *J. Empirical Finance* 18, 2011, 506-521 ([EUR PDF](https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf), [Alpha Architect](https://alphaarchitect.com/swedroe-spotlight-enhancing-momentum-strategies-via-idiosyncratic-momentum/)).
**Formula.** For each stock, regress monthly returns on FF3 (MKT, SMB, HML) over the **prior 36 months**; take residuals `ε`. Residual momentum = **(Σ ε over t−12..t−1) / stdev(ε over t−12..t−1)** — i.e. the *vol-standardized sum of residuals*.
**Effect.** Monthly Sharpe **0.48 vs 0.25** for conventional momentum (≈doubles IR), **~half the volatility** at similar gross return (1.39% vs 1.54%/mo), and avoids nearly all of the 1930s crash (absorbs some of 2009). All three crash-control methods (idio-mom, constant vol-scale, dynamic) cut crashes; **idiosyncratic momentum gives the largest max-drawdown reduction** in cross-market tests.

**Why it's ranked 4th not 1st for you.** Your panel already found residual-vs-SPY momentum's edge came partly from an **anti-beta tilt that fails against QQQ**. So as an *always-on selector* it dilutes your high-beta edge. Best use: a **transition-regime conditioning overlay** — when `panic_t`/near-200dma, prefer the residual-momentum ranking (lower crash beta) for that period's lot; in clean bull regimes, use raw momentum. This buys the crash-reduction exactly where you need it (your worst 3y windows) without paying the anti-beta tax in bulls. You can compute residuals from daily OHLCV by regressing on SPY (1-factor) or a homemade 3-factor proxy.

---

## Implementation priority (what to backtest first, in order)

1. **FIP continuous-information tie-breaker** (§1) — cheapest, most edge-aligned, daily-OHLCV-only. Test if ID separates your top-10.
2. **Panic-state deploy-deferral with rebound redeployment** (§2) — directly targets the worst-3y-window weakness; reuse your existing 200dma/vol regime infra.
3. **Turn-of-month deploy-day default** (§3) — free, pure execution; validate sign on your `cadence_study` panel.
4. **Fresh-high gate `FH ≥ 0.88`** (§5) + **own-trend-intact gate** (§7) — combine into one "is this winner still leading?" eligibility filter.
5. **Residual-momentum overlay, panic-conditional only** (§10) — crash insurance without the bull-market anti-beta tax.
6. **Conviction score-gap sizing, κ small, hard-capped** (§6) — last, smallest expected lift, real tail cost.

**Cross-cutting caveat.** Every one of these has been published, hence partly arbitraged; effect sizes above are *historical full-sample* and decay out-of-sample. Validate each on your PIT panel against your *actual objective* (rolling-window win-rate vs QQQ **and** the worst-3y-window), not against Sharpe (which your review shows is the wrong target for this book). Adopt only ideas that improve the worst window without lowering the median — i.e. genuinely buy tail insurance without selling your concentrated-momentum return.

---

### Sources
- Da, Gurun & Warachka (2014), *RFS* — Frog in the Pan: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2370931 · https://www3.nd.edu/~zda/Frog.pdf · https://rpc.cfainstitute.org/research/cfa-digest/2015/03/frog-in-the-pan-continuous-information-and-momentum-digest-summary
- Barroso & Santa-Clara (2015), *JFE* — Momentum has its moments: https://www.researchgate.net/publication/256017573_Momentum_Has_Its_Moments
- Daniel & Moskowitz (2016), *JFE* — Momentum Crashes: https://www.nber.org/system/files/working_papers/w20439/w20439.pdf · https://www.aqr.com/Insights/Research/Journal-Article/Momentum-Crashes
- Moreira & Muir (2017), *JF* — Volatility-Managed Portfolios: https://amoreira2.github.io/alan-moreira.github.io/VolPortfolios_published.pdf · https://www.nber.org/system/files/working_papers/w22208/w22208.pdf
- Blitz, Huij & Martens (2011), *JEmpFin* — Residual Momentum: https://repub.eur.nl/pub/22252/ResidualMomentum-2011.pdf · https://alphaarchitect.com/swedroe-spotlight-enhancing-momentum-strategies-via-idiosyncratic-momentum/
- Combining low-vol & momentum (2024), *Applied Economics*: https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2337806
- George & Hwang (2004), *JF* — 52-Week High & Momentum: https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2004.00695.x
- Ariel (1987), *JFE* — Monthly effect; Etula, Rinne, Suominen & Vaittinen (2020), *RFS* — Dash for Cash: https://academic.oup.com/rfs/article/33/1/75/5494694 · https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2528692
- Moskowitz, Ooi & Pedersen (2012), *JFE* — Time Series Momentum: https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- Acceleration effect & Gamma factor (2021), *Physica A*: https://www.sciencedirect.com/science/article/abs/pii/S0378437120307196 · https://www.cxoadvisory.com/momentum-investing/buying-on-impulse-change-in-momentum/
- AQR — Factor Momentum Everywhere: https://images.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Factor-Momentum-Everywhere-JPM-Quant-19.pdf
