# PHOENIX-5 advanced research — four new approaches (round 2)

This round pursued four genuinely different methods beyond re-combining the
existing daily prices. All work is isolated under `phoenix5/` on the
`claude/phoenix-strategy-improvements-cdfs91` branch and **does not touch the
live/production strategy or main**. Honest summary up front, including what
failed — failures here are as informative as the wins.

| Workstream | Outcome | Verdict |
|---|---|---|
| 1. Causal sleeve factory (`factory/`) | Builds a robust but weak orthogonal diversifier (OOS Sharpe 0.6–0.85, corr 0.19). Deflated-Sharpe prob = 0.63 after 23 trials. Under the no-leverage rule it adds only +0.01 Sharpe at a −4pp CAGR cost. | **Negative** — no usable new return |
| 2. Parameter bagging (`bagging/`) | Reveals ORION/HELIOS canonical OOS Sharpes are partly parameter luck. **De-lucked PHOENIX OOS Sharpe ≈ 1.88 (vs 2.15), MDD −25% (vs −18%).** | **Most important finding** — recalibrates expectations + offers a robustness-hardened production option |
| 3. Meta-labeling (`metalabel/`) | Purged-CV AUC = 0.52 (coin-flip). Helps IS by memorization, **hurts OOS** (2.15→2.00, CAGR 36→15%). | **Negative** — no signal at this horizon |
| 4. New-data signals (folded into factory) | External downloads blocked (sandboxed network). Mined under-used existing data: VIX-term carry, bond cross-section. VIX/bond streams were weak; credit-carry family was the only robust part, and it overlaps CREDLO/MOSAIC. | **Mostly negative** |

The one durable, mechanism-supported improvement from *all* the research remains
**5X-TURBO** (intraday-RV accelerated overlay), documented in `README.md`:
OOS Sharpe 2.25, CAGR 37.6%, MDD −17.0%, strictly dominating production. Round 2
did not beat it — but it sharpened how much to trust it (see #2).

---

## 1. Causal sleeve factory (`phoenix5/factory/factory.py`)

Generalizes MOSAIC into an auto-generated library of 23 candidate streams across
four orthogonal-to-PHOENIX families — bond cross-section (carry / momentum /
credit mean-reversion using `features.parquet`), single-asset cross-asset TSMOM,
VIX-term-structure carry (SVXY gated by a causal VIXY-contango proxy), and
credit-carry timing — combined by a **strictly trailing-window (causal)
selector** (monthly, keep trailing-252d-Sharpe > floor, inverse-vol weight).

Honest results:
- The credit-carry family (HY/IG/EM/JNK vs duration) is the only robust part
  (OOS Sharpe 0.6–0.66, small IS→OOS gap) — and it overlaps CREDLO/MOSAIC.
- The bond cross-section streams were largely **negative** OOS; single-asset
  TSMOM was weak except GLD (one lucky asset).
- FACTORY meta-stream: full Sharpe 0.60, OOS 0.85, vol 3%, **corr 0.19** to
  PHOENIX. Looks like a nice diversifier — but its **deflated Sharpe**
  (Bailey–López de Prado, accounting for 23 trials) gives only a **0.63
  probability the true Sharpe is > 0**. We cannot confidently distinguish it
  from multiple-testing luck.
- Under the no-leverage constraint, adding FACTORY to production moves OOS
  Sharpe 2.19→2.20 while cutting CAGR 37→33% — a risk-reduction lever, not
  "more money and less risk." **Rejected** for the stated goal.

Lesson: the existing daily price data is mined out for new *return*. A real new
sleeve needs new *data* (options-implied vol, futures term structure,
fundamentals) — which this sandboxed environment cannot download.

## 2. Parameter bagging (`phoenix5/bagging/`)

For each rules-based sleeve, ran a grid of nearby parameters and equal-weight
averaged the resulting return streams (VANGUARD 45 variants, ORION 9, HELIOS 18).
Each canonical stream was reproduced exactly first, confirming the real sleeve
logic was driven (not reimplemented).

| Sleeve | canonical IS→OOS gap | bagged IS→OOS gap |
|---|---|---|
| VANGUARD | +0.05 (already robust) | +0.06 (bagging neutral) |
| ORION | **−0.46** | −0.23 (halved) |
| HELIOS | **−0.24** | −0.05 (nearly gone) |

A large *negative* IS→OOS gap (OOS >> IS) is a red flag for *favorable*
parameter luck. Bagging pulls both ORION and HELIOS toward believable levels.

**De-lucked PHOENIX** (`bagged_blend.py`) — production blend + overlay with
bagged sleeves:

| | canonical sleeves | bagged sleeves |
|---|---|---|
| OOS Sharpe | 2.15 | **1.88** |
| OOS CAGR | 35.6% | 30.3% |
| OOS MDD | −17.6% | **−25.4%** |
| IS→OOS gap | +0.41 | +0.62 |

This is the headline scientific result of round 2: **about 0.27 of production's
OOS Sharpe, and roughly a third of its drawdown protection, rest on lucky
parameter picks in ORION/HELIOS.** The realistic, overfit-robust forward
expectation is closer to **~1.9 OOS Sharpe with deeper (~−25%) drawdowns**.

Implication / option: running the *bagged* sleeves in production would lower the
backtested numbers but should hold up better out-of-sample going forward (that
is the entire point of bagging). It is the more honest production base. The
genuine improvements (RV overlay + parking) applied to the bagged base land at
~1.92 OOS Sharpe (`bagging/bagged_blend.py` + the RV test) — still real, just on
an honest foundation.

## 3. Meta-labeling (`phoenix5/metalabel/metalabel.py`)

Trained an IS-only classifier (logistic + shallow GBM, purged & embargoed 5-fold
CV) to predict whether PHOENIX's forward 5-day return is negative, then sized
exposure down ahead of predicted bad stretches (overlay can only *reduce* risk;
idle capital earns BIL). Strict protocol: features lagged to t−1, model frozen
before OOS, overlay hyperparameters chosen on IS Sharpe only.

Result — a clean **negative**:
- Purged-CV AUC = **0.532 (logit) / 0.517 (GBM)** — barely above 0.5. There is
  essentially no learnable signal for PHOENIX's bad weeks at this horizon.
- The overlay helped IS (Sharpe 2.62→2.91, MDD halved) purely by memorization,
  and **hurt OOS** (Sharpe 2.15→2.00, CAGR 36%→15% from over-de-risking).
- Conclusion: 5-day directional predictability of a diversified book is ~nil
  (efficient-markets, as expected). The *useful* part of "predicting bad times"
  is volatility, which clusters and *is* forecastable — and that is exactly what
  the existing vol-target plus the RV overlay (5X-TURBO) already exploit.

## 4. New-data signals

External market-data downloads are blocked by the sandboxed network policy, so
"new data" became "under-used existing data," folded into the factory: VIX-term
ETFs (VIXY/SVXY) for a vol-carry stream, and the 26-ticker bond panel in
`features.parquet`. Both were weak (see #1). The genuinely new data the strategy
*would* benefit from — options-implied vol/skew, futures term structures,
earnings/fundamentals — is not reachable here and is the recommended next step
if a data source is provided.

---

## Where this leaves us

- **Best honest improvement:** 5X-TURBO (RV overlay) — real, mechanism-backed,
  OOS 2.25 vs 2.15 on the canonical base.
- **Biggest caveat, newly quantified:** the canonical base is ~0.27 Sharpe
  optimistic; budget for ~1.9 realized OOS Sharpe and deeper drawdowns.
- **Robustness option:** adopt bagged sleeves for a lower-backtest but
  sturdier-forward production strategy.
- **Exhausted on current data:** new orthogonal return (factory) and
  short-horizon timing (meta-labeling). Further gains need new *data*, not new
  math.

Reproduce: `python3 phoenix5/factory/factory.py`,
`python3 phoenix5/bagging/bag_sleeves.py && python3 phoenix5/bagging/bagged_blend.py`,
`python3 phoenix5/metalabel/metalabel.py`.
