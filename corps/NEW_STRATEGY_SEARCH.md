# Search for a novel high-Sharpe corporate strategy — a documented negative result

**Objective set:** a new, proprietary corporate-bond strategy with **Sharpe ≈ 3
and CAGR ≥ 10%**, honest, unbiased, not overfit.

**Outcome: the target was not met, and the evidence says it is not reachable in
this instrument set.** Nine strategy families were designed and screened; one
survived in-sample and then **failed its one-shot out-of-sample test**. The best
honest strategy remains the existing dislocation-reversion book, improved by one
change found here (an issuer concentration cap).

This document records the search in full — including everything that failed —
because the negative result is the finding.

---

## 1. What was built to make the test honest

Two pieces of infrastructure came first, and one of them changed our published
numbers.

**Fast research engine** (`corps/research/engine2.py`): a compact per-bond
numpy cache (~1 min load vs ~8 min) with a pure-numpy port of the proven event
engine. Honesty rules identical to the muni/GRANITE engine: signal on day *t*
uses data through *t* only; entry at the first customer-**ask** print strictly
after *t*; exit at customer-**bid** prints; point-in-time liquidity gate;
survivorship-complete tape.

**Honest mark-to-market** — and this corrected a real overstatement. Previously
the equity curve used *linear intra-trade attribution*, which smooths the path.
Re-marking every position daily at its **actual mid prints** (stale marks held
flat) gives:

| book | published maxDD | honest maxDD | monthly Sharpe vs T-bill |
|---|--:|--:|--:|
| full universe ≥3pt | −14.1% | **−32.7%** | 0.41 |
| focused ≤5y | −15.4% | **−31.3%** | 0.46 |
| ≥4pt | −18.4% | **−37.2%** | 0.38 |
| focused ≤5y + issuer cap | — | **−30.0%** | **0.58** |

Total return and CAGR are unaffected — they were always realized from bid/ask
fills. Only the *path* was smoothed. **The strategy's real risk is about twice
what we published, and its risk-adjusted return is ordinary.** That correction
also set the honest bar for this search: beating Sharpe ~0.5, not ~2.

## 2. How candidates were generated and screened

Six independent "trading lens" agents (microstructure/liquidity provision,
cross-sectional momentum, carry/rolldown, issuer relative value, event-driven,
portfolio engineering) each proposed 3–4 point-in-time computable specs — **24
specs**. An adversarial synthesizer killed specs that needed unavailable data or
hid a look-ahead trap, merged duplicates, and ranked the survivors into **8
test-first sleeves**.

**The anti-overfitting protocol, fixed before any result was seen:**

- Design and all iteration on **IS 2003–2015 only**.
- Each sleeve carried a **pre-registered kill gate** (usually monotonicity in
  its own knobs). Failing the gate = dead, **no retuning**.
- Every sleeve's control **replays its full filter chain** with random timing,
  so the comparison isolates timing skill from universe selection and carry.
- **One-shot OOS** on 2016–2025, run once, reported as printed.

## 3. In-sample screening — 7 of 9 died

| sleeve | thesis | IS verdict |
|---|---|---|
| COILSPRING | mild spread-widening reversion | **dead** — excess +0.10%, flat across widening depth; −1.05%/trade outright |
| ENDGAME | index-exclusion pull-to-par | **dead** — Sharpe −0.98 |
| BALLAST-K | short-crossover carry ladder | **dead** — Sharpe −0.40; turnover eats the carry |
| TWINS-R | cheap vs same-issuer siblings | **dead** — timing signal *real* (excess +0.40%, p<0.001) but −0.40%/trade outright: cheap-vs-siblings bonds are cheap for a reason |
| ANGELFALL-M | fallen-angel forced selling | **dead** — 49 events in 13y, excess ≈ 0 |
| CREST | issuer 6-1 spread-tightening momentum | **dead** — Sharpe −0.04, CAGR +1.37% (below T-bill) |
| ANCHOR | short money-good paper held to redemption | **dead** — see §4 |
| **DEBUT** | new-issue concession | alive — Sharpe 0.58, but excess only +0.15% (mostly carry, little timing alpha) |
| **FLOWBACK-S** | volume-confirmed fire-sale reversal | **alive** — excess **+0.40%, p<0.001**, cleanly **monotone in both knobs** (3×/1.0pt → +0.18%; 4×/1.25 → +0.40%; 6×/2.0 → +0.88%) |

## 4. ANCHOR, and the flaw it exposed

ANCHOR was the most promising structural idea: every short-horizon sleeve died
paying the bid-ask **twice**, so hold short-dated money-good paper to
**redemption** — pay the spread once, and let mark volatility collapse as the
price pins to par. This is what real cash-plus desks run.

It failed, and instructively: Sharpe is **monotone increasing in maturity**
(≤1y −0.73, ≤2y 0.13, ≤3y 0.45) — the *opposite* of the pull-to-par thesis.
Short paper doesn't carry enough to cover even a once-paid entry spread.

**Disclosed flaw:** the deliberately punitive "last-bid-only" variant scored
*better* than par redemption (Sharpe 0.43 vs 0.13). Cause: our coupon proxy is
median **yield**, but premium bonds (most short money-good paper in the
falling-rate 2003–2015 era) have coupon > yield. Income is understated while
redemption at exactly 100.0 forces a phantom capital loss from an above-par
purchase. For GRANITE this nets out against the matched control; ANCHOR *is* a
pull-to-par trade, so no control absorbs it. True ANCHOR performance lies
between the two figures and **cannot be pinned down without actual coupons**,
which the OSBAP daily panel does not carry. Both brackets are far below
GRANITE-C, so the verdict is unaffected.

## 5. The one-shot out-of-sample test (2016–2025)

Specs frozen from IS. Run once. Reported as printed.

| sleeve | OOS excess vs control | p | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|--:|--:|
| FLOWBACK-S | +0.07% | **0.253** | +2.46% | 0.14 | −22.9% |
| DEBUT | +0.16% | 0.011 | +2.91% | 0.19 | −23.4% |
| **GRANITE-C** | **+2.65%** | **<0.001** | **+7.24%** | **0.51** | −28.5% |
| combined (frozen rules) | — | — | +3.33% | 0.21 | −19.5% |

**FLOWBACK-S failed.** Its in-sample edge was significant *and* monotone — the
two things that usually indicate a real effect — and it still did not survive.
That is precisely why the one-shot OOS wall exists, and it is why we report the
failure rather than reopening the spec.

## 6. Why combination could not rescue it

| | IS correlations | OOS correlations |
|---|---|---|
| GRANITE ↔ DEBUT | 0.57 | **0.69** |
| GRANITE ↔ FLOWBACK | 0.37 | **0.70** |
| FLOWBACK ↔ DEBUT | 0.35 | **0.85** |

Every surviving sleeve is **long the same risk factor — credit beta** — and the
correlations *rose* out-of-sample, exactly when diversification was needed.
Combination cut drawdown (−30% → −19.5%) but produced a Sharpe **below the best
single sleeve**, because vol-targeting to 5% and dilution by weaker sleeves cost
more than the diversification paid.

## 7. Why Sharpe 3 is not reachable here — the mechanism

Three independent constraints, each demonstrated above rather than asserted:

1. **The bid-ask is paid twice.** A round trip costs ~0.5–1.5 points. At
   30–50 day holds that is a 5–15%/yr drag — it killed COILSPRING, BALLAST-K,
   CREST and FLOWBACK-S. Only ~1-year holds amortize it, and long holds carry
   credit beta, which caps Sharpe.
2. **There is one tradable risk factor.** Long-only cash corporates with no
   shorting means every sleeve is long credit. High Sharpe from combination
   requires uncorrelated streams; they do not exist in this instrument set.
3. **Vol-targeting trades CAGR for Sharpe.** It raises Sharpe only by sitting in
   T-bills, which lowers CAGR. Without leverage the two targets are in direct
   opposition.

**The honest frontier:** GRANITE-C delivers **Sharpe ~0.5 (OOS) / 0.86 (IS)** at
**CAGR +7.24%**. Reaching CAGR 10% requires ~1.4× leverage, which leaves Sharpe
unchanged at ~0.5 and deepens maxDD toward −40%. Sharpe 3 is roughly **6×** the
best strategy that survives honest validation. Getting there would require
instruments this dataset does not contain — CDS or futures to hedge the credit
and rates factors, short positions to build relative-value pairs, or
repo leverage on a genuinely low-vol book.

## 8. What this search did produce

One real improvement, found in the GRANITE experiments and confirmed OOS:

**The issuer concentration cap (max 1 concurrent position per issuer)** raises
Sharpe from 0.46 to **0.58** (full sample) and cuts maxDD from −31.3% to
**−30.0%** — on *half* the trades. OOS it delivers **+2.65% excess (p<0.001),
CAGR +7.24%, Sharpe 0.51**. Diversification across issuers beats raw breadth.
This is now the recommended operating point.

Two supporting findings, also OOS-confirmed:

- **Depth × duration stack** (≥4pt & ≤5y): excess IS +5.91% / OOS +3.92%, best
  CAGR (+6.33%) — but maxDD −38.9%. More alpha, more crisis concentration.
- **Capacity is real**: the excess survives in the *most liquid* quartile
  (+3.54%), so it is not an illiquidity artifact.

## 9. Round two — creative extensions, and the one that survived

A second experimental round attacked the constraints of §7 directly. Everything
below was IS-designed with pre-registered kill gates, red-teamed by an
adversarial agent before implementation, and (for the survivor) given exactly
one OOS look — the program's second, disclosed as such.

### 9.1 Coupon recovery — fixing the dataset's missing field

The panel omits coupons; the median-yield proxy distorted any pull-to-par
accounting (§4). Every (clean price, yield, maturity) triple embeds the coupon
via the bond identity `P = c·A + 100(1+y)^-T`, so we inverted it per bond-day
(qualifying days: mat ≥ 3, price 70–130, y < 25%) and took per-bond medians.
Validation: on near-par bonds — where the identity is exact — recovered coupon
matches yield to **0.049**; premium bonds have c > y in 99.95% of cases,
discount bonds c < y in 100.00%; within-bond daily IQR median 0.09. An
adversarially-proposed accrued-interest correction was tested and **rejected by
the par-anchor test** (it tripled the error — clean prices already remove the
accrual sawtooth). Residual disclosed bias: premium callables carry
yield-to-worst, understating income (conservative). 53,265 of 55,545 bonds
recovered; the rest keep the legacy proxy.

### 9.2 MAGNET — killed by its own pre-registered gates

The most attractive structural idea: graft the proven dislocation *entry* onto
a hold-to-redemption *exit* (spread paid once, no exit-timing decision). The
red-team demanded — and got — a patched redemption classifier (no par credit at
the sample edge or for early-faded tapes), dual controls (same-bond
random-timing; random-bond matched-month for adverse selection), a same-entries
GRANITE-exit benchmark, and bracket invariance. Results:

- **Depth gate FAILED**: excess vs the adverse-selection control is +3%
  (p<0.001) but *flat* across 1/2/3/4-pt depth (+3.38/+3.34/+3.11/+3.05) —
  whatever it is, it is not dislocation alpha.
- **Maturity gradient RISING** (+0.15/+1.15/+1.60% annualized for ≤1/≤2/≤3y) —
  the pre-registered artifact signature: repackaged carry/credit beta.
- **Benchmark decisive**: on *identical entries*, the ordinary GRANITE 1-year
  bid-sale exit beat hold-to-redemption (+5.37% vs +3.58% annualized, Sharpe
  0.92, maxDD −11.8%). The redemption exit adds nothing. Killed.

### 9.3 Two more clean kills

- **Vol-managed sizing** (Moreira-Muir, frozen rule): Sharpe 0.86 → 0.69.
  Vol-timing de-levers exactly before the recovery months that pay for the
  crisis losses. Dead.
- **Calendar concentration** (two pre-registered hypotheses only): December
  t=+0.68, quarter-end t=−0.92. Both null. Closed.

### 9.4 GRANITE-CL — the survivor: limit-entry discipline

One frozen parameter, transparent economics: after the ≥3-pt dislocation
signal, accept the fill **only if the ask is ≤ the latest prior mid + 0.25** —
never chase a price that has already bounced. The matched control faces the
same fill cap (the FLOWBACK lesson). On the issuer-capped ≤5y book:

| window | trades | win | mean/trade | excess vs control | p | CAGR | Sharpe | maxDD |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| IS 2003–2015 | 2,018 | 82% | +12.62% | +4.56% | <0.001 | +11.45% | 0.99 | −37.6% |
| **OOS 2016–2025 (one-shot)** | 2,733 | 79% | **+10.98%** | **+5.48%** | **<0.001** | **+8.05%** | 0.56 | −31.3% |
| full 2003–2025 | 4,592 | 80% | +10.54% | +4.58% | <0.001 | **+8.88%** | **0.73** | −37.3% |

**The OOS excess is *higher* than in-sample** — no decay, unlike FLOWBACK-S.
The mechanism is coherent with everything else in this program: the limit cap
is an implicit depth filter (it keeps only fills where the dislocation is still
fully available), and depth is the one knob that was monotone from the start.

GRANITE-CL is now the best honest operating point: **+10.5% per trade, +4.6%
excess (p<0.001), CAGR +8.9%, Sharpe 0.73, on ~200 trades/yr** — versus
GRANITE-C's +6.6% CAGR / 0.58 Sharpe. Cost: fewer fills (a third of the book)
and a deeper crisis drawdown (−37.3% at honest marks; the limit book
concentrates in the deepest dislocations).

### 9.5 Final verdict on the Sharpe-3 / CAGR-10 target — unchanged

GRANITE-CL clears **CAGR 10% in-sample** (+11.45%) and lands at **+8.05% OOS**
with Sharpe 0.56. The §7 mechanism stands: without leverage, shorting, or
derivatives, Sharpe ~0.6–1.0 at CAGR ~8–11% is the honest frontier of this
instrument set, and this program's best point sits on it.

## 10. Reproduce

```bash
python corps/research/engine2.py build       # compact cache
python corps/research/augment_cache.py       # lagged medians, volumes
python corps/research/augment2.py            # cs60/med15/spr60/qvmed90
python corps/research/augment3.py            # ytw
python corps/research/granite_experiments.py # stack / liquidity / issuer cap
python corps/research/sleeves_events.py      # COILSPRING, FLOWBACK-S, ENDGAME
python corps/research/sleeves_portfolio.py   # BALLAST-K, CREST
python corps/research/sleeves_wave3.py       # TWINS-R, DEBUT, ANGELFALL-M
python corps/research/anchor.py              # ANCHOR + redemption sensitivity
python corps/research/combine_is.py          # sleeve combination (IS)
python corps/research/oos_validate.py        # ONE-SHOT OOS — run once
# round two:
python corps/research/augment4_coupon.py     # coupon recovery (validated)
python corps/research/magnet.py              # MAGNET + kill gates
python corps/research/variants.py            # vol-managed / limit-entry / calendar
python corps/research/oos2.py                # GRANITE-CL one-shot OOS — run once
```
