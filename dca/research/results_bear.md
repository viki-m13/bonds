# Bear-market behavior for the biweekly DCA picker

Code: `research/signals_bear.py` (run `python research/signals_bear.py`;
scorecards land in `research/scorecards/bear_*.json`).
Protocol: `protocol.evaluate_signal`, k=3, every=10, 5 bps. All signals are
trailing-only (rolling windows, expanding max, forward-only state machines);
regime features from `regime.build_regime`.

## DATA NOTE (important for cross-session comparisons)

The PIT price panel (`data/pit/panel_*.parquet`) was re-downloaded and
rebuilt mid-session (2026-06-12 18:38 UTC) by a concurrent process. All
numbers below are from the **current** panel. On this panel the naive 6m
momentum baseline is `win_qqq=59% med_vs_qqq=+6.0% worst_vs_qqq=-25.7%`
(the `73%/+14%/-40%` quoted in RESEARCH_PROTOCOL.md is from the previous
panel). Every comparison below is internally consistent (re-run end to end
after the rebuild). Random-pick control on the current panel (12 draws):
win_qqq mean 7.8% (max 9.0%), med -14.9%, worst -45.5% — all candidates
clear it by a wide margin.

## Design

* **Bull sleeve**: 6m momentum (`mom126`), the standing baseline.
* **Bear mask (base)**: SPY < 200dma OR breadth_200_ma10 < 0.40
  (27.3% of sample days). On bear days the biweekly contribution buys the
  *bear sleeve* instead of momentum; no holdings are touched.
* **Bear sleeves tested**
  * `lowvol_rs` — lowest 126d vol among names with positive 12m RS vs SPY
  * `defq` — rank(low vol) + rank(shallow drawdown-from-ATH)
  * `rebound` — "quality rebounders": long-term uptrend intact
    (above 400d MA OR positive 24m return) AND 30–60% below ATH, deepest
    discount first
  * `mom_lowbeta` — 6m momentum within the low-beta half
  * `cash` — buy nothing in bears (trend-gate control; cash deploys at the
    next risk-on signal date)

## 1. What to buy in bears

| config (switch @ base bear mask) | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | p10 | GFC | recov 09-12 | bear22 | vol18 | full mult |
|---|---|---|---|---|---|---|---|---|---|---|
| ref: mom126 always (baseline)    | 59% | 86% | +6.0% | -25.7% | -.153 | -.18 | -.03 | -.16 | -.09 | 13.3 |
| sw_cash (trend gate)             | 60% | 86% | +5.1% | -29.7% | -.147 | +.01 | -.09 | -.19 | -.09 | 16.6 |
| **sw_rebound**                   | **61%** | **91%** | **+5.7%** | **-24.5%** | **-.113** | **-.03** | **+.03** | **-.08** | **-.08** | **15.4** |
| sw_lowvol_rs                     | 53% | 88% | +1.6% | -21.5% | -.137 | -.08 | -.05 | -.16 | -.09 | 11.7 |
| sw_defq                          | 53% | 89% | +1.3% | -21.5% |  | -.09 | -.04 | -.14 | -.09 | 11.7 |
| sw_mom_lowbeta                   | 55% | 83% | +2.4% | -22.0% |  | -.16 | -.02 | -.12 | -.10 | 12.1 |

* **Quality rebounders win.** `sw_rebound` is the only bear sleeve that
  improves the baseline on essentially every axis: best win_spy of anything
  tested (91%), best p10 (-11.3% vs -15.3%), GFC -3% vs -18%, 2022 bear
  -8% vs -16%, recovery 2009-12 flips positive (+3%), and median is kept
  (+5.7% vs +6.0%). DCA logic: the contribution keeps buying, but it buys
  quality names at 30-60% panic discounts instead of bear-market momentum
  leaders (which in risk-off are typically defensives/energy that lag the
  recovery).
* Sample picks: 2008-12 → MA, RRC, AAPL (-54..58% off ATH); 2009-03 → GME,
  AAPL, MA; 2020-03 → TDG, AES, CMG; 2022-10 → SLB, HAL, BBWI. Economically
  sensible, not survivorship artifacts (and the random control carries the
  same universe bias).
* Defensive sleeves (lowvol/defq/low-beta) trim the worst window a bit
  (-21.5%) but give up most of the median — they buy "safety" exactly when
  DCA should be buying cheap. Run full-time they are disasters
  (med -21%, win_qqq 5%).
* Pure trend gate (hold cash in bears) is strictly dominated by
  `sw_rebound`: worst gets *worse* (-29.7%) because cash deployed at the
  first risk-on date misses the rebound's first leg.
* A low-vol fallback for periods when no rebounder qualifies
  (`rebound_fb`) changes nothing — with ~500 members the 30-60% band is
  never empty on bear days.

**Sensitivity** (all on current panel): dd band 20-50 / 30-60 / 40-70 /
25-75 → med +4.8/+5.7/+5.3/+5.3, worst -23.6/-24.5/-26.3/-26.5;
shallow-discount-first scoring: med +4.4, worst -20.2 (more conservative
variant if tail matters most); strict quality (400dma AND 24m>0): +4.8/-25.4;
bear mask trend-only vs breadth 0.35/0.45: all within ±0.3% of headline;
schedule offset 3/7: +5.9/-23.3, +5.2/-23.5; k=5 dilutes (med +2.0%).
Nothing is knife-edge; 30-60 deep-first is the modal best.

## 2. Recovery triggers (early switch back to bull sleeve)

Triggers (forward-only latch, reset on each fresh bear entry):
breadth thrust (<0.30 → >0.50 within 42d), VIX pct3y falling <0.75 after
>0.90, SPY reclaiming 50dma while spy_dd < -20%.

| on sw_rebound | extra bull days | med | worst | GFC |
|---|---|---|---|---|
| none (base mask only) | — | +5.7% | -24.5% | -.03 |
| breadth thrust | +4.9% | +5.3% | -24.5% | -.03 |
| VIX relax | +9.8% | +2.6% | -26.1% | **-.20** |
| SPY>50dma deep-dd | +2.6% | +4.8% | -24.4% | -.12 |
| all three | +14.7% | +2.6% | -26.1% | -.20 |

**Verdict: skip recovery triggers.** None helps; VIX relaxation is actively
harmful (it flips to the momentum sleeve during 2008 bear rallies, GFC
-3% → -20%). The reason the 200dma's lateness hurt the *trend gate* is that
cash sat idle; once the bear sleeve keeps buying rebounders, there is
nothing for an early trigger to fix — momentum scores are stale junk at the
bottom anyway. Same conclusion for lowvol/defq sleeves (no trigger moved
them more than +0.5% med).

## 3. Sell / exit triggers

| config | win_qqq | win_spy | med | worst | p10 | GFC | full mult |
|---|---|---|---|---|---|---|---|
| no sells (control = sw_rebound)        | 61% | 91% | +5.7% | -24.5% | -.113 | -.03 | 15.4 |
| HY-OAS panic liquidation (on sw_reb)   | 55% | 70% | +1.9% | **-62.6%** | -.322 | -.08 | 10.6 |
| HY-OAS panic liquidation (on mom126)   | 48% | 64% | -1.6% | -62.0% |  | -.28 | 10.2 |
| 300dma -15% stop (on mom126)           | 57% | 82% | +2.0% | -42.0% | -.186 | -.33 | 11.0 |
| 300dma -15% stop (on sw_rebound)       | 39% | 79% | -4.9% | -33.4% | -.242 | -.05 | 7.4 |

(Panic rule: sell all when HY OAS > trailing-3y 95th pct AND SPY<200dma,
hold cash — buys suspended — until a recovery trigger or base risk-on;
9.7% of days. Stop: sell any holding closing >15% below its own 300d MA;
fires on 10% of member-days.)

**Verdict: no sell rule survives.** The panic liquidation *triples* the
worst window (-24.5% → -62.6%): it dumps everything into the 2008Q4/2020
holes and the recycled proceeds rebuy too high; windows that start just
before a panic are crippled. The per-stock stop fights the rebound sleeve
directly (it sells the discounted names the sleeve just bought) and on
plain momentum it degrades GFC to -33% and halves the median. The only
metric any sell rule improved was a few isolated regime cells — never the
distribution. DCA's edge in bears is *accumulating*; both exits forfeit it.

## 4. VIX-scaled aggressiveness

Overlay: when VIX pct3y > 0.9 (11.6% of days), buy highest-beta quality
rebound candidates (uptrend intact, >=20% off ATH, ranked by 252d beta).

| config | win_qqq | win_spy | med | worst | GFC | covid | bear22 |
|---|---|---|---|---|---|---|---|
| sw_rebound (no overlay)        | 61% | 91% | +5.7% | -24.5% | -.03 | +.06 | -.08 |
| **sw_rebound + vixaggro(0.9)** | **61%** | **90%** | **+6.8%** | **-25.7%** | -.07 | **+.13** | -.08 |
| sw_rebound + vixaggro(0.85)    | 60% | 88% | +7.5% | -26.8% | -.07 | +.13 | -.08 |
| vixaggro on plain mom126       | 60% | 85% | +8.1% | -29.5% | -.11 | +.13 | -.16 |
| beta within 30-60 band only    | 61% | 89% | +6.5% | -26.9% | -.07 | +.13 | -.08 |

The high-beta overlay genuinely adds median (+1 to +2%, mostly from COVID
+6% → +13%) at a small cost in worst/GFC. It is a legitimate dial:
vixaggro(0.9) on top of sw_rebound is the best median config that keeps
worst ≤ baseline. Stacking recovery triggers on top
(`bear_stack_rebound_rec_all_vixaggro`) re-imports the VIX-relax damage
(GFC -20%) — don't.

## RECOMMENDATION

**Bear sleeve: quality rebounders. Switch: SPY<200dma OR breadth<0.40.
No sells. No early-recovery triggers. Optional high-VIX beta tilt.**

1. Risk-on (73% of days): top-3 by 6m momentum, as today.
2. Risk-off: keep buying every period — top-3 quality names (above 400d MA
   or positive 24m return) that are 30-60% below ATH, deepest discount
   first. This is `bear_sw_rebound`:
   `win_qqq 61% / win_spy 91% / med +5.7% / worst -24.5% / p10 -11.3%`
   vs baseline `59% / 86% / +6.0% / -25.7% / p10 -15.3%`, with every
   stress regime improved (GFC -18%→-3%, recovery -3%→+3%, 2022 -16%→-8%).
3. Optionally add the VIX>90th-pct high-beta tilt for `med +6.8%` at
   worst -25.7% (`bear_sw_rebound_fb_vixaggro`).
4. **Do not add sell rules** — the best-intentioned one (HY-OAS panic exit)
   takes the worst window from -24.5% to -62.6%; the 300dma stop costs
   3-10% of median. Control (no sells) wins on every distributional metric.
5. **Do not add recovery triggers** — with a bear sleeve that keeps buying,
   re-entering momentum early only buys bear-rally leaders (GFC -3%→-20%
   with the VIX trigger).

Caveat: the rebound sleeve buys distressed names, the corner of the
universe where missing delisted tickers (~57% coverage in 2005) flatter
results most. The quality gate (long-term uptrend intact) excludes most
death-spiral profiles and the random control carries the same bias, but
this caveat belongs on any graduation audit (`audit.audit_builder`).
