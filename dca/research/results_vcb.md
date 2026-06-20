# VCB family: volatility-compression / breakout signals — results

Date: 2026-06-12. Builders in `research/signals_vcb.py`; scorecards in
`research/scorecards/vcb_*.json`. Protocol: biweekly DCA, top-k, 5 bps,
244-window grid + 8 regimes. Benchmarks to beat: naive 6m momentum k=3
(win_qqq 73%, win_spy 88%, med +14.0%, worst -39.7%); random control
(win_qqq 13%, med -13.8%).

## Headline: the family FAILS the 85% bar

No VCB variant reaches win_qqq ≥ 85%. Worse: every compression overlay
*degrades* a plain momentum+uptrend control (71% win_qqq). The compression
contribution is consistently negative on win rate, mildly positive on
worst-case. All variants beat the random control, so they carry real signal
— it is just momentum doing the work.

## Results table (grid windows; k=3 biweekly unless noted)

| name | k | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | regime notes |
|---|---|---|---|---|---|---|
| vcb_volcomp (rv20/rv120 inv, uptrend gate) | 3 | 16% | 35% | -17.6% | -52.3% | picks low-beta laggards; loses everywhere except sideways |
| vcb_volcomp_rank (rank comp + rank 6m mom) | 3 | 36% | 67% | -7.5% | -29.9% | OK 2013-17; loses GFC/recovery/2022 |
| vcb_volcomp_rank | 1 | 35% | 65% | -7.9% | -34.9% | |
| vcb_volcomp_rank | 5 | 30% | 68% | -8.4% | -27.0% | |
| vcb_range (20d/120d HL range) | 3 | 14% | 49% | -14.5% | -56.9% | same low-vol trap as vcb_volcomp |
| vcb_bbw (BB-width trailing-1y pctile) | 3 | 15% | 42% | -16.8% | -51.6% | |
| vcb_donchian (≤2% off 252d hi + low-vol tercile) | 3 | 5% | 32% | -20.6% | -52.5% | avg only ~25 candidates/day; quiet leaders = utilities/staples |
| vcb_atr (ATR20/ATR120 + mom gate) | 3 | 5% | 34% | -18.0% | -53.1% | worst of family |
| vcb_basebreak (fresh 252d-hi after ≥63d base) | 3 | 23% | 50% | -13.2% | -42.1% | event signal, ~18 names/day; too sparse/noisy |
| vcb_sqz_exp (squeeze 1m ago + 1m ret>0) | 3 | 28% | 66% | -8.8% | -31.6% | best of the "pure" spec signals |
| vcb_sqz_exp30 (sqz_pct=0.30) | 1 | 36% | 69% | -6.8% | -33.7% | |
| vcb_sqz_exp30 | 2 | 40% | 73% | -3.1% | -32.3% | only +ve regimes: covid, ai_bull |
| vcb_sqz_exp30 | 3 | 39% | 73% | -4.5% | -32.3% | |
| vcb_sqz_exp30 | 5 | 35% | 72% | -6.8% | -31.9% | |
| vcb_sqz_exp_tight (score by squeeze depth) | 3 | 14% | 44% | -16.3% | -38.1% | scoring by tightness ≪ scoring by expansion ret |
| vcb_mom_comp_tilt (w=0.25) | 1 | 45% | 75% | -2.8% | -40.3% | |
| vcb_mom_comp_tilt (w=0.25) | 3 | 48% | 80% | -2.7% | -27.2% | mild everywhere, wins nothing decisively |
| vcb_mom_comp_tilt (w=0.25) | 5 | 45% | 79% | -3.2% | -24.7% | best worst-case of family |
| vcb_ownhist_comp_mom (own-1y pctile ≤0.3) | 3 | 48% | 81% | -0.4% | -24.7% | best tail control; median ~flat vs QQQ |
| vcb_comp_gate_mom (xsec gate 0.5) | 3 | 58% | 84% | +4.7% | -30.3% | |
| **vcb_comp_gate30_mom** (gate 0.3) | 1 | **70%** | 82% | **+12.9%** | -42.8% | wins bulls (+33% '13-17, +33% AI bull), loses recovery '09-12 (-39%), GFC (-22%), 2022 (-17%) |
| **vcb_comp_gate30_mom** | 2 | **69%** | **84%** | +9.9% | -35.3% | same shape, softer tails |
| vcb_comp_gate30_mom | 3 | 64% | 84% | +5.7% | -30.9% | |
| vcb_comp_gate30_mom | 5 | 54% | 84% | +1.2% | -29.0% | |
| vcb_comp_gate30_mom, monthly (every=21) | 1 | 59% | 81% | +4.3% | -36.3% | monthly cadence hurts (-11pp win_qqq) |
| vcb_comp_gate30_mom, monthly | 2 | 56% | 84% | +2.9% | -29.1% | |
| vcb_comp_gate30_mom, monthly | 3 | 55% | 83% | +2.8% | -29.9% | |
| — control: pure 6m mom + uptrend gate (no comp) | 3 | 71% | 88% | +11.6% | -36.7% | what the family must beat to justify itself; it doesn't |
| — control: near-252d-high, 12m-mom score, no vol cond | 3 | 36% | 78% | -3.9% | -31.1% | 52wk-high proximity itself is weak here |
| — control: momentum + vol-EXPANSION tilt w=0.25 | 3 | 45% | 77% | -2.6% | -27.3% | expansion hurts as much as compression ⇒ vol info ~orthogonal noise that dilutes momentum |

## Parameter sensitivity (honest)

* **Compression dose is monotonically bad for win rate.** comp-gate
  threshold 0.3/0.4/0.5/0.6/0.7 → win_qqq 64/61/58/58/57%; tilt weight
  0/0.1/0.25/0.5/1.0 → 71/59/48/41/36%. The optimum is always "least
  compression", i.e. the family converges back to plain momentum.
* Vol windows: 10/60 ≈ 20/120 (61% vs 58% at gate 0.5); anything anchored
  to 252d long windows is worse (20/252: 49%, 60/252: 43%).
* Squeeze depth: stricter squeeze = worse (sqz_pct 0.1/0.2/0.3 → 25/28/39%).
  Shorter expansion lag (10d) worse than 21d (26% vs 28%).
* Own-history percentile gate: pct_max 0.2/0.3/0.4/0.5 → 37/48/53/53%.
* k: event-sparse signals (donchian/basebreak, 18–25 candidates/day) do not
  improve at small k; comp_gate30 peaks at k=1–2 and decays at k=5.
* Cadence: every=21 monthly is strictly worse than biweekly for the best
  variant (59% vs 70% at k=1).

## Per-regime weaknesses (best variant, vcb_comp_gate30_mom k=2)

* Loses every high-vol / turning-point regime: GFC -20%, recovery 2009-12
  -28% (worst — compressed names lag hardest off a bottom, when the junky
  high-vol rebound leads), vol_2018 -11%, bear_2022 -15% vs QQQ.
* Wins persistent-trend regimes: bull 2013-17 +23%, sideways 15-16 +10%,
  covid +9%, AI bull +28% vs QQQ. Classic momentum profile; the
  compression gate adds nothing the uptrend gate didn't already have.

## Top-3 recommendations

1. **Do not graduate this family.** Volatility compression — in every form
   tried (RV ratio, HL-range ratio, BB-width percentile, ATR ratio, Donchian
   quiet-leader, base breakout, squeeze→expansion) — is at best a neutral,
   usually negative, overlay on momentum in this S&P-500 biweekly-DCA
   setting. The symmetric failure of compression AND expansion tilts says
   short-horizon vol structure carries ~no cross-sectional alpha here.
2. **If a VCB member must be kept**: `vcb_comp_gate_mom(gate=0.3)` at k=1–2
   biweekly (win_qqq 69–70%, med +10–13%) — but prefer the simpler pure
   momentum+uptrend control which beats it (71%) with one less parameter.
   Keep `vcb_ownhist_comp_mom` only as a tail-risk variant (worst -24.7%,
   best in family) if worst-case ever becomes the binding objective.
3. **Where the residual value is**: the compression gate consistently
   improves worst_vs_qqq by ~5–12pp at the cost of median. Future work:
   use compression not for *selection* but as a regime/sizing input (e.g.
   sell-rule or cash-buffer when the portfolio's own vol expands), and
   combine momentum with quality-of-trend measures instead of vol level.
