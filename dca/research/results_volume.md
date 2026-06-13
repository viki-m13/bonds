# Volume / Accumulation signal family — results

Biweekly DCA, top-k, 5 bps, full grid (244 windows + 8 regimes), per
`RESEARCH_PROTOCOL.md`. Builders in `research/signals_volume.py`; sweeps in
`research/sweep_volume{,2,3}.py`; scorecards in `research/scorecards/vol_*`.
All normalizations are trailing rolling windows or within-row cross-sectional
ranks (causality contract respected).

Reference points: naive 6m momentum **73% / +14% / -40%** (win_qqq /
med_vs_qqq / worst_vs_qqq); random-pick control ~13% / -14%.

## Results (sorted by win_qqq; k=3 unless noted)

| name | k | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq | regime notes (vs QQQ) |
|---|---|---|---|---|---|---|
| vol_veto_chaikin_k1 | 1 | 66% | 80% | +10.6% | -31.6% | weak: GFC -20%, vol2018, bear22; strong: covid, ai_bull +37% |
| vol_veto_updown_k1 | 1 | 65% | 82% | +10.0% | -33.0% | weak: GFC, vol2018, bear22; strong: bull13-17, covid, ai_bull +41% |
| vol_veto_chaikin | 3 | 64% | 82% | +8.2% | -36.3% | weak: GFC, recovery09-12, vol2018, bear22 |
| vol_veto_updown | 3 | 63% | 82% | +8.6% | -33.6% | weak: GFC, recovery09-12, vol2018, bear22 |
| vol_veto_fp | 3 | 63% | 82% | +7.2% | -33.5% | same pattern |
| vol_veto_chaikin_k2 | 2 | 57% | 82% | +5.7% | -27.2% | same pattern |
| vol_veto_updown_k2 | 2 | 57% | 84% | +5.2% | -30.4% | same pattern |
| vol_tilt_hv_w25_k1 | 1 | 56% | 84% | +3.2% | -26.3% | weak: GFC -18%, vol2018, bear22 |
| vol_tilt_hv_w25_k2 | 2 | 53% | 85% | +1.7% | -22.7% | weak: GFC, vol2018, bear22; mildest tails |
| vol_tilt_hv_w25 | 3 | 52% | 78% | +1.1% | -26.2% | weak: GFC, recovery, bear22 |
| vol_hv_x_mom_k1 | 1 | 50% | 84% | +0.5% | -20.7% | best worst-case in family; weak GFC/covid/bear22 |
| vol_tilt_updown_w25 | 3 | 50% | 77% | +0.1% | -29.6% | weak: GFC, recovery, vol2018, bear22 |
| vol_tilt_fp_w25 | 3 | 50% | 73% | +0.0% | -29.4% | also weak covid |
| vol_tilt_chaikin_w25 | 3 | 50% | 73% | -0.1% | -28.4% | weak: GFC, recovery, vol2018, bear22 |
| vol_tilt_hv_w50 | 3 | 50% | 79% | +0.2% | -24.3% | weak: GFC, recovery, bear22 |
| vol_veto_updown_k5 | 5 | 50% | 84% | +0.2% | -24.6% | diversification kills the edge |
| vol_veto_chaikin_k5 | 5 | 47% | 81% | -1.0% | -24.1% | — |
| vol_tilt_obvdiv_w25 | 3 | 45% | 76% | -1.3% | -34.0% | — |
| vol_hv_x_mom | 3 | 45% | 80% | -2.1% | -24.2% | weak: GFC, recovery, bear22 |
| vol_hv_x_mom_k2 | 2 | 45% | 82% | -2.2% | -23.0% | weak: GFC, bear22 only |
| vol_tilt_hv_w25_k5 | 5 | 45% | 81% | -2.7% | -25.3% | — |
| vol_tilt_updown_w50 | 3 | 44% | 76% | -1.7% | -27.8% | — |
| vol_gate_hv_q80 | 3 | 41% | 77% | -3.1% | -30.2% | also loses bull13-17 |
| vol_hv_x_mom_k5 | 5 | 40% | 82% | -3.5% | -26.9% | — |
| vol_updown_x_mom | 3 | 38% | 76% | -3.8% | -27.5% | — |
| vol_hv_interact | 3 | 38% | 76% | -6.6% | -32.5% | flattest regimes: nothing very bad, nothing good |
| vol_gate_updown_q90 | 3 | 29% | 72% | -8.1% | -29.5% | — |
| vol_fp_63_m15 | 3 | 28% | 57% | -11.8% | -44.0% | loses ai_bull too |
| vol_fp_p_mom | 3 | 27% | 66% | -8.2% | -27.9% | — |
| vol_fp_x_mom | 3 | 27% | 66% | -8.1% | -28.2% | — |
| vol_chaikin_x_mom | 3 | 27% | 58% | -10.2% | -26.2% | — |
| vol_chaikin_p_mom | 3 | 26% | 58% | -10.3% | -26.6% | — |
| vol_obvdiv_x_mom | 3 | 21% | 61% | -9.7% | -32.8% | — |
| vol_updown_63_dlr | 3 | 21% | 59% | -11.3% | -47.2% | — |
| vol_obv_div_126 | 3 | 21% | 66% | -13.2% | -49.5% | weak in every trending regime |
| vol_fp_126_m2 | 3 | 16% | 49% | -14.7% | -45.6% | — |
| vol_hv_gate_v10 | 3 | 16% | 43% | -16.0% | -46.2% | — |
| vol_chaikin_63 | 3 | 15% | 27% | -19.5% | -57.7% | weak in 7/8 regimes |
| vol_gate_updown_q80 | 3 | 14% | 66% | -11.4% | -33.6% | — |
| vol_dryup_90 | 3 | 14% | 43% | -16.3% | -53.5% | — |
| vol_gate_fp_q80 | 3 | 13% | 54% | -14.3% | -35.9% | — |
| vol_chaikin_21 | 3 | 13% | 30% | -17.8% | -48.6% | — |
| vol_updown_63 | 3 | 12% | 52% | -14.8% | -46.7% | — |
| vol_hv_gate_v20 | 3 | 11% | 37% | -16.3% | -48.3% | — |
| vol_obv_trend_63 | 3 | 11% | 58% | -13.8% | -45.9% | — |
| vol_fp_63_m2 | 3 | 10% | 46% | -15.8% | -45.4% | — |
| vol_dryup | 3 | 10% | 41% | -17.2% | -52.4% | — |
| vol_updown_21 | 3 | 9% | 39% | -16.0% | -47.4% | — |
| vol_updown_21_dlr | 3 | 9% | 39% | -16.2% | -47.6% | — |
| vol_gate_chaikin_q80 | 3 | 8% | 22% | -18.7% | -50.3% | worst in family; weak in all 8 regimes |

Legend for variant names: `hv` = abnormal volume (20d/120d mean) x recent
return (Gervais et al.); `updown` = log up-volume/down-volume; `obv` = signed
volume trend / range-position divergence; `chaikin` = CLV-weighted money flow;
`dryup` = low 10d volume near 126d highs in uptrend; `fp` = net count of
high-volume accumulation minus distribution days; `tilt` = mom126 rank + w x
accum rank; `veto` = mom126 rank with net-distribution names demoted; `gate` =
accum rank within top momentum names; `x_mom` / `p_mom` = rank product / sum.

## Parameter sensitivity (honest read)

* **Window length:** 63d beats 21d everywhere (updown 12% vs 9%, chaikin 15%
  vs 13%), but both are deep underwater. Dollar-volume weighting helps only
  at 63d (+9pp) — noise-level.
* **Footprint threshold:** mult 1.5 >> 2.0 (28% vs 10%) — the "big-money"
  2x cutoff is too rare in mega-caps; sensitivity this large means the raw
  signal is fragile.
* **Blend weight:** results degrade monotonically as volume weight rises
  (w=0 i.e. pure mom 73% > w=.25 ~50-52% > w=.5 ~44-50% > rank-product ~27-45%).
  Volume information is *monotonically harmful* to the momentum core here.
* **k:** k=1 is best for every promising blend; k=5 destroys the edge
  (vetoes drop 65%→50%). Same concentration profile as the momentum family.
* **Gating direction matters:** momentum-gated-accumulation (pick accum among
  top-momentum) is far worse than accumulation-vetoed momentum — i.e. volume
  is usable only as a mild *filter*, never as the *selector*.

## Per-regime weaknesses

Every variant above 40% win_qqq is just diluted momentum and inherits
momentum's exact weak regimes: **GFC 2007-09 (-18..-20% vs QQQ), recovery
2009-12, vol 2018, bear 2022** — the volume overlay does not fix a single one
of them. Pure volume/accumulation signals are additionally weak in trending
bull regimes (2013-17, AI bull), i.e. they have no strong regime at all. The
only genuine contribution of volume interactions is tail compression:
`vol_hv_x_mom_k1` worst_vs_qqq -20.7% and `vol_tilt_hv_w25_k2` -22.7% vs -40%
for naive momentum — bought at the cost of ~20pp of win rate and the entire
median edge.

## Top-3 (within family) and recommendation

1. **vol_veto_chaikin, k=1** — 66% / 80% / +10.6% / -31.6%. 6m momentum with
   Chaikin-net-distribution names demoted. Best of family.
2. **vol_veto_updown, k=1** — 65% / 82% / +10.0% / -33.0%. Nearly identical;
   veto construction is robust across all three accumulation measures.
3. **vol_tilt_hv_w25, k=2** — 53% / 85% / +1.7% / -22.7%. Only as a
   tail-risk-reduction component (worst case roughly halved vs momentum).

**Family verdict: do not graduate.** Nothing here approaches the 85% bar, and
every variant is strictly dominated by plain `mom_12_1` (80% / +24.6% /
-30.9%): the best volume "signals" are just momentum minus information. All
standalone accumulation measures beat the random control (so they are not
pure survivorship noise) but carry no incremental selection power in a
large-cap, high-attention universe — consistent with the literature that
volume effects concentrate in small/neglected names. If anything survives to
the composite stage, it is the *veto construction* (demote heavy-distribution
names) as a cheap overlay knob, and abnormal-volume interaction as a
worst-case damper; neither should drive selection.
