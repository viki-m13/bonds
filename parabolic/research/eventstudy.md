# Event study: OHLCV precursors of parabolic 6m runs (IS/OOS)

Outcome: fwd6 = close[d+126]/close[d]-1; parabolic = fwd6 > +50%. Member-masked PIT S&P 500 panel 5647d x 720 tickers. IS = signal date < 2016-01-01, OOS >= 2016-01-01. Features sign-aligned so higher = more bullish (flipped: age_52w_high, bbw_pctile, corr_120, dist_52w_high, fip, tightness_20, vcp_contraction, vol_dryup).

Base rate P(parabolic): IS 2.71% (n=43,921), OOS 3.31% (n=53,247).

`IC_*` = mean per-date Spearman rank-IC of the feature vs fwd6. `topP_*` = P(parabolic) in the feature's top cross-sectional decile. `lift_oos` = topP_oos / base_oos. `topX_*` = top-decile mean fwd6 minus same-date cross-sectional mean (market-neutral excess). `t_oos` = t-stat of the OOS IC series.

| feature | IC_is | IC_oos | t_oos | topP_is | topP_oos | lift_oos | topX_is | topX_oos |
|---|---|---|---|---|---|---|---|---|
| beta_120 | -0.022 | +0.072 | +4.5 | 7.0% | 10.8% | 3.2x | -0.69% | +5.59% |
| dist_52w_low | +0.004 | +0.042 | +4.9 | 6.0% | 6.8% | 2.1x | -0.09% | +3.13% |
| ep_gap_20 | -0.014 | +0.041 | +3.4 | 6.1% | 7.8% | 2.3x | +1.02% | +2.57% |
| vol_20d | -0.017 | +0.027 | +2.1 | 9.0% | 10.9% | 3.3x | +1.24% | +3.09% |
| adr_cc | -0.017 | +0.027 | +2.0 | 9.6% | 11.5% | 3.5x | +1.25% | +3.53% |
| max_dret_21 | -0.015 | +0.022 | +2.0 | 7.1% | 9.0% | 2.7x | +0.11% | +2.76% |
| ret_6m | -0.003 | +0.015 | +1.4 | 3.8% | 4.8% | 1.5x | -0.39% | +1.69% |
| fip | +0.019 | +0.011 | +1.7 | 3.6% | 3.4% | 1.0x | +0.73% | +0.62% |
| dist_52w_high | -0.002 | +0.011 | +0.9 | 8.7% | 9.5% | 2.9x | +0.62% | +1.40% |
| ret_3m | -0.014 | +0.008 | +0.8 | 4.3% | 4.9% | 1.5x | -0.68% | +1.20% |
| rs_line | +0.001 | +0.008 | +0.7 | 1.4% | 2.6% | 0.8x | -1.27% | +0.56% |
| gap_1d | +0.004 | +0.007 | +0.7 | 4.0% | 4.6% | 1.4x | +0.06% | -0.28% |
| mansfield_rs | -0.013 | +0.006 | +0.6 | 3.4% | 4.2% | 1.3x | -1.20% | +0.97% |
| rs_ibd_raw | -0.003 | +0.005 | +0.4 | 3.8% | 4.5% | 1.4x | -0.95% | +1.30% |
| px_vs_sma200 | -0.006 | +0.005 | +0.4 | 3.5% | 4.1% | 1.2x | -0.80% | +0.90% |
| vol_shock | +0.002 | +0.002 | +0.4 | 3.9% | 3.6% | 1.1x | +0.54% | +0.14% |
| at_new_high | -0.013 | +0.001 | +0.2 | 1.3% | 1.8% | 0.5x | -2.25% | +0.86% |
| vcp_contraction | +0.005 | -0.001 | -0.1 | 3.4% | 3.4% | 1.0x | +0.30% | -0.11% |
| vol_dryup | +0.001 | -0.001 | -0.1 | 3.2% | 3.5% | 1.1x | -0.19% | -0.43% |
| rs_line_new_high | -0.007 | -0.001 | -0.2 | 1.2% | 3.2% | 1.0x | -2.16% | +0.76% |
| px_vs_sma50 | -0.018 | -0.001 | -0.1 | 4.5% | 5.1% | 1.6x | -0.26% | +0.86% |
| bbw_pctile | +0.006 | -0.002 | -0.4 | 2.7% | 3.5% | 1.1x | -0.24% | -0.17% |
| sma200_slope | +0.000 | -0.004 | -0.3 | 2.9% | 4.5% | 1.4x | -1.54% | +1.13% |
| ret_1m | -0.018 | -0.004 | -0.4 | 4.6% | 5.3% | 1.6x | +0.01% | +0.90% |
| mom_12_1 | -0.005 | -0.005 | -0.4 | 2.6% | 4.9% | 1.5x | -1.95% | +1.40% |
| accum_25 | -0.006 | -0.005 | -0.8 | 2.5% | 3.1% | 0.9x | -0.49% | +0.03% |
| ret_12m | -0.009 | -0.007 | -0.6 | 2.8% | 4.7% | 1.4x | -1.86% | +1.22% |
| trend_template | +0.003 | -0.008 | -1.0 | 1.4% | 1.9% | 0.6x | -7.17% | -3.69% |
| nearness_52wh | +0.002 | -0.011 | -0.9 | 1.0% | 1.6% | 0.5x | -1.86% | -0.54% |
| tightness_20 | +0.015 | -0.017 | -1.5 | 0.2% | 0.7% | 0.2x | -0.88% | -1.81% |
| age_52w_high | -0.004 | -0.018 | -1.7 | 1.4% | 1.9% | 0.6x | -1.57% | -0.23% |
| corr_120 | -0.002 | -0.060 | -5.6 | 5.0% | 2.3% | 0.7x | +0.40% | -1.62% |
| ep_gap_vol_20 | +nan | +nan | +nan | nan% | nan% | nanx | +nan% | +nan% |

## Robust shortlist (OOS IC>0, sign-stable IS->OOS, OOS lift>=1.3x, |t_oos|>=1.5)

dist_52w_low
