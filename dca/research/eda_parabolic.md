# EDA: What precedes parabolic single-stock runs (PIT S&P 500, 2005–2025)

Script: `research/eda_parabolic.py` (run from `dca/`). Panel: `data.build_panel()`,
5647 days x 725 tickers, member-mask applied everywhere. All characteristics use
information through close of day d only (trailing windows; no shift(-k), no
full-sample statistics). Forward outcomes: `close[d+126]/close[d]-1` (fwd6) and
`close[d+252]/close[d]-1` (fwd12). Decile/profile analyses sample every 21 trading
days (251 dates, 97,168 pooled member-obs); rank-IC series sampled every 10 days.
Regime = SPY close vs its 200dma on day d (trailing).

## 0. Base rates

| metric | value |
|---|---|
| P(fwd 6m > +50%)  ["parabolic"] | 3.0% |
| P(fwd 12m > +100%) | 1.7% |
| median fwd 6m | +6.3% |
| mean fwd 6m | +6.2% |

So a stock-month has a ~1-in-33 chance of a parabolic 6m run. Everything below is
measured as lift over that base rate and as excess fwd-6m return vs the same-date
cross-sectional mean (`mean_excess6`), which removes market timing.

## 1. Headline: the most promising characteristics

Ranked by a combination of P(parabolic) decile gradient, excess-return spread,
and year-by-year IC stability (full tables in sections below).

| rank | characteristic | top-vs-bottom decile P(parab) | top-decile excess fwd6 | mean IC (fwd6) | IC>0 years /21 | verdict |
|---|---|---|---|---|---|---|
| 1 | `vol_20d` (20d realized vol) | 0.4% -> 10.0% (23x) | +2.3% | +0.004 | 10 | best parabolic screen; pure tail-spreader, median return unchanged |
| 2 | `beta_120` (vs SPY) | 0.8% -> 9.1% (12x) | +2.8% | +0.023 | 11 | best combined: fat tail AND positive mean; IC positive in both regimes |
| 3 | `max_dret_21` (largest daily move, 21d) | 0.7% -> 8.1% (12x) | +1.6% | +0.003 | 11 | vol cousin; a recent >4% day is a strong precursor signature |
| 4 | `dd_from_ath` (drawdown from ATH) | 9.6% in deepest decile vs 1.1% near-ATH | +0.8% (deepest) | +0.016 | 13 | huge but REGIME-CONDITIONAL: works only when SPY<200dma |
| 5 | `dist_52w_low` (bounce off 52w low) | 2.9% -> 6.4% | +1.75% | +0.021 | 12 | the only "strength" measure with positive IC in risk-on AND parabolic-prone top decile |
| 6 | `corr_120` (corr vs SPY) | 3.5% -> 2.1% (anti-parabolic) | +0.7% | +0.030 (best) | 13 | most stable mean-return factor (IC +0.025 risk-on / +0.048 risk-off); selects steady compounders, not rockets |
| 7 | `dist_52w_high` | 9.2% (far below high) vs 1.3% (at high) | +1.2% (far) | -0.005 | 9 | same story as dd_from_ath, slightly weaker; risk-off only |
| 8 | `age_52w_high` (days since 52w high) | 1.6% -> 5.6% | +0.3% | +0.011 | 12 | mild; stale-high names (6m+) modestly better and more parabolic-prone |

**Useless / unstable on this universe:**

| characteristic | mean IC | IC IR | comment |
|---|---|---|---|
| `vol_ratio_20_60` (vol compression) | -0.003 | -0.03 | flat decile profile; "coiling" adds nothing once vol level is known |
| `volu_trend` (20d/120d volume) | -0.005 | -0.06 | flat; tiny U-shape in P(parab) at both extremes only |
| `updown_21` (up-day share) | -0.004 | -0.03 | noise |
| `higher_lows6` (rising 10d lows, 60d) | -0.005 | -0.04 | noise; sign flips with regime |
| `range_contr` (20d/120d high-low range) | -0.010 | -0.10 | weak but *consistent* negative: contracted ranges slightly GOOD (D1 excess +0.9%), i.e. signal is range expansion = bad, not coiling = rocket |
| `ret_1m` | -0.012 | -0.07 | short-term reversal, small |
| `mom_12_1`, `ret_12m` | -0.005 / -0.008 | -0.02 / -0.04 | unconditional 12m momentum has ~zero IC vs fwd6 here; swings -0.31 (2009) to +0.15 (2015) |
| `ret_6m` | +0.006 | +0.03 | near zero unconditionally; +0.024 risk-on / -0.054 risk-off |

Reading: the *level and recency of volatility* (vol_20d, beta, max_dret_21) is by far
the strongest unconditional precursor of parabolic moves — parabolic runs come from
high-energy names, not from quiet "coiled springs" (compression measures are dead).
The second axis is *where the stock sits in its cycle*: deep drawdown + already
bounced off the 52w low. Classic momentum is a regime-conditional, not unconditional,
edge at the 6m horizon.


## 2. Decile tables (selected)

Per-date cross-sectional deciles (1 = lowest characteristic value, 10 = highest),
pooled over 251 monthly dates. `mean_excess6` = fwd6 minus same-date cross-sectional
mean. `_on` / `_off` = SPY above / below its 200dma on the signal date.

### vol_20d

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0492 | 0.0548 | -0.0143 | 0.0043 | -0.0063 | 0.0020 | -0.0395 | 0.0137 |
| 2 | 0.0533 | 0.0558 | -0.0103 | 0.0058 | -0.0049 | 0.0031 | -0.0265 | 0.0166 |
| 3 | 0.0539 | 0.0570 | -0.0097 | 0.0078 | -0.0049 | 0.0052 | -0.0259 | 0.0177 |
| 4 | 0.0600 | 0.0623 | -0.0036 | 0.0125 | -0.0023 | 0.0067 | -0.0052 | 0.0363 |
| 5 | 0.0600 | 0.0622 | -0.0036 | 0.0168 | -0.0015 | 0.0108 | -0.0110 | 0.0418 |
| 6 | 0.0656 | 0.0644 | 0.0020 | 0.0255 | 0.0012 | 0.0175 | 0.0069 | 0.0579 |
| 7 | 0.0656 | 0.0616 | 0.0019 | 0.0299 | 0.0019 | 0.0224 | 0.0021 | 0.0621 |
| 8 | 0.0685 | 0.0611 | 0.0050 | 0.0407 | 0.0021 | 0.0316 | 0.0116 | 0.0782 |
| 9 | 0.0729 | 0.0600 | 0.0093 | 0.0581 | 0.0036 | 0.0446 | 0.0239 | 0.1089 |
| 10 | 0.0862 | 0.0530 | 0.0227 | 0.1004 | 0.0109 | 0.0823 | 0.0619 | 0.1691 |

*Monotone 23x lift in P(parab). Note med_fwd6 in D10 (5.3%) is *below* D1 (5.5%): vol buys you the right tail at the cost of a fatter left tail — a lottery-ticket axis, needs a second filter.*

### beta_120

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0444 | 0.0502 | -0.0187 | 0.0077 | -0.0131 | 0.0051 | -0.0402 | 0.0142 |
| 2 | 0.0501 | 0.0550 | -0.0130 | 0.0075 | -0.0086 | 0.0040 | -0.0262 | 0.0218 |
| 3 | 0.0545 | 0.0573 | -0.0087 | 0.0113 | -0.0037 | 0.0076 | -0.0252 | 0.0265 |
| 4 | 0.0545 | 0.0563 | -0.0086 | 0.0137 | -0.0067 | 0.0077 | -0.0132 | 0.0374 |
| 5 | 0.0598 | 0.0606 | -0.0034 | 0.0190 | -0.0034 | 0.0137 | -0.0024 | 0.0402 |
| 6 | 0.0634 | 0.0629 | 0.0003 | 0.0252 | -0.0008 | 0.0178 | 0.0039 | 0.0543 |
| 7 | 0.0687 | 0.0672 | 0.0056 | 0.0311 | 0.0046 | 0.0213 | 0.0086 | 0.0700 |
| 8 | 0.0723 | 0.0637 | 0.0092 | 0.0411 | 0.0061 | 0.0308 | 0.0195 | 0.0824 |
| 9 | 0.0715 | 0.0607 | 0.0083 | 0.0544 | 0.0052 | 0.0429 | 0.0159 | 0.0991 |
| 10 | 0.0914 | 0.0620 | 0.0284 | 0.0906 | 0.0199 | 0.0738 | 0.0578 | 0.1544 |

*The one characteristic where mean, tail and excess all line up: D10 mean fwd6 9.1% vs D1 4.4%, P(parab) 9.1% vs 0.8%. Works in both regimes (excess_on +2.0%, excess_off +5.8%).*

### max_dret_21

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0508 | 0.0552 | -0.0127 | 0.0069 | -0.0076 | 0.0031 | -0.0272 | 0.0216 |
| 2 | 0.0563 | 0.0602 | -0.0072 | 0.0085 | -0.0009 | 0.0053 | -0.0281 | 0.0218 |
| 3 | 0.0564 | 0.0595 | -0.0072 | 0.0133 | -0.0045 | 0.0080 | -0.0151 | 0.0349 |
| 4 | 0.0580 | 0.0598 | -0.0055 | 0.0146 | -0.0020 | 0.0097 | -0.0166 | 0.0347 |
| 5 | 0.0601 | 0.0609 | -0.0035 | 0.0196 | -0.0021 | 0.0133 | -0.0089 | 0.0454 |
| 6 | 0.0636 | 0.0607 | 0.0001 | 0.0267 | -0.0005 | 0.0199 | 0.0007 | 0.0538 |
| 7 | 0.0682 | 0.0612 | 0.0046 | 0.0327 | 0.0024 | 0.0254 | 0.0137 | 0.0636 |
| 8 | 0.0696 | 0.0609 | 0.0060 | 0.0440 | 0.0018 | 0.0348 | 0.0185 | 0.0819 |
| 9 | 0.0728 | 0.0581 | 0.0091 | 0.0544 | 0.0047 | 0.0418 | 0.0216 | 0.1032 |
| 10 | 0.0795 | 0.0547 | 0.0159 | 0.0812 | 0.0085 | 0.0649 | 0.0403 | 0.1420 |

*A single recent big up-day (D10 median max move ~6%+) marks names with 8.1% parabolic probability — energy begets energy.*

### dd_from_ath

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0711 | 0.0328 | 0.0077 | 0.0963 | -0.0084 | 0.0773 | 0.0692 | 0.1728 |
| 2 | 0.0644 | 0.0528 | 0.0009 | 0.0536 | -0.0063 | 0.0384 | 0.0263 | 0.1132 |
| 3 | 0.0687 | 0.0627 | 0.0052 | 0.0392 | 0.0011 | 0.0286 | 0.0225 | 0.0840 |
| 4 | 0.0652 | 0.0629 | 0.0017 | 0.0265 | 0.0026 | 0.0183 | -0.0012 | 0.0607 |
| 5 | 0.0651 | 0.0613 | 0.0016 | 0.0230 | 0.0007 | 0.0145 | 0.0040 | 0.0547 |
| 6 | 0.0610 | 0.0613 | -0.0023 | 0.0161 | -0.0021 | 0.0105 | -0.0032 | 0.0382 |
| 7 | 0.0622 | 0.0602 | -0.0014 | 0.0155 | 0.0022 | 0.0126 | -0.0138 | 0.0274 |
| 8 | 0.0620 | 0.0635 | -0.0016 | 0.0136 | 0.0027 | 0.0103 | -0.0181 | 0.0260 |
| 9 | 0.0594 | 0.0585 | -0.0040 | 0.0095 | 0.0035 | 0.0081 | -0.0316 | 0.0161 |
| 10 | 0.0560 | 0.0596 | -0.0074 | 0.0113 | 0.0040 | 0.0099 | -0.0524 | 0.0138 |

*Deciles ordered by drawdown value, so D1 = deepest drawdown. D1: P(parab) 9.6% but median fwd6 only +3.3% — classic distressed lottery. The entire effect is risk-off: excess_off +6.9% / P_off 17.3% vs excess_on -0.8%.*

### dist_52w_high

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0759 | 0.0439 | 0.0123 | 0.0918 | -0.0058 | 0.0696 | 0.0798 | 0.1787 |
| 2 | 0.0711 | 0.0606 | 0.0077 | 0.0505 | 0.0007 | 0.0372 | 0.0347 | 0.1047 |
| 3 | 0.0686 | 0.0620 | 0.0051 | 0.0356 | 0.0018 | 0.0244 | 0.0177 | 0.0821 |
| 4 | 0.0657 | 0.0618 | 0.0022 | 0.0293 | 0.0005 | 0.0183 | 0.0084 | 0.0732 |
| 5 | 0.0657 | 0.0635 | 0.0023 | 0.0234 | 0.0028 | 0.0172 | 0.0015 | 0.0468 |
| 6 | 0.0625 | 0.0618 | -0.0011 | 0.0186 | 0.0013 | 0.0136 | -0.0090 | 0.0396 |
| 7 | 0.0584 | 0.0579 | -0.0051 | 0.0154 | -0.0030 | 0.0118 | -0.0142 | 0.0297 |
| 8 | 0.0579 | 0.0605 | -0.0052 | 0.0126 | 0.0003 | 0.0115 | -0.0244 | 0.0178 |
| 9 | 0.0595 | 0.0591 | -0.0071 | 0.0109 | 0.0004 | 0.0097 | -0.0361 | 0.0162 |
| 10 | 0.0500 | 0.0531 | -0.0107 | 0.0129 | 0.0009 | 0.0127 | -0.0564 | 0.0113 |

*Same shape as dd_from_ath (D1 = far below 52w high): P(parab) 9.2% vs 1.3% at-the-high; risk-off only (excess_off +8.0% vs on -0.6%).*

### dist_52w_low

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0481 | 0.0482 | -0.0130 | 0.0291 | -0.0171 | 0.0225 | 0.0054 | 0.0579 |
| 2 | 0.0547 | 0.0584 | -0.0075 | 0.0191 | -0.0091 | 0.0148 | -0.0001 | 0.0348 |
| 3 | 0.0670 | 0.0611 | -0.0056 | 0.0302 | -0.0075 | 0.0133 | 0.0053 | 0.0945 |
| 4 | 0.0554 | 0.0545 | -0.0062 | 0.0174 | -0.0073 | 0.0124 | 0.0001 | 0.0385 |
| 5 | 0.0578 | 0.0601 | -0.0036 | 0.0203 | -0.0032 | 0.0157 | -0.0052 | 0.0377 |
| 6 | 0.0626 | 0.0598 | 0.0006 | 0.0231 | 0.0012 | 0.0177 | 0.0043 | 0.0483 |
| 7 | 0.0670 | 0.0625 | 0.0035 | 0.0274 | 0.0045 | 0.0193 | -0.0015 | 0.0616 |
| 8 | 0.0674 | 0.0617 | 0.0040 | 0.0300 | 0.0045 | 0.0215 | 0.0007 | 0.0649 |
| 9 | 0.0733 | 0.0648 | 0.0099 | 0.0386 | 0.0118 | 0.0312 | -0.0026 | 0.0679 |
| 10 | 0.0811 | 0.0611 | 0.0175 | 0.0639 | 0.0215 | 0.0563 | -0.0064 | 0.0880 |

*D10 = biggest bounce off 52w low: P(parab) 6.4%, excess +1.75%, and — uniquely among the cycle measures — positive in risk-ON too (excess_on +2.2%). "Already turned" beats "still falling".*

### age_52w_high

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0530 | 0.0522 | -0.0070 | 0.0164 | 0.0025 | 0.0150 | -0.0434 | 0.0219 |
| 2 | 0.0627 | 0.0616 | -0.0043 | 0.0216 | -0.0012 | 0.0174 | -0.0188 | 0.0360 |
| 3 | 0.0632 | 0.0580 | 0.0008 | 0.0228 | 0.0017 | 0.0154 | -0.0008 | 0.0510 |
| 4 | 0.0598 | 0.0569 | -0.0044 | 0.0221 | -0.0058 | 0.0136 | -0.0008 | 0.0554 |
| 5 | 0.0589 | 0.0596 | -0.0046 | 0.0234 | -0.0029 | 0.0160 | -0.0105 | 0.0525 |
| 6 | 0.0578 | 0.0561 | -0.0040 | 0.0249 | -0.0051 | 0.0163 | 0.0002 | 0.0597 |
| 7 | 0.0712 | 0.0625 | 0.0076 | 0.0337 | -0.0010 | 0.0208 | 0.0423 | 0.0855 |
| 8 | 0.0720 | 0.0658 | 0.0086 | 0.0354 | 0.0084 | 0.0291 | 0.0095 | 0.0611 |
| 9 | 0.0679 | 0.0677 | 0.0043 | 0.0423 | 0.0057 | 0.0356 | -0.0053 | 0.0667 |
| 10 | 0.0660 | 0.0524 | 0.0031 | 0.0556 | -0.0022 | 0.0449 | 0.0269 | 0.1003 |

*Stocks 6m+ past their last 52w high (D7-10) carry modest positive excess and 2-3x parabolic lift vs fresh-high names.*

### corr_120

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0553 | 0.0476 | -0.0078 | 0.0345 | -0.0083 | 0.0234 | -0.0087 | 0.0722 |
| 2 | 0.0624 | 0.0555 | -0.0007 | 0.0360 | -0.0022 | 0.0265 | 0.0037 | 0.0728 |
| 3 | 0.0594 | 0.0553 | -0.0038 | 0.0323 | -0.0048 | 0.0235 | 0.0024 | 0.0687 |
| 4 | 0.0582 | 0.0496 | -0.0049 | 0.0334 | -0.0042 | 0.0258 | -0.0109 | 0.0603 |
| 5 | 0.0570 | 0.0528 | -0.0062 | 0.0302 | -0.0061 | 0.0226 | -0.0050 | 0.0610 |
| 6 | 0.0642 | 0.0610 | 0.0011 | 0.0319 | 0.0013 | 0.0224 | 0.0015 | 0.0698 |
| 7 | 0.0669 | 0.0619 | 0.0038 | 0.0307 | 0.0049 | 0.0238 | -0.0005 | 0.0576 |
| 8 | 0.0675 | 0.0637 | 0.0043 | 0.0282 | 0.0045 | 0.0226 | 0.0042 | 0.0511 |
| 9 | 0.0702 | 0.0686 | 0.0071 | 0.0250 | 0.0069 | 0.0201 | 0.0093 | 0.0457 |
| 10 | 0.0699 | 0.0705 | 0.0069 | 0.0209 | 0.0077 | 0.0151 | 0.0039 | 0.0440 |

*Monotone in mean/median fwd6 (D10 median 7.1% vs D1 4.8%) but *anti*-parabolic (P(parab) 2.1% vs 3.5%). A stabilizer, not a rocket-finder.*

### ret_6m

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0718 | 0.0472 | 0.0084 | 0.0696 | -0.0083 | 0.0476 | 0.0754 | 0.1571 |
| 2 | 0.0647 | 0.0600 | 0.0012 | 0.0385 | -0.0050 | 0.0251 | 0.0279 | 0.0952 |
| 3 | 0.0649 | 0.0629 | 0.0016 | 0.0277 | -0.0020 | 0.0163 | 0.0162 | 0.0740 |
| 4 | 0.0605 | 0.0566 | -0.0029 | 0.0233 | -0.0031 | 0.0132 | -0.0001 | 0.0644 |
| 5 | 0.0594 | 0.0603 | -0.0040 | 0.0192 | -0.0043 | 0.0129 | 0.0000 | 0.0455 |
| 6 | 0.0587 | 0.0601 | -0.0047 | 0.0174 | -0.0016 | 0.0126 | -0.0135 | 0.0362 |
| 7 | 0.0622 | 0.0591 | -0.0012 | 0.0194 | 0.0008 | 0.0155 | -0.0094 | 0.0358 |
| 8 | 0.0604 | 0.0602 | -0.0029 | 0.0203 | 0.0022 | 0.0193 | -0.0256 | 0.0224 |
| 9 | 0.0596 | 0.0575 | -0.0039 | 0.0223 | 0.0031 | 0.0208 | -0.0326 | 0.0286 |
| 10 | 0.0716 | 0.0630 | 0.0083 | 0.0439 | 0.0177 | 0.0423 | -0.0366 | 0.0445 |

*Flat-to-U unconditionally. Regime split is stark: D10 excess_on +1.8% vs excess_off -3.7%; D1 excess_off +7.5% (bear-market junk rally).*

### mom_12_1

| decile | mean_fwd6 | med_fwd6 | mean_excess6 | P_parab | excess6_on | P_parab_on | excess6_off | P_parab_off |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0658 | 0.0395 | 0.0024 | 0.0738 | -0.0101 | 0.0570 | 0.0524 | 0.1434 |
| 2 | 0.0685 | 0.0606 | 0.0052 | 0.0431 | 0.0012 | 0.0306 | 0.0222 | 0.0957 |
| 3 | 0.0669 | 0.0628 | 0.0035 | 0.0311 | 0.0022 | 0.0211 | 0.0127 | 0.0725 |
| 4 | 0.0692 | 0.0682 | 0.0059 | 0.0266 | 0.0048 | 0.0167 | 0.0116 | 0.0657 |
| 5 | 0.0642 | 0.0634 | 0.0008 | 0.0195 | 0.0011 | 0.0147 | 0.0022 | 0.0388 |
| 6 | 0.0605 | 0.0598 | -0.0028 | 0.0173 | -0.0010 | 0.0116 | -0.0079 | 0.0416 |
| 7 | 0.0595 | 0.0585 | -0.0039 | 0.0171 | -0.0023 | 0.0130 | -0.0087 | 0.0354 |
| 8 | 0.0564 | 0.0572 | -0.0069 | 0.0145 | -0.0029 | 0.0102 | -0.0240 | 0.0277 |
| 9 | 0.0591 | 0.0578 | -0.0043 | 0.0179 | 0.0002 | 0.0152 | -0.0246 | 0.0281 |
| 10 | 0.0634 | 0.0533 | 0.0001 | 0.0391 | 0.0065 | 0.0350 | -0.0344 | 0.0498 |

*Same as ret_6m but weaker. Unconditional 12-1 momentum does not predict fwd 6m returns in this universe at the mean; the parabolic mass sits at BOTH extremes (D1 7.4%, D10 3.9%).*

## 3. Regime interaction (SPY vs 200dma) — the single biggest modifier

From the decile tables and the IC split below, the panel-wide pattern:

| characteristic (top-decile tilt) | excess fwd6, SPY>200dma | excess fwd6, SPY<200dma | P(parab) off-regime |
|---|---|---|---|
| deep drawdown from ATH (D1) | -0.8% | **+6.9%** | 17.3% (vs 3.0% base) |
| far below 52w high (D1) | -0.6% | **+8.0%** | 17.9% |
| 6m winner (ret_6m D10) | **+1.8%** | -3.7% | 4.5% |
| 6m loser (ret_6m D1) | -0.8% | **+7.5%** | 15.7% |
| high beta (D10) | +2.0% | **+5.8%** | 15.4% |
| high vol_20d (D10) | +1.1% | **+6.2%** | 16.9% |
| big bounce off 52w low (D10) | **+2.2%** | -0.6% | 8.8% |

Takeaways:
- In risk-off, *everything* that points at beaten-down, high-vol names lights up:
  P(parabolic) for deep-drawdown high-beta names is ~5-6x base rate. This is the
  2009/2020 rebound effect and it repeats across the regime years.
- In risk-on, the only reliable tilts are: bounce off 52w low (+2.2%), beta (+2.0%),
  6m strength (+1.8%), correlation (+0.7% top decile, IC +0.025).
- Momentum and reversal are mirror images across the regime switch; an unconditional
  strategy on either will average near zero IC.

## 4. Rank IC (Spearman, characteristic at d vs fwd 6m), biweekly dates

| feature | mean_IC | IC_IR | pct_pos | IC_spy_above200 | IC_spy_below200 |
|---|---|---|---|---|---|
| corr_120 | 0.0300 | 0.2023 | 0.5837 | 0.0250 | 0.0478 |
| beta_120 | 0.0232 | 0.0882 | 0.5114 | 0.0091 | 0.0729 |
| dist_52w_low | 0.0212 | 0.1406 | 0.5570 | 0.0301 | -0.0078 |
| dd_from_ath | 0.0157 | 0.0723 | 0.5551 | 0.0361 | -0.0503 |
| age_52w_high | 0.0110 | 0.0690 | 0.5095 | 0.0057 | 0.0286 |
| ret_6m | 0.0056 | 0.0314 | 0.5627 | 0.0241 | -0.0543 |
| vol_20d | 0.0042 | 0.0187 | 0.4981 | -0.0138 | 0.0628 |
| max_dret_21 | 0.0026 | 0.0136 | 0.4981 | -0.0108 | 0.0461 |
| vol_ratio_20_60 | -0.0026 | -0.0255 | 0.4620 | -0.0079 | 0.0146 |
| updown_21 | -0.0035 | -0.0291 | 0.4962 | 0.0049 | -0.0307 |
| ret_3m | -0.0036 | -0.0216 | 0.5266 | 0.0148 | -0.0632 |
| volu_trend | -0.0046 | -0.0598 | 0.4734 | -0.0081 | 0.0069 |
| mom_12_1 | -0.0048 | -0.0247 | 0.5209 | 0.0019 | -0.0266 |
| higher_lows6 | -0.0054 | -0.0408 | 0.4791 | 0.0095 | -0.0537 |
| dist_52w_high | -0.0054 | -0.0260 | 0.5133 | 0.0168 | -0.0772 |
| ret_12m | -0.0080 | -0.0404 | 0.5228 | 0.0025 | -0.0421 |
| range_contr | -0.0103 | -0.0960 | 0.4753 | -0.0045 | -0.0289 |
| ret_1m | -0.0116 | -0.0744 | 0.4810 | -0.0005 | -0.0477 |

### By year (mean IC within year)

| feature | 2005 | 2006 | 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ret_1m | 0.025 | -0.059 | 0.075 | -0.096 | -0.062 | -0.065 | -0.018 | -0.035 | -0.017 | 0.001 | 0.050 | -0.023 | 0.067 | 0.019 | -0.049 | 0.030 | -0.084 | -0.030 | 0.019 | -0.008 | 0.021 |
| ret_3m | 0.063 | -0.067 | 0.138 | -0.093 | -0.051 | -0.081 | -0.044 | -0.074 | 0.023 | -0.043 | 0.071 | -0.059 | 0.097 | 0.026 | 0.004 | -0.011 | -0.091 | -0.001 | 0.075 | 0.026 | 0.022 |
| ret_6m | 0.082 | -0.063 | 0.143 | -0.031 | -0.155 | -0.048 | -0.088 | -0.029 | 0.070 | -0.028 | 0.120 | -0.094 | 0.090 | 0.009 | 0.123 | -0.114 | -0.104 | 0.003 | 0.105 | 0.080 | 0.057 |
| ret_12m | 0.063 | -0.099 | 0.118 | -0.008 | -0.308 | -0.005 | 0.008 | 0.015 | 0.027 | -0.052 | 0.152 | -0.196 | 0.137 | -0.074 | 0.130 | -0.226 | -0.070 | -0.028 | 0.116 | 0.101 | 0.043 |
| mom_12_1 | 0.057 | -0.089 | 0.101 | 0.012 | -0.285 | 0.010 | 0.015 | 0.019 | 0.028 | -0.056 | 0.140 | -0.185 | 0.125 | -0.088 | 0.144 | -0.230 | -0.050 | -0.022 | 0.113 | 0.111 | 0.039 |
| dist_52w_high | -0.013 | -0.020 | 0.153 | 0.050 | -0.322 | -0.068 | 0.098 | -0.004 | -0.037 | -0.008 | 0.190 | -0.146 | 0.061 | 0.051 | 0.088 | -0.254 | -0.066 | -0.004 | 0.073 | 0.096 | -0.026 |
| dist_52w_low | 0.126 | -0.152 | 0.114 | -0.181 | 0.137 | 0.004 | -0.115 | -0.030 | 0.103 | -0.037 | 0.059 | -0.063 | 0.144 | -0.061 | 0.043 | 0.133 | -0.067 | -0.017 | 0.139 | 0.053 | 0.118 |
| age_52w_high | 0.007 | 0.107 | -0.113 | -0.010 | 0.153 | 0.010 | -0.046 | 0.012 | 0.016 | 0.021 | -0.111 | 0.176 | -0.066 | 0.027 | -0.084 | 0.212 | 0.094 | 0.027 | -0.098 | -0.073 | -0.037 |
| vol_20d | 0.141 | -0.115 | -0.071 | -0.212 | 0.267 | 0.083 | -0.186 | 0.034 | 0.113 | -0.034 | -0.207 | 0.095 | 0.053 | -0.134 | -0.089 | 0.317 | -0.076 | -0.010 | 0.087 | -0.097 | 0.143 |
| vol_ratio_20_60 | -0.011 | 0.032 | -0.026 | -0.021 | -0.002 | 0.005 | 0.019 | -0.010 | -0.040 | -0.014 | 0.008 | 0.001 | -0.020 | -0.007 | -0.011 | 0.013 | -0.020 | 0.013 | 0.010 | 0.007 | 0.021 |
| volu_trend | -0.036 | 0.011 | -0.064 | -0.040 | 0.040 | -0.027 | 0.015 | 0.018 | -0.021 | -0.022 | 0.017 | 0.004 | -0.020 | -0.015 | -0.002 | 0.035 | 0.034 | -0.003 | -0.040 | 0.015 | 0.001 |
| updown_21 | 0.035 | -0.039 | 0.098 | -0.075 | -0.055 | -0.065 | -0.001 | -0.018 | -0.035 | 0.011 | 0.053 | -0.011 | 0.058 | 0.008 | -0.003 | -0.025 | -0.045 | -0.032 | 0.021 | 0.021 | 0.029 |
| max_dret_21 | 0.111 | -0.113 | -0.064 | -0.198 | 0.215 | 0.065 | -0.143 | 0.020 | 0.089 | -0.013 | -0.137 | 0.087 | 0.058 | -0.107 | -0.092 | 0.257 | -0.092 | 0.001 | 0.068 | -0.080 | 0.134 |
| beta_120 | 0.148 | -0.094 | -0.037 | -0.216 | 0.262 | 0.110 | -0.263 | 0.018 | 0.183 | -0.073 | -0.218 | 0.244 | 0.184 | -0.168 | -0.108 | 0.368 | -0.083 | 0.008 | 0.160 | -0.072 | 0.204 |
| corr_120 | 0.020 | 0.038 | 0.018 | -0.064 | 0.020 | 0.087 | -0.092 | -0.016 | 0.038 | -0.072 | 0.051 | 0.176 | 0.189 | -0.103 | -0.028 | 0.045 | -0.037 | 0.039 | 0.185 | -0.024 | 0.168 |
| range_contr | -0.011 | -0.003 | 0.012 | -0.043 | -0.128 | -0.008 | 0.019 | -0.048 | -0.040 | 0.019 | 0.054 | -0.003 | -0.009 | -0.016 | 0.029 | -0.012 | 0.019 | 0.001 | -0.026 | -0.018 | -0.004 |
| higher_lows6 | 0.008 | -0.062 | 0.071 | -0.075 | -0.059 | -0.074 | 0.002 | -0.060 | -0.011 | -0.017 | 0.031 | -0.043 | 0.056 | 0.042 | 0.039 | -0.033 | -0.081 | -0.038 | 0.081 | 0.041 | 0.077 |
| dd_from_ath | 0.001 | 0.002 | 0.182 | 0.058 | -0.313 | -0.010 | 0.179 | -0.068 | -0.067 | 0.061 | 0.213 | -0.122 | 0.060 | 0.090 | 0.168 | -0.250 | -0.057 | -0.024 | 0.094 | 0.139 | 0.002 |

| feature | years_IC>0 | n_years |
|---|---|---|
| ret_1m | 9 | 21 |
| ret_3m | 10 | 21 |
| ret_6m | 11 | 21 |
| ret_12m | 11 | 21 |
| mom_12_1 | 13 | 21 |
| dist_52w_high | 9 | 21 |
| dist_52w_low | 12 | 21 |
| age_52w_high | 12 | 21 |
| vol_20d | 10 | 21 |
| vol_ratio_20_60 | 10 | 21 |
| volu_trend | 10 | 21 |
| updown_21 | 9 | 21 |
| max_dret_21 | 11 | 21 |
| beta_120 | 11 | 21 |
| corr_120 | 13 | 21 |
| range_contr | 7 | 21 |
| higher_lows6 | 10 | 21 |
| dd_from_ath | 13 | 21 |

Stability reading:
- `corr_120` is the most *stable* (IC IR 0.20, 13/21 years positive, positive in both
  regimes) but its payoff is mean return, not parabolic capture.
- `beta_120` / `vol_20d` / `max_dret_21` have huge but oscillating yearly ICs
  (beta: +0.37 in 2020, -0.26 in 2011) — their sign tracks the market regime, hence
  near-zero unconditional IR. Use them conditioned, or for the parabolic tail only.
- `dist_52w_low` and `dd_from_ath`/`mom_12_1` are mirror cyclicals; `dist_52w_low`
  is the most regime-robust of the trendish ones (only -0.008 in risk-off).
- All compression/volume/pattern features (`vol_ratio_20_60`, `volu_trend`,
  `updown_21`, `higher_lows6`, `range_contr`) are dead in every cut — drop them.

## 5. Profile of the top-1% forward-6m winners

Per sampled date, stocks with fwd6 in the top 1% cross-section (1,091 pooled obs;
median fwd6 of the group +65%). `med_pct_winners` = median cross-sectional
percentile of the characteristic among winners on the signal date (0.50 = looks
like the median stock).

| feature | med_pct_winners | med_raw_winners | med_raw_all |
|---|---|---|---|
| ret_1m | 0.461 | 0.007 | 0.012 |
| ret_3m | 0.473 | 0.028 | 0.032 |
| ret_6m | 0.495 | 0.066 | 0.061 |
| ret_12m | 0.388 | 0.057 | 0.115 |
| mom_12_1 | 0.399 | 0.055 | 0.107 |
| dist_52w_high | 0.233 | -0.162 | -0.087 |
| dist_52w_low | 0.702 | 0.402 | 0.277 |
| age_52w_high | 0.648 | 104.500 | 66.000 |
| vol_20d | 0.862 | 0.359 | 0.238 |
| vol_ratio_20_60 | 0.484 | 0.940 | 0.962 |
| volu_trend | 0.485 | 0.964 | 0.970 |
| updown_21 | 0.452 | 0.524 | 0.524 |
| max_dret_21 | 0.798 | 0.045 | 0.030 |
| beta_120 | 0.782 | 1.323 | 0.981 |
| corr_120 | 0.368 | 0.511 | 0.560 |
| range_contr | 0.429 | 0.376 | 0.394 |
| higher_lows6 | 0.438 | 3.000 | 3.000 |
| dd_from_ath | 0.183 | -0.335 | -0.139 |

The median future top-1% winner on its signal date:
- sits at the **86th percentile of 20d realized vol** (36% ann. vs 24% for the median stock)
- **78th percentile beta** (1.32 vs 0.98), had a **4.5% best day** in the last month (80th pct)
- is **-34% from its ATH** (18th pct of drawdown) and **-16% below its 52w high** (23rd pct)
- yet has **already bounced +40% off its 52w low** (70th pct) and its 52w high is ~5 months stale
- has **no momentum signature**: ret_1m/3m/6m percentiles 0.46-0.50, ret_12m 0.39
- is a **low-correlation** name (37th pct, 0.51 vs 0.56)
- volume trend, up-day ratio, compression, higher-lows: all ~50th pct — invisible on those axes.

So the archetype is a volatile, high-beta, idiosyncratic name that crashed, has already
turned off its low, but hasn't yet rebuilt any conventional momentum reading. The
characteristics that *don't* distinguish winners are exactly the chart-pattern ones.

## 6. Biweekly relative-strength persistence (formation F periods, payoff at lag L)

Mean Spearman IC of formation-return rank vs the single 10-day-period return L
biweekly periods ahead (members only, ~530 grid dates):

| formation | L=1 | L=2 | L=3 | L=4 | L=6 | L=9 | L=13 |
|---|---|---|---|---|---|---|---|
| F=1p(10d) | -0.0190 | -0.0016 | 0.0009 | -0.0008 | 0.0015 | -0.0033 | -0.0094 |
| F=2p(20d) | -0.0129 | -0.0036 | -0.0006 | -0.0104 | 0.0085 | -0.0062 | 0.0013 |
| F=3p(30d) | -0.0114 | -0.0030 | -0.0088 | -0.0090 | 0.0060 | -0.0027 | 0.0044 |
| F=6p(60d) | -0.0145 | -0.0009 | -0.0021 | -0.0052 | 0.0020 | 0.0048 | 0.0116 |
| F=13p(130d) | -0.0035 | 0.0032 | 0.0021 | 0.0051 | 0.0119 | 0.0102 | 0.0121 |
| F=26p(260d) | 0.0053 | 0.0078 | 0.0083 | 0.0076 | 0.0098 | 0.0094 | 0.0120 |

Takeaways:
- **Lag 1 is reversal territory** for every formation up to 6 months: F=1 (10d)
  IC -0.019 at L=1; even 6m strength is -0.0035 in the immediately following period.
  A biweekly rebalance that buys this period's winners for next period fights a
  systematic headwind — skip the most recent 1-2 weeks in any formation window.
- **Persistence lives at long formation + long lag**: 12m formation (F=26) is
  positive at every lag and rising (+0.012 by L=13 ≈ 6 months out); 6m formation
  turns positive from L=2 and peaks at L=6-13 (+0.012).
- Sweet spot at biweekly cadence: **form on 6-12 months, skip ~2 weeks, expect the
  payoff to accrue over the following 3-6 months** — i.e. exactly the 12-1-style
  construction, and consistent with slow rebalancing (holding winners for months,
  not re-ranking every period).
- Magnitudes are small per 10d period (IC ~0.01) but compound across ~13 periods.

## 7. Bottom line for signal construction

1. **Parabolic moves are found on the high-energy axis, not the quiet one.**
   vol_20d / beta_120 / max_dret_21 give 12-23x lift in P(fwd6 > +50%) from bottom
   to top decile. Compression ("coiled spring") measures are empirically dead here.
2. **beta_120 is the best single all-weather characteristic** (only one with
   positive excess mean AND fat right tail AND positive IC in both regimes).
3. **Condition on regime.** Below the 200dma, buy deep-drawdown high-vol names
   (P(parab) 17-18%, excess +7-8%); above it, buy strength — bounce off 52w low,
   beta, 6m return, correlation. A regime switch roughly doubles the usable IC of
   every cycle characteristic.
4. **dist_52w_low ("already turned") is the most regime-robust trend measure** and
   beats dist_52w_high/momentum as an unconditional input.
5. **corr_120 is a stability anchor** (best IC IR) to mix with the lottery axes —
   it points away from parabolic names, so use it to control the left tail, not to
   find rockets.
6. **At biweekly cadence use 6-12m formation with a 1-2 week skip**; expect payoff
   over 3-6 months; do not chase 2-week winners (reliable short-term reversal).

## Caveats

- Universe survivorship: panel covers ~57% of 2005 members rising to ~99% today;
  high-vol deep-drawdown losers that *delisted* are under-represented, so the
  risk-off rebound effects (sec. 3) are likely overstated — verify any strategy
  vs the random-pick control in `protocol.evaluate_signal`.
- A handful of recycled-ticker series survive the 30-day-gap guard (e.g. PTV, HAR
  post-2017); rank statistics are robust to them, raw means less so.
- fwd6/fwd12 measured close-to-close from d; the engine executes at next open —
  effect sizes here are upper bounds before costs/slippage.
- Decile means pool overlapping 6m windows sampled monthly; per-year ICs are the
  better guide to statistical reliability than pooled t-stats.
