# Basket backtest — do these signals catch pre-parabolic names?

Pick top-10 eligible names every 21 trading days; outcome = forward 126d return; parabolic = >+50%. IS rebalance date < 2016-01-01, OOS >=. Random control = 200 random top-k baskets from the same eligible pool on the same dates (survivorship-matched).

`hit` = P(pick goes parabolic); `lift` = hit/base; `excess_vs_univ` = basket mean fwd6 minus universe mean; `excess_vs_rand` = minus random-basket mean; `rand_pctile` = mean percentile of the basket within the random distribution (0.5 = no edge).

## ignition

| split | n | hit | base | lift | mean_fwd6 | exc_univ | exc_rand | t_rand | beat_rand | rand_pctile | strat_ann | rand_ann |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IS | 127 | 11.6% | 4.2% | 2.79x | +8.3% | +2.5% | +2.5% | +0.8 | 52% | 0.52 | +17.4% | +11.8% |
| OOS | 119 | 12.9% | 4.9% | 2.65x | +12.0% | +4.3% | +4.3% | +1.4 | 54% | 0.57 | +25.5% | +18.5% |

Year-by-year (pooled splits):

| year | n | hit | base | exc_vs_rand |
|---|---|---|---|---|
| 2005 | 7 | 10% | 4% | +8.6% |
| 2006 | 12 | 7% | 2% | -6.1% |
| 2007 | 12 | 6% | 2% | -2.5% |
| 2008 | 12 | 6% | 3% | -8.2% |
| 2009 | 12 | 36% | 20% | +18.4% |
| 2010 | 12 | 9% | 5% | -1.6% |
| 2011 | 12 | 8% | 1% | -4.6% |
| 2012 | 12 | 16% | 4% | +12.5% |
| 2013 | 12 | 18% | 3% | +10.1% |
| 2014 | 12 | 6% | 1% | +0.7% |
| 2015 | 12 | 6% | 1% | +2.8% |
| 2016 | 12 | 14% | 4% | +1.8% |
| 2017 | 12 | 12% | 3% | +4.7% |
| 2018 | 12 | 2% | 1% | -2.8% |
| 2019 | 12 | 5% | 1% | -0.5% |
| 2020 | 12 | 32% | 22% | +7.7% |
| 2021 | 12 | 15% | 3% | +5.9% |
| 2022 | 12 | 8% | 2% | +7.3% |
| 2023 | 12 | 5% | 3% | -0.0% |
| 2024 | 12 | 8% | 2% | +0.3% |
| 2025 | 11 | 27% | 8% | +19.6% |

## ignition_beta

| split | n | hit | base | lift | mean_fwd6 | exc_univ | exc_rand | t_rand | beat_rand | rand_pctile | strat_ann | rand_ann |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IS | 127 | 9.9% | 4.2% | 2.39x | +7.1% | +1.2% | +1.2% | +0.4 | 48% | 0.49 | +14.7% | +11.8% |
| OOS | 119 | 16.1% | 4.9% | 3.30x | +15.3% | +7.6% | +7.5% | +2.0 | 68% | 0.66 | +32.8% | +18.5% |

## ignition_noregime

| split | n | hit | base | lift | mean_fwd6 | exc_univ | exc_rand | t_rand | beat_rand | rand_pctile | strat_ann | rand_ann |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IS | 127 | 11.3% | 4.2% | 2.71x | +7.4% | +1.6% | +1.6% | +0.5 | 50% | 0.51 | +15.4% | +11.8% |
| OOS | 119 | 11.8% | 4.9% | 2.44x | +11.1% | +3.4% | +3.3% | +1.1 | 55% | 0.56 | +23.4% | +18.5% |

## practitioner_breakout

| split | n | hit | base | lift | mean_fwd6 | exc_univ | exc_rand | t_rand | beat_rand | rand_pctile | strat_ann | rand_ann |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IS | 119 | 2.6% | 1.3% | 1.95x | +4.2% | -0.7% | -0.7% | -0.5 | 45% | 0.46 | +8.5% | +9.8% |
| OOS | 116 | 2.9% | 1.8% | 1.67x | +7.0% | +1.4% | +1.4% | +0.9 | 59% | 0.56 | +14.6% | +13.4% |

## pure_energy

| split | n | hit | base | lift | mean_fwd6 | exc_univ | exc_rand | t_rand | beat_rand | rand_pctile | strat_ann | rand_ann |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IS | 127 | 10.2% | 2.7% | 3.75x | +6.8% | +1.2% | +1.2% | +0.3 | 49% | 0.47 | +14.1% | +11.1% |
| OOS | 119 | 15.7% | 3.3% | 4.78x | +15.5% | +8.7% | +8.7% | +1.9 | 63% | 0.63 | +33.4% | +13.1% |
