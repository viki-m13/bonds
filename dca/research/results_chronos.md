# Chronos-bolt re-ranking experiment — results

Design: see header of `signals_chronos.py`. Monthly signal dates 2016-01 →
2026-06; candidate set = top-30 members by 189d-skip-21 momentum; treatment =
chronos-bolt-small median 42d log-price forecast; control = the momentum
value itself on the same candidates; identical ffill/eligibility treatment.
3,761 (date,ticker) forecasts cached in `chronos_scores.parquet`.

| arm | k | win_qqq | win_spy | med_vs_qqq | worst_vs_qqq |
|---|---|---|---|---|---|
| chronos re-rank | 1 | 40% | 52% | -14.9% | -69.2% |
| momentum control | 1 | 60% | 69% | +11.4% | -60.3% |
| chronos re-rank | 2 | 35% | 54% | -12.7% | -64.6% |
| momentum control | 2 | 61% | 73% | +6.8% | -56.3% |
| chronos re-rank | 3 | 38% | 62% | -4.8% | -58.2% |
| momentum control | 3 | 64% | 76% | +17.6% | -54.9% |

(Grid windows starting before 2016 hold cash until the first signal in both
arms, hence the large worst-case numbers; the arm-vs-arm delta is the
experiment.)

## Verdict

**No.** Chronos re-ranking is decisively worse than the plain momentum it
re-ranks, at every concentration level (-15 to -22pp win-rate, median swings
of -17 to -26pp). The pretrained forecaster's 42-day median extrapolation
adds noise, not signal, on large-cap daily series — consistent with the ML
agent's finding that OHLCV cross-sections contain little beyond the momentum
direction itself.

Caveats: chronos-bolt-small only (CPU constraint); log-price inputs with
instance normalization (the defensible configuration — raw returns collapse
to zero-drift forecasts); the model's pretraining corpus may overlap the
evaluation period, which would bias *toward* finding skill — making the
negative result stronger.

Not advanced. Foundation-model forecasting is excluded from the final
strategy.
