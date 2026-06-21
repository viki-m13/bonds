# TimesFM forecast sleeve on HL crypto — verdict: no edge

Ran Google TimesFM 2.5 (200M) zero-shot to forecast each HL coin's forward 5-day
return at weekly rebalances (71 weeks computed on CPU before stopping — enough for a
robust information-coefficient read).

## Result
- Cross-sectional IC (TimesFM forecast vs realized fwd-5d return): **mean -0.006,
  median +0.023, t-stat -0.22, 54% of weeks positive** — indistinguishable from zero.
- Forecast vs trailing-20d return correlation: **-0.17** (mild contrarian, so it isn't
  merely re-deriving momentum — it just has no cross-sectional signal).

## Conclusion
TimesFM's zero-shot forecasts carry **no usable cross-sectional information** on which
crypto coins outperform, so a TimesFM sleeve would be noise and does NOT improve
STRATA's OOS. A general-purpose time-series foundation model has no special edge on
crypto returns beyond the factors STRATA already captures — consistent with the
broader finding that crypto cross-sectional returns are near-unpredictable except via
trend/carry/BAB/flow. Model loads + runs locally (timesfm 2.5 + torch); the negative
is honest, not an infra failure.
