# Orthogonal price-factor legs — certifiable diversification (honest)

Price legs have full history, so a low-corr leg positive on BOTH OOS and pre-HL is CERTIFIABLE (unlike carry).

## Standalone factor legs

| leg | HL | IS | OOS | pre-HL | corr→TIDE(HL) | maxDD |
|---|---|---|---|---|---|---|
| TIDE | +2.23 | +2.20 | +2.29 | +1.35 | +1.00 | -9% |
| LOTTERY | +0.19 | +0.48 | -0.21 | -0.36 | -0.29 | -28% |
| LOWBETA | +1.46 | +1.41 | +1.56 | +0.39 | +0.15 | -14% |
| STREV | -0.91 | -1.04 | -0.74 | -0.75 | -0.57 | -34% |
| SKEW | -0.78 | -0.54 | -1.15 | -0.89 | -0.31 | -31% |

## Verdict

- Certifiable diversifiers (corr<0.5, OOS>0, pre-HL>0): **LOWBETA**.
- TIDE alone: OOS +2.29, pre-HL +1.35. **TIDE+LOWBETA: OOS +2.45 (+0.16), pre-HL +1.25 (-0.09), HL +2.53, full +1.53, maxDD -8%, WF +2.4, +1.4, +3.4, +2.9.**
- **Combo does NOT robustly beat TIDE on both OOS and pre-HL — diversification benefit is not certified.**
- (Carry remains the most orthogonal leg but is HL-only; these price legs are the test of whether a *certifiable* second leg exists in price data. Beyond this: L4 order flow.)
