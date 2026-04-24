# APEX Progression Log

Starting point: OOS Sharpe 0.87 (v1)
Final: OOS Sharpe 1.98 (v30)
Progression: +1.11 SR (+127% improvement)

## Version timeline

| Version | # Sleeves | OOS SR | Key change |
|---|---|---|---|
| v1 | 7 | 0.87 | Initial (trend-focused) |
| v7 EW | 7 | 0.97 | Equal-weight blend |
| v14 | 11 | 1.34 | + DualBear/Calendar/VRP/Crypto |
| v15 | 11 | 1.55 | + INVERSE + MULTI_CRYPTO |
| v17 | 14 | 1.55 | + FOMC/PCA/BUY_FEAR/VOL_OF_VOL |
| v18 | 16 | 1.63 | + HMM/Breakout52/Divergence |
| v20 | 24 | 1.65 | + 6 macro sleeves |
| v22 | 8 | 1.76 | **Simplified to 8 strongest** |
| v23 | 4 | 1.89 | **Ultra-lean 4** |
| v24 | 3 | 1.91 | **Minimum 3 (exhaustive search)** |
| v28 | 4 | 1.97 | **+ACCEL_MOM (novel math)** |
| v30 | 6 | 1.98 | **+SKEW_MOM + HURST (novel math)** |

## Final v30 architecture

6 LETF sleeves (50% book, equal-weight) + MULTI_CRYPTO (50%):
- PX_HELIOS: Phoenix-exact 6m momentum on 13 unlevered → 3x LETF
- HMM_REGIME: 3-state Gaussian HMM on SPY returns
- DIVERGENCE: SPY-QQQ relative strength regime
- ACCEL_MOM: 2nd-derivative momentum (novel)
- SKEW_MOM: skewness-signed momentum (novel)
- HURST: Hurst exponent regime (novel)
- MULTI_CRYPTO: BTC + ETH + SOL

## vs Phoenix comparison (2010-2026)

| Metric | Phoenix | APEX v30 |
|---|---|---|
| Full SR | 2.33 | 1.53 |
| OOS SR | 2.14 | **1.98** |
| IS SR | 2.52 | 1.05 |
| 2008 MDD | -1.3% | -7.5% |
| 2020 COVID SR | 3.35 | 3.25 |
| 2022 SR | +0.22 | -1.35 |

**Gap to Phoenix OOS: 0.16 SR** (was 1.27 at v1).

## Key lessons

1. **Simple beats complex**: 6 strong uncorrelated sleeves > 24 weak ones
2. **Novel math helps**: ACCEL_MOM, SKEW_MOM, HURST added +0.07 over base
3. **Correlation matters more than SR**: low-corr sleeves win
4. **Crypto is the ultimate diversifier**: 0.05 correlation to LETFs
5. **Causal signals only**: avoid lookahead bias (SavGol bug caught)
