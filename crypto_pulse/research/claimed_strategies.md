# Publicly-claimed strategies on crypto + downside diversification

Net of 4.5bps taker, IS=first60/OOS=last40 of HL era. IBS_MR and ROTMOM are long-only; shown vol-targeted to 12% for comparison, with raw exposure noted.

## Standalone (vol-targeted to 12%)

| strategy | Sharpe | IS | OOS | CAGR | maxDD | Calmar | avg expo |
|---|---|---|---|---|---|---|---|
| IBS mean-reversion | **-0.08** | +0.05 | -0.29 | -2% | -18% | -0.10 | 90% |
| rotational momentum | **+0.63** | +0.76 | +0.37 | +8% | -23% | 0.36 | 67% |
| grand stack (ref) | **+1.59** | +1.21 | +2.15 | +28% | -10% | 2.65 | ~100% |

Correlation to grand stack: IBS -0.17, ROT -0.18

## Downside mitigation — blend grand stack + IBS dip-buyer

Equal-risk blends, vol-targeted to 12%. The low-exposure long-only MR sleeve buys oversold crashes, so it should cushion the directional book's drawdowns.

| book | Sharpe | CAGR | maxDD | Calmar | worst day |
|---|---|---|---|---|---|
| grand stack alone | **+1.59** | +28% | -10% | 2.65 | -6.0% |
| grand + IBS (50/50 risk) | **+0.24** | +2% | -14% | 0.16 | -3.0% |
| grand + IBS + ROT (1/3 each) | **+0.50** | +6% | -14% | 0.41 | -3.4% |

## Verdict

- IBS mean-reversion on crypto: standalone Sharpe -0.08 (IS/OOS in table), correlation to the book -0.17, avg exposure 90%. It does not improve the blended Calmar here.
