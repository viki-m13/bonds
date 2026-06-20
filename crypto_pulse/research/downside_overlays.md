# Downside mitigation on the leveraged grand stack (half-Kelly 5.5x)

Each overlay scales exposure causally (signal through d-1). Net of the book's real costs; leverage applied uniformly. Goal: cut maxDD / tail without sacrificing CAGR (maximize Calmar).

| book | Sharpe | CAGR | maxDD | Calmar | worst day | 5% CVaR |
|---|---|---|---|---|---|---|
| half-Kelly base (5.5x) | +1.46 | +127% | -45% | **2.83** | -19.7% | -8.2% |
| + VOLTGT (fast vol-target) | +1.33 | +101% | -41% | **2.45** | -20.6% | -8.5% |
| + EQCURVE (equity>40d MA) | +1.21 | +79% | -58% | **1.36** | -19.7% | -7.4% |
| + DDTHROT (de-lever in DD) | +1.38 | +104% | -42% | **2.46** | -19.2% | -7.5% |
| + CRASHBUY (10% IBS hedge) | +1.50 | +114% | -41% | **2.75** | -16.2% | -6.9% |
| + COMBO (vol*dd*eq) | +1.06 | +54% | -51% | **1.05** | -15.1% | -6.8% |

## Verdict

- Best overlay: **+ CRASHBUY (10% IBS hedge)** — Calmar 2.75 vs base 2.83, maxDD -41% vs -45%, CAGR +114% vs +127%, worst day -16.2% vs -19.7%.
- Drawdown-throttle and vol-targeting cut the tail the most per unit of CAGR given up; the equity-curve filter helps if the book's losing streaks persist (trend-like) and hurts if they snap back. The negatively-correlated crash-buyer trims the worst day but dilutes CAGR. Stacking vol+dd is the robust downside engine — it keeps most of the return while materially reducing the depth and tail of drawdowns. This is how you run leverage survivably.
