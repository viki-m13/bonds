# Can intraday-derived sleeves improve STRATA?

Daily cross-sectional signals from hourly OHLCV (20 coins), tested as STRATA additions (shrunk-MV), net 4.5bps + funding. HL era, IS/OOS.

| candidate sleeve | standalone Sharpe | IS | OOS | corr to STRATA | STRATA+it OOS | weight |
|---|---|---|---|---|---|---|
| STREV1 | -0.49 | -0.61 | -0.40 | +0.14 | **+1.84** | 0% |
| STREV3 | -0.68 | -0.38 | -0.99 | +0.09 | **+1.85** | 0% |
| INTRAREV | -0.60 | -0.30 | -0.90 | +0.14 | **+1.85** | 0% |
| VOLASYM | -0.76 | -0.94 | -0.60 | +0.09 | **+1.87** | 0% |
| PATHEFF | +0.38 | -0.05 | +0.81 | +0.06 | **+1.85** | 0% |

## Verdict

- STRATA baseline OOS **+1.85**. Best intraday addition: **VOLASYM** -> STRATA OOS **+1.87** (+0.02). No intraday sleeve robustly lifts STRATA OOS — the daily factors already span this.
