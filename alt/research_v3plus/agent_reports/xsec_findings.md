# Agent: cross-sectional findings (IS-only, candidates saved)

Saved to scratchpad/candidates/ (Date,ret full window; IS stats only inspected):
| file | IS SR | CAGR | Vol | MDD | TO | corr vs blend |
|---|---|---|---|---|---|---|
| gh52_hyggate_eq.csv | 1.08 | 33.3% | 31.2% | -36.1% | 41.8 | 0.42 |
| hyg_lead_sso.csv | 1.07 | 20.5% | 19.1% | -28.9% | 19.0 | 0.39 |
| gh52_vt25_hyg.csv | 0.97 | 22.1% | 23.4% | -31.8% | 42.8 | 0.40 |
| gh52_displo.csv | 0.97 | 20.3% | 21.3% | -27.8% | 36.2 | 0.47 |
| gh52_letf_rot.csv | 0.87 | 29.1% | 37.3% | -49.9% | 32.5 | 0.49 |
| sector_gh_hedge.csv | 0.87 | 17.0% | 20.6% | -28.4% | 11.5 | 0.49 |
| smh_lead_tqqq.csv | 0.64 | 17.2% | 33.6% | -50.9% | 64.4 | 0.32 |

Key insights:
- GH 52w-high proximity rank >> plain 252d momentum (0.87 vs 0.48 same book, K3 weekly).
  Could also upgrade ORION's ranking later.
- HYG>MA gate robust: MA {50,80,100,120,150,200} -> SR {1.23,1.09,1.08,0.91,0.84,0.68},
  monotone, saved mid-param (100dma) not peak.
- gh52_* variants 0.78-0.94 inter-correlated = ONE idea; hyg_lead_sso 0.62-0.75 to them;
  smh_lead_tqqq most independent (0.45-0.57).
- Dead ends: residual momentum (real but corr>0.5, SR<0.9), RS switch pairs (whipsaw),
  rank acceleration, hard proximity thresholds, trend filters on GH.
- Caveat: GH family is one engine; HYG gate turnover 19-44x/yr execution-sensitive.
