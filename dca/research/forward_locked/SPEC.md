# LOCKED-GTAA-v1 — Forward-Locked Strategy (FROZEN 2026-06-17)

A commitment device to test "can a rule beat QQQ?" with **zero selection bias**.
After 22 experiments where every in-sample "beat" reduced to bias (survivorship /
regime / parameter / universe), the only honest test left is to lock a rule with
**no remaining degrees of freedom** and judge it on data dated **strictly after
the freeze**. Full spec + runnable evaluator: `locked_gtaa_v1.py`
(`param_hash=8e4def0d77eba04d`). Nothing in it may change.

## The rule (textbook defaults, no optimization)
- **Universe** (objective: major investable asset classes, oldest/largest liquid
  ETF each; *includes likely losers* — commodities, EM, long bonds — so it is NOT
  cherry-picked): SPY QQQ IWM EFA EEM VNQ GLD DBC TLT IEF. **+BTC variant** adds
  BTC-USD as one more asset, treated identically (so the crypto question is
  itself tested forward, not assumed).
- **Signal**: trailing 12-month return. **Eligible** if 12m return > 0 and > IEF.
- **Hold**: equal-weight top-3 eligible; unfilled slots → IEF; none → 100% IEF.
- Monthly rebalance, long-only, no leverage.
- **Benchmarks**: QQQ buy&hold (primary), SPY, 60/40.

## Success criteria (judged ONLY post-freeze, ≥3y)
- **Primary (risk-adjusted):** maxDD ≤ 0.7× QQQ maxDD **and** Calmar ≥ QQQ Calmar.
- **Secondary (honesty):** CAGR within 3pp of QQQ.
- **Stretch (a genuine bias-free beat, NOT expected):** CAGR ≥ QQQ.

## Honest prior (recorded before any forward data exists)
I expect this to **track or modestly trail QQQ on return with materially lower
drawdown** — I do NOT expect a raw-return beat. If one happens, it is real,
because nothing was fit.

## In-sample context (CONTAMINATED — not a claim; for reference only)
| | CAGR | maxDD | Sharpe | Calmar |
|---|---|---|---|---|
| LOCKED-GTAA (no crypto) | 9.5% | −23% | 0.79 | 0.41 |
| LOCKED-GTAA +BTC | 18.5% | −29% | 0.97 | 0.64 |
| QQQ buy&hold | 13.6% | −50% | 0.76 | 0.27 |
| 60/40 | 8.5% | −29% | 0.95 | 0.29 |

## How to judge it (future)
Run `evaluate(include_btc=False, start='2026-06-17')` (and the +BTC variant) once
≥3y of post-freeze data exist. Score against the criteria above. Do not re-tune.
