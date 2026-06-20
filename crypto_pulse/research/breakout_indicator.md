# Breakout indicator experiment — TTM Squeeze (best-effort from the X posts)

Volatility-squeeze breakout (BB inside KC -> release -> trade momentum direction). HL era, net of 4.5bps + funding, IS/OOS. I'm inferring this is the indicator from @onlybreakouts/@polishquant — correct me if not.

Squeeze fires on ~1665 coin-days (2.7% of coin-days).

| variant | Sharpe | IS | OOS | maxDD | corr to STRATA |
|---|---|---|---|---|---|
| A. TS-directional breakout | **+0.26** | +0.86 | -0.74 | -18% | +0.11 |
| B. XS market-neutral | **-0.89** | -1.52 | +0.04 | -36% | +0.05 |

## Add breakout (TS) to STRATA

- NOT admitted (IS +0.86/OOS -0.74, corr +0.11). The squeeze breakout overlaps existing trend/squeeze sleeves or is cost-blocked.

## Verdict

- This is the canonical volatility-squeeze breakout. If the posts use different parameters (ORB, Donchian-N, ATR-multiple, volume filter) or a different indicator, paste the rules and I'll match them exactly.
