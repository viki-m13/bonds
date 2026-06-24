# Hardening the funding-carry leg — battery + tail-taming (honest)

Carry can't get pre-HL validation (funding starts ~2023), but here's its HL-internal overfit battery and tail-taming.

## A) Carry overfit battery (HL era)

- **Lookback plateau:** 3d:+1.23 | 5d:+1.40 | 7d:+1.46 | 10d:+1.48 | 14d:+1.55 | 21d:+1.56 | 30d:+1.46
  -> 7/7 lookbacks > 1.0 Sharpe.
- **Coin bootstrap** (15× 70% subsets): median +1.52, 5th-pct +0.76, min +0.30.
- **Shuffle null** (permute funding across coins): real +1.55 vs null max +0.93, mean +0.09.
- **Walk-forward** (4 folds): +1.71, -0.49, +2.30, +2.49 (3/4 positive).

## B) Tail-taming overlays on carry

| overlay | HL | OOS | dOOS | CAGR | maxDD |
|---|---|---|---|---|---|
| base carry(14d) | +1.55 | +2.41 | +0.00 | +24% | -18% |
| +volgate | +1.39 | +2.19 | -0.21 | +21% | -14% |
| +ddfloor | +1.45 | +2.24 | -0.17 | +17% | -8% |
| +winsor | +1.09 | +1.58 | -0.82 | +16% | -18% |
| +volgate+ddfloor | +1.17 | +1.96 | -0.45 | +13% | -8% |

## C) TIDE + tamed carry

- Best tail-tamer: **+ddfloor** (carry maxDD -18% -> -8%, OOS +2.41 -> +2.24).
- TIDE alone: HL +2.23, OOS +2.29, maxDD -9%.
- **TIDE + tamed carry: HL +2.46, OOS +2.80 (+0.52), CAGR +42%, maxDD -9%, WF +2.6, +0.9, +3.1, +3.0.**

## Verdict

- **Carry is shakier than TIDE on its own battery** — treat with extra caution.
- **Tail-taming works:** +ddfloor cuts carry maxDD to -8% while keeping OOS +2.24. TIDE + tamed carry: OOS +2.80 (vs +2.29 alone), maxDD -9%.
- **Still phase-2, not certified:** the hard blocker is unchanged — no pre-HL funding data means no independent-regime confirmation. But carry now has an internal battery + a tamed tail, so it's a *stronger, better-understood* phase-2 sleeve to paper-trade. Certified book stays TIDE alone.
