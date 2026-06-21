# SUMMIT — Prop Market-Neutral Alpha Engine (SHORTING + LEVERAGE)

> **Status: ARCHIVED for future implementation. NOT the current deployment.**
> Requires shorting + leverage/margin — out of scope for the current long-only,
> no-margin mandate. Recorded here so we can pick it up and improve it later.
> Build/validate scripts: `dca/research/exp73_factorzoo.py`, `exp74_alpha.py`,
> `exp75_reality.py`, `exp76_mlls.py`. Data: PIT survivorship-clean Tiingo +
> SEC fundamentals/revenue + Form-4 insider. Window 2012–2025 (ML 2015–2025).

## What it is
A dollar-neutral long/short book that strips market beta to harvest pure alpha.
Two engines, both ~zero correlation to QQQ:

1. **Linear multi-factor L/S** — equal-weight combine of 11 single-factor
   decile L/S sleeves: value (B/M, E/P), Novy-Marx gross profitability (GP/A),
   sales/assets, Piotroski, ROA, ROE, momentum 12-1, 52w-high, buyback
   (−share issuance), revenue acceleration, insider cluster.
2. **ML L/S** — walk-forward HistGBM (trained only on past data) predicting
   cross-sectional return terciles from all 36 features, long top decile /
   short bottom decile.

## Validated results (PIT survivorship-clean)

| Engine | ann return | Sharpe (gross) | Sharpe (net 6%/yr borrow) | corr QQQ | maxDD |
|---|--:|--:|--:|--:|--:|
| Single best factor — Value (B/M) | 21.0% | 1.38 | — | −0.15 | — |
| Linear 11-factor L/S | 12–14% | 1.55–1.75 | ~1.0 | −0.09 | −16% |
| **ML L/S (decile)** | **46.4%** | **2.00** | **1.74** | **−0.01** | −40% |

**Portable alpha (overlay on QQQ beta):**
- QQQ + 1× linear alpha → 30% / Sharpe 1.66
- QQQ + 1× ML alpha (net) → **59% / Sharpe 2.01**
- QQQ + 2× alpha → 41–79% / Sharpe ~1.83–2.03

Regime robustness (ML L/S net Sharpe): 2015-18 **1.89**, 2019-21 0.66, 2022-25 **2.57** — never negative (linear book WAS negative 2017-20).

## Honest caveats (do not strip)
- Sharpe 2.0 is **gross of commissions / slippage / market impact**; decile L/S
  turns over ~100%+/month across ~200 small/mid-cap names. Realistic net is
  materially lower.
- Requires **shorting hard-to-borrow small/mid-caps** (borrow cost modeled at
  6%/yr; squeeze + availability risk on top) and **leverage/margin** for the
  overlay.
- Pure L/S max drawdown ~−40%.
- Feature-set choice used some full-sample judgment (training itself is OOS).

## Turnover & cost-aware analysis — DEPLOYMENT-READY (exp78)
Realistic net-of-cost results for the ML L/S, incl 6%/yr borrow on the short leg.
Rank-buffering (hold a name until it leaves a 2× band) is the key efficiency lever
— it cuts turnover AND raises Sharpe (less noise trading).

| Config | turnover/mo | gross Sharpe | net @10bps | net @20bps | net ann |
|---|--:|--:|--:|--:|--:|
| monthly decile | 160% | 2.02 | 1.68 | 1.60 | 37% |
| monthly + buffer2× | 99% | 2.22 | 1.91 | 1.85 | 43% |
| quarterly decile | 80% | 1.98 | 1.67 | 1.63 | 38% |
| **quarterly + buffer2× (DEPLOY)** | **58%** | 2.12 | 1.83 | **1.80** | **43%** |
| monthly quintile | 137% | 1.74 | 1.35 | 1.26 | 25% |

**Cost robustness (quarterly+buffer2×, incl borrow):** net Sharpe 1.87 / 1.85 /
1.83 / 1.80 / 1.77 at 0 / 5 / 10 / 20 / 30 bps-per-side. maxDD ~−37%.
=> **Deployable config: quarterly rebalance, decile L/S, 2× rank buffer — net
Sharpe ~1.8 after realistic costs.** Capacity: equal-weight decile of the liquid
(>$3, >$5 short) PIT universe = a few hundred names/side; size limited by ADV of
the smaller names (cap each position at a few % of 20-day ADV; large AUM should
tilt to the larger-cap half). Borrow: restrict shorts to easy-to-borrow to realize
the modeled 6%/yr.

## v2 — borrow-aware shorts: the realized improvement (exp108–109)
Worked the roadmap empirically on the deployable book (quarterly + buffer2×,
net 10bps + 6%/yr borrow, 2015–2025, dev 2015–22 / locked holdout 2023–25):

| Variant | net CAGR | net Sharpe | maxDD | dev Sh | holdout Sh | corr QQQ |
|---|--:|--:|--:|--:|--:|--:|
| v1 base (short ≥$5) | 44% | 1.83 | −37% | 1.65 | 2.30 | −0.30 |
| **v2 — short ≥$10 mcap (DEPLOY)** | **52%** | **2.16** | **−34%** | **1.90** | **2.83** | −0.28 |
| sector-neutral ML | 23% | 1.01 | −63% | 0.67 | 1.87 | −0.43 |
| ensemble ML + linear | 35% | 1.43 | −58% | 0.97 | 2.69 | −0.34 |
| v2 + vol-target 12% | 36% | 2.02 | −27% | 1.75 | 2.75 | −0.17 |

**What worked — borrow-aware shorting.** Restricting the short leg to ≥$10 mcap
names (the only ones reliably borrowable at the modeled ~6%/yr) lifts net Sharpe
**1.83 → 2.16** with *no* drawdown cost (−37% → −34%) and holds OOS (holdout
2.30 → 2.83). Micro-cap shorts added squeeze risk and noise, not edge — so the
deployability constraint and the performance improvement point the same way.
This is now the deployable SUMMIT (full-sample compounded: CAGR 62% /
Sharpe 2.16 / maxDD −34% / corr −0.28).

**What did NOT transfer — sector-neutralization.** It helped the *linear* factor
composite (Sharpe 0.83→0.93, exp99) but actively *hurts* the ML book
(1.83→1.01, DD →−63%): the gradient-booster already prices sector context, so
demeaning within sector strips real signal. Likewise the ML+linear ensemble
dilutes (corr 0.76) → 1.43. Neither is adopted.

**Optional — vol-targeting.** A trailing-6m realized-vol target of 12% (cap 2×)
on v2 cuts maxDD to −27% for a modest Sharpe give-back (2.16→2.02); the book's
native vol is ~24%/yr. Use it if drawdown, not raw return, is the binding
constraint — you can re-lever a vol-targeted book to restore return.

## Roadmap (remaining)
1. ~~Turnover/cost-aware construction~~ — DONE (quarterly + buffer2×).
2. ~~Sector neutralization~~ — TESTED, rejected for the ML book (hurts).
3. ~~Borrow-aware shorting~~ — DONE, adopted as v2 (the main win).
4. ~~Vol-targeting~~ — TESTED, available as optional DD-control overlay.
5. **New alpha data** — 13F institutional flows, short-interest/FTD, options
   skew, **analyst estimate-revisions** (the one durable factor SUMMIT lacks;
   needs paid data) — the remaining high-EV, un-built lever.
6. **Ensemble** ML + linear — TESTED, dilutes (corr 0.76); not adopted.
