# SUMMIT on ETF universes (sector / all-types / leveraged + inverse)

Question: how does SUMMIT do if it picks ETFs instead of stocks — sector ETFs,
a huge all-types ETF pool, and with leveraged/inverse ETFs added?
244-window grid vs QQQ/SPY DCA, 5 bps. `python research/etf_universe_study.py`.

## Result — SUMMIT does NOT work on ETFs

| universe | beat QQQ | beat SPY | median vs QQQ | worst | grew to | random beats QQQ | what it ends up holding |
|---|---|---|---|---|---|---|---|
| Sector ETFs (~20) | 11% | 30% | −18.3% | −58% | 4.3× | 3% | SMH 44%, XLF 25%, XLE 23% |
| All unlevered ETFs (218) | 1% | 8% | −21.0% | −59% | 3.9× | 2% | **QQQ 44%, SPY 33%, GLD 13%** |
| All ETFs + leveraged + inverse (254) | 1% | 3% | −21.4% | −56% | 4.1× | 1% | **QQQ 41%, SPY 33%, GLD 17%** |

(SUMMIT on the S&P 500, for comparison: 93% beat QQQ, +28.8% median, −10.6%
worst, 20.0×.)

It loses to a plain QQQ DCA in ~89-99% of windows on every ETF universe.

## Why — the mega-cap tilt degenerates on ETFs

SUMMIT's edge on stocks is "momentum **× a strong dollar-volume (mega-cap)
tilt**" — pick the biggest leaders out of ~500 names. On ETFs the
highest-dollar-volume "names" are **SPY and QQQ themselves**, so the tilt makes
SUMMIT buy SPY + QQQ + GLD and a few big sector funds. That is just a
worse-diversified version of the benchmark (QQQ/SPY diluted with GLD, EEM,
bonds — which drag), so it can't beat a pure QQQ DCA. The random-pick control
also beats QQQ only 1-3% of the time: **most ETFs underperformed QQQ over
2009-2026**, so the pool itself is a losing draw against QQQ.

## Leveraged ETFs don't get picked — and don't rescue it

Even with TQQQ / SOXL / UPRO in the pool, the size tilt favors SPY/QQQ (far
higher dollar volume), so the leveraged funds are essentially never selected.
Turning the tilt **off** to let momentum chase them:

| size tilt | beat QQQ | median | worst | grew to | top holdings |
|---|---|---|---|---|---|
| 5.0 (default) | 1% | −21.4% | −56% | 4.1× | QQQ 41%, SPY 33%, GLD 17% |
| 1.0 | 2% | −26.0% | −66% | 3.1× | GLD 26%, QQQ 25%, USO 23% |
| 0.0 (pure momentum) | 32% | −10.4% | −50% | 8.0× | XLE 46%, USO 45%, GBTC 7% |

With no size tilt it becomes a commodity/energy/crypto momentum rotation
(XLE, USO, GLD, GBTC) — better than the tilted version (32% beat QQQ, 8×) but
still loses to QQQ DCA two-thirds of the time, with −50% drawdowns. It does NOT
turn into a TQQQ compounding machine; the multi-horizon momentum + regime logic
spreads picks across whatever ran hardest, which over this window was oil and
gold as often as tech.

## Verdict

**SUMMIT is a stock-selection strategy and should stay on stocks.** Its size
tilt is meaningful for equities (mega-cap stocks lead) but degenerate for ETFs
(the biggest "name" is the index fund itself), and ETF momentum rotation in
general underperformed simply dollar-cost-averaging into QQQ over this era.
Leveraged ETFs are available but neither selected by default nor a rescue when
forced in. If the goal is a leveraged strategy, the repo's purpose-built
**PHOENIX / APEX** (LETF ensembles with proper risk management) are the right
tools — running SUMMIT's stock logic on ETFs is the wrong instrument for the
job, and the numbers say so plainly.
