# CRT / Daily Stock Guide — honest DCA vs SPY-DCA vs QQQ-DCA

An independent, reproducible monthly-DCA reading of the **CRT / Daily Stock
Guide** stock picker (site: [dailystockguide.com](https://dailystockguide.com/),
repo: `viki-m13/crt`), using the strategy's **own real point-in-time return
streams** — no re-tuning — benchmarked against **both** S&P-DCA (as the site
does) **and** QQQ-DCA (the growth benchmark the site omits).

Live factsheet: [`docs/crt_dca.html`](../../../docs/crt_dca.html) ·
data: [`docs/crt_dca_data.json`](../../../docs/crt_dca_data.json) ·
builder: [`build_crt_dca.py`](build_crt_dca.py).

## Why this exists

The site headlines *"a monthly DCA strategy that beat S&P-DCA in 100% of
10-year windows."* The statement is **true on its data**, but it answers the
wrong question. A picker that tilts to high-beta growth *should* beat the
cap-weighted S&P. The honest tests are: (1) does it beat a **growth** benchmark
(QQQ-DCA)? and (2) **when** — is the edge uniform or front-loaded? This analysis
adds exactly those two views, plus the contrast with a clean walk-forward
retrain that already lives in the CRT repo.

## Method (one convention throughout)

- **DCA:** contribute **$1 at the start of each month, earn that month's
  return.** Money multiple = terminal value ÷ dollars contributed; "per year" =
  annualised money-weighted IRR.
- **Strategy & S&P streams:** taken verbatim from the production
  `experiments/monthly_dca/v5` data the site ships
  (`experiments/docs/monthly-dca/data.json → dca_investor.growth`: `r` =
  strategy monthly return net of 10 bps, `s` = SPY monthly total return).
  Reconstruction reproduces the site **exactly** (v5 money multiple 2450.5×,
  S&P 5.31×).
- **QQQ stream:** total-return adjusted close from this repo's
  `data/etfs/QQQ.csv`, aligned to the same month-ends. (Consistent total-return
  basis with the site's SPY.)
- **Nothing is re-tuned.** This only re-slices and re-benchmarks existing data.

## Headline results — production stream, 2003-03 → 2026-04 (278 months)

| DCA into | money multiple | return/yr (IRR) | worst account drawdown |
|---|---:|---:|---:|
| **Strategy (v5 / E2)** | **2450.5×** | **54.6%** | −56.0% |
| QQQ-DCA | 10.3× | 17.0% | −36.8% |
| S&P-DCA | 5.31× | 12.5% | −38.2% |

Rolling-window beat rates (money-weighted IRR per window):

| held for | windows | beat S&P-DCA | beat QQQ-DCA | typical strat/yr | S&P/yr | QQQ/yr |
|---|---:|---:|---:|---:|---:|---:|
| 3 years | 243 | 96% | 93% | 34.4% | 12.3% | 17.0% |
| 5 years | 219 | 99% | 99% | 47.6% | 12.4% | 16.8% |
| 10 years | 159 | **100%** | **100%** | 52.9% | 12.7% | 17.5% |

So on the production stream the picker clears **even QQQ-DCA** on every horizon.
The catch is *where* the edge comes from.

## The edge is front-loaded — era by era (DCA within each era)

| era | strategy/yr | S&P/yr | QQQ/yr | lead vs S&P | lead vs QQQ |
|---|---:|---:|---:|---:|---:|
| **2003–2009** | **125.1%** | 1.1% | 5.8% | +124.0 pp | **+119.3 pp** |
| 2010–2015 | 27.7% | 13.0% | 18.0% | +14.7 pp | +9.7 pp |
| 2016–2020 | 51.9% | 16.7% | 29.0% | +35.2 pp | +22.9 pp |
| 2021–2026 | 27.1% | 16.9% | 20.8% | +10.2 pp | +6.2 pp |

The 2003–2009 GFC-recovery era carries essentially the entire *magnitude*; the
lead over QQQ fades to single digits by the most recent era. Mechanically this
is a **crash-recovery / high-beta amplifier** — also why the account fell ~−56%
peak-to-trough. The repeatable, steady-state edge over a growth benchmark is a
fraction of the 55%/yr headline.

## The other layer — a clean walk-forward retrain

The CRT repo's own `research/validation/REPORT.md` retrains a *comparable*
recipe (same K, blend, LGBM, vol-target, regime gate, costs) on strictly
point-in-time membership with no cached-universe survivorship. Run through the
identical DCA:

| clean retrain | strategy/yr | vs S&P-DCA | vs QQQ-DCA |
|---|---:|---:|---:|
| S&P 500 PIT, 2007–2024 | 11.7% | −1.2 pp | **−8.7 pp** |
| NASDAQ-100 PIT, 2019–2025 | 8.9% | −7.5 pp | **−11.1 pp** |

On clean PIT data the DCA edge over a growth benchmark **disappears**. The truth
sits between the two layers and depends on how much of the production stream's
2003–2009 magnitude is durable out of sample.

## Bottom line (consistent with `../VALIDATION_METHODOLOGY.md`)

The Daily Stock Guide picker is **real but front-loaded and beta-driven**, not a
durable 55%/yr selection edge. Benchmarked honestly:

1. It does beat S&P-DCA *and* QQQ-DCA on the production stream — but the edge is
   ~+120 pp/yr in 2003–2009 fading to ~+6 pp vs QQQ in 2021–2026.
2. A clean walk-forward retrain loses to QQQ-DCA by ~9–11 pp/yr.

The site's methodology hygiene is genuinely strong (walk-forward + 7-month
embargo + PIT membership + measured survivorship correction + frank
front-loading/drawdown disclosures). The three honest presentation upgrades this
work recommends: **(1) add the QQQ-DCA benchmark, (2) lead with the four-era
table instead of the 100%/10y headline, (3) publish the clean-retrain number
next to the 55% one.**

## Reproduce

```bash
git clone --depth 1 https://github.com/viki-m13/crt /tmp/crt
python dca/research/crt_dca/build_crt_dca.py    # writes docs/crt_dca_data.json
# env overrides: CRT_ROOT=/path/to/crt BONDS_ROOT=/path/to/bonds
```

*Source streams © their authors; re-used here for honest validation. Past
performance does not predict future results. Research, not financial advice.*
