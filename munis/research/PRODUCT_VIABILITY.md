# Can the muni execution edge be sold as a product?

The retail execution penalty measured on this tape is real and survives every
control we could think to apply. This document records what happened when we
tried to turn it into something someone would pay for, and why the answer
came back **no**.

Measurement code lives in the `crt` repo under `research/muni_edge/`
(`market_check.py`, `product_shape.py`, `price_check_control.py`,
`who_needs_it.py`, `can_they_act.py`); the shipping implementation of the
rules is `api/_munisignal.py` and `api/_muniaudit.py` there. Everything below
runs on this repo's `munis/data/trades` tape — 3,085 bonds, 5.0m prints.

## 1. The edge is real

Retail customers are penalised on both sides of the market. Buying, they pay
**+0.311 points** ($311 per $100k) versus the day's institutional benchmark;
selling, they receive **+0.176 points** less.

Controls that it survives:

- **Symmetry.** Penalised on *both* sides, which drift cannot produce — a
  trending market pushes one side only.
- **Intraday.** Restricting the comparison to prints within 60 minutes of
  each other still gives **+0.156 points**, `t = +26.6` clustered across
  2,351 bonds.
- **Breadth.** 84.8% of 2,631 bonds show retail executing worse.

In yield terms retail gives up roughly 5.7bp buying and 5.3bp selling.

## 2. The segment is growing, not shrinking

An earlier no-go call rested on the assumption that individual muni buying is
a declining retail habit. Measured on this tape, that is false:

| | retail share of trades |
|---|---|
| first three years (2013–15) | 84.2% |
| last three years (2024–26) | 92.9% |

Trend **+0.74 points per year**. The second assumption — that dislocations
occur in lots too small for professional buyers — is also false. MSRB
research finds institutional dealers accounted for **44% of odd-lot customer
trades in 2024**, and judges it likely that **≥50%** of odd-lot customer
transactions were with institutional clients, driven by the growth of muni
separately managed accounts (~**$1.3tn across ~180 managers**, ~30% of the
$4tn market, up from 13–14% penetration in 2018).

So the professional buyer trades in exactly the lot sizes where the signal
fires.

## 3. Two candidate products, measured on how often they fire

**The nightly list** — scan the tape, publish bonds trading ≥3 points under
their own 60-day trend:

| | 250-day window |
|---|---|
| dislocations found | 1,627 across 625 distinct bonds |
| per day | median **1**, mean 6.5, max 195 |
| days with an empty list | **104/250 (42%)** |
| discount when it fires | median 3.82 points |

Nothing to show two days in five. That is a churn machine, not a
subscription.

**The pre-trade price check** — advisor pastes a CUSIP and the quoted price,
gets *within limit* / *too rich* and by how much. Replaying all 385,610
retail-size customer buys in the window: **58.4% too rich**, 33.0% within
limit, 8.7% no opinion. Median overpayment 1.13 points.

### The control that keeps this honest

The limit is set off the *prior* mark, so in a rising market a fair buy clears
it mechanically and the check would look valuable while measuring nothing but
the trend. Two controls:

- **Drift is nil.** Median day-over-day change in mid **+0.0000 points**,
  mean +0.0226 — an order of magnitude too small to produce a +0.64 gap.
- **Symmetry again.** Buys print **+0.640** points above the prior mark,
  sells **+0.190** below it. Clustered one vote per bond: buys +0.693 across
  3,081 bonds, `t = +74.3`; sells +0.345 across 3,068, `t = +47.6`.

The reference mid is often built from customer prints that already contain a
markup, so the measured penalty is if anything understated.

### Who the customer is (median points paid over the prior mark, buys)

| lot size | trades | median | past the 0.25 cap | t |
|---|---|---|---|---|
| under $25k | 150,351 | +0.490 | 60.3% | +61.4 |
| $25k–$100k | 196,764 | **+0.742** | 67.0% | +73.6 |
| $100k–$500k | 97,597 | +0.500 | 61.5% | +61.0 |
| $500k–$1m | 11,958 | +0.156 | 42.9% | +26.2 |
| $1m+ | 28,085 | **+0.000** | 29.1% | +10.9 |

Monotone above $100k, and **exactly zero** for block buyers — they already
trade at the mark. Every dollar of value sits in the odd-lot band.

## 4. Why it still fails as a business

**The verdict is less actionable than it looks.** A "too rich" flag is only
worth money if declining the quote has an alternative. Looking forward ten
trading days from each of 245,430 flagged buys:

| | share of flagged buys |
|---|---|
| the bond printed a customer buy again | 96.0% |
| next print was cheaper | 66.5% |
| **next print beat the flagged excess** | **30.2%** |
| median improvement when cheaper | 1.001 pts = $250 on a $25k lot |
| *best of the ten days beat the excess* | *55.3% — look-ahead, unattainable* |

The headline uses the **next** print. Taking the minimum over the window
assumes foresight nobody has. And even 30% is generous: it ignores the cost
of not owning the bond meanwhile, and assumes the later print was available
to *this* buyer, which the tape cannot confirm.

**The buyer is not the beneficiary.** The $250–274 a flag is worth accrues to
the *client*; the advisor pays the subscription. Execution-quality tools
therefore sell as compliance and defensibility, not as savings — a much
narrower pitch.

**The compliance version is already sold.** BondWave's Effi platform ships
both candidate products: Transaction Quality Analysis (trade cost against
marketplace peers, with a per-calculation archive and management reporting —
i.e. the best-execution record) and a pre-trade market calculator, backed by
licensed DPC DATA municipal content, proprietary trade benchmarks and muni
curves, and ICE Bonds integration.

Our one scarce asset is solved EMMA extraction — past the image-rendered
CUSIPs and the TLS-fingerprint block that stops python-requests. That is a
months-long lead, not a moat, against a vendor licensing the data outright.

## Conclusion

Sound measurement, growing segment, genuinely useful tool — and a bad
business at the price that would justify building it. It would flip only on
owned distribution to odd-lot muni buyers, which we do not have.

The muni work's value is therefore as a **credential**: a fully controlled,
externally checkable piece of market-microstructure analysis, every figure of
which can be verified against emma.msrb.org. That is worth keeping and worth
showing. It is not worth selling as a subscription.
