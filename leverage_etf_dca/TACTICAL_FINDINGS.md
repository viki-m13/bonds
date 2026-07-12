# PULSE — tactical QQQ-core sibling switch (unleveraged): the one that passed

Response to: *"a model that strategically/tactically invests in QQQ and other ETFs so we
ultimately outperform DCA into QQQ only."* Scripts: `scripts/tactical.py`.

## The idea that survived (and why it dodges the structural wall)
Every failed strategy **left tech** to seek an edge and paid for it in the bull. The
sibling switch never leaves tech beta: hold **QQQ by default**, switch the core to
**SMH (semiconductors)** only while SMH's relative momentum vs QQQ (blended 3/6/12-month
return of the SMH/QQQ ratio) is positive. One decision monthly. Unleveraged. ~2 switches/yr.

## Results (DCA $1000/mo, 10bps/side, no look-ahead)
| era | ratio vs QQQ-DCA |
|---|--:|
| dot-com 1999–03 (pre-SMH: holds QQQ) | 1.00 |
| 2006–09 (semis LAG) | 0.95 |
| 2010–14 | 0.97 |
| 2015–19 | 1.06 |
| 2020–26 (semis lead) | 1.81 |
| **full 2006–26 continuous** | **2.06–2.17×** |
| **full 1999–26 continuous** | **1.99–2.10×** |

**Lump-sum risk (2006–26):** CAGR **20.0%** (QQQ 15.9%, SMH B&H 19.6%) at Sharpe **0.90 =
QQQ's** and maxDD **−50% = QQQ's** (SMH B&H: 0.84, −57%). The switch delivers semi-level
return at QQQ-level risk.

## The gauntlet (what killed every predecessor)
- **Phase-robustness: PASSES.** Full 2006–26 by rebalance day: ME 2.17 / d4 1.43 /
  d9 1.62 / d14 1.98 — beats QQQ-DCA on **every** trade day. (1999–26: 1.32–2.10.)
- **Parameter-family robustness: PASSES.** Every lookback works (3m: 1.78, 6m: 2.41,
  12m: 1.56, blends 1.95–2.17), and a *different signal family* (ratio above its own
  10-month MA) gives 1.88×. Not one magic parameter. Shipped config = the pre-registered
  3/6/12 blend, not the in-sample best.
- **Cost stress: PASSES.** 25bps/side: 1.99×; even 50bps/side: 1.74×. (41 switches in 20y.)
- **Mechanism through dot-com (proxy): BENIGN.** SMH data starts 2005, so the crash is
  tested via QQQ↔XLK (1998+): 0.97× through 1999–2003 — within-tech switching neither
  protects nor destroys in a crash (the pair is too correlated to bleed much).

## Honest caveats (what this is and is not)
1. **The edge is "semis lead tech," adaptively harvested.** Excess return is concentrated
   in semis-led eras (esp. 2020–26 AI cycle). If semis stop leading, the switch degrades
   gracefully to ≈QQQ (0.95–0.97 in semis-lagging eras) — that's the design, but the big
   numbers are not guaranteed forward.
2. **Zero crash protection.** maxDD −50% (2008), −81% if a dot-com repeats. It is a pure
   return-selection overlay on full tech beta. Pair with AEGIS/defensive sleeves separately.
3. **Semis through 2000–02 untested** (no pre-2005 SMH/SOX data in repo; external sources
   blocked by network policy). Semis fell harder than QQQ then; the momentum lag (~2–4
   months) would have cost a real but bounded bleed before exiting to QQQ. The XLK proxy
   bounds the mechanism risk but not the semis-specific severity.
4. Composability (future work): the VOLT vol-target dial could ride this switched core
   (vol-targeted SOXL↔TQQQ) — untested, adds leverage.

## Rejected in the same sweep (for the record)
- **Credit gate on unleveraged QQQ: 0.64–0.73× full — FAILED** (exiting an unleveraged
  compounder costs too much; the gate only pays when it controls *leverage*, as in VOLT).
- **Faber 10m-MA trend switch on QQQ: 0.58× full — FAILED** (whipsaw, 2010s).
- **No-sell contribution router (QQQ/SMH/GLD/TLT): 1.04–1.07× — marginal**, not worth it.
- **3-way QQQ/SMH/XLK: 2.11×** — no better than the simpler 2-way.
