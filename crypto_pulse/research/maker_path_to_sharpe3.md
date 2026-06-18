# The maker path to Sharpe 3+ — critical investigation on real HL L2 data

Sharpe = edge/bet x sqrt(bets/year). Daily taker books cap ~1.1-1.4 (too few
bets, edge decayed). The ONLY structural route to 3+ is high-frequency, which at
taker fees is suicide (VELOCITY: gross Sharpe 15-60, dead at taker). So 3+ must be
a MAKER strategy: capture spread+rebate thousands of times/day. Tested on 25 min
of real HL L2+trades (queue-aware fill sim, maker_sim.py), per-fill bps at the
top rebate tier (-0.3 bps):

| policy | NET/fill (bps) | note |
|---|---|---|
| naive both-sides quoting | **-2.04** | adverse selection (-2.6) dwarfs spread+rebate (+0.5) |
| + order-book-imbalance filter | -0.93 | filter cuts adverse selection; still negative |
| + reversal-alpha quoting (the creative idea) | **-2.3 (FAILED)** | at the per-fill SECOND scale the flow is momentum-TOXIC, not reversal-favourable; the VELOCITY reversal edge is a 1-min CROSS-SECTIONAL effect, not a per-fill one |
| + fast exit (~2s instead of 10s) | -0.50 | toxic move plays out over seconds; speed dodges most of it |

**The real signal — per-coin (2s exit, imbalance-filtered, rebate):**

| coin | quoted spread (bps) | NET/fill (bps) |
|---|---|---|
| LTC | 1.38 | **+0.51** |
| DOGE | 1.07 | **+0.57** |
| XRP | 0.96 | **+0.28** |
| BTC | 0.17 | -0.16 |
| SOL | 0.29 | **-1.94** (toxic) |

**Quoted spread is the discriminator.** On wider-spread coins (~1 bps+) a fast,
imbalance-filtered maker at the rebate tier is **net-positive per fill**; on the
tightest majors (BTC/SOL) adverse selection wins. So the honest Sharpe-3 candidate
is a **SELECTIVE FAST MAKER**: quote only wider-spread coins, imbalance-filter,
exit in ~2s, top rebate tier — high breadth (thousands of fills/day) x small
positive per-fill edge.

## Honest caveats (why this is a candidate, not a result)
- 25 min, single-digit-to-dozens of fills/coin: the per-coin positives are **not
  yet statistically reliable** (ETH +0.88 on 2 fills is noise).
- Requires the **top maker-rebate tier** (gated on exchange-wide maker share —
  pro-MM only) and **genuine fast execution** (the 2s edge assumes you can quote/
  cancel/exit faster than toxic flow; HL has ~0.2s blocks + address rate limits).
- Real frictions not modelled: queue competition, inventory risk over the 2s
  holds, requote costs, latency.

## To actually validate
1. Run record_l2.py FORWARD for several days across ~20 wider-spread coins.
2. Confirm the selective-fast-MM per-fill is robustly positive with thousands of
   fills and a real t-stat.
3. If it holds, it's a **market-making deployment** (rebate tier + low-latency
   infra), NOT a retail directional bot.

Bottom line: a genuine structural path to Sharpe 3+ exists (selective fast maker
on wider-spread coins at the rebate tier), and the real data shows positive
per-fill economics on the right coins — but it is an infrastructure/rebate game
that needs forward data and pro execution to confirm, not a backtestable taker
strategy. The naive and reversal-alpha makers are net-negative; the deployable
TAKER book stays the 3-sleeve trend+carry+order-flow ~1.1-1.4.
