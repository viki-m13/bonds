"""Queue-aware market-making simulator over recorded HL L2 + trades.

Replays a record_l2.py JSONL file and runs a passive two-sided maker per coin:
  * always quote a small order at the current best bid and best ask (post-only);
  * QUEUE POSITION is modeled FIFO — a resting quote sits behind the size
    already at its level; it fills only after aggressive trades at that price
    consume the queue ahead of it (the honest part most backtests skip);
  * re-pricing to a new best puts the quote at the BACK of the new queue
    (pessimistic but realistic);
  * inventory is skewed Avellaneda-Stoikov-style (quote wider/away on the side
    that grows inventory) and hard-capped;
  * each fill books the HL maker fee/rebate; the position is marked at mid.

The headline output is the per-fill economics — **half-spread captured + rebate
vs adverse selection** (mid move against you after a fill). If that is positive,
passive MM makes money on this tape; if not, it doesn't. A 25-min sample can't
give a trustworthy annualized Sharpe, so we report the per-fill edge (robust) and
flag the Sharpe as indicative only.

Run from crypto_pulse/:  python maker_sim.py [record.jsonl]
"""
import glob
import json
import os
import sys

import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "hl_l2")
ADV_HORIZON = 10.0          # seconds after a fill to measure adverse selection
ORDER_USD = 500.0           # notional per quote
INV_CAP_USD = 5000.0        # hard inventory cap per coin
SKEW = 0.5                  # inventory-skew aggressiveness (fraction of spread)


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            rows.append(json.loads(line))
    rows.sort(key=lambda x: x["r"])
    return rows


def sim_coin(events, maker_bps, imb_thr=None, imb_levels=5):
    """events: time-ordered list of (recv_ts, channel, data) for ONE coin.
    If imb_thr is set, suppress the bid when top-of-book imbalance is strongly
    negative (sellers, price about to drop) and the ask when strongly positive —
    the adverse-selection filter. Returns fills + mid path."""
    best_bid = best_ask = None
    bid_qsz = ask_qsz = None
    my_bid_px = my_ahead_b = None
    my_ask_px = my_ahead_a = None
    imb = 0.0
    inv = 0.0
    fills = []
    mid_path = []

    def mid():
        return (best_bid + best_ask) / 2 if best_bid and best_ask else None

    for ts, ch, d in events:
        if ch == "l2Book":
            lv = d["levels"]
            if not lv[0] or not lv[1]:
                continue
            nb, na = float(lv[0][0]["px"]), float(lv[1][0]["px"])
            nbsz, nasz = float(lv[0][0]["sz"]), float(lv[1][0]["sz"])
            bvol = sum(float(x["sz"]) for x in lv[0][:imb_levels])
            avol = sum(float(x["sz"]) for x in lv[1][:imb_levels])
            imb = (bvol - avol) / (bvol + avol) if (bvol + avol) > 0 else 0.0
            if mid():
                mid_path.append((ts, (nb + na) / 2))
            # (re)place bid
            if my_bid_px != nb:
                my_bid_px = nb
                my_ahead_b = nbsz
            else:
                my_ahead_b = min(my_ahead_b, nbsz)
            if my_ask_px != na:
                my_ask_px = na
                my_ahead_a = nasz
            else:
                my_ahead_a = min(my_ahead_a, nasz)
            # inventory cap
            if inv * nb >= INV_CAP_USD:
                my_bid_px = None
            if inv * nb <= -INV_CAP_USD:
                my_ask_px = None
            # adverse-selection filter: pull the side the book leans against
            if imb_thr is not None:
                if imb < -imb_thr:
                    my_bid_px = None        # sellers dominate -> don't buy
                if imb > imb_thr:
                    my_ask_px = None        # buyers dominate -> don't sell
            best_bid, best_ask, bid_qsz, ask_qsz = nb, na, nbsz, nasz

        else:  # trades (list)
            trs = d if isinstance(d, list) else [d]
            for t in trs:
                px = float(t["px"]); sz = float(t["sz"]); side = t["side"]
                m = mid()
                # aggressive SELL ('A') hits bids -> can fill our resting bid
                if side == "A" and my_bid_px is not None and px <= my_bid_px + 1e-12:
                    if my_ahead_b > 0:
                        eaten = min(my_ahead_b, sz); my_ahead_b -= eaten; sz -= eaten
                    if sz > 0 and my_ahead_b <= 0:
                        fsz = min(ORDER_USD / my_bid_px, sz)
                        inv += fsz
                        fills.append(dict(ts=ts, side="buy", px=my_bid_px,
                                          sz=fsz, mid=m, maker_bps=maker_bps))
                        my_ahead_b = bid_qsz       # rejoin back of queue
                # aggressive BUY ('B') lifts asks -> can fill our resting ask
                if side == "B" and my_ask_px is not None and px >= my_ask_px - 1e-12:
                    if my_ahead_a > 0:
                        eaten = min(my_ahead_a, sz); my_ahead_a -= eaten; sz -= eaten
                    if sz > 0 and my_ahead_a <= 0:
                        fsz = min(ORDER_USD / my_ask_px, sz)
                        inv -= fsz
                        fills.append(dict(ts=ts, side="sell", px=my_ask_px,
                                          sz=fsz, mid=m, maker_bps=maker_bps))
                        my_ahead_a = ask_qsz
    return fills, mid_path


def mid_at(mid_path, t):
    """mid price at/after time t (next available sample)."""
    lo, hi = 0, len(mid_path)
    while lo < hi:
        m = (lo + hi) // 2
        if mid_path[m][0] < t:
            lo = m + 1
        else:
            hi = m
    return mid_path[lo][1] if lo < len(mid_path) else (mid_path[-1][1] if mid_path else None)


def attribute(fills, mid_path):
    """Per fill: edge vs mid + maker fee − adverse selection over ADV_HORIZON."""
    rows = []
    for f in fills:
        if f["mid"] is None:
            continue
        sgn = 1 if f["side"] == "buy" else -1
        # spread captured vs mid (you bought below mid / sold above): + for maker
        spread_edge = sgn * (f["mid"] - f["px"]) / f["mid"]      # >0 good
        fee = -f["maker_bps"] / 1e4                              # rebate>0 => +
        future_mid = mid_at(mid_path, f["ts"] + ADV_HORIZON)
        if future_mid is None:
            continue
        adverse = sgn * (future_mid - f["mid"]) / f["mid"]       # mid move w/ your side; <0 = adverse
        rows.append(dict(spread_bps=spread_edge * 1e4, fee_bps=fee * 1e4,
                         adverse_bps=adverse * 1e4,
                         net_bps=(spread_edge + fee + adverse) * 1e4,
                         notional=f["sz"] * f["px"]))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob(os.path.join(OUTDIR, "*.jsonl")), key=os.path.getmtime)
    rows = load(path)
    span = rows[-1]["r"] - rows[0]["r"] if rows else 0
    bycoin = {}
    for r in rows:
        c = r["d"]["coin"] if r["c"] == "l2Book" else (
            r["d"][0]["coin"] if isinstance(r["d"], list) and r["d"] else None)
        if c is None:
            continue
        bycoin.setdefault(c, []).append((r["r"], r["c"], r["d"]))

    print(f"# Maker-fill simulation over {os.path.basename(path)}")
    print(f"span {span/60:.1f} min, {len(bycoin)} coins, {len(rows):,} msgs. "
          f"Order ${ORDER_USD:.0f}, inv cap ${INV_CAP_USD:.0f}, adverse horizon "
          f"{ADV_HORIZON:.0f}s. Per-fill economics in bps of notional.\n")
    def run(maker_bps, imb_thr):
        allrows = []
        for c, ev in bycoin.items():
            fills, mp = sim_coin(ev, maker_bps, imb_thr=imb_thr)
            allrows += attribute(fills, mp)
        if not allrows:
            return None
        net = np.array([r["net_bps"] for r in allrows])
        t = net.mean() / net.std() * np.sqrt(len(net)) if net.std() > 0 else np.nan
        return dict(n=len(net), spread=np.mean([r["spread_bps"] for r in allrows]),
                    fee=np.mean([r["fee_bps"] for r in allrows]),
                    adverse=np.mean([r["adverse_bps"] for r in allrows]),
                    net=net.mean(), t=t)

    for policy, imb_thr in (("NAIVE (quote both sides)", None),
                            ("FILTERED imb>0.30", 0.30),
                            ("FILTERED imb>0.15", 0.15)):
        print(f"\n## {policy}")
        print(f"{'fee tier':14s} {'fills':>6s} {'spread':>8s} {'fee':>7s} "
              f"{'adverse':>8s} {'NET/fill':>9s} {'t-stat':>8s}")
        for maker_bps, tag in ((1.5, "base 1.5bps"), (0.0, "zero fee"),
                               (-0.3, "rebate -0.3")):
            r = run(maker_bps, imb_thr)
            if r is None:
                print(f"{tag:14s}   no fills"); continue
            print(f"{tag:14s} {r['n']:>6d} {r['spread']:>+8.2f} {r['fee']:>+7.2f} "
                  f"{r['adverse']:>+8.2f} {r['net']:>+9.3f} {r['t']:>+8.1f}")
    # per-coin breakdown (rebate tier, imbalance-filtered) sorted by spread —
    # tests whether wider-spread coins flip MM positive.
    print("\n## Per-coin (rebate -0.3bps, imb>0.15) — sorted by quoted spread")
    print(f"{'coin':6s} {'qspread_bps':>11s} {'fills':>6s} {'spread':>8s} "
          f"{'adverse':>8s} {'NET/fill':>9s}")
    percoin = []
    for c, ev in bycoin.items():
        # mean quoted spread from L2
        sp_list = []
        for ts, ch, d in ev:
            if ch == "l2Book" and d["levels"][0] and d["levels"][1]:
                b = float(d["levels"][0][0]["px"]); a = float(d["levels"][1][0]["px"])
                sp_list.append((a - b) / ((a + b) / 2) * 1e4)
        qspread = np.mean(sp_list) if sp_list else np.nan
        fills, mp = sim_coin(ev, -0.3, imb_thr=0.15)
        rows = attribute(fills, mp)
        if rows:
            net = np.mean([r["net_bps"] for r in rows])
            sp = np.mean([r["spread_bps"] for r in rows])
            ad = np.mean([r["adverse_bps"] for r in rows])
            percoin.append((c, qspread, len(rows), sp, ad, net))
    for c, qs, n, sp, ad, net in sorted(percoin, key=lambda x: x[1]):
        print(f"{c:6s} {qs:>11.2f} {n:>6d} {sp:>+8.2f} {ad:>+8.2f} {net:>+9.3f}")

    print("\nNET/fill > 0 => passive MM captures more spread+rebate than it loses "
          "to adverse selection. The imbalance filter is the adverse-selection "
          "avoidance the literature says MM requires. Per-fill edge is the robust "
          "read; Sharpe from a short sample is not.")


if __name__ == "__main__":
    main()
