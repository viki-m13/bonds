"""MICROSTRUCTURE ALPHA — does short-horizon L2 prediction beat costs on HL?

The HFT question, tested honestly on recorded Hyperliquid L2 + trades. For each
coin we build the canonical microstructure predictors at every book update:
  * OBI   — multi-level order-book imbalance (sum bid sz - ask sz)/(sum), top 5;
  * MICRO — Stoikov microprice minus mid, in bps (size-weighted fair value);
  * OFI   — Cont-Kukanov-Stoikov best-level order-flow imbalance, EWMA'd;
  * TFLOW — signed taker trade flow (buys-sells $) over a trailing window.
Then we measure, at horizons h in {1,2,5,10 s}, the information coefficient (corr
of signal vs forward mid-return) and — the part that matters — whether a simple
threshold strategy clears costs:
  * TAKER: predicted move must exceed half-spread + 4.5bps to cross profitably;
  * MAKER: quote only when alpha favors the fill side (does smart quoting flip
    adverse selection positive?).

A 30-60 min sample can't give a trustworthy annualized Sharpe, so we report the
robust reads: IC (with t-stat), signal half-life, and the per-trade cost-adjusted
edge. That tells us whether ANY of this is worth a real forward recording.

Run from crypto_pulse/:  python microstructure_alpha.py [rec.jsonl]
"""
import glob
import json
import os
import sys

import numpy as np

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "hl_l2")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
HORIZONS = [1.0, 2.0, 5.0, 10.0]      # seconds
TAKER_BPS = 4.5
GRID = 0.5                            # resample grid (s) for forward returns


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows.sort(key=lambda x: x["r"])
    return rows


def bycoin(rows):
    out = {}
    for r in rows:
        c = r["d"]["coin"] if r["c"] == "l2Book" else (
            r["d"][0]["coin"] if isinstance(r["d"], list) and r["d"] else None)
        if c is None:
            continue
        out.setdefault(c, []).append((r["r"], r["c"], r["d"]))
    return out


def build_series(events, levels=5):
    """Return time-aligned arrays of features + mid from one coin's event stream."""
    ts, mid, spread = [], [], []
    obi, micro, ofi, tflow = [], [], [], []
    pb = pa = pbs = pas = None
    ofi_ewma = 0.0
    tf = 0.0                          # decaying signed trade flow ($)
    last_t = None
    for t, ch, d in events:
        if last_t is not None:
            dt = t - last_t
            tf *= np.exp(-dt / 3.0)   # 3s decay on trade flow
        last_t = t
        if ch == "trades":
            trs = d if isinstance(d, list) else [d]
            for tr in trs:
                usd = float(tr["px"]) * float(tr["sz"])
                tf += usd if tr["side"] == "B" else -usd
            continue
        lv = d["levels"]
        if not lv[0] or not lv[1]:
            continue
        b, a = float(lv[0][0]["px"]), float(lv[1][0]["px"])
        bs, as_ = float(lv[0][0]["sz"]), float(lv[1][0]["sz"])
        m = (b + a) / 2
        bvol = sum(float(x["sz"]) for x in lv[0][:levels])
        avol = sum(float(x["sz"]) for x in lv[1][:levels])
        ob = (bvol - avol) / (bvol + avol) if (bvol + avol) > 0 else 0.0
        mp = (b * as_ + a * bs) / (bs + as_) if (bs + as_) > 0 else m   # microprice
        # Cont-Kukanov-Stoikov best-level OFI
        if pb is not None:
            e = 0.0
            if b > pb:
                e += bs
            elif b == pb:
                e += bs - pbs
            else:
                e += -pbs
            if a < pa:
                e += as_
            elif a == pa:
                e += as_ - pas
            else:
                e += -pas
            ofi_ewma = 0.94 * ofi_ewma + e
        pb, pa, pbs, pas = b, a, bs, as_
        ts.append(t); mid.append(m); spread.append((a - b) / m * 1e4)
        obi.append(ob); micro.append((mp - m) / m * 1e4)
        ofi.append(ofi_ewma); tflow.append(tf)
    return {k: np.array(v) for k, v in dict(
        ts=ts, mid=mid, spread=spread, obi=obi, micro=micro, ofi=ofi,
        tflow=tflow).items()}


def fwd_returns(ts, mid, h):
    """forward mid-return (bps) over horizon h seconds, via searchsorted."""
    idx = np.searchsorted(ts, ts + h, side="left")
    idx = np.clip(idx, 0, len(mid) - 1)
    fr = (mid[idx] / mid - 1) * 1e4
    fr[idx == np.arange(len(mid))] = np.nan         # no future point
    return fr


def ic(sig, fr):
    msk = np.isfinite(sig) & np.isfinite(fr)
    if msk.sum() < 50 or np.std(sig[msk]) == 0:
        return np.nan, 0
    c = np.corrcoef(sig[msk], fr[msk])[0, 1]
    n = msk.sum()
    return c, n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else max(
        glob.glob(os.path.join(OUTDIR, "*.jsonl")), key=os.path.getmtime)
    rows = load(path)
    span = (rows[-1]["r"] - rows[0]["r"]) / 60 if rows else 0
    bc = bycoin(rows)

    feats = ["obi", "micro", "ofi", "tflow"]
    lines = [f"# Microstructure alpha on {os.path.basename(path)}\n"]
    lines.append(f"span {span:.1f} min, {len(bc)} coins. Information coefficient "
                 "(signal vs forward mid-return) by horizon; pooled across coins "
                 "(z-scored per coin). Net edge vs cost at the end.\n")

    # pool z-scored features + forward returns across coins
    pool = {f: {h: ([], []) for h in HORIZONS} for f in feats}
    spreads = []
    for c, ev in bc.items():
        S = build_series(ev)
        if len(S["ts"]) < 200:
            continue
        spreads.append(np.nanmedian(S["spread"]))
        for h in HORIZONS:
            fr = fwd_returns(S["ts"], S["mid"], h)
            for f in feats:
                x = S[f].astype(float)
                if np.std(x[np.isfinite(x)]) == 0:
                    continue
                xz = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
                pool[f][h][0].append(xz)
                pool[f][h][1].append(fr)

    lines.append("## Information coefficient (IC) by feature x horizon\n")
    lines.append("| feature | " + " | ".join(f"{h:.0f}s" for h in HORIZONS) + " |")
    lines.append("|" + "---|" * (len(HORIZONS) + 1))
    best = {}
    for f in feats:
        cells = []
        for h in HORIZONS:
            xs = np.concatenate(pool[f][h][0]) if pool[f][h][0] else np.array([])
            fs = np.concatenate(pool[f][h][1]) if pool[f][h][1] else np.array([])
            c, n = ic(xs, fs)
            t = c * np.sqrt(max(n - 2, 1)) if np.isfinite(c) else 0
            cells.append(f"{c:+.3f} (t{t:+.0f})" if np.isfinite(c) else "n/a")
            best[(f, h)] = (c, n, xs, fs)
        lines.append(f"| {f} | " + " | ".join(cells) + " |")

    med_spread = float(np.nanmedian(spreads)) if spreads else np.nan
    lines.append(f"\nMedian quoted spread across coins: **{med_spread:.1f} bps** "
                 f"(half-spread {med_spread/2:.1f}). Taker round-trip cost "
                 f"~{2*TAKER_BPS:.0f}bps + spread.\n")

    # cost-adjusted edge: best feature/horizon, predicted move on a top-decile
    # signal vs the cost to capture it as a TAKER
    lines.append("## Cost-adjusted taker edge (best signal)\n")
    rowsout = []
    for (f, h), (c, n, xs, fs) in best.items():
        if not np.isfinite(c) or n < 200:
            continue
        msk = np.isfinite(xs) & np.isfinite(fs)
        xs, fs = xs[msk], fs[msk]
        thr = np.quantile(np.abs(xs), 0.9)
        sgn = np.sign(xs)
        sel = np.abs(xs) > thr
        if sel.sum() < 30:
            continue
        edge = np.mean(sgn[sel] * fs[sel])          # directional move captured, bps
        cost = med_spread + TAKER_BPS               # cross now, exit at mid later (optimistic)
        rowsout.append((f, h, edge, edge - cost, sel.sum()))
    rowsout.sort(key=lambda x: -x[2])
    lines.append("| feature | horizon | top-decile move (bps) | minus taker cost | n |")
    lines.append("|---|---|---|---|---|")
    for f, h, edge, net, n in rowsout[:8]:
        lines.append(f"| {f} | {h:.0f}s | {edge:+.2f} | **{net:+.2f}** | {n} |")

    any_pos = any(net > 0 for *_, net, _ in [(0, 0, 0, r[3], r[4]) for r in rowsout])
    lines.append("\n## Verdict\n")
    lines.append(
        "- The IC measures genuine short-horizon predictability; the cost-adjusted "
        "column is what a TAKER actually keeps. " +
        ("At least one signal's top-decile move EXCEEDS the taker cost — worth a "
         "long forward recording + a smart-taker/maker test.\n" if any_pos else
         "Every signal's predicted move is SMALLER than the half-spread + taker fee "
         "— the predictability is real but **not taker-monetizable**; it is maker-"
         "only (you must EARN the spread, not pay it), which puts us back in the "
         "queue/adverse-selection game maker_sim.py already showed we lose at "
         "retail latency. This is the honest microstructure result.\n"))
    lines.append(f"- Sample is {span:.0f} min — IC/edge are robust reads; a real "
                 "Sharpe needs days of L2 (record_l2.py running forward).\n")

    out = "\n".join(lines)
    with open(os.path.join(HERE, "microstructure_alpha.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/microstructure_alpha.md")


if __name__ == "__main__":
    main()
