"""Extract representative GRANITE-XL trades with their full surrounding tape
for the strategy page's "anatomy of a trade" section.

Selection protocol (stated on the page — representative, not cherry-picked):
percentile picks by realized return among tape-dense trades (p90 winner,
p50 median, p10 loser), plus the best COVID-2020 entry and one hard-stop
stale loser. Tape = entry-90d .. exit+60d: mid prints, trailing-60d median,
customer-buy (ask) and customer-sell (bid) prints, entry/exit markers.

  python corps/research/trade_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from flowml import xl_fills  # noqa: E402

DOCS = ROOT.parent / "docs"


def density(bonds, f):
    b = bonds[f.six]
    day = b["day"]
    j0 = np.searchsorted(day, f.entry_day - 90)
    j1 = np.searchsorted(day, f.exit_day + 60, side="right")
    return j1 - j0


def tape(bonds, f, label, note):
    b = bonds[f.six]
    day = b["day"]
    j0 = np.searchsorted(day, f.entry_day - 90)
    j1 = np.searchsorted(day, f.exit_day + 60, side="right")
    rows = []
    for i in range(j0, j1):
        d = pd.Timestamp(int(day[i]), unit="D").strftime("%Y-%m-%d")
        def fv(x):
            return round(float(x), 3) if np.isfinite(x) else None
        rows.append({"d": d, "m": fv(b["mid"][i]), "a": fv(b["s_px"][i]),
                     "b": fv(b["p_px"][i]), "md": fv(b["med60"][i])})
    return {
        "label": label, "note": note, "cusip": f.six,
        "entry_date": pd.Timestamp(f.entry_day, unit="D").strftime("%Y-%m-%d"),
        "entry_px": round(float(f.entry_px), 3),
        "exit_date": pd.Timestamp(f.exit_day, unit="D").strftime("%Y-%m-%d"),
        "exit_px": round(float(f.exit_px), 3),
        "ret": round(float(f.ret), 4), "hold": int(f.hold),
        "coupon": round(float(f.coupon), 2), "stale": bool(f.stale),
        "mat": int(b["mat"][max(np.searchsorted(day, f.entry_day) - 1, 0)]),
        "tape": rows,
    }


def main():
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    fills = xl_fills(bonds)
    dense = [f for f in fills if density(bonds, f) >= 60]
    dense.sort(key=lambda f: f.ret)
    n = len(dense)
    print(f"XL fills: {len(fills)}, tape-dense: {n}", flush=True)

    picks = []
    p90 = dense[int(0.90 * n)]
    p50 = dense[int(0.50 * n)]
    p10 = dense[int(0.10 * n)]
    covid = max((f for f in dense
                 if e2.D("2020-03-01") <= f.entry_day <= e2.D("2020-12-31")),
                key=lambda f: f.ret)
    stales = [f for f in dense if f.stale and f.ret < 0]
    stale = stales[len(stales) // 2] if stales else None

    picks.append(tape(bonds, p90, "90th-percentile outcome",
                      "A deep forced-seller dislocation that fully reverts — the strategy's target trade."))
    picks.append(tape(bonds, p50, "Median outcome",
                      "The typical trade: modest dislocation, recovery to the pre-event median, exit at the first qualifying bid."))
    picks.append(tape(bonds, covid, "COVID 2020 entry",
                      "Crisis vintage: the March-2020 liquidation priced short-dated credit for defaults that never came."))
    picks.append(tape(bonds, p10, "10th-percentile outcome",
                      "A loser: the dislocation was information, not flow — the price kept sliding and the exit realized a loss."))
    if stale is not None:
        picks.append(tape(bonds, stale, "Hard-stop stale exit",
                          "The failure mode the engine never hides: the bond stopped trading and the position exits at the last available bid."))

    for p in picks:
        print(f"  {p['label']:26} {p['cusip']} {p['entry_date']}→{p['exit_date']} "
              f"ret={p['ret']*100:+.1f}% hold={p['hold']}d tape={len(p['tape'])}", flush=True)
    sel = {"protocol": "Percentile picks by realized return among tape-dense trades "
                       "(>=60 prints in window), plus best COVID-2020 entry and a "
                       "median stale hard-stop loser. Not cherry-picked.",
           "trades": picks}
    (DOCS / "trade_examples_corp.json").write_text(json.dumps(sel, default=float))
    print("wrote docs/trade_examples_corp.json", flush=True)


if __name__ == "__main__":
    main()
