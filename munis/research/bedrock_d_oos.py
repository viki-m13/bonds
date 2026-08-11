"""BEDROCK Sleeve D — the ONE-SHOT muni OOS test. Run once, report as printed.

Spec frozen (BEDROCK_RESEARCH.md §3/§8): KEYSTONE price_discount(3.0) + limit
cap + REAL issuer cap, lagged-mid recovery exits; cliff zone = signal-day mid
in [threshold-3, threshold+1), threshold = 100 - 0.25*ceil(yrs). OOS window
2023-01-01 .. 2025-04-08 (censor-safe: data_end - 455d). Reported: cliff book
vs complement book (paired conventions), cap-matched control excess on
hold-matched 1y fills, and the same for the complement.

  python munis/research/bedrock_d_oos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest as bt  # noqa: E402
from strategies import FACTORIES  # noqa: E402
from limit_transfer import load_bonds, limit_filter, limit_control  # noqa: E402
from keystone_xl import issuer_cap  # noqa: E402
from bedrock_d import cliff_map, in_zone, exit_lagged, MAXH  # noqa: E402

OOS_LO = pd.Timestamp("2023-01-01")
OOS_HI = pd.Timestamp("2025-04-08")          # censor-safe (2026-07-07 - 455d)
RNG = np.random.default_rng(211)


def stats(bonds, fills, tag, ctl=True, lo=OOS_LO, hi=OOS_HI):
    if not fills:
        print(f"  {tag:28} n=0", flush=True)
        return {"n": 0}
    r = np.array([f.ret for f in fills])
    out = {"n": len(fills), "mean": float(r.mean()), "win": float((r > 0).mean()),
           "hold": float(np.mean([f.hold_days for f in fills]))}
    if ctl:
        c = limit_control(bonds, fills, lo, hi)
        if len(c):
            by = {}
            for f in fills:
                by.setdefault(f.six, []).append(f.ret)
            keys = list(by); boots = []
            for _ in range(2000):
                ks = RNG.choice(len(keys), size=len(keys), replace=True)
                sm = np.concatenate([np.asarray(by[keys[q]]) for q in ks]).mean()
                boots.append(sm - RNG.choice(c, size=len(c), replace=True).mean())
            out["ctl"] = float(c.mean())
            out["excess"] = float(r.mean() - c.mean())
            out["excess_p"] = float((np.array(boots) <= 0).mean())
    print(f"  {tag:28} n={out['n']:5} mean={out['mean']*100:+6.2f}% "
          f"win={out['win']*100:3.0f}% hold={out['hold']:4.0f}d"
          + (f" excess={out.get('excess', float('nan'))*100:+5.2f}% "
             f"p={out.get('excess_p', float('nan')):.4f}" if 'excess' in out else ""),
          flush=True)
    return out


def main():
    bonds = load_bonds()
    mats = cliff_map()
    fn = FACTORIES["price_discount"](discount=3.0)
    out = {}
    print("=" * 64, flush=True)
    print("BEDROCK D — ONE-SHOT OOS 2023-01-01 .. 2025-04-08 (censor-safe)", flush=True)
    print("=" * 64, flush=True)
    base = bt.run_signal(bonds, fn, min_hold=365, max_hold=MAXH,
                         date_lo=OOS_LO, date_hi=OOS_HI)
    stack = issuer_cap(limit_filter(bonds, base))
    flags = [in_zone(bonds, mats, f) for f in stack]
    cliff = [f for f, fl in zip(stack, flags) if fl is True]
    rest = [f for f, fl in zip(stack, flags) if fl is False]
    print(f"stack {len(stack)} -> cliff {len(cliff)}, complement {len(rest)}", flush=True)

    print("\n[1y-hold fills, hold-matched excess]", flush=True)
    out["cliff_1y"] = stats(bonds, cliff, "CLIFF zone")
    out["rest_1y"] = stats(bonds, rest, "complement")

    print("\n[lagged recovery-exit books (paired conventions, no ctl)]", flush=True)
    out["cliff_xl"] = stats(bonds, exit_lagged(bonds, cliff), "CLIFF recovery book", ctl=False)
    out["rest_xl"] = stats(bonds, exit_lagged(bonds, rest), "complement recovery book", ctl=False)

    p = Path(__file__).resolve().parent / "results" / "bedrock_d_oos.json"
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
