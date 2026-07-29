"""IS-ONLY screens of the shortlisted portfolio sleeves (wave 2).

  BALLAST-K  short-crossover carry ladder with knife veto (base/CAGR layer)
  CREST      issuer-level 6-1 spread-tightening momentum (winners sleeve)

Monthly mechanics via portfolio.run_portfolio (buys at ask, sells at bid,
dead bonds force-exited at last bid). Design window 2003-2015 ONLY.

  python corps/research/sleeves_portfolio.py [ballast crest]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2      # noqa: E402
import portfolio as pf    # noqa: E402

IS_LO, IS_HI = e2.D("2003-01-01"), e2.D("2015-12-31")


# ------------------------------------------------------------------ BALLAST-K

def gate_ballast(b):
    mid = b["mid"]; med60 = b.get("med60")
    if med60 is None:
        return None
    rf = getattr(gate_ballast, "_rf", None)
    if rf is None or len(rf) != len(b["day"]):
        rf = e2.load_rf(b["day"]) * 100.0
    knife = (mid >= med60 - 2.0) & (mid >= np.nan_to_num(b["med15"], nan=-1e9) - 1.0)
    return ((b["mat"] >= 1) & (b["mat"] <= 4) & (b["cs"] >= 0.015) & (b["cs"] <= 0.05)
            & (mid >= 85) & (mid <= 100.5)
            & (np.nan_to_num(b["qvmed90"]) * np.maximum(b["act90"], 1) >= 1.0)  # >=$1mm traded/90d ($MM units)
            & (b["ytw"] >= rf + 2.0) & knife)


def score_cs(b, t, i):
    c = float(b["cs"][i])
    return c if np.isfinite(c) else None


# ---------------------------------------------------------------------- CREST

def make_score_crest(bonds):
    imap = {}
    for six in bonds:
        imap.setdefault(six[:6], []).append(six)

    def issuer_med_cs(iss, lo, hi, t):
        vals = []
        for p in imap.get(iss, []):
            bp = bonds[p]
            day = bp["day"]
            j0 = np.searchsorted(day, t - hi); j1 = np.searchsorted(day, t - lo)
            seg = bp["cs"][j0:j1]
            seg = seg[np.isfinite(seg)]
            if len(seg) >= 3:
                vals.append(float(np.median(seg)))
        return float(np.median(vals)) if vals else None

    cache = {}

    def score(b, t, i):
        cs = float(b["cs"][i])
        if not (np.isfinite(cs) and 0.015 <= cs <= 0.05):
            return None
        if not (2 <= int(b["mat"][i]) <= 10):
            return None
        iss = b["_six"][:6]
        key = (iss, t)
        if key not in cache:
            recent = issuer_med_cs(iss, 21, 42, t)
            old = issuer_med_cs(iss, 126, 168, t)
            cache[key] = (old - recent) if (recent is not None and old is not None) else None
        v = cache[key]
        return v   # spread TIGHTENING = positive score
    return score


def run(name, bonds, score_fn, gate, top_n, hold_rank):
    from combine import save_fills
    print(f"\n[{name}] IS 2003-2015 top_n={top_n}", flush=True)
    closed = pf.run_portfolio(bonds, score_fn, IS_LO, IS_HI, top_n=top_n,
                              hold_until_rank=hold_rank, extra_gate=gate)
    fills = pf.positions_to_fills(closed)
    if not fills:
        print("  no fills", flush=True); return None
    save_fills(fills, ROOT / "research" / f"fills_{name.lower().replace('-','')}_is.json")
    days, nav, daily = e2.mtm_nav(bonds, fills)
    ps = e2.perf_stats(days, nav, daily)
    ps["n_roundtrips"] = len(fills)
    print(f"  n={len(fills)} cagr={ps['cagr']*100:+.2f}% maxdd={ps['maxdd']*100:.1f}% "
          f"sharpe_m={ps['sharpe_m']} sharpe_a={ps['sharpe_a']} "
          f"vol_m={ps['vol_m_ann']*100:.1f}%", flush=True)
    ctl = pf.random_control_portfolio(bonds, score_fn, IS_LO, IS_HI, top_n,
                                      n_reps=3, extra_gate=gate)
    if ctl:
        ps["ctl_cagr"] = float(np.mean([c["cagr"] for c in ctl]))
        cs_ = [c["sharpe_m"] for c in ctl if c["sharpe_m"] is not None]
        ps["ctl_sharpe_m"] = float(np.mean(cs_)) if cs_ else None
        print(f"  control: cagr={ps['ctl_cagr']*100:+.2f}% sharpe_m={ps['ctl_sharpe_m']}", flush=True)
    return ps


def main():
    want = set(sys.argv[1:]) or {"ballast", "crest"}
    bonds = e2.load_cache()
    for six, b in bonds.items():
        b["_six"] = six
    print(f"loaded {len(bonds)} bonds", flush=True)
    out = {}
    if "ballast" in want:
        out["ballast_k"] = run("BALLAST-K", bonds, score_cs, gate_ballast, 60, 180)
    if "crest" in want:
        out["crest"] = run("CREST", bonds, make_score_crest(bonds), None, 50, 100)
    p = ROOT / "research" / "sleeves_is.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update({k: v for k, v in out.items() if v})
    p.write_text(json.dumps(old, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
