"""S4 follow-up (BEDROCK_RESEARCH.md §8i): [SZ] size-matched drawdown null and
[W] S4 as a WEIGHT rather than a filter, on the full GRANITE-XL book.

  python corps/research/s4_weight.py [oos]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import engine2 as e2  # noqa: E402
from bedrock_v import cl_fills, real_coupons, exit_lagged  # noqa: E402
from combos import depth_of  # noqa: E402
from s4_transfer import split_s4, IS, OOS  # noqa: E402

RNG = np.random.default_rng(8181)


def nav_stats(bonds, fl, weights):
    days, nav, daily = e2.mtm_nav(bonds, fl, weights=weights)
    return e2.perf_stats(days, nav, daily)


def main():
    oos_mode = len(sys.argv) > 1 and sys.argv[1] == "oos"
    lo, hi = OOS if oos_mode else IS
    bonds = e2.load_cache()
    issuers = {}
    for six in bonds:
        issuers.setdefault(six[:6], []).append(six)
    print(f"loaded {len(bonds)} bonds", flush=True)
    base_f = cl_fills(bonds, lo, hi)
    elig, ok, bad = split_s4(bonds, issuers, base_f)
    ok_ids = {(f.six, f.entry_day) for f in ok}
    print(f"entries {len(base_f)}; sibling-eligible {len(elig)}; S4 pass {len(ok)}",
          flush=True)
    out = {}

    def dep(fl):
        return [float(np.clip(depth_of(bonds, f) / 3.0, 0.5, 2.0)) for f in fl]

    A = real_coupons(bonds, exit_lagged(bonds, base_f))
    psA = nav_stats(bonds, A, dep(A))
    print(f"\nA GRANITE-XL full: cagr={psA['cagr']*100:+.2f}% "
          f"sh={psA['sharpe_m']:.2f} dd={psA['maxdd']*100:.1f}% n={len(A)}", flush=True)
    out["A"] = psA

    if not oos_mode:
        # ---- [SZ] size-matched drawdown null ------------------------------
        B = real_coupons(bonds, exit_lagged(bonds, elig))
        C = real_coupons(bonds, exit_lagged(bonds, ok))
        wB, wC = dep(B), dep(C)
        ddC = nav_stats(bonds, C, wC)["maxdd"]
        n = len(C)
        print(f"\n[SZ] size-matched null: 400 random {n}-trade subsamples of B "
              f"(n={len(B)})", flush=True)
        dds = []
        idx = np.arange(len(B))
        for _ in range(400):
            pick = RNG.choice(idx, size=n, replace=False)
            fl = [B[i] for i in pick]; w = [wB[i] for i in pick]
            dds.append(nav_stats(bonds, fl, w)["maxdd"])
        dds = np.array(dds)
        pval = float((dds <= ddC).mean())     # share of random books at least as bad
        out["SZ"] = {"dd_s4": float(ddC), "dd_random_mean": float(dds.mean()),
                     "dd_random_p05": float(np.quantile(dds, 0.05)),
                     "dd_random_p50": float(np.quantile(dds, 0.50)),
                     "p_at_least_as_bad": pval, "n_sub": n}
        print(f"  S4 book maxDD      : {ddC*100:6.1f}%", flush=True)
        print(f"  random {n}-trade   : mean {dds.mean()*100:6.1f}%  "
              f"p05 {np.quantile(dds,0.05)*100:6.1f}%  median {np.quantile(dds,0.50)*100:6.1f}%",
              flush=True)
        print(f"  share of random books at least as deep: {pval*100:.1f}% "
              f"-> {'concentration artefact' if pval > 0.05 else 'S4-specific'}",
              flush=True)

    # ---- [W] S4 as a weight ------------------------------------------------
    print("\n[W] S4 as a WEIGHT on the full book (depth weights x k if S4-pass):",
          flush=True)
    ks = [1.5, 2.0, 3.0] if not oos_mode else [
        json.loads((ROOT / "research" / "s4_weight_is.json").read_text())["best_k"]]
    wA = dep(A)
    for k in ks:
        w = [wA[i] * (k if (f.six, f.entry_day) in ok_ids else 1.0)
             for i, f in enumerate(A)]
        ps = nav_stats(bonds, A, w)
        gate = (ps["cagr"] >= psA["cagr"] and ps["sharpe_m"] >= psA["sharpe_m"]
                and ps["maxdd"] >= psA["maxdd"] - 0.02)
        ps["admit"] = bool(gate)
        out[f"W_{k}"] = ps
        print(f"  k={k:<4} cagr={ps['cagr']*100:+6.2f}% sh={ps['sharpe_m']:5.2f} "
              f"dd={ps['maxdd']*100:6.1f}%  "
              f"[{'ADMIT' if gate else 'reject'}]", flush=True)

    if not oos_mode:
        adm = [k for k in ks if out[f"W_{k}"]["admit"]]
        if adm:
            best = max(adm, key=lambda k: out[f"W_{k}"]["sharpe_m"])
            out["best_k"] = best
            print(f"\nIS survivor: k={best} -> one OOS look", flush=True)
        else:
            print("\nno k passes the frozen gates -> no OOS look", flush=True)
    p = ROOT / "research" / (f"s4_weight_{'oos' if oos_mode else 'is'}.json")
    p.write_text(json.dumps(out, default=float))
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
