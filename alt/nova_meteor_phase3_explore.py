"""METEOR Phase 3 — tight refinement around Phase 2 winner.

Phase 2 found:
  lb=120, tn=3, cap=1.0, rb=21, ov=5.5, pf=0.30, nw=15, pw=378, nfm=0.33
  FULL SR 0.917, Ret 56.15%, MDD -52.6%, NAVx 96.0, Calmar 1.067
  (IS_SR 0.810 / OOS_SR 1.003 — OOS > IS, strong robustness)

Phase 3 refines the grid at finer resolution:
  - pdot_win ∈ {252, 315, 378, 441, 504, 630}
  - nav_floor_mult ∈ {0.20, 0.25, 0.30, 0.33, 0.40, 0.50}
  - pdot_floor ∈ {0.25, 0.28, 0.30, 0.32, 0.35}
  - overlay_base ∈ {5.0, 5.25, 5.5, 5.75, 6.0, 6.5}
  - nav_win ∈ {10, 12, 15, 17, 20, 25}
  - rebal ∈ {17, 19, 21, 23, 25}
  - top_n ∈ {2, 3, 4}

Step 1: each axis swept independently (anchor fixed).
Step 2: joint sweep of the 3 most sensitive axes (ov × pw × pf) × nfm.

No vol scaling. No look-ahead. Aim: push CAGR up or MDD down from the
Phase 2 winner while keeping robust IS/OOS agreement.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from nova_meteor_deep_explore import prep, run, stats, eval_cfg, summary

ROOT = Path("/home/user/bonds")
RESULTS = ROOT / "data/results"
IS_END = pd.Timestamp("2020-01-01")


def main():
    ctx = prep()

    ANCHOR = dict(lookback=120, top_n=3, cap=1.0, rebal=21, overlay_base=5.5,
                  pdot_floor=0.30, nav_win=15, pdot_win=378, skip_recent=0,
                  nav_floor_mult=0.33)

    # Step 1 — one axis at a time (finer than phase-1)
    S1 = [ANCHOR]
    for v in [21, 42, 63, 126, 189, 252, 315, 378, 441, 504, 630, 756]:
        S1.append({**ANCHOR, "pdot_win": v})
    for v in [0.15, 0.20, 0.25, 0.30, 0.33, 0.40, 0.50, 0.67, 0.75]:
        S1.append({**ANCHOR, "nav_floor_mult": v})
    for v in [0.20, 0.22, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40]:
        S1.append({**ANCHOR, "pdot_floor": v})
    for v in [4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 7.0]:
        S1.append({**ANCHOR, "overlay_base": v})
    for v in [5, 7, 10, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30]:
        S1.append({**ANCHOR, "nav_win": v})
    for v in [15, 17, 19, 21, 23, 25, 28, 31]:
        S1.append({**ANCHOR, "rebal": v})
    for v in [2, 3, 4, 5]:
        S1.append({**ANCHOR, "top_n": v})
    for v in [90, 100, 110, 120, 130, 140, 150, 180, 210]:
        S1.append({**ANCHOR, "lookback": v})
    for v in [0, 5, 10, 15, 21]:
        S1.append({**ANCHOR, "skip_recent": v})

    seen = set(); u1 = []
    for c in S1:
        k = tuple(sorted(c.items()))
        if k in seen: continue
        seen.add(k); u1.append(c)

    print(f"Step 1 (one-axis fine): {len(u1)} configs")
    r1 = []
    for i, c in enumerate(u1):
        r1.append(eval_cfg(ctx, c))
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(u1)}")
    df1 = pd.DataFrame(r1)
    df1 = summary(df1, "Step 1 — top 20 by Calmar")

    # Step 2 — joint sweep over the high-leverage axes
    S2 = []
    for ov in [5.0, 5.25, 5.5, 5.75, 6.0]:
        for pw in [252, 315, 378, 441, 504]:
            for pf in [0.25, 0.28, 0.30, 0.32, 0.35]:
                for nfm in [0.25, 0.33, 0.40, 0.50]:
                    for nw in [12, 15, 17, 20]:
                        S2.append({**ANCHOR, "overlay_base": ov,
                                   "pdot_win": pw, "pdot_floor": pf,
                                   "nav_floor_mult": nfm, "nav_win": nw})
    seen = set(); u2 = []
    for c in S2:
        k = tuple(sorted(c.items()))
        if k in seen: continue
        seen.add(k); u2.append(c)

    print(f"\nStep 2 (5-axis joint): {len(u2)} configs")
    r2 = []
    for i, c in enumerate(u2):
        r2.append(eval_cfg(ctx, c))
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(u2)}")
    df2 = pd.DataFrame(r2)
    df2 = summary(df2, "Step 2 — top 20 by Calmar")

    df1["step"] = 1; df2["step"] = 2
    full = pd.concat([df1, df2], ignore_index=True)
    full.to_csv(RESULTS / "nova_meteor_phase3_grid.csv", index=False)

    print("\n=== 55%+ CAGR, MDD>-55%, min_sr>0.60 ===")
    robust = full[(full.FULL_Ret >= 55) & (full.FULL_MDD > -55)
                  & (full.min_sr > 0.60)]
    print(f"{len(robust)} configs")
    cols = ["lookback","top_n","rebal","overlay_base","pdot_floor","nav_win",
            "pdot_win","nav_floor_mult","IS_SR","OOS_SR","FULL_SR","FULL_Ret",
            "FULL_MDD","NAVx","calmar","min_sr","SR_gap","step"]
    print(robust.sort_values("calmar", ascending=False).head(30)[cols].to_string(index=False))

    print("\n=== 55%+ CAGR, MDD>-50%, min_sr>0.55 (tighter MDD) ===")
    tight = full[(full.FULL_Ret >= 55) & (full.FULL_MDD > -50)
                 & (full.min_sr > 0.55)]
    print(f"{len(tight)} configs")
    print(tight.sort_values("calmar", ascending=False).head(20)[cols].to_string(index=False))

    print("\n=== 60%+ CAGR any MDD, ranked by Calmar ===")
    hi = full[full.FULL_Ret >= 60]
    print(f"{len(hi)} configs")
    print(hi.sort_values("calmar", ascending=False).head(15)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
