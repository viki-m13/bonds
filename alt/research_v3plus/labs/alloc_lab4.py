"""Round 4: finer deadband grid + cross-allocator robustness of the fixed throttle.
All printed metrics: 2014-01-02..2018-12-31 ONLY. Then save the chosen
candidate's FULL-period series without inspecting post-2018 numbers."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/bonds/alt")
import phoenix_production as prod
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from alloc_lab import build_W, seg_metrics, SEG_START, SEG_END, TC_W_BPS
from alloc_lab3 import overlay

SCRATCH = Path("/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")


def run_on(sleeve_df, W, name, **kw):
    raw = (sleeve_df.loc[W.index] * W).sum(axis=1)
    net, mult = overlay(raw, **kw)
    dw = W.diff().abs().sum(axis=1).fillna(0.0)
    net = net - dw * mult * (TC_W_BPS / 1e4)
    m = seg_metrics(net, W, name)
    m["avg_mult"] = round(float(mult.loc[SEG_START:SEG_END].mean()), 3)
    return m, net


def main():
    sleeve_df = prod.load_sleeve_returns()
    W = build_W(sleeve_df, "A", fit_years=4)

    print("=== FINE DEADBAND GRID, baseline allocator (2014-2018 ONLY) ===")
    rows = []
    for s in (-0.02, -0.03, -0.04, -0.05, -0.06, -0.07):
        for f in (-0.10, -0.12, -0.14, -0.15, -0.16, -0.18):
            if f >= s:
                continue
            m, _ = run_on(sleeve_df, W, f"db {s}->{f}", dd_start=s, dd_floor=f)
            rows.append(m)
    g = pd.DataFrame(rows)
    print(g.to_string(index=False))
    # plateau view: pivot SR
    g["s"] = g["variant"].str.split().str[1].str.split("->").str[0].astype(float)
    g["f"] = g["variant"].str.split().str[1].str.split("->").str[1].astype(float)
    print("\nSR pivot (rows=dd_start, cols=dd_floor):")
    print(g.pivot(index="s", columns="f", values="SR").to_string())

    print("\n=== throttle db(-0.05,-0.15) across allocator specs (2014-2018 ONLY) ===")
    specs = {
        "A4(base)": build_W(sleeve_df, "A", fit_years=4),
        "A3": build_W(sleeve_df, "A", fit_years=3),
        "A5": build_W(sleeve_df, "A", fit_years=5),
        "Q4": build_W(sleeve_df, "Q", fit_years=4),
        "shrink30": build_W(sleeve_df, "A", fit_years=4, shrink=0.3),
        "semicov": build_W(sleeve_df, "A", fit_years=4, downside=True),
    }
    rows2 = []
    for nm, Wv in specs.items():
        m0, _ = run_on(sleeve_df, Wv, f"{nm} + inert", fixed=False)
        m1, _ = run_on(sleeve_df, Wv, f"{nm} + db(-.05,-.15)", dd_start=-0.05, dd_floor=-0.15)
        rows2 += [m0, m1]
    print(pd.DataFrame(rows2).to_string(index=False))

    # ---- save the recommended candidate: baseline allocator + deadband throttle
    _, net = run_on(sleeve_df, W, "final", dd_start=-0.05, dd_floor=-0.15)
    out = net.loc[SEG_START:]  # 2014 onward, FULL period; post-2018 never inspected
    (SCRATCH / "candidates").mkdir(exist_ok=True)
    out.rename("ret").rename_axis("Date").to_csv(SCRATCH / "candidates/alloc_deadband_ddthrottle.csv")
    print(f"\nSaved candidates/alloc_deadband_ddthrottle.csv  rows={len(out)}  "
          f"({out.index[0].date()} .. [end not printed])")


if __name__ == "__main__":
    main()
