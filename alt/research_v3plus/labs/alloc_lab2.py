"""Round 2: verify DD-throttle inertness, corrected throttle, combo variants.
All printed metrics: 2014-01-02..2018-12-31 ONLY."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/bonds/alt")
import phoenix_production as prod
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from alloc_lab import (build_W, run_variant, seg_metrics, apply_overlay_v,
                       SEG_START, SEG_END, TC_W_BPS)


def apply_overlay_fixed_dd(raw, dd_floor=-0.10, gate_pct=0.99, gate_mult=0.5):
    """Same as production overlay but with the CORRECTED linear DD throttle:
    mult = (1 - dd/dd_floor), i.e. 1 at HWM, 0 at the floor."""
    scaled = raw.copy()
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(prod.DD_WIN, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = (1 - dd / dd_floor).clip(0.0, 1.0)
    sv = scaled.rolling(prod.GATE_VOL_WIN).std()
    sv_thr = sv.rolling(prod.GATE_LOOKBACK, min_periods=60).quantile(gate_pct)
    g = pd.Series(np.where(sv <= sv_thr, 1.0, gate_mult), index=raw.index)
    total_app = (dd_mult * g).shift(2).fillna(1.0)
    tc = total_app.diff().abs().fillna(0.0) * (prod.TC_BPS_PER_LEV_CHG / 1e4)
    return raw * total_app - tc, total_app


def main():
    sleeve_df = prod.load_sleeve_returns()

    # ---- verify production dd_mult is identically 1 on the whole path
    net, state, _Wy = prod.run_strategy(sleeve_df)
    print("prod dd_mult unique values:", sorted(state["dd_mult"].unique())[:5],
          " (n unique =", state["dd_mult"].nunique(), ")")

    W_base = build_W(sleeve_df, "A", fit_years=4)
    rows = []

    def add(name, W, **ov):
        net, _ = run_variant(sleeve_df, W, **ov)
        rows.append(seg_metrics(net, W, name))

    add("base_A4 (baseline)", W_base)

    # --- corrected DD throttle on baseline allocator, floor grid
    for ddf in (-0.08, -0.10, -0.12, -0.15):
        raw = (sleeve_df.loc[W_base.index] * W_base).sum(axis=1)
        net, mult = apply_overlay_fixed_dd(raw, dd_floor=ddf)
        dw = W_base.diff().abs().sum(axis=1).fillna(0.0)
        net = net - dw * mult * (TC_W_BPS / 1e4)
        rows.append(seg_metrics(net, W_base, f"FIXED-ddthrottle {ddf}"))

    # --- combos / refinements
    add("shrink_d15", build_W(sleeve_df, "A", fit_years=4, shrink=0.15))
    add("momo_k10", build_W(sleeve_df, "A", fit_years=4, mom_k=0.10))
    add("shrink30+momo15", build_W(sleeve_df, "A", fit_years=4, shrink=0.3, mom_k=0.15))
    add("semicov+shrink30", build_W(sleeve_df, "A", fit_years=4, downside=True, shrink=0.3))
    add("semicov+momo15", build_W(sleeve_df, "A", fit_years=4, downside=True, mom_k=0.15))

    W_semi = build_W(sleeve_df, "A", fit_years=4, downside=True)
    W_sh30 = build_W(sleeve_df, "A", fit_years=4, shrink=0.3)
    add("ens(base,semicov)", (W_base + W_semi) / 2)
    add("ens(base,semicov,shr30)", (W_base + W_semi + W_sh30) / 3)

    # window-ensemble (3y,4y,5y annual) — window robustness ensembling
    W_a3 = build_W(sleeve_df, "A", fit_years=3)
    W_a5 = build_W(sleeve_df, "A", fit_years=5)
    add("ens(A3,A4,A5)", (W_a3 + W_base + W_a5) / 3)

    df = pd.DataFrame(rows)
    print("\n=== ROUND 2 (2014-01-02 .. 2018-12-31 ONLY) ===")
    print(df.to_string(index=False))

    # year-by-year SR of top contenders within segment (stability check)
    print("\n=== per-year Sharpe within 2014-2018 (stability) ===")
    contenders = {
        "base_A4": (W_base, {}),
        "semicov": (W_semi, {}),
        "ens(base,semicov)": ((W_base + W_semi) / 2, {}),
        "shrink_d30": (W_sh30, {}),
        "momo_k15": (build_W(sleeve_df, "A", fit_years=4, mom_k=0.15), {}),
    }
    out = {}
    for nm, (W, ov) in contenders.items():
        net, _ = run_variant(sleeve_df, W, **ov)
        r = net.loc[SEG_START:SEG_END]
        out[nm] = {y: round(prod._sharpe(g), 2) for y, g in r.groupby(r.index.year)}
    print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
