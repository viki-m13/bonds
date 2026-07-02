"""Round 3: shape of the corrected DD throttle + deadband variant + gate interaction.
All printed metrics: 2014-01-02..2018-12-31 ONLY."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/bonds/alt")
import phoenix_production as prod
sys.path.insert(0, "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad")
from alloc_lab import build_W, seg_metrics, SEG_START, SEG_END, TC_W_BPS


def overlay(raw, dd_start=0.0, dd_floor=-0.10, gate_pct=0.99, gate_mult=0.5,
            fixed=True):
    """Corrected DD throttle with optional deadband: mult=1 for dd>=dd_start,
    linear to 0 at dd_floor. fixed=False reproduces the inert production one."""
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(prod.DD_WIN, min_periods=30).max()
    dd = cum / hwm - 1
    if fixed:
        dd_mult = ((dd_floor - dd) / (dd_floor - dd_start)).clip(0.0, 1.0)
    else:
        dd_mult = (1 + dd / dd_floor).clip(0.0, 1.0)
    sv = raw.rolling(prod.GATE_VOL_WIN).std()
    sv_thr = sv.rolling(prod.GATE_LOOKBACK, min_periods=60).quantile(gate_pct)
    g = pd.Series(np.where(sv <= sv_thr, 1.0, gate_mult), index=raw.index)
    total_app = (dd_mult * g).shift(2).fillna(1.0)
    tc = total_app.diff().abs().fillna(0.0) * (prod.TC_BPS_PER_LEV_CHG / 1e4)
    return raw * total_app - tc, total_app


def main():
    sleeve_df = prod.load_sleeve_returns()
    W = build_W(sleeve_df, "A", fit_years=4)
    raw = (sleeve_df.loc[W.index] * W).sum(axis=1)
    dw = W.diff().abs().sum(axis=1).fillna(0.0)

    def run(name, **kw):
        net, mult = overlay(raw, **kw)
        net = net - dw * mult * (TC_W_BPS / 1e4)
        m = seg_metrics(net, W, name)
        m["avg_mult"] = round(float(mult.loc[SEG_START:SEG_END].mean()), 3)
        return m, net

    rows = []
    m, _ = run("baseline (inert dd)", fixed=False); rows.append(m)
    # pure linear-from-HWM throttle, floor sweep
    for f in (-0.10, -0.12, -0.15, -0.20, -0.25, -0.30, -0.40):
        m, _ = run(f"lin dd 0->{f}", dd_floor=f); rows.append(m)
    # deadband throttle: full exposure until dd_start, linear to floor
    for s, f in [(-0.03, -0.12), (-0.05, -0.15), (-0.05, -0.20),
                 (-0.08, -0.20), (-0.08, -0.25), (-0.10, -0.25)]:
        m, _ = run(f"deadband {s}->{f}", dd_start=s, dd_floor=f); rows.append(m)
    print("=== DD THROTTLE SHAPE (2014-2018 ONLY) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    # gate interaction at two plateau throttles
    rows2 = []
    for s, f in [(0.0, -0.15), (-0.05, -0.20)]:
        for gp in (0.97, 0.99):
            for gm in (0.25, 0.5):
                m, _ = run(f"dd({s},{f}) gp{gp} gm{gm}",
                           dd_start=s, dd_floor=f, gate_pct=gp, gate_mult=gm)
                rows2.append(m)
    print("\n=== GATE x FIXED THROTTLE (2014-2018 ONLY) ===")
    print(pd.DataFrame(rows2).to_string(index=False))

    # per-year SR within segment for the leading candidates
    print("\n=== per-year Sharpe within 2014-2018 ===")
    cands = {"inert(base)": dict(fixed=False),
             "lin0->-0.15": dict(dd_floor=-0.15),
             "lin0->-0.20": dict(dd_floor=-0.20),
             "db-0.05->-0.20": dict(dd_start=-0.05, dd_floor=-0.20),
             "db-0.08->-0.25": dict(dd_start=-0.08, dd_floor=-0.25)}
    out = {}
    for nm, kw in cands.items():
        _, net = run(nm, **kw)
        r = net.loc[SEG_START:SEG_END]
        out[nm] = {y: round(prod._sharpe(g), 2) for y, g in r.groupby(r.index.year)}
    print(pd.DataFrame(out).to_string())


if __name__ == "__main__":
    main()
