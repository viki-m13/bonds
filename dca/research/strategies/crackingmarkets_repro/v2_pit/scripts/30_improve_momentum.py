"""Momentum sleeve improvement — dev/holdout disciplined.

All candidate levers are independently documented (no parameter mining):
  - narrower/megacap universe (top-100 dollar-ADV)     [size/liquidity tilt]
  - risk-adjusted ranking mom/vol                      [Sharpe momentum]
  - 12-2 skip window                                   [reversal avoidance]
  - top-50 breadth vs top-20                           [idiosyncratic dilution]
  - absolute-momentum gate: book -> cash when SPY<200DMA  [Antonacci]
  - Barroso-Santa-Clara vol scaling to 15% target

Selection on DEV (2000-2015) Sharpe only; HOLDOUT (2016-2026) reported but
not used for selection. Costs + MOC-t execution throughout.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_tiingo, fill_cost_bps, riskfree_daily, stats, fmt, OUT

t0 = time.time()
C = pd.read_parquet(os.path.join(OUT, "clean_close.parquet")).astype("float64")
DADV = pd.read_parquet(os.path.join(OUT, "clean_dadv.parquet")).astype("float64")
UNIV = pd.read_parquet(os.path.join(OUT, "univ_mask_monthly.parquet"))
rets = C.pct_change(fill_method=None)
month_ends = UNIV.index
BOOK = 1_000_000.0

ac, _ = load_tiingo()
spy = ac["SPY"].astype("float64").dropna()
spy_gate = (spy > spy.rolling(200, min_periods=200).mean())


def run(n_top=20, skip=21, look=252, univ_top=500, rank="plain", gate=False,
        volscale=False, lag_days=0, start="2000-01-31", costs=True):
    me = month_ends[(month_ends >= pd.Timestamp(start))
                    & (month_ends <= C.index[-1])]
    daily, prev_hold, expo = [], pd.Index([]), []
    port_r_hist = pd.Series(dtype=float)
    vol252 = rets.rolling(252, min_periods=200).std()
    for k in range(1, len(me)):
        d_sig, d_next = me[k - 1], me[k]
        univ = UNIV.loc[:d_sig].iloc[-1]
        univ = univ[univ].index
        p_sig = C.index.searchsorted(d_sig)
        if p_sig < look + 5:
            continue
        if univ_top < 500:
            adv = DADV.iloc[p_sig][univ].dropna()
            univ = adv.nlargest(univ_top).index
        mom = (C.iloc[p_sig - skip][univ] / C.iloc[p_sig - look][univ] - 1).dropna()
        alive = C.iloc[p_sig][univ].dropna().index
        mom = mom[mom.index.isin(alive)]
        if rank == "sharpe":
            v = vol252.iloc[p_sig][mom.index] * np.sqrt(252)
            mom = (mom / v.replace(0, np.nan)).dropna()
        if len(mom) < n_top * 2:
            continue
        hold = mom.nlargest(n_top).index
        gate_on = True
        if gate:
            g = spy_gate.loc[:d_sig]
            gate_on = bool(g.iloc[-1]) if len(g) else True
        lev = 1.0
        if volscale and len(port_r_hist) > 130:
            rv = port_r_hist.iloc[-126:].std() * np.sqrt(252)
            lev = float(np.clip(0.15 / max(rv, 1e-4), 0.3, 2.0))
        w_scale = lev if gate_on else 0.0
        i_from = p_sig + 1 + lag_days
        i_to = min(C.index.searchsorted(d_next) + lag_days, len(C) - 1)
        seg = rets.iloc[i_from:i_to + 1][hold]
        port = seg.mean(axis=1).fillna(0) * w_scale
        cur_hold = hold if w_scale > 0 else pd.Index([])
        if costs:
            enter = cur_hold.difference(prev_hold)
            exit_ = prev_hold.difference(cur_hold)
            w = max(w_scale, 0.5) / n_top
            cost_frac = 0.0
            adv_row, px_row = DADV.iloc[p_sig], C.iloc[p_sig]
            for t in list(enter) + list(exit_):
                a, px = float(adv_row.get(t, np.nan)), float(px_row.get(t, np.nan))
                if not np.isfinite(px):
                    px, a = 50.0, 5e6
                cost_frac += w * fill_cost_bps("auction", px, BOOK * w, a) / 1e4
            if len(port):
                port.iloc[0] -= cost_frac
        daily.append(port)
        prev_hold = cur_hold
        port_r_hist = pd.concat([port_r_hist, port])
        expo.append(pd.Series(w_scale, index=port.index))
    r = pd.concat(daily)
    e = pd.concat(expo)
    return r[~r.index.duplicated()], e[~e.index.duplicated()]


if __name__ == "__main__":
    rf = riskfree_daily(C.index)
    DEV_END = "2015-12-31"
    variants = [
        ("base 12-1 top20 u500",                dict()),
        ("12-2 top50 u500",                     dict(n_top=50, skip=42)),
        ("12-2 top50 u100 megacap",             dict(n_top=50, skip=42, univ_top=100)),
        ("12-2 top20 u100 megacap",             dict(n_top=20, skip=42, univ_top=100)),
        ("12-2 top50 u500 sharpe-rank",         dict(n_top=50, skip=42, rank="sharpe")),
        ("12-2 top50 u100 sharpe-rank",         dict(n_top=50, skip=42, univ_top=100, rank="sharpe")),
        ("12-2 top50 u500 + SPY gate",          dict(n_top=50, skip=42, gate=True)),
        ("12-2 top50 u500 sharpe + gate",       dict(n_top=50, skip=42, rank="sharpe", gate=True)),
        ("12-2 top50 u500 sharpe + gate + vs",  dict(n_top=50, skip=42, rank="sharpe", gate=True, volscale=True)),
        ("12-2 top50 u100 sharpe + gate",       dict(n_top=50, skip=42, univ_top=100, rank="sharpe", gate=True)),
    ]
    print(f"{'variant':44s} {'DEV Sh(d)':>9} {'CAGR':>7} | {'HOLD Sh(d)':>10} {'CAGR':>7} {'maxDD':>7}")
    best, results, expos = None, {}, {}
    for label, kw in variants:
        r, e = run(**kw)
        d = r.loc[:DEV_END]
        h = r.loc["2016-01-01":]
        sd, sh = stats(d, rf, ""), stats(h, rf, "")
        print(f"{label:44s} {sd['Sharpe_d']:9.2f} {sd['CAGR']*100:6.1f}% | "
              f"{sh['Sharpe_d']:10.2f} {sh['CAGR']*100:6.1f}% {sh['maxDD']*100:6.1f}%")
        results[label] = r
        expos[label] = e
        if best is None or sd["Sharpe_d"] > best[1]:
            best = (label, sd["Sharpe_d"])
    print(f"\nDEV pick: {best[0]}")
    pd.DataFrame({"mom_best": results[best[0]],
                  "mom_base": results["base 12-1 top20 u500"]}) \
        .to_parquet(os.path.join(OUT, "sleeveB_momentum_best.parquet"))
    pd.DataFrame({"mom_best": expos[best[0]]}) \
        .to_parquet(os.path.join(OUT, "sleeveB_momentum_expo.parquet"))
    print(f"saved -> out/sleeveB_momentum_best.parquet t={time.time()-t0:.0f}s")
