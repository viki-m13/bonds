"""Sleeve B — 12-1 cross-sectional momentum, top-N equal weight, monthly,
on the survivorship-clean PIT liquidity universe (delisting-inclusive).

Honesty features:
  - universe mask from PREVIOUS month-end (no same-bar universe selection)
  - execution either at the signal month-end close (MOC ~3:55 signal) or at
    the NEXT day's close (fully lagged) -- both reported
  - full costs: auction fills, per-name impact vs its own dollar-ADV at
    trade time (book size $1M default), turnover measured exactly
  - delisted holdings: position frozen at last price into cash (their final
    crash IS in the adjusted series), charged an exit cost

Variants: plain 12-1; vol-scaled (Barroso 126d, cap 2x); skip-month robust
(12-2); N in {20, 50}.
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import fill_cost_bps, riskfree_daily, stats, fmt, OUT

t0 = time.time()
C = pd.read_parquet(os.path.join(OUT, "clean_close.parquet"))
DADV = pd.read_parquet(os.path.join(OUT, "clean_dadv.parquet"))
UNIV = pd.read_parquet(os.path.join(OUT, "univ_mask_monthly.parquet"))
C = C.astype("float64")
rets = C.pct_change(fill_method=None)
month_ends = UNIV.index
BOOK = 1_000_000.0


def run(n_top=20, skip=21, look=252, volscale=False, lag_days=0,
        start="2000-01-31", costs=True):
    """Monthly momentum book. Returns net daily returns Series."""
    me = month_ends[(month_ends >= pd.Timestamp(start))
                    & (month_ends <= C.index[-1])]
    daily = []
    prev_hold = pd.Index([])
    lev_hist = []
    port_r_hist = pd.Series(dtype=float)
    for k in range(1, len(me)):
        d_sig, d_next = me[k - 1], me[k]
        univ = UNIV.loc[:d_sig].iloc[-1]          # prior month-end universe
        univ = univ[univ].index
        p_sig = C.index.searchsorted(d_sig)
        if p_sig < look + 5:
            continue
        px_now = C.iloc[p_sig - skip][univ]
        px_then = C.iloc[p_sig - look][univ]
        mom = (px_now / px_then - 1).dropna()
        # must still be alive at signal date
        alive = C.iloc[p_sig][univ].dropna().index
        mom = mom[mom.index.isin(alive)]
        if len(mom) < n_top * 2:
            continue
        hold = mom.nlargest(n_top).index
        # leverage from vol-scaling (target 15% ann on 126d realized)
        lev = 1.0
        if volscale and len(port_r_hist) > 130:
            rv = port_r_hist.iloc[-126:].std() * np.sqrt(252)
            lev = float(np.clip(0.15 / max(rv, 1e-4), 0.3, 2.0))
        # holding-period daily returns (exec lag pushes entry by lag_days)
        i_from = p_sig + 1 + lag_days
        i_to = C.index.searchsorted(d_next) + lag_days
        i_to = min(i_to, len(C) - 1)
        seg = rets.iloc[i_from:i_to + 1][hold]
        port = seg.mean(axis=1).fillna(0) * lev   # EW; dead names ffilled->0
        # ---- costs on the rebalance ----
        if costs:
            enter = hold.difference(prev_hold)
            exit_ = prev_hold.difference(hold)
            w = lev / n_top
            cost_frac = 0.0
            adv_row = DADV.iloc[p_sig]
            px_row = C.iloc[p_sig]
            for t in list(enter) + list(exit_):
                adv = float(adv_row.get(t, np.nan))
                px = float(px_row.get(t, np.nan))
                if not np.isfinite(px):
                    px, adv = 50.0, 5e6          # delisted exit: worst tier
                bps = fill_cost_bps("auction", px, BOOK * w, adv)
                cost_frac += w * bps / 1e4
            if len(port):
                port.iloc[0] -= cost_frac
        daily.append(port)
        prev_hold = hold
        port_r_hist = pd.concat([port_r_hist, port])
        lev_hist.append(lev)
    r = pd.concat(daily)
    r = r[~r.index.duplicated()]
    return r, np.mean(lev_hist)


if __name__ == "__main__":
    rf = riskfree_daily(C.index)
    print("Sleeve B: clean-universe momentum (2000-2026), $1M book")
    res = {}
    for label, kw in [
        ("12-1 top20 MOC t, costed",      dict(n_top=20, lag_days=0)),
        ("12-1 top20 close t+1, costed",  dict(n_top=20, lag_days=1)),
        ("12-1 top20 FREE",               dict(n_top=20, costs=False)),
        ("12-1 top50 MOC t, costed",      dict(n_top=50, lag_days=0)),
        ("12-2 top20 MOC t, costed",      dict(n_top=20, skip=42)),
        ("12-1 top20 volscaled, costed",  dict(n_top=20, volscale=True)),
        ("12-1 top50 volscaled, costed",  dict(n_top=50, volscale=True)),
    ]:
        r, lev = run(**kw)
        st = stats(r, rf, label)
        print(fmt(st) + f"  avg lev {lev:.2f}")
        res[label] = r
    keep = pd.DataFrame({
        "mom20": res["12-1 top20 MOC t, costed"],
        "mom20_lag": res["12-1 top20 close t+1, costed"],
        "mom50": res["12-1 top50 MOC t, costed"],
        "mom20_vs": res["12-1 top20 volscaled, costed"],
        "mom50_vs": res["12-1 top50 volscaled, costed"],
    })
    keep.to_parquet(os.path.join(OUT, "sleeveB_momentum.parquet"))
    print(f"saved -> out/sleeveB_momentum.parquet  t={time.time()-t0:.0f}s")
