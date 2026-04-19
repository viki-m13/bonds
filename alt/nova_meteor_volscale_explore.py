"""METEOR + vol-targeting explore.

Adds a THIRD multiplier on top of PDOT and NAV-trend: a realized-vol
target.

    vol_mult = min(1, vol_target / realised_vol_60d(strategy))

Realised vol is computed on the strategy's OWN pre-overlay return
series over a rolling 60-day window, annualised. When vol is below
target, multiplier is capped at 1 (we don't create leverage above
overlay_base from the vol layer). When vol is above target, multiplier
shrinks proportionally.

Tests whether adding vol targeting improves Calmar vs the pure
PDOT+NAV-trend METEOR (rb=21, tn=3, ov=4.5, dd=0.30, nw=20).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

EQUITY = ["TQQQ","UPRO","SOXL","TECL","FAS","TMF","UGL","LABU","EDC","YINN",
          "ERX","NUGT","DRN","UCO","TYD","QLD","SSO","UBT"]
CRYPTO = ["BTC_USD","ETH_USD"]
IS_END = pd.Timestamp("2020-01-01")


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def prep():
    btc = load_etf("BTC_USD"); spy = load_etf("SPY")
    dates = spy.loc[btc.index.min():].index
    universe = EQUITY + CRYPTO
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
    rets = prices.pct_change().fillna(0).values
    avail = prices.notna().values
    P = prices.values
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0).values
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy_a = spy.reindex(dates).ffill()
    reg_eq = ((spy_a > spy_a.rolling(200).mean()) & (vix < 30)
              ).shift(1).fillna(False).astype(float).values
    btc_a = btc.reindex(dates).ffill()
    reg_bt = (btc_a > btc_a.rolling(200).mean()).shift(1).fillna(False).astype(float).values
    rf = (load_fred("DGS3MO").reindex(dates).ffill() / 100.0 / 252.0).fillna(0).values
    return dates, universe, P, rets, avail, bil, reg_eq, reg_bt, rf


def run(dates, universe, P, R, avail, bil, reg_eq, reg_bt, rf,
        lookback, top_n, cap, rebal, overlay_base, dd_floor, nav_win,
        vol_target, vol_win=60, tc_bps=15.0):
    """vol_target in annualised units (0.60 = 60%). Set very high to disable."""
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])
    n = len(dates); m = len(universe)
    current = np.zeros(m)
    port = np.zeros(n)
    gross_hist = np.zeros(n)   # pre-overlay gross returns for vol estimate
    nav = np.ones(n + 1)
    last_idx = -rebal
    tc_pending = 0.0

    for i in range(n):
        if i > lookback and i - last_idx >= rebal:
            live = avail[i - 1]
            mom = np.where(live, P[i - 1] / P[i - 1 - lookback] - 1.0, np.nan)
            valid = ~np.isnan(mom) & (mom > 0)
            order = np.argsort(-np.where(valid, mom, -np.inf))
            picks = [int(k) for k in order if valid[k]][:top_n]
            new = np.zeros(m)
            if picks:
                w = min(1.0 / len(picks), cap)
                for k in picks: new[k] = w
            tc_pending = float(np.abs(new - current).sum())
            current = new
            last_idx = i

        eff = current.copy()
        geq = reg_eq[i]; gbt = reg_bt[i]
        off_eq = current[eq_idx].sum() * (1 - geq)
        off_bt = current[cr_idx].sum() * (1 - gbt)
        eff[eq_idx] = current[eq_idx] * geq
        eff[cr_idx] = current[cr_idx] * gbt
        gross = (R[i] * eff).sum() + (off_eq + off_bt) * bil[i]
        invested = float(eff.sum())
        gross_hist[i] = gross

        # PDOT
        if i > 0:
            lo = max(1, i - 251)
            hwm = nav[lo:i + 1].max()
            dd = (nav[i] / hwm) - 1.0
        else:
            dd = 0.0
        pdot = max(0.0, 1.0 + dd / dd_floor)

        # NAV-trend
        if i > nav_win and nav[i - nav_win] > 0:
            nav_mom = nav[i] / nav[i - nav_win] - 1.0
        else:
            nav_mom = 0.0
        nav_floor = dd_floor / 2.0
        trend_mult = max(0.0, min(1.0, 1.0 + nav_mom / nav_floor)) if nav_mom < 0 else 1.0

        # Vol-target layer: realised vol on the pre-overlay gross series
        if i > vol_win:
            v = gross_hist[i - vol_win:i].std() * np.sqrt(252)
            vol_mult = min(1.0, vol_target / v) if v > 1e-6 else 1.0
        else:
            vol_mult = 1.0

        overlay_t = overlay_base * pdot * trend_mult * vol_mult

        tc_today = 0.0
        if last_idx == i and tc_pending > 0:
            tc_today = tc_pending * (tc_bps / 1e4) * overlay_t
            tc_pending = 0.0

        levered = overlay_t * gross - (overlay_t - 1.0) * invested * rf[i] - tc_today
        port[i] = levered
        nav[i + 1] = nav[i] * (1 + levered)

    return pd.Series(port, index=dates)


def stats(r):
    if len(r) < 2 or r.std() == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    return sr, ar * 100, av * 100, mdd * 100, c.iloc[-1]


def main():
    dates, universe, P, R, avail, bil, reg_eq, reg_bt, rf = prep()

    # Hold the winning METEOR cell fixed, sweep vol_target + vol_win.
    # Also include vol_target=99.0 (effectively disabled) as baseline.
    cfgs = []
    base = dict(lookback=120, top_n=3, cap=1.0, rebal=21, overlay_base=4.5,
                dd_floor=0.30, nav_win=20)
    for vt in [99.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00, 1.20]:
        for vw in [30, 60, 90]:
            cfgs.append({**base, "vol_target": vt, "vol_win": vw})

    # Also sweep overlay at each vol_target — with vol targeting, we
    # might tolerate higher base overlay.
    for ov in [4.5, 5.5, 7.0, 10.0]:
        for vt in [0.40, 0.50, 0.60, 0.70, 0.80]:
            for vw in [60]:
                cfgs.append({**base, "overlay_base": ov,
                             "vol_target": vt, "vol_win": vw})

    print(f"{len(cfgs)} configs")
    rows = []
    for c in cfgs:
        p = run(dates, universe, P, R, avail, bil, reg_eq, reg_bt, rf, **c)
        s_is = stats(p.loc[:IS_END]); s_os = stats(p.loc[IS_END:]); s_fu = stats(p)
        rows.append({
            "ov": c["overlay_base"], "vt": c["vol_target"], "vw": c["vol_win"],
            "IS_SR": round(s_is[0], 3), "IS_Ret": round(s_is[1], 2), "IS_MDD": round(s_is[3], 1),
            "OOS_SR": round(s_os[0], 3), "OOS_Ret": round(s_os[1], 2), "OOS_MDD": round(s_os[3], 1),
            "FULL_SR": round(s_fu[0], 3), "FULL_Ret": round(s_fu[1], 2),
            "FULL_Vol": round(s_fu[2], 1), "FULL_MDD": round(s_fu[3], 1),
            "NAVx": round(s_fu[4], 1),
        })
    df = pd.DataFrame(rows)
    df["SR_gap"] = (df.IS_SR - df.OOS_SR).abs()
    df["min_sr"] = df[["IS_SR","OOS_SR"]].min(axis=1)
    df["calmar"] = df.FULL_Ret / -df.FULL_MDD
    df.to_csv(RESULTS / "nova_meteor_voltarget_sweep.csv", index=False)

    print("\n=== Baseline (vt=99.0 = no vol target) vs vol-target layer ===")
    print(df.sort_values(["ov","vt","vw"]).to_string(index=False))

    print("\n=== Top 15 by Calmar ===")
    print(df.sort_values("calmar", ascending=False).head(15).to_string(index=False))

    print("\n=== 50%+ CAGR configs ranked by Calmar ===")
    c = df[df.FULL_Ret >= 50]
    print(f"{len(c)} configs")
    print(c.sort_values("calmar", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
