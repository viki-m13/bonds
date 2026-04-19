"""METEOR v2 exploration — simpler mechanics, focus on PDOT + NAV-trend.

After v1 showed vol-parity + dual-horizon filter over-fit exposure away,
v2 isolates the two mechanics that matter most for the MDD/CAGR frontier:

  (A) Path-Dependent Overlay Throttle (PDOT): de-lever vs rolling 252d HWM
      drawdown, linear between 0 and -dd_floor.
  (B) NAV-TREND CONFIRMATION: additional multiplier that goes to 0 when the
      strategy's own short-horizon NAV momentum is negative. Asymmetric:
      de-levers FAST on loss streaks but re-levers quickly once NAV prints
      a fresh 20d high.

Per-name selection is single-horizon momentum (top-N, equal weight, user-
chosen cap) — i.e. the same mechanics as v1, just with the dynamic overlay
stacked on top.

Grid: (lookback, top_n, cap, rebal, overlay_base, dd_floor, nav_trend_win).
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
        tc_bps=15.0):
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])
    n = len(dates); m = len(universe)
    current = np.zeros(m)
    port = np.zeros(n)
    nav = np.ones(n + 1)  # nav[0] = 1.0 baseline
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
                w = 1.0 / len(picks)
                w = min(w, cap)
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

        # PDOT throttle: DD vs rolling 252d HWM of NAV
        if i > 0:
            lo = max(1, i - 251)
            hwm = nav[lo:i + 1].max()
            dd = (nav[i] / hwm) - 1.0
        else:
            dd = 0.0
        pdot = max(0.0, 1.0 + dd / dd_floor)

        # NAV-trend: short-horizon NAV return. If negative, multiplier shrinks
        # linearly from 1 at 0% down to 0 at -nav_floor (= dd_floor/2).
        if i > nav_win and nav[i - nav_win] > 0:
            nav_mom = nav[i] / nav[i - nav_win] - 1.0
        else:
            nav_mom = 0.0
        nav_floor = dd_floor / 2.0
        trend_mult = max(0.0, min(1.0, 1.0 + nav_mom / nav_floor)) if nav_mom < 0 else 1.0

        overlay_t = overlay_base * pdot * trend_mult

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
    cfgs = []
    for lb in [60, 90, 120, 180]:
        for tn in [2, 3]:
            for cap in [0.50, 1.00]:
                for rb in [5, 10]:
                    for ov in [2.0, 2.5, 3.0, 4.0, 5.0]:
                        for dd in [0.15, 0.20, 0.25, 0.30]:
                            for nw in [10, 20, 40]:
                                cfgs.append((lb, tn, cap, rb, ov, dd, nw))

    print(f"Evaluating {len(cfgs)} configs...")
    rows = []
    for idx, c in enumerate(cfgs):
        lb, tn, cap, rb, ov, dd, nw = c
        p = run(dates, universe, P, R, avail, bil, reg_eq, reg_bt, rf,
                lb, tn, cap, rb, ov, dd, nw)
        s_is = stats(p.loc[:IS_END]); s_os = stats(p.loc[IS_END:]); s_fu = stats(p)
        rows.append({
            "lb": lb, "tn": tn, "cap": cap, "rb": rb, "ov": ov, "dd_f": dd, "nw": nw,
            "IS_SR": round(s_is[0], 3), "IS_Ret": round(s_is[1], 2), "IS_MDD": round(s_is[3], 1),
            "OOS_SR": round(s_os[0], 3), "OOS_Ret": round(s_os[1], 2), "OOS_MDD": round(s_os[3], 1),
            "FULL_SR": round(s_fu[0], 3), "FULL_Ret": round(s_fu[1], 2),
            "FULL_Vol": round(s_fu[2], 1), "FULL_MDD": round(s_fu[3], 1),
            "NAVx": round(s_fu[4], 1),
        })
        if (idx + 1) % 200 == 0: print(f"  {idx+1}/{len(cfgs)} done")

    df = pd.DataFrame(rows)
    df["SR_gap"] = (df["IS_SR"] - df["OOS_SR"]).abs()
    df["min_sr"] = df[["IS_SR","OOS_SR"]].min(axis=1)
    df["calmar"] = df["FULL_Ret"] / -df["FULL_MDD"]
    df.to_csv(RESULTS / "nova_meteor_v2_grid.csv", index=False)

    print(f"\n=== 50%+ CAGR AND MDD > -50% AND robust (IS/OOS SR > 0.5, gap < 0.4) ===")
    c50 = df[(df["FULL_Ret"] >= 50) & (df["FULL_MDD"] > -50)
             & (df["IS_SR"] > 0.5) & (df["OOS_SR"] > 0.5) & (df["SR_gap"] < 0.4)]
    print(f"{len(c50)} configs")
    if len(c50):
        print(c50.sort_values("calmar", ascending=False).head(20).to_string(index=False))

    print(f"\n=== 50%+ CAGR AND MDD > -60% ===")
    c60 = df[(df["FULL_Ret"] >= 50) & (df["FULL_MDD"] > -60)]
    print(f"{len(c60)} configs")
    if len(c60):
        print(c60.sort_values("calmar", ascending=False).head(20).to_string(index=False))

    print("\n=== Top 10 by Calmar (Ret/|MDD|), full period ===")
    print(df.sort_values("calmar", ascending=False).head(10).to_string(index=False))

    print("\n=== Top 10 by min(IS,OOS) Sharpe (robustness) ===")
    print(df.sort_values("min_sr", ascending=False).head(10).to_string(index=False))

    print("\n=== Top 10 by FULL Sharpe ===")
    print(df.sort_values("FULL_SR", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
