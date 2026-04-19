"""METEOR deep exploration — push for higher CAGR & lower MDD.

Adds mechanical knobs we hadn't searched before:

  - PDOT_WIN: rolling-HWM window for the drawdown throttle.
    Original: 252 (1 year). Try 63, 126, 252, 504 (3m..2y).
  - SKIP_RECENT: classic academic 12-1 momentum skips the last month
    to dodge short-term reversal. Try 0, 10, 21, 42.
  - NAV_FLOOR_MULT: in the NAV-trend asymmetric multiplier, the loss
    floor was hardcoded at dd_floor/2. Try 0.25, 0.50, 0.75, 1.00 of
    dd_floor.
  - TOP_N range extended to 1..6.
  - PDOT_FLOOR: also decoupled from dd_floor (default: same, but try
    0.20, 0.25, 0.30, 0.35, 0.40 independently).

No vol scaling — we've established it doesn't help.

Two-phase search:
  Phase 1 (one-axis-at-a-time): anchor at the known METEOR cell
    (lb=120, tn=3, cap=1.0, rb=21, ov=4.5, dd=0.30, nw=20,
     pdot_win=252, skip=0, nav_floor_mult=0.5), vary one knob.
  Phase 2 (joint): sweep the most productive dimensions together.

Output: data/results/nova_meteor_deep_grid.csv

No look-ahead (signal uses t-1 close, throttles use NAV through t-1).
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
        lookback, top_n, cap, rebal, overlay_base, pdot_floor, nav_win,
        pdot_win=252, skip_recent=0, nav_floor_mult=0.5, tc_bps=15.0):
    """
    Name signal = P[t-1-skip] / P[t-1-skip-lookback] - 1.
    PDOT  = max(0, 1 + DD_t / pdot_floor), DD vs rolling pdot_win HWM.
    NAV-trend = if nav_mom<0: max(0, 1 + nav_mom/(pdot_floor*nav_floor_mult)), else 1.
    """
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])
    n = len(dates); m = len(universe)
    current = np.zeros(m)
    port = np.zeros(n)
    nav = np.ones(n + 1)
    last_idx = -rebal
    tc_pending = 0.0
    nav_floor = pdot_floor * nav_floor_mult
    start_req = max(lookback + skip_recent + 2, 2)

    for i in range(n):
        if i > start_req and i - last_idx >= rebal:
            live = avail[i - 1 - skip_recent]
            num = P[i - 1 - skip_recent]
            den = P[i - 1 - skip_recent - lookback]
            mom = np.where(live, num / den - 1.0, np.nan)
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

        if i > 0:
            lo = max(1, i - (pdot_win - 1))
            hwm = nav[lo:i + 1].max()
            dd = (nav[i] / hwm) - 1.0
        else:
            dd = 0.0
        pdot = max(0.0, 1.0 + dd / pdot_floor)

        if i > nav_win and nav[i - nav_win] > 0:
            nav_mom = nav[i] / nav[i - nav_win] - 1.0
        else:
            nav_mom = 0.0
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


def eval_cfg(ctx, cfg):
    p = run(*ctx, **cfg)
    s_is = stats(p.loc[:IS_END]); s_os = stats(p.loc[IS_END:]); s_fu = stats(p)
    return {
        **{k: cfg[k] for k in ["lookback","top_n","cap","rebal","overlay_base",
                                "pdot_floor","nav_win","pdot_win","skip_recent","nav_floor_mult"]},
        "IS_SR": round(s_is[0], 3), "IS_Ret": round(s_is[1], 2), "IS_MDD": round(s_is[3], 1),
        "OOS_SR": round(s_os[0], 3), "OOS_Ret": round(s_os[1], 2), "OOS_MDD": round(s_os[3], 1),
        "FULL_SR": round(s_fu[0], 3), "FULL_Ret": round(s_fu[1], 2),
        "FULL_Vol": round(s_fu[2], 1), "FULL_MDD": round(s_fu[3], 1),
        "NAVx": round(s_fu[4], 1),
    }


def summary(df, label):
    df = df.copy()
    df["SR_gap"] = (df.IS_SR - df.OOS_SR).abs()
    df["min_sr"] = df[["IS_SR","OOS_SR"]].min(axis=1)
    df["calmar"] = df.FULL_Ret / -df.FULL_MDD
    print(f"\n=== {label} ===")
    print(df.sort_values("calmar", ascending=False).head(20).to_string(index=False))
    return df


def main():
    ctx = prep()

    BASE = dict(lookback=120, top_n=3, cap=1.0, rebal=21, overlay_base=4.5,
                pdot_floor=0.30, nav_win=20, pdot_win=252, skip_recent=0,
                nav_floor_mult=0.5)

    cfgs = [BASE]

    # Phase 1 — one axis at a time
    P1 = []
    for v in [30, 45, 60, 75, 90, 105, 120, 150, 180, 210, 240, 300, 360]:
        P1.append({**BASE, "lookback": v})
    for v in [1, 2, 3, 4, 5, 6]:
        P1.append({**BASE, "top_n": v})
    for v in [0.33, 0.50, 0.67, 1.0]:
        P1.append({**BASE, "cap": v})
    for v in [10, 15, 17, 19, 21, 23, 25, 28, 31, 42, 63]:
        P1.append({**BASE, "rebal": v})
    for v in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]:
        P1.append({**BASE, "overlay_base": v})
    for v in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        P1.append({**BASE, "pdot_floor": v})
    for v in [5, 7, 10, 15, 20, 30, 45, 60, 90]:
        P1.append({**BASE, "nav_win": v})
    for v in [21, 42, 63, 126, 189, 252, 378, 504, 756]:
        P1.append({**BASE, "pdot_win": v})
    for v in [0, 5, 10, 15, 21, 31, 42]:
        P1.append({**BASE, "skip_recent": v})
    for v in [0.25, 0.33, 0.50, 0.67, 0.75, 1.00]:
        P1.append({**BASE, "nav_floor_mult": v})

    # Dedup
    seen = set(); unique = []
    for c in [BASE] + P1:
        key = tuple(sorted(c.items()))
        if key in seen: continue
        seen.add(key); unique.append(c)
    cfgs = unique

    print(f"Phase 1: {len(cfgs)} configs")
    rows = []
    for idx, c in enumerate(cfgs):
        rows.append(eval_cfg(ctx, c))
        if (idx+1) % 20 == 0: print(f"  {idx+1}/{len(cfgs)}")
    df1 = pd.DataFrame(rows)
    df1 = summary(df1, "Phase 1 one-axis sweep — top 20 by Calmar")

    # Phase 2 — joint sweep in the most productive dimensions
    # Based on phase 1, joint-sweep the top axes likely to matter:
    # rebal × overlay × pdot_floor × pdot_win × nav_floor_mult × top_n
    # Keep lookback at 120 (phase 1 reliably picks 120), cap=1.0, skip=0.
    P2 = []
    for tn in [2, 3, 4]:
        for rb in [19, 21, 25, 31]:
            for ov in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]:
                for pf in [0.25, 0.30, 0.35, 0.40]:
                    for pw in [126, 189, 252, 378]:
                        for nfm in [0.33, 0.50, 0.75]:
                            for nw in [15, 20, 30]:
                                P2.append({**BASE, "top_n": tn, "rebal": rb, "overlay_base": ov,
                                           "pdot_floor": pf, "pdot_win": pw,
                                           "nav_floor_mult": nfm, "nav_win": nw})

    seen = set(); unique2 = []
    for c in P2:
        key = tuple(sorted(c.items()))
        if key in seen: continue
        seen.add(key); unique2.append(c)
    print(f"\nPhase 2: {len(unique2)} configs")
    rows2 = []
    for idx, c in enumerate(unique2):
        rows2.append(eval_cfg(ctx, c))
        if (idx+1) % 500 == 0: print(f"  {idx+1}/{len(unique2)}")
    df2 = pd.DataFrame(rows2)
    df2 = summary(df2, "Phase 2 joint sweep — top 20 by Calmar")

    # Write all
    df1["phase"] = 1; df2["phase"] = 2
    full = pd.concat([df1, df2], ignore_index=True)
    full.to_csv(RESULTS / "nova_meteor_deep_grid.csv", index=False)

    # Final summary: best 50%+ CAGR with robust IS/OOS
    print("\n=== 50%+ CAGR, MDD > -55%, min_sr > 0.55, gap < 0.5 ===")
    robust = full[(full.FULL_Ret >= 50) & (full.FULL_MDD > -55)
                  & (full.min_sr > 0.55) & (full.SR_gap < 0.5)]
    print(f"{len(robust)} configs")
    print(robust.sort_values("calmar", ascending=False).head(25).to_string(index=False))

    print("\n=== All >=50% CAGR AND MDD > -50% ===")
    elite = full[(full.FULL_Ret >= 50) & (full.FULL_MDD > -50)]
    print(f"{len(elite)} configs")
    print(elite.sort_values("calmar", ascending=False).head(25).to_string(index=False))

    print("\n=== All >=55% CAGR (any MDD), ranked by min_sr ===")
    hi = full[full.FULL_Ret >= 55]
    print(f"{len(hi)} configs")
    print(hi.sort_values("min_sr", ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
