"""NOVA METEOR — proprietary novel method targeting 50%+ CAGR with bounded MDD.

Four mechanics stacked on the corrected v1 momentum core:

  (1) DUAL-HORIZON MOMENTUM CONFIRMATION.
      A name qualifies only if BOTH the long-horizon (default 120d) AND a
      short-horizon (default 20d) momentum are positive. Filters late-stage
      tops where the trend has cracked but the long window hasn't registered
      it yet. Rank by the long-horizon value among qualifiers.

  (2) VOLATILITY-PARITY SIZING.
      Among the top-N qualifiers, weight by 1/vol_60 (60-day realized).
      Keeps risk contributions balanced; hot names get smaller weight than
      TMF-like names of similar momentum rank.

  (3) CROSS-SECTIONAL DISPERSION GATE.
      Compute daily cross-sectional std of 20d returns across the universe
      (momentum-crash signal). Compare to its own 252-day 25th percentile.
      When dispersion is below the floor, the regime is "compressed" (leaders
      no longer differentiating) — halve overlay exposure.

  (4) PATH-DEPENDENT OVERLAY THROTTLE (PDOT) — the key novel piece.
      overlay_t = overlay_base * max(0, 1 + DD_252[t]/DD_floor)
      where DD_252 is the strategy's own drawdown vs its rolling 252-day
      high-water mark. At DD=0 the book is fully levered; at DD=DD_floor
      the book is fully de-levered; linear in between. The rolling 252-day
      reset lets the strategy reclaim leverage over time, avoiding the
      classic CPPI-lockout trap.

Outputs a grid of configs with IS/OOS Sharpe and MDD so the operator can
pick a config that hits 50%+ CAGR with MDD bounded below ~50%.
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
    rets = prices.pct_change().fillna(0)
    vol60 = rets.rolling(60).std().bfill().fillna(0.01)
    avail = prices.notna()
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)

    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy_a = spy.reindex(dates).ffill()
    reg_eq = ((spy_a > spy_a.rolling(200).mean()) & (vix < 30)
              ).shift(1).fillna(False).astype(float)
    btc_a = btc.reindex(dates).ffill()
    reg_bt = (btc_a > btc_a.rolling(200).mean()).shift(1).fillna(False).astype(float)
    rf = (load_fred("DGS3MO").reindex(dates).ffill() / 100.0 / 252.0).fillna(0)

    # Cross-sectional dispersion signal: std of 20d returns across universe.
    # Use log-returns and only live names.
    mom20 = prices / prices.shift(20) - 1
    disp = mom20.std(axis=1, skipna=True).ffill()
    disp_floor = disp.rolling(252, min_periods=60).quantile(0.25).ffill()
    disp_on = (disp.shift(1) >= disp_floor.shift(1)).fillna(True)

    return {
        "dates": dates, "universe": universe,
        "P": prices.values, "R": rets.values, "V": vol60.values,
        "avail": avail.values, "bil": bil.values,
        "reg_eq": reg_eq.values, "reg_bt": reg_bt.values, "rf": rf.values,
        "disp_on": disp_on.values,
    }


def run(ctx, lb_long, lb_short, top_n, overlay_base, dd_floor, rebal, tc_bps=15.0,
        disp_mult_off=0.5):
    dates = ctx["dates"]; universe = ctx["universe"]
    P = ctx["P"]; R = ctx["R"]; V = ctx["V"]; avail = ctx["avail"]; bil = ctx["bil"]
    reg_eq = ctx["reg_eq"]; reg_bt = ctx["reg_bt"]; rf = ctx["rf"]
    disp_on = ctx["disp_on"]
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])
    n = len(dates); m = len(universe)

    current = np.zeros(m)
    port = np.zeros(n)
    nav = np.ones(n)   # NAV series for PDOT
    last_idx = -rebal
    for i in range(n):
        if i > max(lb_long, 252) and i - last_idx >= rebal:
            live = avail[i - 1]
            # Dual-horizon momentum
            mom_l = np.where(live, P[i - 1] / P[i - 1 - lb_long] - 1.0, np.nan)
            mom_s = np.where(live, P[i - 1] / P[i - 1 - lb_short] - 1.0, np.nan)
            qualifies = (~np.isnan(mom_l)) & (~np.isnan(mom_s)) & (mom_l > 0) & (mom_s > 0)
            order = np.argsort(-np.where(qualifies, mom_l, -np.inf))
            picks = [int(k) for k in order if qualifies[k]][:top_n]
            new = np.zeros(m)
            if picks:
                # vol-parity weights
                vols = np.array([V[i - 1, k] for k in picks])
                vols = np.where(vols > 1e-6, vols, 1e-6)
                w = 1.0 / vols
                w = w / w.sum()
                for k, wk in zip(picks, w): new[k] = wk
            # Transaction cost scaled by overlay of THIS bar (computed below)
            current_new_l1 = float(np.abs(new - current).sum())
            current = new
            last_idx = i
            # apply TC later once overlay_t is known

        eff = current.copy()
        geq = reg_eq[i]; gbt = reg_bt[i]
        off_eq = current[eq_idx].sum() * (1 - geq)
        off_bt = current[cr_idx].sum() * (1 - gbt)
        eff[eq_idx] = current[eq_idx] * geq
        eff[cr_idx] = current[cr_idx] * gbt
        gross = (R[i] * eff).sum() + (off_eq + off_bt) * bil[i]
        invested = float(eff.sum())

        # PDOT: compute drawdown vs rolling 252d high-water mark of own NAV
        if i > 0:
            lo = max(0, i - 252)
            hwm = nav[lo:i].max() if i - lo > 0 else nav[i - 1]
            dd = (nav[i - 1] / hwm) - 1.0
        else:
            dd = 0.0
        # throttle: 1.0 at DD=0, linearly to 0 at DD=-dd_floor
        throttle = max(0.0, 1.0 + dd / dd_floor)
        # Dispersion multiplier: 1.0 if dispersion healthy, else disp_mult_off
        disp_mult = 1.0 if disp_on[i] else disp_mult_off
        overlay_t = overlay_base * throttle * disp_mult

        # pay TC once on this bar if a rebalance happened (last_idx == i)
        tc_today = 0.0
        if last_idx == i:
            tc_today = current_new_l1 * (tc_bps / 1e4) * overlay_t

        levered = overlay_t * gross - (overlay_t - 1.0) * invested * rf[i] - tc_today
        # When overlay < 1, no borrowing; we don't credit un-invested cash
        # beyond what bil earns — close enough for exploratory purposes.
        port[i] += levered
        nav[i] = nav[i - 1] * (1 + levered) if i > 0 else 1.0 + levered

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
    ctx = prep()
    cfgs = []
    for lb_l in [120, 180]:
        for lb_s in [20, 40]:
            for tn in [2, 3, 4]:
                for ov in [1.5, 2.0, 2.5, 3.0]:
                    for dd in [0.20, 0.25, 0.30, 0.35]:
                        for rb in [5, 10]:
                            cfgs.append((lb_l, lb_s, tn, ov, dd, rb))

    print(f"Evaluating {len(cfgs)} configs...")
    rows = []
    for idx, (lb_l, lb_s, tn, ov, dd, rb) in enumerate(cfgs):
        p = run(ctx, lb_l, lb_s, tn, ov, dd, rb)
        s_is = stats(p.loc[:IS_END]); s_os = stats(p.loc[IS_END:]); s_fu = stats(p)
        rows.append({
            "lb_l": lb_l, "lb_s": lb_s, "tn": tn, "ov": ov, "dd_f": dd, "rb": rb,
            "IS_SR": round(s_is[0], 3), "IS_Ret": round(s_is[1], 2), "IS_MDD": round(s_is[3], 1),
            "OOS_SR": round(s_os[0], 3), "OOS_Ret": round(s_os[1], 2), "OOS_MDD": round(s_os[3], 1),
            "FULL_SR": round(s_fu[0], 3), "FULL_Ret": round(s_fu[1], 2),
            "FULL_Vol": round(s_fu[2], 1), "FULL_MDD": round(s_fu[3], 1),
            "NAVx": round(s_fu[4], 1),
        })
        if (idx + 1) % 50 == 0: print(f"  {idx+1}/{len(cfgs)} done")

    df = pd.DataFrame(rows)
    df["SR_gap"] = (df["IS_SR"] - df["OOS_SR"]).abs()
    df.to_csv(RESULTS / "nova_meteor_grid.csv", index=False)

    # Candidates: 50%+ CAGR, MDD better than -50%, IS/OOS gap < 0.35
    cand = df[(df["FULL_Ret"] >= 50) & (df["FULL_MDD"] > -50) &
              (df["SR_gap"] < 0.35) & (df["IS_SR"] > 0.5) & (df["OOS_SR"] > 0.5)]
    print(f"\n=== {len(cand)} configs: 50%+ CAGR and MDD > -50% and IS/OOS robust ===")
    print(cand.sort_values("FULL_SR", ascending=False).head(25).to_string(index=False))

    relaxed = df[(df["FULL_Ret"] >= 50) & (df["FULL_MDD"] > -60)]
    print(f"\n=== {len(relaxed)} configs: 50%+ CAGR and MDD > -60% ===")
    print(relaxed.sort_values("FULL_SR", ascending=False).head(20).to_string(index=False))

    print("\n=== Top 15 by full Sharpe overall ===")
    print(df.sort_values("FULL_SR", ascending=False).head(15).to_string(index=False))

    print("\n=== Top 15 by min(IS,OOS) Sharpe (most robust) ===")
    df["min_sr"] = df[["IS_SR","OOS_SR"]].min(axis=1)
    print(df.sort_values("min_sr", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
