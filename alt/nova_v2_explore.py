"""Exploratory search for an honest 50%-CAGR NOVA variant.

Everything here uses the CORRECTED mechanics:
  - momentum signal lagged 1 bar
  - regime gates lagged 1 bar
  - IS / OOS split at 2020-01-01 with both metrics reported

Search axes (wider than the original grid):
  - lookback_days: 20, 40, 60, 90, 120, 180, 240
  - top_n:         1, 2, 3, 4
  - cap:           0.50, 0.67, 1.00   (higher cap = more concentration)
  - rebal_days:    2, 3, 5, 10
  - overlay_lev:   1.0, 1.3, 1.5, 1.7, 2.0   (post-gate portfolio leverage
                    at short-rate funding cost)

Reports honest IS/OOS Sharpe and full-window metrics. A config is 'candidate'
if IS Sharpe > 0.4, OOS Sharpe > 0.4, IS-OOS Sharpe gap < 0.5, and full-window
CAGR >= 45%. Target: 50% CAGR honest.

Output: data/results/nova_v2_grid.csv
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
    reg_eq = ((spy_a > spy_a.rolling(200).mean()) & (vix < 30)).shift(1).fillna(False).astype(float).values
    btc_a = btc.reindex(dates).ffill()
    reg_bt = (btc_a > btc_a.rolling(200).mean()).shift(1).fillna(False).astype(float).values
    rf_daily = (load_fred("DGS3MO").reindex(dates).ffill() / 100.0 / 252.0).fillna(0).values

    return dates, universe, P, rets, avail, bil, reg_eq, reg_bt, rf_daily


def run(dates, universe, P, rets, avail, bil, reg_eq, reg_bt, rf,
        lookback, top_n, cap, rebal, overlay, tc_bps=15.0):
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])
    n = len(dates); m = len(universe)
    current = np.zeros(m)
    port = np.zeros(n)
    last_idx = -rebal
    for i in range(n):
        if i > lookback and i - last_idx >= rebal:
            live = avail[i - 1]
            denom = P[i - 1 - lookback]
            num = P[i - 1]
            momo = np.where(live, num / denom - 1.0, np.nan)
            valid = ~np.isnan(momo) & (momo > 0)
            order = np.argsort(-np.where(valid, momo, -np.inf))
            picks = [int(k) for k in order if valid[k]][:top_n]
            new = np.zeros(m)
            if picks:
                w = 1.0 / len(picks)
                w = min(w, cap)
                for k in picks: new[k] = w
            tc = np.sum(np.abs(new - current)) * (tc_bps / 1e4) * overlay
            port[i] -= tc
            current = new
            last_idx = i
        eff = current.copy()
        geq = reg_eq[i]; gbt = reg_bt[i]
        off_eq = current[eq_idx].sum() * (1 - geq)
        off_bt = current[cr_idx].sum() * (1 - gbt)
        eff[eq_idx] = current[eq_idx] * geq
        eff[cr_idx] = current[cr_idx] * gbt
        gross = (rets[i] * eff).sum() + (off_eq + off_bt) * bil[i]
        # Portfolio overlay: apply leverage to gross exposure, financing
        # cost on (overlay - 1) * invested at short rate. Cash bucket already
        # earns bil so no double financing cost on it.
        invested = eff.sum()  # <= 1
        levered = overlay * gross - (overlay - 1) * invested * rf[i]
        port[i] += levered
    return pd.Series(port, index=dates)


def stats(r):
    if len(r) < 2 or r.std() == 0:
        return 0.0, 0.0, 0.0, 0.0
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    return sr, ar * 100, av * 100, mdd * 100


def main():
    dates, universe, P, rets, avail, bil, reg_eq, reg_bt, rf = prep()
    cfgs = []
    for lb in [20, 40, 60, 90, 120, 180, 240]:
        for tn in [1, 2, 3, 4]:
            for cap in [0.50, 0.67, 1.00]:
                for rb in [2, 3, 5, 10]:
                    for ov in [1.0, 1.3, 1.5, 1.7, 2.0]:
                        cfgs.append((lb, tn, cap, rb, ov))
    print(f"Evaluating {len(cfgs)} configs...")
    rows = []
    for idx, (lb, tn, cap, rb, ov) in enumerate(cfgs):
        p = run(dates, universe, P, rets, avail, bil, reg_eq, reg_bt, rf,
                lb, tn, cap, rb, ov)
        s_is = stats(p.loc[:IS_END])
        s_os = stats(p.loc[IS_END:])
        s_fu = stats(p)
        rows.append({
            "lb": lb, "tn": tn, "cap": cap, "rb": rb, "ov": ov,
            "IS_SR": round(s_is[0], 3), "IS_Ret": round(s_is[1], 2), "IS_MDD": round(s_is[3], 1),
            "OOS_SR": round(s_os[0], 3), "OOS_Ret": round(s_os[1], 2), "OOS_MDD": round(s_os[3], 1),
            "FULL_SR": round(s_fu[0], 3), "FULL_Ret": round(s_fu[1], 2),
            "FULL_Vol": round(s_fu[2], 1), "FULL_MDD": round(s_fu[3], 1),
        })
        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(cfgs)} done")

    df = pd.DataFrame(rows)
    df["SR_gap"] = (df["IS_SR"] - df["OOS_SR"]).abs()
    df.to_csv(RESULTS / "nova_v2_grid.csv", index=False)

    cand = df[
        (df["IS_SR"] > 0.4) &
        (df["OOS_SR"] > 0.4) &
        (df["SR_gap"] < 0.5) &
        (df["FULL_Ret"] >= 45.0)
    ].sort_values("FULL_SR", ascending=False)

    print(f"\n=== {len(cand)} honest candidates hit >=45% CAGR with IS/OOS robustness ===")
    print(cand.head(25).to_string(index=False))

    print("\n=== Top 15 by IS Sharpe (for selection) ===")
    print(df.sort_values("IS_SR", ascending=False).head(15).to_string(index=False))

    print("\n=== Top 15 by min(IS,OOS) Sharpe (most robust) ===")
    df["min_sr"] = df[["IS_SR", "OOS_SR"]].min(axis=1)
    print(df.sort_values("min_sr", ascending=False).head(15).to_string(index=False))

    print("\n=== Top 15 by Full-window CAGR (may be overfit) ===")
    print(df.sort_values("FULL_Ret", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
