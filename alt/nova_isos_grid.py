"""IS/OOS grid search on CORRECTED NOVA backtest.

Splits 2014-09 (BTC inception) into:
  IS  = 2014-09 .. 2019-12 (~5.3y)
  OOS = 2020-01 .. present (~6.2y)

Grid over (lookback, top_n, cap). Fixed regime gates (SPY>200dma & VIX<30,
BTC>200dma) — those were chosen ex-ante and not further swept here to keep
the selection surface small. Reports top configs by IS Sharpe and their OOS
performance, so a human can pick the parameters."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"

EQUITY = ["TQQQ","UPRO","SOXL","TECL","FAS","TMF","UGL","LABU","EDC","YINN",
          "ERX","NUGT","DRN","UCO","TYD","QLD","SSO","UBT"]
CRYPTO = ["BTC_USD","ETH_USD"]


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def backtest_vectorized(lookback, top_n, cap, tc_bps=15.0):
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

    n = len(dates); m = len(universe)
    eq_idx = np.array([universe.index(t) for t in EQUITY])
    cr_idx = np.array([universe.index(t) for t in CRYPTO])

    current = np.zeros(m)
    port = np.zeros(n)
    last_idx = -5
    rebal = 5

    for i in range(n):
        if i > lookback and i - last_idx >= rebal:
            live = avail[i - 1]
            denom = P[i - 1 - lookback]
            num = P[i - 1]
            momo = np.where(live, num / denom - 1.0, np.nan)
            # rank positive momentum desc
            valid = ~np.isnan(momo) & (momo > 0)
            order = np.argsort(-np.where(valid, momo, -np.inf))
            positive = [k for k in order if valid[k]][:top_n]
            new = np.zeros(m)
            if positive:
                w = 1.0 / len(positive)
                w = min(w, cap)
                for k in positive: new[k] = w
            tc = np.sum(np.abs(new - current)) * (tc_bps / 1e4)
            port[i] -= tc
            current = new
            last_idx = i
        eff = current.copy()
        geq = reg_eq[i]; gbt = reg_bt[i]
        off_eq = current[eq_idx].sum() * (1 - geq)
        off_bt = current[cr_idx].sum() * (1 - gbt)
        eff[eq_idx] = current[eq_idx] * geq
        eff[cr_idx] = current[cr_idx] * gbt
        port[i] += (rets[i] * eff).sum() + (off_eq + off_bt) * bil[i]

    return pd.Series(port, index=dates)


def stats(r):
    ar = r.mean() * 252
    av = r.std() * np.sqrt(252)
    sr = ar / av if av > 0 else 0
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    return sr, ar * 100, av * 100, mdd * 100, c.iloc[-1]


def main():
    IS_END = pd.Timestamp("2020-01-01")
    rows = []
    configs = []
    for lb in [10, 20, 30, 60, 90, 120]:
        for tn in [2, 3, 4, 5]:
            for cap in [0.25, 0.33, 0.50]:
                configs.append((lb, tn, cap))
    print(f"Evaluating {len(configs)} configs...")
    for idx, (lb, tn, cap) in enumerate(configs):
        p = backtest_vectorized(lb, tn, cap)
        is_p = p.loc[:IS_END]
        oos_p = p.loc[IS_END:]
        s_is = stats(is_p); s_oos = stats(oos_p); s_full = stats(p)
        rows.append({
            "lb": lb, "tn": tn, "cap": cap,
            "IS_SR": round(s_is[0], 3), "IS_Ret": round(s_is[1], 2), "IS_MDD": round(s_is[3], 1),
            "OOS_SR": round(s_oos[0], 3), "OOS_Ret": round(s_oos[1], 2), "OOS_MDD": round(s_oos[3], 1),
            "FULL_SR": round(s_full[0], 3), "FULL_Ret": round(s_full[1], 2), "FULL_MDD": round(s_full[3], 1),
        })
        if (idx + 1) % 10 == 0:
            print(f"  {idx+1}/{len(configs)} done")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/results/nova_isos_grid.csv", index=False)
    print("\n=== Top 15 by IS Sharpe ===")
    print(df.sort_values("IS_SR", ascending=False).head(15).to_string(index=False))
    print("\n=== Most robust (min of IS and OOS Sharpe) ===")
    df["min_sr"] = df[["IS_SR", "OOS_SR"]].min(axis=1)
    print(df.sort_values("min_sr", ascending=False).head(15).to_string(index=False))
    print("\n=== Top 15 by OOS Sharpe (overfitting check) ===")
    print(df.sort_values("OOS_SR", ascending=False).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
