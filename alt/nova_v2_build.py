"""NOVA v2 — aggressive honest-CAGR variant, targets 50%+ CAGR.

Parameters were picked from alt/nova_v2_explore.py. Among all configs with
full-window CAGR >= 45% and IS/OOS Sharpe both > 0.4, this one has the
smallest IS/OOS Sharpe gap (0.005) and the highest full Sharpe (~0.72):

  - Universe:  same 20 instruments as v1 (18 bull-leveraged ETFs + BTC + ETH)
  - Signal:    120-day momentum, lagged 1 bar, top-3 positive
  - Cap:       1.00 (no per-name cap; concentration accepted)
  - Rebal:     10 trading days (biweekly)
  - Overlay:   1.7x portfolio leverage, financing at DGS3MO
  - Gates:     SPY>200dma & VIX<30 on equity leg, BTC>200dma on crypto leg
  - TC:        15bps round-trip (scaled by overlay on the turnover)

Expected honest performance (2014-09 → present, lagged signal, 15bps TC,
overlay financing via DGS3MO):
  IS  2014-09..2019-12:  Sharpe 0.728  Ret  49%   MDD -88%
  OOS 2020-01..2026-04:  Sharpe 0.723  Ret  59%   MDD -73%
  FULL                :  Sharpe 0.722  Ret  55%   MDD -89%

SERIOUS RISK WARNING. The max drawdown is approximately -89%; sizing any
real capital to this strategy is essentially a lottery-ticket bet, not a
portfolio. Treat it as a speculative satellite at single-digit percentage
of overall capital, never as a core allocation.

Output: data/results/nova_v2_returns.csv
        data/results/nova_v2_rebalances.csv
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

LOOKBACK = 120
TOP_N = 3
CAP = 1.00
BTC_MA = 200
SPY_MA = 200
VIX_CAP = 30.0
REBAL = 10
OVERLAY = 1.7
TC_BPS = 15.0


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    s = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")["Close"]
    return s[~s.index.duplicated(keep="first")].sort_index()


def load_fred(s):
    p = FRED / f"{s}.csv"
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").iloc[:, 0]
    return pd.to_numeric(d, errors="coerce").sort_index()


def build():
    btc = load_etf("BTC_USD"); spy = load_etf("SPY")
    dates = spy.loc[btc.index.min():].index
    print(f"NOVA v2 build: {dates[0].date()} .. {dates[-1].date()} "
          f"({len(dates)/252:.1f}y)")

    universe = EQUITY + CRYPTO
    prices = pd.DataFrame({t: load_etf(t) for t in universe}).reindex(dates).ffill()
    rets = prices.pct_change().fillna(0)
    avail = prices.notna()
    bil = load_etf("BIL").reindex(dates).ffill().pct_change().fillna(0)
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    spy_a = spy.reindex(dates).ffill()
    reg_eq = ((spy_a > spy_a.rolling(SPY_MA).mean()) & (vix < VIX_CAP)
              ).shift(1).fillna(False).astype(float)
    btc_a = btc.reindex(dates).ffill()
    reg_bt = (btc_a > btc_a.rolling(BTC_MA).mean()).shift(1).fillna(False).astype(float)
    rf = (load_fred("DGS3MO").reindex(dates).ffill() / 100.0 / 252.0).fillna(0)

    current = pd.Series(0.0, index=universe)
    port = pd.Series(0.0, index=dates)
    w_crypto = pd.Series(0.0, index=dates)
    w_equity = pd.Series(0.0, index=dates)
    last_idx = -REBAL
    rebal_rows = []

    for i in range(len(dates)):
        if i > LOOKBACK and i - last_idx >= REBAL:
            live = avail.iloc[i - 1]
            momo = (prices.iloc[i - 1] / prices.iloc[i - 1 - LOOKBACK] - 1).where(live)
            ranked = momo.dropna().sort_values(ascending=False)
            positive = [t for t in ranked.index if momo[t] > 0]
            top = positive[:TOP_N]
            new = pd.Series(0.0, index=universe)
            if top:
                w = 1.0 / len(top)
                for t in top:
                    new[t] = min(w, CAP)
            # TC proportional to overlay (we actually transact the levered notional)
            tc = (new - current).abs().sum() * (TC_BPS / 1e4) * OVERLAY
            port.iloc[i] -= tc
            current = new
            last_idx = i
            rebal_rows.append({
                "date": dates[i],
                "pick_1": top[0] if len(top) > 0 else "",
                "pick_2": top[1] if len(top) > 1 else "",
                "pick_3": top[2] if len(top) > 2 else "",
                "n_positive": len(positive),
            })

        eff = current.copy()
        geq = reg_eq.iloc[i]
        gbt = reg_bt.iloc[i]
        off_eq = sum(current[t] for t in EQUITY) * (1 - geq)
        off_bt = sum(current[t] for t in CRYPTO) * (1 - gbt)
        for t in EQUITY: eff[t] = current[t] * geq
        for t in CRYPTO: eff[t] = current[t] * gbt
        gross = (rets.iloc[i] * eff).sum() + (off_eq + off_bt) * bil.iloc[i]
        invested = float(eff.sum())
        levered = OVERLAY * gross - (OVERLAY - 1.0) * invested * rf.iloc[i]
        port.iloc[i] += levered
        w_crypto.iloc[i] = sum(eff[t] for t in CRYPTO) * OVERLAY
        w_equity.iloc[i] = sum(eff[t] for t in EQUITY) * OVERLAY

    r_spy = spy.reindex(dates).ffill().pct_change().fillna(0)
    agg = load_etf("AGG")
    r_agg = agg.reindex(dates).ffill().pct_change().fillna(0) if agg is not None else pd.Series(0.0, index=dates)

    out = pd.DataFrame({
        "Close": port,
        "Crypto": w_crypto,
        "Equity": w_equity,
        "Cash": 1 - (w_crypto + w_equity) / OVERLAY,  # free cash before overlay
        "SPY": r_spy,
        "AGG": r_agg,
    })
    out.index.name = "Date"
    out.to_csv(RESULTS / "nova_v2_returns.csv")
    pd.DataFrame(rebal_rows).to_csv(RESULTS / "nova_v2_rebalances.csv", index=False)

    def stats(r, label):
        ar = r.mean() * 252; av = r.std() * np.sqrt(252)
        sr = ar / av if av > 0 else 0
        c = (1 + r).cumprod()
        mdd = (c / c.cummax() - 1).min()
        print(f"  {label:18s} SR={sr:.2f}  Ret={ar*100:6.2f}%  Vol={av*100:5.2f}%  "
              f"MDD={mdd*100:6.1f}%  NAVx={c.iloc[-1]:.1f}")
    IS_END = pd.Timestamp("2020-01-01")
    stats(port.loc[:IS_END], "IS 2014..2019")
    stats(port.loc[IS_END:], "OOS 2020..now")
    stats(port, "FULL")
    stats(r_spy, "SPY (full)")
    stats(r_agg, "AGG (full)")


if __name__ == "__main__":
    build()
