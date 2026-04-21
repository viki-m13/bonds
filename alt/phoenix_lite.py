"""PHOENIX-LITE — single-cadence leveraged-ETF rotation.

Same objective as PHOENIX (honest Sharpe uplift on a broad LETF universe) but
stripped to ONE uniform rebalance frequency with no mid-period overlays.

Rules:
  * At close[t-1]: compute 63-day total return on each LETF in the universe.
  * Regime gate (all three must pass):
      - SPY > 200d SMA (and SMA upward sloping over 20d)
      - HY OAS 20d slope < 1.0 bp/day
      - VIX < 30
  * If regime OK at rebal day: top-K momentum names, equal-weight; hold N days.
  * If regime OFF: 100% BIL; hold N days (re-check at next rebal day).
  * Execute at next-day open. 10 bps/side TC on |dw|.
  * NO daily vol scaling. NO overlays. NO intra-cadence decisions.

Universe (17 LETFs, same as PHOENIX): broad, not cherry-picked.
IS: 2010-03-11 .. 2018-12-31. OOS: 2019-01-02 .. 2026-04-02.

Grid-search N (cadence) and K (breadth) on IS only; one-shot OOS.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

UNIVERSE = ["TQQQ", "UPRO", "QLD", "SSO", "SOXL", "TECL", "FAS", "ERX",
            "DRN", "EDC", "YINN", "UCO", "UGL", "NUGT", "TMF", "UBT", "TYD"]
IS_END = "2018-12-31"
OOS_START = "2019-01-02"
TC_BPS = 10.0  # one-way


def load_etf(t: str):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(s: str):
    p = FRED / f"{s}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"sharpe": 0, "cagr": 0, "vol": 0, "mdd": 0, "sortino": 0, "calmar": 0, "navx": 1}
    mu = r.mean() * 252
    sd = r.std() * np.sqrt(252)
    sr = mu / sd if sd > 0 else 0
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    cagr = c.iloc[-1] ** (1 / yrs) - 1 if c.iloc[-1] > 0 else -1
    neg = r[r < 0]
    sortino = mu / (neg.std() * np.sqrt(252)) if len(neg) > 0 and neg.std() > 0 else 0
    return {"sharpe": float(sr), "cagr": float(cagr), "vol": float(sd),
            "mdd": float(dd), "sortino": float(sortino),
            "calmar": float(cagr / abs(dd)) if dd < 0 else 0,
            "navx": float(c.iloc[-1])}


def backtest(close: pd.DataFrame, opn: pd.DataFrame, regime_ok: pd.Series,
             mom63_lag: pd.DataFrame, N: int, K: int) -> pd.Series:
    """Run a single-cadence strategy.

    At every N-th trading day, set new target weights based on regime + momentum.
    Hold those weights for N days, no mid-period changes.
    """
    idx = opn.index
    cols = list(UNIVERSE) + ["BIL"]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    current = pd.Series(0.0, index=cols)
    current["BIL"] = 1.0  # start in cash

    for i, dt in enumerate(idx):
        is_rebal = (i % N == 0)
        if is_rebal:
            if regime_ok.iloc[i]:
                # Pick top K by 63d momentum (using lagged signal = close[t-1])
                m = mom63_lag.iloc[i].dropna()
                # only names with valid price today (to be tradable at open)
                tradable = [t for t in UNIVERSE if not np.isnan(opn[t].iloc[i]) and t in m.index]
                m = m[tradable]
                # must be positive momentum to hold it
                m = m[m > 0].nlargest(K)
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    w_each = 1.0 / len(m)
                    for t in m.index:
                        new_w[t] = w_each
                    new_w["BIL"] = 0.0
                else:
                    new_w["BIL"] = 1.0
                current = new_w
            else:
                current = pd.Series(0.0, index=cols)
                current["BIL"] = 1.0
        W.iloc[i] = current.values

    # Compute returns: open[t] -> open[t+1] on held positions
    opn_ret = opn[UNIVERSE].pct_change().shift(-1).fillna(0)  # open[t+1]/open[t] - 1, indexed at t
    # BIL ~ cash, treat as open-to-open on BIL (which we have as an ETF)
    bil_ret = opn["BIL"].pct_change().shift(-1).fillna(0) if "BIL" in opn.columns else pd.Series(0.0, index=idx)

    port_ret = (W[UNIVERSE] * opn_ret[UNIVERSE]).sum(axis=1) + W["BIL"] * bil_ret

    # Transaction costs: paid when weights change (at each rebalance)
    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))  # first row full turnover
    tc = dW * (TC_BPS / 1e4)
    # Cost hits the day weights change (at open[t] -- so reduce port_ret on that day)
    port_ret = port_ret - tc

    # port_ret indexed at t represents return earned open[t] -> open[t+1].
    # Shift by +1 so index aligns with the period-end of the return.
    port_ret = port_ret.shift(1).fillna(0.0)

    return port_ret, W, dW


def main():
    # Load prices
    close_d, opn_d = {}, {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf(t)
        if df is not None:
            close_d[t] = df["Close"]
            opn_d[t] = df["Open"]
    close = pd.DataFrame(close_d)
    opn = pd.DataFrame(opn_d)

    # Common calendar
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) &
                  (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    # Macro regime inputs
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    vix = load_fred("VIXCLS").reindex(dates).ffill()

    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime_ok = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)

    # 63-day total return on each LETF, lagged 1 day (close[t-1])
    mom63 = close[UNIVERSE].pct_change(63)
    mom63_lag = mom63.shift(1)

    # Grid search N, K on IS
    print("PHOENIX-LITE grid search (IS Sharpe):")
    print(f"  Universe: {len(UNIVERSE)} LETFs; {len(dates)} trading days total")
    print(f"  IS: {dates[0].date()} to {IS_END} | OOS: {OOS_START} to {dates[-1].date()}")
    print()

    rows = []
    cache = {}
    for N in [1, 3, 5, 10, 21, 42, 63]:
        for K in [1, 2, 3, 4, 5]:
            port, W, dW = backtest(close, opn, regime_ok, mom63_lag, N, K)
            port.index = dates
            cache[(N, K)] = port
            m_is = metrics(port.loc[:IS_END])
            m_oos = metrics(port.loc[OOS_START:])
            m_full = metrics(port)
            # avg rebalances/yr
            rb_per_yr = (len(dates) / N) / (len(dates) / 252)
            # avg turnover /yr
            tov_ann = dW.sum() / (len(dates) / 252)
            rows.append({
                "N": N, "K": K,
                "is_sr": round(m_is["sharpe"], 3),
                "oos_sr": round(m_oos["sharpe"], 3),
                "full_sr": round(m_full["sharpe"], 3),
                "is_cagr": round(m_is["cagr"], 3),
                "oos_cagr": round(m_oos["cagr"], 3),
                "full_cagr": round(m_full["cagr"], 3),
                "full_mdd": round(m_full["mdd"], 3),
                "full_vol": round(m_full["vol"], 3),
                "rb_per_yr": round(rb_per_yr, 1),
                "tov_ann": round(tov_ann, 2),
                "gap": round(abs(m_is["sharpe"] - m_oos["sharpe"]), 3),
            })

    g = pd.DataFrame(rows).sort_values("is_sr", ascending=False)
    print(g.to_string(index=False))
    print()

    # Pick best by IS Sharpe, evaluate OOS once
    best = g.iloc[0]
    N_star = int(best["N"]); K_star = int(best["K"])
    port = cache[(N_star, K_star)]
    m_is = metrics(port.loc[:IS_END])
    m_oos = metrics(port.loc[OOS_START:])
    m_full = metrics(port)

    print(f"=== CHOSEN by IS: N={N_star}  K={K_star} ===")
    print(f"  Rebalance every {N_star} trading days (~{252/N_star:.0f}x/yr)")
    print(f"  Hold top {K_star} LETF(s) by 63d momentum when regime ON, else BIL")
    print()
    print(f"  {'win':10s}  {'SR':>5s} {'CAGR':>6s} {'Vol':>5s} {'MDD':>6s} {'Calmar':>6s} {'Sortino':>7s}")
    for name, m in [("FULL", m_full), ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {name:10s}  {m['sharpe']:5.2f} {m['cagr']*100:5.1f}% "
              f"{m['vol']*100:5.1f}% {m['mdd']*100:5.1f}% "
              f"{m['calmar']:6.2f} {m['sortino']:7.2f}")
    print(f"  IS-OOS gap: {abs(m_is['sharpe']-m_oos['sharpe']):.2f}")

    # Save
    out = {
        "strategy": "PHOENIX-LITE",
        "params": {"N": N_star, "K": K_star, "mom_lb": 63, "tc_bps": TC_BPS,
                   "universe_size": len(UNIVERSE)},
        "full": m_full, "is": m_is, "oos": m_oos,
        "is_oos_gap": round(abs(m_is["sharpe"] - m_oos["sharpe"]), 4),
    }
    (RESULTS / "phoenix_lite_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": port.index, "ret": port.values}).to_csv(
        RESULTS / "phoenix_lite_returns.csv", index=False)
    g.to_csv(RESULTS / "phoenix_lite_grid.csv", index=False)
    print()
    print("Saved phoenix_lite_metrics.json, _returns.csv, _grid.csv")


if __name__ == "__main__":
    main()
