"""SOLAR — Multi-timeframe trend consensus on a broad LETF universe.

Hard rules:
  * SINGLE uniform rebalance cadence N (no overlays, no mid-period).
  * Long-only LETFs; survivors equal-weighted; if none survive, 100% BIL.
  * Signal uses close[t-1]; execute at open[t]; 10 bps/side TC on |dw|.
  * NO daily vol scaling.
  * Universe: 17 broad LETFs (no cherry-picking).

Core idea — trend consensus:
  At rebalance day t, for each LETF compute raw momentum at horizons
  {21, 63, 252} trading days (lagged 1 day so close[t-1] is used).
  Optionally convert to a Sharpe-like score: return / (vol * sqrt(h)).
  Count how many horizons have a POSITIVE signal. Require a minimum
  number of horizons to pass ("consensus"). Optionally further filter
  to top-half of the surviving set by the average horizon score.
  Equal-weight the survivors; else BIL.

Grid (IS only):
  * cadence N in {5, 10, 21, 42, 63}
  * consensus threshold (>=): 2 or 3
  * signal type: "raw" or "sharpe" (ret / rolling-vol)
  * top-half filter: False or True
  * horizon set fixed: (21, 63, 252)

Pick winner by IS Sharpe, evaluate OOS once. Honest reporting.
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
IS_START = "2010-03-11"
IS_END = "2018-12-31"
OOS_START = "2019-01-02"
OOS_END = "2026-04-02"
TC_BPS = 10.0  # one-way
HORIZONS = (21, 63, 252)


def load_etf(t: str):
    p = ETF / f"{t}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) == 0:
        return {"sharpe": 0, "cagr": 0, "vol": 0, "mdd": 0,
                "sortino": 0, "calmar": 0, "navx": 1}
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
            "calmar": float(cagr / abs(dd)) if dd < 0 else 0.0,
            "navx": float(c.iloc[-1])}


def build_signals(close: pd.DataFrame, signal_type: str) -> dict:
    """For each horizon, build a per-ticker signal DataFrame (lagged by 1 day)."""
    sigs = {}
    for h in HORIZONS:
        ret_h = close[UNIVERSE].pct_change(h)
        if signal_type == "sharpe":
            daily = close[UNIVERSE].pct_change()
            vol = daily.rolling(h).std() * np.sqrt(h)  # horizon-scaled vol
            s = ret_h / vol.replace(0, np.nan)
        else:  # raw
            s = ret_h
        sigs[h] = s.shift(1)  # use close[t-1]
    return sigs


def backtest(opn: pd.DataFrame, sigs: dict, N: int, consensus: int,
             use_top_half: bool) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    idx = opn.index
    cols = list(UNIVERSE) + ["BIL"]
    W = pd.DataFrame(0.0, index=idx, columns=cols)
    current = pd.Series(0.0, index=cols)
    current["BIL"] = 1.0

    # pre-stack signals for vectorized access
    # For each date, we'll check per-horizon sign and build a score.
    h_arr = list(HORIZONS)
    sig_frames = [sigs[h] for h in h_arr]

    for i, dt in enumerate(idx):
        if (i % N) != 0:
            W.iloc[i] = current.values
            continue

        # Signals today (lagged = close[t-1])
        scores_per_h = []
        positive_flags = []
        for sf in sig_frames:
            row = sf.iloc[i]
            scores_per_h.append(row)
            positive_flags.append(row > 0)
        score_df = pd.concat(scores_per_h, axis=1)  # index: tickers, cols: horizons
        score_df.columns = h_arr

        # Tradable: non-NaN open today, non-NaN score at all horizons
        tradable = []
        for t in UNIVERSE:
            if np.isnan(opn[t].iloc[i]):
                continue
            if score_df.loc[t].isna().any():
                continue
            tradable.append(t)

        if not tradable:
            new_w = pd.Series(0.0, index=cols); new_w["BIL"] = 1.0
            current = new_w
            W.iloc[i] = current.values
            continue

        sub = score_df.loc[tradable]
        pos_count = (sub > 0).sum(axis=1)
        survivors = sub.index[pos_count >= consensus].tolist()

        if use_top_half and len(survivors) > 1:
            # Use average z-score across horizons as composite quality score
            # z-score per horizon across tradable set, then mean
            zed = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
            zed = zed.fillna(0)
            comp = zed.mean(axis=1)
            comp_s = comp.loc[survivors].sort_values(ascending=False)
            keep_n = max(1, len(comp_s) // 2)
            survivors = comp_s.head(keep_n).index.tolist()

        new_w = pd.Series(0.0, index=cols)
        if survivors:
            w_each = 1.0 / len(survivors)
            for t in survivors:
                new_w[t] = w_each
        else:
            new_w["BIL"] = 1.0
        current = new_w
        W.iloc[i] = current.values

    # returns: open[t] -> open[t+1]
    opn_ret = opn[UNIVERSE].pct_change().shift(-1).fillna(0)
    bil_ret = opn["BIL"].pct_change().shift(-1).fillna(0) if "BIL" in opn.columns else pd.Series(0.0, index=idx)
    port_ret = (W[UNIVERSE] * opn_ret[UNIVERSE]).sum(axis=1) + W["BIL"] * bil_ret

    dW = W.diff().abs().sum(axis=1).fillna(W.abs().sum(axis=1))
    tc = dW * (TC_BPS / 1e4)
    port_ret = port_ret - tc

    # align: port_ret at t is the return earned from open[t] to open[t+1]. Shift +1.
    port_ret = port_ret.shift(1).fillna(0.0)
    return port_ret, W, dW


def main():
    close_d, opn_d = {}, {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf(t)
        if df is not None:
            close_d[t] = df["Close"]
            opn_d[t] = df["Open"]
    close = pd.DataFrame(close_d)
    opn = pd.DataFrame(opn_d)

    # Common calendar from SPY
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp(IS_START)) &
                  (dates <= pd.Timestamp(OOS_END))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    print(f"SOLAR — multi-timeframe trend consensus")
    print(f"  Universe: {len(UNIVERSE)} LETFs | Horizons: {HORIZONS}")
    print(f"  IS: {dates[0].date()} to {IS_END} | OOS: {OOS_START} to {dates[-1].date()}")
    print(f"  TC: {TC_BPS} bps/side; single uniform cadence; long-only; no vol scaling")
    print()

    # Pre-build signals for both types (reuse across cadences)
    sig_raw = build_signals(close, "raw")
    sig_shp = build_signals(close, "sharpe")

    rows = []
    cache = {}
    for signal_type, sigs in [("raw", sig_raw), ("sharpe", sig_shp)]:
        for N in [5, 10, 21, 42, 63]:
            for consensus in [2, 3]:
                for use_top_half in [False, True]:
                    port, W, dW = backtest(opn, sigs, N, consensus, use_top_half)
                    port.index = dates
                    key = (signal_type, N, consensus, use_top_half)
                    cache[key] = port
                    m_is = metrics(port.loc[:IS_END])
                    m_oos = metrics(port.loc[OOS_START:])
                    m_full = metrics(port)
                    tov_ann = dW.sum() / (len(dates) / 252)
                    rows.append({
                        "signal": signal_type,
                        "N": N,
                        "consensus": consensus,
                        "top_half": use_top_half,
                        "is_sr": round(m_is["sharpe"], 3),
                        "oos_sr": round(m_oos["sharpe"], 3),
                        "full_sr": round(m_full["sharpe"], 3),
                        "is_cagr": round(m_is["cagr"], 3),
                        "oos_cagr": round(m_oos["cagr"], 3),
                        "full_cagr": round(m_full["cagr"], 3),
                        "full_mdd": round(m_full["mdd"], 3),
                        "full_vol": round(m_full["vol"], 3),
                        "tov_ann": round(tov_ann, 2),
                        "gap": round(abs(m_is["sharpe"] - m_oos["sharpe"]), 3),
                    })

    g = pd.DataFrame(rows).sort_values("is_sr", ascending=False)
    print("Top 15 configs by IS Sharpe:")
    print(g.head(15).to_string(index=False))
    print()

    best = g.iloc[0]
    signal_star = str(best["signal"])
    N_star = int(best["N"])
    cons_star = int(best["consensus"])
    th_star = bool(best["top_half"])
    port = cache[(signal_star, N_star, cons_star, th_star)]

    m_is = metrics(port.loc[:IS_END])
    m_oos = metrics(port.loc[OOS_START:])
    m_full = metrics(port)

    print(f"=== CHOSEN by IS Sharpe ===")
    print(f"  signal_type = {signal_star}")
    print(f"  cadence N   = {N_star} trading days  (~{252/N_star:.1f}x/yr)")
    print(f"  consensus   = {cons_star} of {len(HORIZONS)} horizons positive")
    print(f"  top_half    = {th_star}")
    print()
    print(f"  {'win':6s}  {'SR':>5s} {'CAGR':>7s} {'Vol':>6s} {'MDD':>7s} {'Calmar':>6s} {'Sortino':>7s}")
    for name, m in [("FULL", m_full), ("IS", m_is), ("OOS", m_oos)]:
        print(f"  {name:6s}  {m['sharpe']:5.2f} {m['cagr']*100:6.1f}% "
              f"{m['vol']*100:5.1f}% {m['mdd']*100:6.1f}% "
              f"{m['calmar']:6.2f} {m['sortino']:7.2f}")
    print(f"  IS-OOS gap: {abs(m_is['sharpe'] - m_oos['sharpe']):.2f}")

    out = {
        "strategy": "SOLAR",
        "params": {
            "signal_type": signal_star,
            "cadence_N": N_star,
            "consensus": cons_star,
            "top_half_filter": th_star,
            "horizons": list(HORIZONS),
            "universe": UNIVERSE,
            "tc_bps_per_side": TC_BPS,
        },
        "full": m_full,
        "is": m_is,
        "oos": m_oos,
        "is_oos_gap": round(abs(m_is["sharpe"] - m_oos["sharpe"]), 4),
        "is_start": IS_START, "is_end": IS_END,
        "oos_start": OOS_START, "oos_end": OOS_END,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "solar_metrics.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame({"Date": port.index, "ret": port.values}).to_csv(
        RESULTS / "solar_returns.csv", index=False)
    g.to_csv(RESULTS / "solar_grid.csv", index=False)
    print()
    print("Saved solar_metrics.json, solar_returns.csv, solar_grid.csv")


if __name__ == "__main__":
    main()
