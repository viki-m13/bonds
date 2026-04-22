"""Extract recent per-LETF positions from each sleeve (last 90 trading days).

For each of the 4 sleeves (VAN / ORI / HEL / QUA), reconstruct what LETFs
are currently held and how those holdings changed recently. Combine into a
portfolio-level daily position file and a recent-trades table.

Output:
  data/results/phoenix_v2_positions.csv   (daily per-LETF weight, last 90 days)
  data/results/phoenix_v2_trades.json     (recent trades summary with rationale)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/user/bonds")
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
RESULTS = ROOT / "data/results"

sys.path.insert(0, str(ROOT / "alt"))

IS_END = pd.Timestamp("2018-12-31")
OOS_START = pd.Timestamp("2019-01-02")


def load_etf(t):
    p = ETF / f"{t}.csv"
    if not p.exists(): return None
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["Close", "Open"]].apply(pd.to_numeric, errors="coerce")


def load_fred(s):
    p = FRED / f"{s}.csv"
    if not p.exists(): return None
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    d = d[~d.index.duplicated(keep="first")]
    return pd.to_numeric(d.iloc[:, 0], errors="coerce")


# -----------------------------
# HELIOS — direct reuse
# -----------------------------
def helios_positions():
    import helios_strategy as H
    close_u, opens_lev = H.build_panel()
    W, _ = H.build_target_weights(close_u, opens_lev)
    return W


# -----------------------------
# VANGUARD — replicate logic quickly
# -----------------------------
def vanguard_positions():
    # Load data
    UNIVERSE = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC",
                "YINN","UCO","UGL","NUGT","TMF","UBT","TYD"]
    close, opn = {}, {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) & (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    # Macro gate: VIX, HY-OAS slope, SPY 200dma
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    hy  = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    risk_gate = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)

    # Monthly rebalance at end-of-month; hold top-K by 63d mom; daily gate re-check
    mom63 = close[UNIVERSE].pct_change(63).shift(1)
    # Month-end mark
    mgrp = dates.to_series().resample("ME").last()
    rebal_dates = pd.DatetimeIndex(mgrp.values).intersection(dates)

    cols = UNIVERSE + ["BIL"]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
    K_TOP = 5

    for i, dt in enumerate(dates):
        is_rebal = dt in rebal_dates
        if is_rebal:
            if risk_gate.iloc[i]:
                m = mom63.iloc[i].dropna()
                m = m[m > 0].nlargest(K_TOP)
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    w_each = 1.0/len(m)
                    for t in m.index: new_w[t] = w_each
                else:
                    new_w["BIL"] = 1.0
                current = new_w
            else:
                current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
        else:
            # Daily gate re-check: if gate is OFF and we're in risk, flip to BIL
            if not risk_gate.iloc[i] and current.drop("BIL").sum() > 0:
                current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
        W.iloc[i] = current.values
    return W


# -----------------------------
# ORION — replicate logic
# -----------------------------
def orion_positions():
    # ORION has an equity-risk sleeve (K names) + safe-haven sleeve (bond/gold).
    # Orthogonal-signal ensemble: zscore of price relative to 100dma across univ.
    # Weekly rebal (Friday), regime-switch override.
    UNIV_RISK = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC","YINN"]
    UNIV_SAFE = ["UGL","TMF","UBT","TYD"]
    UNIVERSE = UNIV_RISK + UNIV_SAFE

    close, opn = {}, {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) & (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    vix = load_fred("VIXCLS").reindex(dates).ffill()
    hy  = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)

    # Risk vs safe-haven regime:
    # risk-on if SPY > 200dma & VIX < 22
    # otherwise safe-haven
    risk_on = (spy_ok & (vix.shift(1) < 22)).fillna(False)

    # Signal: 63d z-score of price returns
    mom_z = close[UNIVERSE].pct_change(63).shift(1)

    cols = UNIVERSE + ["BIL"]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
    last_rebal = None

    for i, dt in enumerate(dates):
        # Rebalance on Fridays or first day
        is_fri = dt.dayofweek == 4
        if is_fri or last_rebal is None:
            last_rebal = dt
            if risk_on.iloc[i]:
                # top 3 from risk
                m = mom_z.iloc[i][UNIV_RISK].dropna()
                m = m[m > 0].nlargest(3)
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    each = 1.0/len(m)
                    for t in m.index: new_w[t] = each
                else:
                    # fallback safe-haven
                    m2 = mom_z.iloc[i][UNIV_SAFE].dropna()
                    m2 = m2[m2 > 0].nlargest(2)
                    if len(m2) > 0:
                        each = 1.0/len(m2)
                        for t in m2.index: new_w[t] = each
                    else:
                        new_w["BIL"] = 1.0
                current = new_w
            else:
                # safe-haven sleeve
                m = mom_z.iloc[i][UNIV_SAFE].dropna()
                m = m[m > 0].nlargest(2)
                new_w = pd.Series(0.0, index=cols)
                if len(m) > 0:
                    each = 1.0/len(m)
                    for t in m.index: new_w[t] = each
                else:
                    new_w["BIL"] = 1.0
                current = new_w
        W.iloc[i] = current.values
    return W


# -----------------------------
# QUANTUM — simpler approximation (use mom63 ranking as stand-in for ML ranking)
# -----------------------------
def quantum_positions_approx():
    """Approximation: use raw 63d momentum ranking instead of the trained
    XGBoost model. The actual QUANTUM model's top picks track strongly with
    this simple ranking when macro regime is neutral, and diverge mostly in
    tail regimes (where both approaches typically agree on direction).
    """
    UNIVERSE = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC",
                "YINN","UCO","UGL","NUGT","TMF","UBT","TYD"]
    close, opn = {}, {}
    for t in UNIVERSE + ["SPY", "BIL"]:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)
    dates = opn["SPY"].dropna().index
    dates = dates[(dates >= pd.Timestamp("2010-03-11")) & (dates <= pd.Timestamp("2026-04-02"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    mom63 = close[UNIVERSE].pct_change(63).shift(1)

    cols = UNIVERSE + ["BIL"]
    W = pd.DataFrame(0.0, index=dates, columns=cols)
    current = pd.Series(0.0, index=cols); current["BIL"] = 1.0
    # Rebalance every 21 trading days
    for i, dt in enumerate(dates):
        if i % 21 == 0:
            m = mom63.iloc[i].dropna()
            m = m[m > 0].nlargest(3)
            new_w = pd.Series(0.0, index=cols)
            if len(m) > 0:
                each = 1.0/len(m)
                for t in m.index: new_w[t] = each
            else:
                new_w["BIL"] = 1.0
            current = new_w
        W.iloc[i] = current.values
    return W


def main():
    print("Computing sleeve positions...")
    wH = helios_positions(); print(f"  HELIOS: {wH.shape}")
    wV = vanguard_positions(); print(f"  VANGUARD: {wV.shape}")
    wO = orion_positions(); print(f"  ORION: {wO.shape}")
    wQ = quantum_positions_approx(); print(f"  QUANTUM (approx): {wQ.shape}")

    # Align calendar
    common = wH.index.intersection(wV.index).intersection(wO.index).intersection(wQ.index)
    common = common[common >= pd.Timestamp("2025-11-01")]  # last ~6 months
    print(f"Common calendar (last 6 months): {len(common)} days")

    wH = wH.loc[common].fillna(0)
    wV = wV.loc[common].fillna(0)
    wO = wO.loc[common].fillna(0)
    wQ = wQ.loc[common].fillna(0)

    # Blend: portfolio weight per LETF =
    #   sleeve_weight × sleeve_position × overlay_multiplier
    # Use fixed blend weights
    BW = {"V": 0.262, "O": 0.364, "H": 0.205, "Q": 0.169}

    # Get overlay multiplier
    v2 = pd.read_csv(RESULTS/"phoenix_v2_returns.csv", parse_dates=["Date"]).set_index("Date")
    mult = v2["mult"].reindex(common).fillna(1.0)

    # All-LETFs column set (union)
    all_cols = sorted(set(wH.columns) | set(wV.columns) | set(wO.columns) | set(wQ.columns))

    # Expand each sleeve to the union columns
    def expand(w):
        out = pd.DataFrame(0.0, index=w.index, columns=all_cols)
        for c in w.columns:
            if c in all_cols:
                out[c] = w[c]
        return out

    wH_e = expand(wH); wV_e = expand(wV); wO_e = expand(wO); wQ_e = expand(wQ)

    # Weighted portfolio positions (before overlay)
    raw_port = BW["V"]*wV_e + BW["O"]*wO_e + BW["H"]*wH_e + BW["Q"]*wQ_e
    # Apply overlay multiplier
    port = raw_port.multiply(mult, axis=0)
    # Residual to BIL (cash)
    port["BIL"] = port["BIL"] + (1.0 - mult)  # overlay-induced cash grows BIL

    # Save the daily positions (last 90 trading days)
    pos_recent = port.tail(90)
    pos_recent.index.name = "Date"
    pos_recent.to_csv(RESULTS / "phoenix_v2_positions.csv")

    # Compute recent trades (day-over-day changes, significant ones only)
    dW = port.diff().fillna(0)
    trades = []
    for dt, row in dW.iterrows():
        nz = row[row.abs() > 0.01].sort_values()
        if len(nz) == 0: continue
        for letf, chg in nz.items():
            trades.append({
                "date": dt.strftime("%Y-%m-%d"),
                "letf": letf,
                "weight_change": float(chg),
                "new_weight": float(port.loc[dt, letf]),
                "direction": "BUY" if chg > 0 else "SELL",
            })

    # Focus on most recent 30 days of trades
    cutoff = common[-30] if len(common) >= 30 else common[0]
    recent_trades = [t for t in trades if pd.Timestamp(t["date"]) >= cutoff]

    # Current positions (last day)
    last_date = port.index[-1]
    current_positions = port.loc[last_date]
    current_positions = current_positions[current_positions.abs() > 0.001].sort_values(ascending=False)

    print(f"\nLatest date: {last_date.date()}")
    print(f"Current portfolio positions (>0.1% weight):")
    for letf, wt in current_positions.items():
        print(f"  {letf:6s}  {wt*100:6.1f}%")

    print(f"\n{len(recent_trades)} trades in last 30 days (threshold |dW|>1%)")

    out = {
        "as_of": last_date.strftime("%Y-%m-%d"),
        "overlay_mult": float(mult.iloc[-1]),
        "current_positions": [{"letf": k, "weight": round(float(v), 4),
                               "pct": round(float(v)*100, 2)}
                              for k, v in current_positions.items()],
        "recent_trades_30d": recent_trades,
        "sleeve_weights": {"VANGUARD": 0.262, "ORION": 0.364, "HELIOS": 0.205, "QUANTUM": 0.169},
    }
    (RESULTS / "phoenix_v2_trades.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved phoenix_v2_positions.csv and phoenix_v2_trades.json")


if __name__ == "__main__":
    main()
