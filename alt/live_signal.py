"""PHOENIX — live signal generator (canonical production version).

Runs after US market close. Fetches latest prices, computes today's target
portfolio, compares to yesterday's positions, generates explicit BUY/SELL
orders, saves state for webapp.

Usage:
    python3 alt/live_signal.py [--skip-fetch]

Output files:
    data/results/live_signal.json      — today's state (read by webapp)
    data/results/live_positions.csv    — running log of daily positions
    data/results/live_trades.csv       — running log of trades

PHOENIX (canonical 5-sleeve, daily vol-targeted):
    Blend (fixed IS inv-vol): VAN 23.6% · ORI 32.7% · HEL 18.5% · QUA 15.2% · CRY 10.1%
    Vol target: 20% annualized, cap 2.0x, floor 0.25x
    DD throttle: -10% floor, 252d HWM
    Vol gate: 60d vol > 99th pct (252d) → 0.5x
    Backtest: Sharpe 2.37 full / 2.22 OOS, CAGR 57.5%, MDD -23.9%

Universe:
    Risk:    TQQQ UPRO QLD SSO SOXL TECL FAS ERX DRN EDC YINN UCO
    Defens:  UGL NUGT TMF UBT TYD
    Crypto:  IBIT (spot BTC)
    Cash:    BIL

Gate: SPY > 200dma (sloping up) & HY-OAS 20d slope < 1.0 & VIX < 30
"""
from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
R = ROOT / "data/results"

UNIVERSE_RISK = ["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC","YINN","UCO"]
UNIVERSE_SAFE = ["UGL","NUGT","TMF","UBT","TYD"]
UNIVERSE_CRYPTO = ["IBIT"]  # primary spot BTC (low fee); ETHA optional
UNIVERSE = UNIVERSE_RISK + UNIVERSE_SAFE + UNIVERSE_CRYPTO
BIL = "BIL"

BLEND_WEIGHTS = {"VAN": 0.236, "ORI": 0.327, "HEL": 0.185, "QUA": 0.152, "CRY": 0.101}

# Production overlay parameters (must match phoenix_production.py exactly)
TARGET_VOL = 0.15        # portfolio realized-vol target (annualized)
VOL_CAP = 1.0            # max gross exposure — NO margin / borrowing
VOL_FLOOR = 0.25
VOL_WIN = 60
DD_FLOOR = -0.10
DD_WIN = 252
VOL_GATE_PCT = 0.99
VOL_GATE_LOOKBACK = 252


# --------------- Data I/O ---------------
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


def fetch_latest(tickers):
    """Refresh price CSVs via yfinance for each ticker, appending new rows only."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed; skipping fetch.", file=sys.stderr)
        return

    # Today's date (UTC — matches yfinance)
    today = datetime.now(timezone.utc).date()
    # Always fetch last 10 calendar days to merge (handles weekends, missed days,
    # and avoids the start>=end race condition when the CSV was updated minutes ago).
    for t in tickers:
        p = ETF / f"{t}.csv"
        start_candidate = today - timedelta(days=14)
        if p.exists():
            existing = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            last_date = existing.index[-1].date()
            # Start from last_date (not last_date+1) so we overlap a day and always fetch something
            start_candidate = max(start_candidate, last_date)
            # If CSV is already at or beyond today, skip
            if last_date >= today:
                continue
        start = start_candidate.strftime("%Y-%m-%d")
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=False)
        except Exception as e:
            print(f"  {t}: fetch failed ({e})", file=sys.stderr)
            continue
        if df is None or len(df) == 0:
            print(f"  {t}: no new data (market closed or already up to date)")
            continue
        # Flatten multi-index columns if any
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        if p.exists():
            existing = pd.read_csv(p)
            combined = pd.concat([existing, df], ignore_index=True)
            combined = combined.drop_duplicates(subset="Date", keep="last").sort_values("Date")
            combined.to_csv(p, index=False)
        else:
            df.to_csv(p, index=False)
        print(f"  {t}: +{len(df)} new rows, latest={df['Date'].iloc[-1]}")


def fetch_fred_latest():
    """Refresh VIX / HY OAS / rates via FRED API (requires FRED_API_KEY env var)."""
    import os, urllib.request
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("FRED_API_KEY not set; skipping FRED fetch.", file=sys.stderr)
        return

    series = ["VIXCLS", "BAMLH0A0HYM2", "DGS10", "DGS2", "FEDFUNDS"]
    for s in series:
        url = (f"https://api.stlouisfed.org/fred/series/observations"
               f"?series_id={s}&api_key={api_key}&file_type=json")
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = json.loads(resp.read())
        except Exception as e:
            print(f"  {s}: FRED fetch failed ({e})", file=sys.stderr)
            continue
        obs = data.get("observations", [])
        if not obs: continue
        rows = [(o["date"], o["value"]) for o in obs if o["value"] != "."]
        df = pd.DataFrame(rows, columns=["Date", s])
        df["Date"] = pd.to_datetime(df["Date"])
        df[s] = pd.to_numeric(df[s], errors="coerce")
        df = df.dropna().sort_values("Date")
        out = FRED / f"{s}.csv"
        df.to_csv(out, index=False)
        print(f"  {s}: {len(df)} rows, latest={df['Date'].iloc[-1].date()}")


# --------------- Signal logic ---------------
def compute_signals(close, opn, dates):
    """Build target portfolio weight for each date."""
    vix = load_fred("VIXCLS").reindex(dates).ffill()
    hy = load_fred("BAMLH0A0HYM2").reindex(dates).ffill()
    spy = close["SPY"]
    spy_ma = spy.rolling(200).mean()
    spy_ok = (spy > spy_ma) & (spy_ma.diff(20) > 0)
    hy_slope = hy - hy.shift(20)
    regime_full = (spy_ok & (hy_slope < 1.0) & (vix < 30)).shift(1).fillna(False)
    regime_risk_on = (spy_ok & (vix.shift(1) < 22)).fillna(False)

    # 63-day momentum (primary signal)
    mom63 = close[UNIVERSE].pct_change(63).shift(1)
    # 21-day and 252-day for composite
    mom21 = close[UNIVERSE].pct_change(21).shift(1)
    mom252 = close[UNIVERSE].pct_change(252).shift(1)
    rets = close[UNIVERSE].pct_change().fillna(0)
    vol63 = rets.rolling(63).std().shift(1) * np.sqrt(252)
    sh63 = (mom63 * (252/63)) / vol63.replace(0, np.nan)

    # Per-sleeve target weight at the most recent date
    i = -1
    dt = dates[i]
    target = pd.Series(0.0, index=UNIVERSE + [BIL])

    # SLEEVE 1 (VAN analog): top-3 from full risk universe by 63d mom, gated
    if regime_full.iloc[i]:
        m = mom63.iloc[i].reindex(UNIVERSE_RISK + UNIVERSE_SAFE).dropna()
        tradable = [t for t in m.index if not np.isnan(opn[t].iloc[i])]
        m = m[tradable]
        top = m[m > 0].nlargest(3)
        if len(top) > 0:
            for t in top.index:
                target[t] += BLEND_WEIGHTS["VAN"] / len(top)
        else:
            target[BIL] += BLEND_WEIGHTS["VAN"]
    else:
        target[BIL] += BLEND_WEIGHTS["VAN"]

    # SLEEVE 2 (ORI analog): risk-on/safe-haven toggle
    if regime_risk_on.iloc[i]:
        m = mom63.iloc[i].reindex(UNIVERSE_RISK).dropna()
        m = m[m > 0].nlargest(3)
        if len(m) > 0:
            for t in m.index:
                target[t] += BLEND_WEIGHTS["ORI"] / len(m)
        else:
            # safe-haven fallback
            ms = mom63.iloc[i].reindex(UNIVERSE_SAFE).dropna()
            ms = ms[ms > 0].nlargest(2)
            if len(ms) > 0:
                for t in ms.index:
                    target[t] += BLEND_WEIGHTS["ORI"] / len(ms)
            else:
                target[BIL] += BLEND_WEIGHTS["ORI"]
    else:
        ms = mom63.iloc[i].reindex(UNIVERSE_SAFE).dropna()
        ms = ms[ms > 0].nlargest(2)
        if len(ms) > 0:
            for t in ms.index:
                target[t] += BLEND_WEIGHTS["ORI"] / len(ms)
        else:
            target[BIL] += BLEND_WEIGHTS["ORI"]

    # SLEEVE 3 (HEL analog): signal on unleveraged underlyings
    UNDER_MAP = {"UPRO":"SPY","SSO":"SPY","TQQQ":"QQQ","QLD":"QQQ","SOXL":"SMH",
                 "TECL":"XLK","FAS":"XLF","ERX":"XLE","DRN":"VNQ","EDC":"EEM",
                 "YINN":"FXI","UCO":"USO","UGL":"GLD","NUGT":"GLD",
                 "TMF":"TLT","UBT":"TLT","TYD":"IEF"}
    under_mom = {}
    for letf, und in UNDER_MAP.items():
        if und in close.columns:
            u = close[und].pct_change(63).shift(1)
            under_mom[letf] = u.iloc[i] if not np.isnan(u.iloc[i]) else None
        else:
            under_mom[letf] = None
    if regime_full.iloc[i]:
        s = pd.Series({k: v for k, v in under_mom.items() if v is not None}).dropna()
        s = s[s > 0].nlargest(2)
        if len(s) > 0:
            for t in s.index:
                target[t] += BLEND_WEIGHTS["HEL"] / len(s)
        else:
            target[BIL] += BLEND_WEIGHTS["HEL"]
    else:
        target[BIL] += BLEND_WEIGHTS["HEL"]

    # SLEEVE 4 (QUA analog): composite z-score of (21d, 63d, 252d return)
    if regime_full.iloc[i]:
        def zscore_row(df, i):
            row = df.iloc[i]
            mu = row.mean(); sd = row.std()
            return (row - mu) / sd if sd > 0 else row * 0
        z = zscore_row(mom21, i) + zscore_row(mom63, i) + zscore_row(mom252, i)
        # require all-positive momentum
        pos = (mom21.iloc[i] > 0) & (mom63.iloc[i] > 0) & (mom252.iloc[i] > 0)
        z = z[pos].dropna()
        top = z.nlargest(3)
        if len(top) > 0:
            for t in top.index:
                target[t] += BLEND_WEIGHTS["QUA"] / len(top)
        else:
            target[BIL] += BLEND_WEIGHTS["QUA"]
    else:
        target[BIL] += BLEND_WEIGHTS["QUA"]

    # SLEEVE 5 (CRYPTO): weekly TSMOM on IBIT (or ETHA), gated
    if regime_full.iloc[i]:
        m = mom63.iloc[i].reindex(UNIVERSE_CRYPTO).dropna()
        m = m[m > 0]
        if len(m) > 0:
            for t in m.index:
                target[t] += BLEND_WEIGHTS["CRY"] / len(m)
        else:
            target[BIL] += BLEND_WEIGHTS["CRY"]
    else:
        target[BIL] += BLEND_WEIGHTS["CRY"]

    # OVERLAY: compute multiplier (vol target × DD throttle × vol gate).
    # Multiplier > 1.0 means leverage up (vol target scales exposure up in low-vol regimes);
    # Multiplier < 1.0 means de-risk. BIL absorbs any residual.
    mult, overlay_reason = compute_overlay_mult(close, opn, dates)
    # Separate risk-assets from cash
    risk = target.drop(BIL)
    scaled = risk * mult
    gross_risk = float(scaled.sum())
    # BIL = 1 - gross_risk (can be negative if leveraged, which means borrowing)
    # We clip BIL at 0 and allow gross_risk > 1 explicitly
    bil_weight = max(0.0, 1.0 - gross_risk)
    target[BIL] = bil_weight
    for t in scaled.index:
        target[t] = float(scaled[t])

    # Context data for reporting
    context = {
        "as_of": str(dates[-1].date()),
        "regime_gate_pass": bool(regime_full.iloc[i]),
        "regime_risk_on":   bool(regime_risk_on.iloc[i]),
        "spy_vs_200dma":    float(spy.iloc[i] / spy_ma.iloc[i] - 1) if not np.isnan(spy_ma.iloc[i]) else None,
        "vix":              float(vix.iloc[i]),
        "hy_oas_slope_20d": float(hy_slope.iloc[i]),
        "overlay_mult":     float(mult),
        "overlay_reason":   overlay_reason,
        "mom63_ranking": {t: float(mom63.iloc[i][t]) for t in UNIVERSE
                          if t in mom63.columns and not np.isnan(mom63.iloc[i][t])},
    }
    return target, context


def compute_overlay_mult(close, opn, dates):
    """Compute the full production overlay: vol target * DD throttle * vol gate.

    Reads returns history from live_positions.csv. On first run with
    insufficient history, falls back to the canonical backtest returns
    (phoenix_production_returns.csv) so the live signal uses a consistent
    overlay from day one.
    """
    # Prefer backtest history seed if live history is too short
    backtest_file = R / "phoenix_production_returns.csv"
    pos_file = R / "live_positions.csv"

    r_hist = None
    # Try backtest (preferred: it has full history from 2010)
    if backtest_file.exists():
        try:
            bt = pd.read_csv(backtest_file, parse_dates=["Date"]).set_index("Date").sort_index()
            if "raw_ret" in bt.columns:
                r_hist = bt["raw_ret"]
        except Exception:
            pass

    if r_hist is None and pos_file.exists():
        try:
            pos = pd.read_csv(pos_file, parse_dates=["Date"]).set_index("Date").sort_index()
            if "ret" in pos.columns:
                r_hist = pos["ret"]
        except Exception:
            pass

    if r_hist is None or len(r_hist) < VOL_WIN:
        return 1.0, "No sufficient history — multiplier defaulted to 1.0"

    r_hist = r_hist.dropna()

    # 1. Daily vol target multiplier
    rv_ann = r_hist.rolling(VOL_WIN).std() * np.sqrt(252)
    vt_raw = (TARGET_VOL / rv_ann.iloc[-1]) if rv_ann.iloc[-1] > 0 else 1.0
    vt_mult = max(VOL_FLOOR, min(VOL_CAP, float(vt_raw)))

    # Apply vol target to historical returns to compute stacked overlays
    vol_mult_series = (TARGET_VOL / rv_ann).clip(VOL_FLOOR, VOL_CAP).shift(1).fillna(1.0)
    scaled = r_hist * vol_mult_series

    # 2. DD throttle on scaled
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(DD_WIN, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = max(0.0, min(1.0, 1.0 + float(dd.iloc[-1]) / DD_FLOOR))

    # 3. Vol gate on scaled
    sv = scaled.rolling(VOL_WIN).std()
    sv_thr = sv.rolling(VOL_GATE_LOOKBACK, min_periods=60).quantile(VOL_GATE_PCT)
    if not np.isnan(sv_thr.iloc[-1]):
        vol_gate_ok = float(sv.iloc[-1]) <= float(sv_thr.iloc[-1])
    else:
        vol_gate_ok = True
    vol_gate_mult = 1.0 if vol_gate_ok else 0.5

    # Combine
    total = vt_mult * dd_mult * vol_gate_mult

    reason_parts = [f"Vol target {TARGET_VOL*100:.0f}% → mult {vt_mult:.2f}x (realized vol {rv_ann.iloc[-1]*100:.1f}%)"]
    if dd_mult < 1.0:
        reason_parts.append(f"DD throttle {dd_mult*100:.0f}% (current DD {dd.iloc[-1]*100:.1f}%)")
    if vol_gate_mult < 1.0:
        reason_parts.append("Vol-regime gate ACTIVE (60d vol > 99th pct of trailing 252d)")
    if dd_mult >= 1.0 and vol_gate_mult >= 1.0:
        reason_parts.append("All safety gates clear")
    reason = " · ".join(reason_parts)
    return float(total), reason


def compute_trades(prev_weights, target_weights, threshold=0.005):
    """Generate buy/sell orders from weight deltas (threshold 0.5%)."""
    all_tickers = sorted(set(prev_weights.index) | set(target_weights.index))
    trades = []
    for t in all_tickers:
        before = float(prev_weights.get(t, 0.0))
        after = float(target_weights.get(t, 0.0))
        delta = after - before
        if abs(delta) >= threshold:
            trades.append({
                "ticker": t,
                "side": "BUY" if delta > 0 else "SELL",
                "weight_before": round(before, 4),
                "weight_after": round(after, 4),
                "weight_delta": round(delta, 4),
                "delta_pct": round(delta * 100, 2),
            })
    # Sort trades: SELLs first (raise cash), then BUYs, largest moves first within each
    trades.sort(key=lambda t: (t["side"] != "SELL", -abs(t["weight_delta"])))
    return trades


# --------------- Main runner ---------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-fetch", action="store_true", help="Don't download new prices")
    args = p.parse_args()

    universe_plus_underlying = list(set(UNIVERSE + ["SPY","BIL","QQQ","TLT","IEF","GLD","USO",
                                                      "XLK","XLE","XLF","SMH","VNQ","EEM","FXI"]))
    if not args.skip_fetch:
        print("Fetching latest prices via yfinance...")
        fetch_latest(universe_plus_underlying)
        print("Fetching FRED macro series...")
        fetch_fred_latest()

    # Load everything
    close, opn = {}, {}
    for t in universe_plus_underlying:
        df = load_etf(t)
        if df is not None:
            close[t] = df["Close"]; opn[t] = df["Open"]
    close = pd.DataFrame(close); opn = pd.DataFrame(opn)

    dates = opn["SPY"].dropna().index
    # restrict to 2010-start for signal computation
    dates = dates[(dates >= pd.Timestamp("2010-03-11"))]
    close = close.reindex(dates).ffill(limit=5)
    opn = opn.reindex(dates).ffill(limit=5)

    # Compute today's target portfolio
    target, context = compute_signals(close, opn, dates)

    # Read previous day's held weights from log
    pos_file = R / "live_positions.csv"
    if pos_file.exists():
        prev_df = pd.read_csv(pos_file, parse_dates=["Date"]).set_index("Date").sort_index()
        prev_df = prev_df.drop(columns=["ret"], errors="ignore")  # ret is stored separately
        if len(prev_df) > 0:
            prev_row = prev_df.iloc[-1]
            prev_weights = prev_row[prev_row != 0]
        else:
            prev_weights = pd.Series(dtype=float)
    else:
        prev_weights = pd.Series(dtype=float)

    # Compute trades
    trades = compute_trades(prev_weights, target)

    # Build today's row
    today_row = target.copy()
    today_row.name = pd.Timestamp(context["as_of"])

    # Append / update positions log
    if pos_file.exists():
        pos_df = pd.read_csv(pos_file, parse_dates=["Date"]).set_index("Date").sort_index()
    else:
        pos_df = pd.DataFrame()
    # Ensure same columns
    if pos_df.empty:
        pos_df = pd.DataFrame([today_row])
        pos_df.index.name = "Date"
    else:
        today_date = pd.Timestamp(context["as_of"])
        # Remove any existing row for today (overwrite)
        pos_df = pos_df[pos_df.index != today_date]
        today_df = pd.DataFrame([today_row], index=[today_date])
        today_df.index.name = "Date"
        pos_df = pd.concat([pos_df, today_df])
    pos_df = pos_df.fillna(0).sort_index()
    pos_df.to_csv(pos_file)

    # Append trades to running log
    trades_file = R / "live_trades.csv"
    if trades:
        trade_rows = [{"Date": context["as_of"], **t} for t in trades]
        t_df = pd.DataFrame(trade_rows)
        if trades_file.exists():
            existing = pd.read_csv(trades_file)
            # Remove any prior trades for today
            existing = existing[existing["Date"] != context["as_of"]]
            t_df = pd.concat([existing, t_df], ignore_index=True)
        t_df.to_csv(trades_file, index=False)

    # Summary JSON
    current_positions = [
        {"ticker": t, "weight": float(target[t]), "pct": round(float(target[t])*100, 2)}
        for t in target.index if target[t] > 0.001
    ]
    current_positions.sort(key=lambda x: -x["weight"])

    prev_positions = [
        {"ticker": t, "weight": float(prev_weights.get(t, 0)), "pct": round(float(prev_weights.get(t, 0))*100, 2)}
        for t in prev_weights.index if prev_weights.get(t, 0) > 0.001
    ]
    prev_positions.sort(key=lambda x: -x["weight"])

    # Recent 30 days of trades (from the log)
    recent_trades = []
    if trades_file.exists():
        all_trades = pd.read_csv(trades_file, parse_dates=["Date"])
        cutoff = pd.Timestamp(context["as_of"]) - pd.Timedelta(days=30)
        recent = all_trades[all_trades["Date"] >= cutoff].sort_values(["Date"], ascending=False)
        recent_trades = recent.to_dict(orient="records")
        # Fix date field to be stringy
        for t in recent_trades:
            t["Date"] = pd.Timestamp(t["Date"]).strftime("%Y-%m-%d")

    # Price context (current LETF close + 63d momentum for ranking view)
    price_context = {}
    for t in UNIVERSE:
        if t in close.columns and not np.isnan(close[t].iloc[-1]):
            price_context[t] = {
                "price": round(float(close[t].iloc[-1]), 2),
                "mom63": round(float(close[t].iloc[-1] / close[t].iloc[-64] - 1), 4) if len(close[t]) > 64 else None,
                "mom21": round(float(close[t].iloc[-1] / close[t].iloc[-22] - 1), 4) if len(close[t]) > 22 else None,
            }

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": context,
        "previous_positions": prev_positions,
        "target_positions": current_positions,
        "trades_today": trades,
        "recent_trades_30d": recent_trades,
        "price_context": price_context,
        "blend_weights": BLEND_WEIGHTS,
    }
    (R / "live_signal.json").write_text(json.dumps(out, indent=2))

    # Pretty stdout report
    print()
    print(f"{'='*70}")
    print(f"PHOENIX v2 LIVE SIGNAL — {context['as_of']}")
    print(f"{'='*70}")
    print(f"Regime gate:       {'PASS (risk-on)' if context['regime_gate_pass'] else 'FAIL (defensive)'}")
    print(f"Risk-on subgate:   {'ON' if context['regime_risk_on'] else 'OFF'}")
    print(f"SPY vs 200dma:     {context['spy_vs_200dma']*100:+.1f}%" if context['spy_vs_200dma'] is not None else "  (no 200dma yet)")
    print(f"VIX:               {context['vix']:.1f}")
    print(f"HY OAS 20d slope:  {context['hy_oas_slope_20d']:+.2f} bps")
    print(f"Overlay mult:      {context['overlay_mult']*100:.0f}% — {context['overlay_reason']}")
    print()
    print("PREVIOUS POSITIONS:")
    if prev_positions:
        for p in prev_positions:
            print(f"  {p['ticker']:6s}  {p['pct']:>6.2f}%")
    else:
        print("  (none — first run)")
    print()
    print("TARGET POSITIONS FOR TOMORROW'S OPEN:")
    for p in current_positions:
        print(f"  {p['ticker']:6s}  {p['pct']:>6.2f}%")
    print()
    print("TRADES (execute at open, 1-day settlement):")
    if trades:
        for t in trades:
            sign = "+" if t["weight_delta"] > 0 else ""
            print(f"  {t['side']:4s}  {t['ticker']:6s}  {sign}{t['delta_pct']:+6.2f}%  "
                  f"(before {t['weight_before']*100:.2f}% → after {t['weight_after']*100:.2f}%)")
    else:
        print("  (no changes)")
    print()
    print(f"Saved: {R/'live_signal.json'}, {pos_file}, {trades_file}")


if __name__ == "__main__":
    main()
