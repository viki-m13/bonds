"""PHOENIX — live signal generator (canonical, replicates the backtest).

Runs after US market close. Fetches latest prices, calls each canonical
sleeve strategy's build_weights() to get the same daily weight DataFrame the
backtest uses, aggregates them with the locked blend weights, applies the
production overlay (vol target / DD throttle / vol gate), then compares to
yesterday's logged positions and emits BUY/SELL trades.

This is THE source of truth for live execution. The aggregated weights are
mathematically identical to the per-day weights implicitly held inside the
backtest's blended return:

    blended_return[t] = sum_i BLEND_WEIGHTS[i] * sleeve_return_i[t]
                      = sum_t agg_weight[t,j] * o2o_return[t,j]
                      where agg_weight[t,j] = sum_i BLEND_WEIGHTS[i] * W_i[t,j]

So if a trader executes the agg_weights at each market open, their realised
return reproduces the backtest within friction (TC drag).

Usage:
    python3 alt/live_signal.py [--skip-fetch] [--as-of YYYY-MM-DD]

Output files:
    data/results/live_signal.json      — today's state (read by webapp)
    data/results/live_positions.csv    — running log of daily positions
    data/results/live_trades.csv       — running log of trades

PHOENIX v3 (canonical 7-sleeve, walk-forward weights):
    Sleeves: VAN, ORI, HEL, CRY, REV, TOM, BND. QUANTUM retired.
    Blend weights are the current calendar year's walk-forward fit,
    recomputed from phoenix_production.current_blend_weights() — no
    hardcoded weights anywhere.

Production overlay (matches phoenix_production.py exactly):
    EWMA(0.94) vol target 18%, mult in [0.25, 1.0] (no margin)
    No DD throttle (historical formula was inert; corrected version rejected OOS)
    Vol gate: 60d vol > 99th pct (252d) → 0.5x
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
ALT = ROOT / "alt"
sys.path.insert(0, str(ALT))

BIL = "BIL"

import phoenix_production as prod

# Overlay parameters come from the single source of truth.
TARGET_VOL = prod.TARGET_VOL
EWMA_LAMBDA = prod.EWMA_LAMBDA
VOL_CAP = prod.VOL_CAP
VOL_FLOOR = prod.VOL_FLOOR
DD_FLOOR = prod.DD_FLOOR
DD_WIN = prod.DD_WIN
VOL_GATE_PCT = prod.GATE_PCT
VOL_GATE_LOOKBACK = prod.GATE_LOOKBACK
GATE_VOL_WIN = prod.GATE_VOL_WIN

# yfinance symbol -> local CSV basename, where they differ
FETCH_NAME_MAP = {"BTC-USD": "BTC_USD", "ETH-USD": "ETH_USD"}


# --------------- Data fetch helpers (same as before) ---------------
def fetch_latest(tickers):
    """Refresh price CSVs via yfinance with a FULL-HISTORY, fully-ADJUSTED
    re-download for every ticker.

    Why full rewrites: the previous incremental fetch appended RAW
    (auto_adjust=False) rows onto an adjusted history, so every dividend
    after the append seam booked as a fake price loss, and a split would
    have frozen a fake ±N00% day into the record (see PHOENIX_REVIEW.md
    H-1/C-2). Re-downloading the whole adjusted series each day keeps one
    consistent price basis; the sleeve-return freeze in refresh_all.py is
    what protects the published track record from vendor revisions.

    Guard: if the fresh download has fewer than 90% of the existing rows,
    the existing file is kept (protects against partial vendor responses).
    """
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed; skipping fetch.", file=sys.stderr)
        return
    today = datetime.now(timezone.utc).date()
    for t in tickers:
        fname = FETCH_NAME_MAP.get(t, t)
        p = ETF / f"{fname}.csv"
        n_existing = 0
        if p.exists():
            existing = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            n_existing = len(existing)
            if existing.index[-1].date() >= today:
                continue
        try:
            df = yf.download(t, start="2005-01-01", progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  {t}: fetch failed ({e})", file=sys.stderr)
            continue
        if df is None or len(df) == 0:
            print(f"  {t}: no data returned; keeping existing file")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
        df = df.drop_duplicates(subset="Date").sort_values("Date")
        if n_existing and len(df) < 0.9 * n_existing:
            print(f"  {t}: [WARN] fresh download has {len(df)} rows vs "
                  f"{n_existing} existing; keeping existing file")
            continue
        df.to_csv(p, index=False)
        print(f"  {t}: full adjusted rewrite, {len(df)} rows, latest={df['Date'].iloc[-1]}")


def fetch_fred_latest():
    """Refresh VIX / HY OAS / rates via FRED API. Merges, never overwrites."""
    import os, urllib.request
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        print("FRED_API_KEY not set; skipping FRED fetch.", file=sys.stderr)
        return
    series = ["VIXCLS", "BAMLH0A0HYM2", "DGS10", "DGS2", "T10Y2Y", "FEDFUNDS"]
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
        if not obs:
            continue
        rows = [(o["date"], o["value"]) for o in obs if o["value"] != "."]
        new_df = pd.DataFrame(rows, columns=["Date", s])
        new_df["Date"] = pd.to_datetime(new_df["Date"])
        new_df[s] = pd.to_numeric(new_df[s], errors="coerce")
        new_df = new_df.dropna().sort_values("Date")
        out = FRED / f"{s}.csv"
        if out.exists():
            try:
                existing = pd.read_csv(out, parse_dates=["Date"])
                existing[s] = pd.to_numeric(existing[s], errors="coerce")
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined = combined.drop_duplicates(subset="Date", keep="last").sort_values("Date")
                if len(combined) >= len(existing):
                    combined.to_csv(out, index=False)
                    new_rows = len(combined) - len(existing)
                    print(f"  {s}: {new_rows} new rows, {len(combined)} total, latest={combined['Date'].iloc[-1].date()}")
                else:
                    print(f"  {s}: [WARN] merge would shrink file; keeping existing")
            except Exception as e:
                print(f"  {s}: [WARN] merge failed ({e}); keeping existing file")
        else:
            new_df.to_csv(out, index=False)
            print(f"  {s}: wrote {len(new_df)} rows (new file)")


# --------------- Canonical aggregator ---------------
def aggregate_sleeve_weights(as_of_close: pd.Timestamp | None = None) -> tuple[pd.Series, dict]:
    """Run each canonical sleeve's build_weights(live_extend=True) and aggregate.

    Each sleeve is evaluated at index = (as_of_close + 1 BDay), i.e., the
    weight to hold at the next-day open computed from `as_of_close`'s close.
    When as_of_close=None, uses the latest available close date.

    Returns (raw_weights, sleeve_picks) where:
      raw_weights : Series indexed by ticker, summing to ~1.12 (VAN is 1.5x gross)
      sleeve_picks : dict per-sleeve {ticker: weight} for reporting
    """
    import vanguard_strategy
    import orion_strategy
    import helios_strategy
    import phoenix_v2_crypto
    import reversal_strategy
    import tom_strategy
    import bondtrend_strategy

    blend_weights = prod.current_blend_weights()

    sleeves = []
    sleeve_picks = {}

    target_index = (as_of_close + pd.tseries.offsets.BDay()) if as_of_close is not None else None

    def _last_row(W: pd.DataFrame, name: str) -> pd.Series:
        """Return the row of W corresponding to the live trading date."""
        if target_index is not None:
            sub = W.loc[W.index <= target_index]
            if sub.empty:
                raise RuntimeError(f"{name}: no weights on or before {target_index.date()}")
            row = sub.iloc[-1]
            row.name = sub.index[-1]
        else:
            row = W.iloc[-1]
        return row

    # All sleeves are called with live_extend=True so the LAST row is W[t+1]
    # computed from close[t] info — i.e., the weight to hold at next-day open.
    # Every sleeve's signal layer carries its own >=1-bar lag (HELIOS included
    # since the v3 fix), so this is uniform across all seven.
    MODULES = [
        ("VAN", "VANGUARD", lambda: vanguard_strategy.build_weights(live_extend=True)),
        ("ORI", "ORION", lambda: orion_strategy.build_weights(live_extend=True)),
        ("HEL", "HELIOS", lambda: helios_strategy.build_weights(live_extend=True)),
        ("CRY", "CRYPTO", lambda: phoenix_v2_crypto.build_weights(
            use_live_proxy=True, live_extend=True)),
        ("REV", "REVERSAL", lambda: reversal_strategy.build_weights(live_extend=True)),
        ("TOM", "TOM", lambda: tom_strategy.build_weights(live_extend=True)),
        ("BND", "BONDTREND", lambda: bondtrend_strategy.build_weights(live_extend=True)),
    ]
    for tag, name, builder in MODULES:
        W = builder()
        row = _last_row(W, name)
        sleeves.append((tag, row))
        sleeve_picks[name] = {
            "as_of": str(row.name.date()),
            "gross": float(row.sum()),
            "blend_weight": round(float(blend_weights.get(tag, 0.0)), 4),
            "weights": {t: round(float(v), 4) for t, v in row.items() if v > 1e-6},
        }

    # Aggregate: agg[ticker] = sum_i blend_weights[i] * sleeve_w_i[ticker]
    agg = {}
    for tag, row in sleeves:
        w = blend_weights.get(tag, 0.0)
        for t, v in row.items():
            if abs(v) < 1e-12:
                continue
            agg[t] = agg.get(t, 0.0) + w * float(v)
    raw = pd.Series(agg, dtype=float).sort_index()
    return raw, sleeve_picks, blend_weights


def compute_overlay_mult(as_of_close: pd.Timestamp | None = None) -> tuple[float, float, float, float, str]:
    """Read phoenix_production_returns.csv::raw_ret and compute the production
    overlay multiplier exactly as phoenix_production.py does.

    as_of_close : if given, restricts the raw_ret history to dates ≤ this
        date (so backfill mode doesn't peek at future returns).

    Returns (vol_mult, dd_mult, vol_gate_mult, total_mult, reason_string).
    """
    backtest_file = R / "phoenix_production_returns.csv"
    if not backtest_file.exists():
        return 1.0, 1.0, 1.0, 1.0, "No production CSV; multiplier defaulted to 1.0"
    bt = pd.read_csv(backtest_file, parse_dates=["Date"]).set_index("Date").sort_index()
    if "raw_ret" not in bt.columns:
        return 1.0, 1.0, 1.0, 1.0, "raw_ret missing; multiplier defaulted to 1.0"
    r_hist = bt["raw_ret"].dropna()
    if as_of_close is not None:
        r_hist = r_hist.loc[r_hist.index <= as_of_close]
    if len(r_hist) < 60:
        return 1.0, 1.0, 1.0, 1.0, "Insufficient history; multiplier defaulted to 1.0"

    # EWMA vol target (matches phoenix_production.apply_overlay one step ahead:
    # the multiplier we output scales the return realized over the NEXT
    # open->open window, and uses raw returns through the as-of close — the
    # same information set as total_mult.shift(2) in the backtest).
    # TARGET_VOL None = vol target disabled (v3.1); tail overlays remain.
    ew_var = r_hist.pow(2).ewm(alpha=1 - EWMA_LAMBDA).mean()
    ew_vol = (ew_var * 252) ** 0.5
    if TARGET_VOL is None:
        vol_mult_series = pd.Series(1.0, index=r_hist.index)
    else:
        vol_mult_series = (TARGET_VOL / ew_vol).clip(VOL_FLOOR, VOL_CAP)
    vt_mult = float(vol_mult_series.iloc[-1])

    # No DD throttle (matches phoenix_production.apply_overlay: the historical
    # formula was inert due to a sign bug; the corrected version was rejected
    # on out-of-sample evidence).
    scaled = r_hist * vol_mult_series.shift(2).fillna(1.0)
    dd_mult = 1.0

    # Vol gate on scaled returns
    sv = scaled.rolling(GATE_VOL_WIN).std()
    sv_thr = sv.rolling(VOL_GATE_LOOKBACK, min_periods=60).quantile(VOL_GATE_PCT)
    vol_gate_ok = (float(sv.iloc[-1]) <= float(sv_thr.iloc[-1])) if not np.isnan(sv_thr.iloc[-1]) else True
    vol_gate_mult = 1.0 if vol_gate_ok else 0.5

    total = vt_mult * dd_mult * vol_gate_mult
    if TARGET_VOL is None:
        parts = [f"No vol target (v3.1) — EWMA vol {ew_vol.iloc[-1]*100:.1f}%, tail overlays only"]
    else:
        parts = [f"EWMA vol target {TARGET_VOL*100:.0f}% → mult {vt_mult:.2f}x (EWMA vol {ew_vol.iloc[-1]*100:.1f}%)"]
    if vol_gate_mult < 1.0:
        parts.append("Vol-regime gate ACTIVE (60d vol > 99th pct of trailing 252d)")
    if dd_mult >= 1.0 and vol_gate_mult >= 1.0:
        parts.append("All safety gates clear")
    return float(vt_mult), float(dd_mult), float(vol_gate_mult), float(total), " · ".join(parts)


def build_target_portfolio(as_of_close: pd.Timestamp | None = None) -> tuple[pd.Series, dict]:
    """Compute final scaled target weights for the next market open.

    as_of_close : the close date the signal is computed AT. Live execution
        happens at the open of the following business day. None = use the
        latest available market close.

    Returns (target, context).
    target: Series indexed by ticker (incl. BIL), all weights ≥ 0
    context: regime + overlay diagnostics for reporting
    """
    raw, sleeve_picks, blend_weights = aggregate_sleeve_weights(as_of_close=as_of_close)
    vt_mult, dd_mult, vg_mult, total_mult, reason = compute_overlay_mult(as_of_close=as_of_close)

    # Strip BIL out of risk side so the overlay scaling doesn't shrink cash;
    # cash absorbs whatever risk doesn't fill.
    risk = raw[raw.index != BIL]
    raw_bil = float(raw.get(BIL, 0.0))
    scaled_risk = risk * total_mult
    gross_risk = float(scaled_risk.sum())
    # Hard no-margin cap: raw sleeve gross can exceed 1.0 (VANGUARD runs
    # 1.5x internal gross), so scale the whole risk book down if needed.
    if gross_risk > 1.0:
        scaled_risk = scaled_risk / gross_risk
        gross_risk = 1.0
    bil_weight = max(0.0, 1.0 - gross_risk)
    # Note: the canonical raw blend's own BIL contribution (from CRY when off
    # or HEL when off) is already inside `raw_bil`. The simpler interpretation
    # is that the production overlay scales ONLY non-cash exposure; any
    # canonical-cash share inside raw is folded into final BIL via 1-gross_risk.

    target = pd.Series(0.0, index=list(scaled_risk.index) + [BIL], dtype=float)
    for t, v in scaled_risk.items():
        target[t] = float(v)
    target[BIL] = bil_weight

    # Context block
    raw_summary = {t: round(float(v), 4) for t, v in raw.items() if abs(v) > 1e-6}
    context = {
        "as_of": str((as_of_close if as_of_close is not None else _today_close_date()).date()),
        "raw_blend_gross": float(risk.sum() + raw_bil),
        "raw_risk_gross": float(risk.sum()),
        "raw_weights": raw_summary,
        "sleeve_picks": sleeve_picks,
        "blend_weights": {k: round(float(v), 4) for k, v in blend_weights.items()},
        "vol_target_mult": vt_mult,
        "dd_mult": dd_mult,
        "vol_gate_mult": vg_mult,
        "overlay_mult": total_mult,
        "overlay_reason": reason,
    }

    # Add macro/regime snapshot for the webapp dashboards
    macro = _macro_snapshot()
    context.update(macro)
    return target, context


def _today_close_date() -> pd.Timestamp:
    """Latest market close date across SPY (anchor)."""
    p = ETF / "SPY.csv"
    df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
    return df.index[-1]


def _macro_snapshot() -> dict:
    """Snapshot of macro readings (VIX, HY OAS slope, SPY/200dma) for the webapp."""
    out = {"vix": None, "hy_oas_slope_20d": None, "spy_vs_200dma": None,
           "regime_gate_pass": None, "regime_risk_on": None}
    try:
        spy = pd.read_csv(ETF / "SPY.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        spy_c = pd.to_numeric(spy["Close"], errors="coerce")
        ma200 = spy_c.rolling(200).mean()
        if not np.isnan(ma200.iloc[-1]):
            out["spy_vs_200dma"] = float(spy_c.iloc[-1] / ma200.iloc[-1] - 1)
            spy_ok = bool(spy_c.iloc[-1] > ma200.iloc[-1] and ma200.diff(20).iloc[-1] > 0)
        else:
            spy_ok = False
        last_date = spy_c.index[-1]
        for series, key in [("VIXCLS", "vix"), ("BAMLH0A0HYM2", "hy")]:
            p = FRED / f"{series}.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
            v = pd.to_numeric(d.iloc[:, 0], errors="coerce").reindex([last_date], method="ffill").iloc[0]
            if key == "vix":
                out["vix"] = float(v)
            else:
                # 20-day slope
                hy_full = pd.to_numeric(d.iloc[:, 0], errors="coerce")
                slope = (hy_full - hy_full.shift(20)).reindex([last_date], method="ffill").iloc[0]
                out["hy_oas_slope_20d"] = float(slope)
        if out["vix"] is not None and out["hy_oas_slope_20d"] is not None:
            out["regime_gate_pass"] = bool(spy_ok and out["hy_oas_slope_20d"] < 1.0 and out["vix"] < 30)
            out["regime_risk_on"] = bool(spy_ok and out["vix"] < 22)
    except Exception as e:
        print(f"  [WARN] macro snapshot failed: {e}")
    return out


# --------------- Trade comparison ---------------
def compute_trades(prev_weights: pd.Series, target_weights: pd.Series, threshold: float = 0.005):
    """Generate buy/sell orders from weight deltas (default threshold 0.5%)."""
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
    trades.sort(key=lambda t: (t["side"] != "SELL", -abs(t["weight_delta"])))
    return trades


# --------------- Main runner ---------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-fetch", action="store_true", help="Don't download new prices")
    p.add_argument("--as-of", type=str, default=None,
                   help="Generate signal as-of this date (YYYY-MM-DD).")
    args = p.parse_args()

    # Fetch fresh data
    if not args.skip_fetch:
        # Union of all tickers across sleeves + price-context display tickers.
        import vanguard_strategy as van
        import orion_strategy as ori
        import helios_strategy as hel
        import reversal_strategy as rev
        import tom_strategy as tomm
        import bondtrend_strategy as bnd
        u = set()
        u.update(van.LEV_UNIVERSE)            # VAN's full lev universe
        u.update(ori.UNIVERSE)
        u.update(hel.PAIRS.keys())            # HEL underlyings
        u.update(hel.PAIRS.values())          # HEL leveraged expressions
        u.update(rev.PAIRS.keys())
        u.update(rev.PAIRS.values())
        u.update([tomm.VEHICLE, bnd.SIGNAL_TICKER, bnd.LONG_VEHICLE])
        u.update(["IBIT", "ETHA", "BTC-USD", "ETH-USD"])  # crypto ETFs + spot signal history
        u.update(["SPY", "BIL", "TLT"])       # anchors + factsheet benchmarks
        print("Fetching latest prices via yfinance...")
        fetch_latest(sorted(u))
        print("Fetching FRED macro series...")
        fetch_fred_latest()

    as_of_close = pd.Timestamp(args.as_of) if args.as_of else None

    # Compute canonical target portfolio
    target, context = build_target_portfolio(as_of_close=as_of_close)

    # Load previous positions
    pos_file = R / "live_positions.csv"
    if pos_file.exists():
        prev_df = pd.read_csv(pos_file, parse_dates=["Date"]).set_index("Date").sort_index()
        prev_df = prev_df.drop(columns=["ret"], errors="ignore")
        # Always diff against the most recent row STRICTLY BEFORE the signal
        # date. Otherwise, when the cron pipeline runs us twice (backfill writes
        # today's row, then refresh_all re-invokes us with --skip-fetch), the
        # second invocation would read its own freshly-written row back and
        # report trades_today=[] even when the day genuinely had trades.
        prev_df = prev_df[prev_df.index < pd.Timestamp(context["as_of"])]
        if len(prev_df) > 0:
            prev_row = prev_df.iloc[-1]
            prev_weights = prev_row[prev_row != 0]
        else:
            prev_weights = pd.Series(dtype=float)
    else:
        prev_weights = pd.Series(dtype=float)

    trades = compute_trades(prev_weights, target)

    # Append/update positions log
    today_row = target.copy()
    today_date = pd.Timestamp(context["as_of"])
    today_row.name = today_date
    if pos_file.exists():
        pos_df = pd.read_csv(pos_file, parse_dates=["Date"]).set_index("Date").sort_index()
    else:
        pos_df = pd.DataFrame()
    if pos_df.empty:
        pos_df = pd.DataFrame([today_row])
        pos_df.index.name = "Date"
    else:
        pos_df = pos_df[pos_df.index != today_date]
        today_df = pd.DataFrame([today_row], index=[today_date])
        today_df.index.name = "Date"
        # Ensure column union (new sleeves may introduce new tickers like QLD, NUGT)
        all_cols = sorted(set(pos_df.columns) | set(today_df.columns))
        pos_df = pos_df.reindex(columns=all_cols, fill_value=0.0)
        today_df = today_df.reindex(columns=all_cols, fill_value=0.0)
        pos_df = pd.concat([pos_df, today_df])
    pos_df = pos_df.fillna(0).sort_index()
    pos_df.to_csv(pos_file)

    # Append trades
    trades_file = R / "live_trades.csv"
    if trades:
        trade_rows = [{"Date": context["as_of"], **t} for t in trades]
        t_df = pd.DataFrame(trade_rows)
        if trades_file.exists():
            existing = pd.read_csv(trades_file)
            existing = existing[existing["Date"] != context["as_of"]]
            t_df = pd.concat([existing, t_df], ignore_index=True)
        t_df.to_csv(trades_file, index=False)

    # Build summary JSON
    current_positions = [
        {"ticker": t, "weight": float(target[t]), "pct": round(float(target[t]) * 100, 2)}
        for t in target.index if target[t] > 0.001
    ]
    current_positions.sort(key=lambda x: -x["weight"])

    prev_positions = [
        {"ticker": t, "weight": float(prev_weights.get(t, 0)),
         "pct": round(float(prev_weights.get(t, 0)) * 100, 2)}
        for t in prev_weights.index if prev_weights.get(t, 0) > 0.001
    ]
    prev_positions.sort(key=lambda x: -x["weight"])

    recent_trades = []
    if trades_file.exists():
        all_trades = pd.read_csv(trades_file, parse_dates=["Date"])
        cutoff = pd.Timestamp(context["as_of"]) - pd.Timedelta(days=30)
        recent = all_trades[all_trades["Date"] >= cutoff].sort_values(["Date"], ascending=False)
        recent_trades = recent.to_dict(orient="records")
        for t in recent_trades:
            t["Date"] = pd.Timestamp(t["Date"]).strftime("%Y-%m-%d")

    # Price context for display (close + 21/63d momentum on each held ticker)
    price_context = {}
    for t in current_positions:
        sym = t["ticker"]
        if sym == BIL:
            continue
        p = ETF / f"{sym}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["Date"]).set_index("Date").sort_index()
        c = pd.to_numeric(df["Close"], errors="coerce").dropna()
        if len(c) >= 64:
            price_context[sym] = {
                "price": round(float(c.iloc[-1]), 2),
                "mom63": round(float(c.iloc[-1] / c.iloc[-64] - 1), 4),
                "mom21": round(float(c.iloc[-1] / c.iloc[-22] - 1), 4) if len(c) >= 22 else None,
            }

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "context": context,
        "previous_positions": prev_positions,
        "target_positions": current_positions,
        "trades_today": trades,
        "recent_trades_30d": recent_trades,
        "price_context": price_context,
        "blend_weights": context.get("blend_weights", {}),
    }
    (R / "live_signal.json").write_text(json.dumps(out, indent=2))

    # Pretty stdout
    print()
    print("=" * 74)
    print(f"PHOENIX CANONICAL LIVE SIGNAL — {context['as_of']}")
    print("=" * 74)
    print(f"Regime gate:       {'PASS (risk-on)' if context.get('regime_gate_pass') else 'FAIL (defensive)'}")
    print(f"VIX:               {context.get('vix')}")
    print(f"HY OAS 20d slope:  {context.get('hy_oas_slope_20d')}")
    print(f"SPY vs 200dma:     {context.get('spy_vs_200dma')}")
    print(f"Overlay mult:      {context['overlay_mult']*100:.0f}% — {context['overlay_reason']}")
    print(f"Raw blend gross:   {context['raw_blend_gross']*100:.1f}%  (risk={context['raw_risk_gross']*100:.1f}%)")
    print()
    print("CANONICAL SLEEVE PICKS (each is the actual backtest's weights):")
    for sleeve_name, sp in context["sleeve_picks"].items():
        held = " · ".join(f"{t} {v*100:.1f}%" for t, v in sp["weights"].items())
        print(f"  {sleeve_name:8s} (gross {sp['gross']:.2f}, as_of {sp['as_of']}): {held}")
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
    print("TRADES (execute at next market open):")
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
