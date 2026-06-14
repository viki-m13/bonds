"""Validation of the user-submitted 'fair-value z-score' mean-reversion scripts.

The submitted scripts (daily / hourly / 5-min / iShares variants) all share ONE
core signal and ONE evaluation method:

  SIGNAL  (computed per ticker, vs SPY):
    z    = residual of rolling OLS  log(stock) ~ log(SPY)   (5y beta, 2y resid std)
    gate = (z <= -2.75) & (weekly RSI(14) <= 40) & (drawdown from 252d high <= -30%)
           & (20d $volume >= 25M)
  "STATS" (their claim of profitability):
    for horizon n in {252, 756, 1260} trading days:
      fwd = price.shift(-n)/price - 1
      compare fwd on signal-days vs non-signal-days -> P(gain), avg, worst, delta

This module re-implements that EXACTLY, then runs it three ways on the repo's
point-in-time (PIT) S&P 500 panel to isolate survivorship bias, and quantifies
the delisting-drop bias baked into forward_return itself.

Run:  python3 validate_user_strategies.py
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PANEL = os.path.join(ROOT, "data", "pit", "summit_panel.parquet")
SPY_CSV = os.path.join(ROOT, "data", "etfs_extended", "SPY.csv")

# ---- user config (verbatim) ----
BETA_WINDOW = 1260
RESID_STD_WINDOW = 504
Z_THRESH = -2.75
RSI_LEN_W = 14
RSI_MAX = 40.0
DD_LOOKBACK = 252
DD_MIN = 0.30
USE_LIQUIDITY = True
DV_LOOKBACK = 20
MIN_DOLLAR_VOL = 25_000_000
HORIZONS = [("1Y", 252), ("3Y", 756), ("5Y", 1260)]

# user's hand-picked "daily with stats" universe (today's mega-caps)
HANDPICKED = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","JPM","V","MA",
    "UNH","LLY","XOM","CVX","COST","WMT","HD","LIN","ABBV","KO","PEP","MRK",
    "AVGO","ORCL","CSCO","MCD","PM","TMO","ABT","CAT","BA","GE","UPS","IBM",
    "INTC","AMD","ADBE","CRM","QCOM","TXN","LOW","GS","MS","BLK","SMCI",
]


# ---- user's functions, verbatim ----
def weekly_rsi_daily_aligned(close, length=14):
    w = close.resample("W-FRI").last().dropna()
    d = w.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1/length, adjust=False).mean()
    rs = roll_up / (roll_dn + 1e-12)
    rsi = 100 - 100/(1 + rs)
    return rsi.reindex(close.index, method="ffill")


def rolling_alpha_beta(log_s, log_m, window, ridge=1e-9):
    mx = log_m.rolling(window).mean()
    my = log_s.rolling(window).mean()
    cov = (log_m * log_s).rolling(window).mean() - mx * my
    var = (log_m * log_m).rolling(window).mean() - mx * mx
    beta = cov / (var + ridge)
    alpha = my - beta * mx
    return alpha, beta


def forward_return(px, n):
    return px.shift(-n) / px - 1.0


def build_signal(close_t, vol_t, spy):
    """Replicate the user's per-ticker signal exactly (overlap-only)."""
    pair = pd.DataFrame({"s": close_t, "m": spy}).dropna()
    if len(pair) < max(BETA_WINDOW, RESID_STD_WINDOW, DD_LOOKBACK) + 50:
        return None
    log_s = np.log(pair["s"])
    log_m = np.log(pair["m"])
    alpha, beta = rolling_alpha_beta(log_s, log_m, BETA_WINDOW)
    fair = alpha + beta * log_m
    resid = log_s - fair
    resid_std = resid.rolling(RESID_STD_WINDOW).std()
    z = resid / (resid_std + 1e-12)
    rsi_w = weekly_rsi_daily_aligned(pair["s"], RSI_LEN_W)
    dd_high = pair["s"].rolling(DD_LOOKBACK).max()
    dd = pair["s"] / dd_high - 1.0
    ok = (z <= Z_THRESH) & (rsi_w <= RSI_MAX) & (dd <= -DD_MIN)
    if USE_LIQUIDITY and vol_t is not None:
        dv = (pair["s"] * vol_t.reindex(pair.index)).rolling(DV_LOOKBACK).mean()
        ok = ok & (dv >= MIN_DOLLAR_VOL)
    return ok.reindex(close_t.index).fillna(False)


def pooled_stats(close, vol, spy, names, member=None, mask_pit=False,
                 floor_delisting=False):
    """User's stats, pooled across `names`.

    member/mask_pit: if True, a signal only counts on days the ticker was an
      actual index member (PIT-correct).
    floor_delisting: if True, when a signal can't realize its full horizon
      because the ticker stopped trading (delisted/acquired), count the return
      as 'hold to last available price' instead of silently dropping it.
    """
    out = {h: {"sig": [], "non": [], "dropped": 0} for h, _ in HORIZONS}
    n_sig_total = 0
    for t in names:
        if t not in close.columns:
            continue
        ct = close[t].dropna()
        if len(ct) < 400:
            continue
        vt = vol[t] if (vol is not None and t in vol.columns) else None
        sig = build_signal(ct, vt, spy)
        if sig is None:
            continue
        if mask_pit and member is not None and t in member.columns:
            m = member[t].reindex(sig.index).fillna(False)
            sig = sig & m
        n_sig_total += int(sig.sum())
        px = ct.reindex(sig.index)
        for hname, n in HORIZONS:
            fwd = forward_return(px, n)
            elig = px.notna()
            sig_mask = elig & sig
            non_mask = elig & (~sig) & px.shift(-n).notna()
            # signal days: realized forward return, or floored to last price
            sr = fwd[sig_mask]
            if floor_delisting:
                # where shift(-n) is NaN but the series still has a later point,
                # use last available price as exit (hold-to-delisting)
                last_px = px.dropna().iloc[-1] if px.notna().any() else np.nan
                missing = sig_mask & px.shift(-n).isna()
                floored = (last_px / px[missing] - 1.0)
                sr = pd.concat([fwd[sig_mask & px.shift(-n).notna()], floored])
            else:
                out[hname]["dropped"] += int((sig_mask & px.shift(-n).isna()).sum())
                sr = fwd[sig_mask & px.shift(-n).notna()]
            out[hname]["sig"].extend(sr.dropna().tolist())
            out[hname]["non"].extend(fwd[non_mask].dropna().tolist())
    res = {"n_signal_days": n_sig_total}
    for hname, _ in HORIZONS:
        s = np.array(out[hname]["sig"], float)
        no = np.array(out[hname]["non"], float)
        res[hname] = {
            "sig_n": len(s),
            "sig_pgain": float((s > 0).mean()) if len(s) else np.nan,
            "sig_avg": float(s.mean()) if len(s) else np.nan,
            "sig_worst": float(s.min()) if len(s) else np.nan,
            "non_n": len(no),
            "non_pgain": float((no > 0).mean()) if len(no) else np.nan,
            "non_avg": float(no.mean()) if len(no) else np.nan,
            "delta_pgain": (float((s > 0).mean()) - float((no > 0).mean()))
            if len(s) and len(no) else np.nan,
            "delta_avg": (float(s.mean()) - float(no.mean()))
            if len(s) and len(no) else np.nan,
            "dropped_delisting": out[hname]["dropped"],
        }
    return res


def count_episodes(close, vol, spy, names, member=None, mask_pit=False, gap=21):
    """Distinct signal 'episodes' (>= gap trading days apart) vs raw signal-days,
    to expose overlapping-sample inflation in the user's n."""
    raw = 0
    episodes = 0
    for t in names:
        if t not in close.columns:
            continue
        ct = close[t].dropna()
        if len(ct) < 400:
            continue
        vt = vol[t] if (vol is not None and t in vol.columns) else None
        sig = build_signal(ct, vt, spy)
        if sig is None:
            continue
        if mask_pit and member is not None and t in member.columns:
            sig = sig & member[t].reindex(sig.index).fillna(False)
        days = list(sig.index[sig.values])
        raw += len(days)
        last = None
        for d in days:
            if last is None or (d - last).days >= gap * 1.4:
                episodes += 1
                last = d
    return raw, episodes


def fmt(r):
    def p(x): return "  NA  " if pd.isna(x) else f"{100*x:5.1f}%"
    def rr(x): return "   NA   " if pd.isna(x) else f"{100*x:+7.2f}%"
    lines = []
    for hname, _ in HORIZONS:
        d = r[hname]
        lines.append(
            f"  {hname}: SIGNAL n={d['sig_n']:6d} P(gain)={p(d['sig_pgain'])} "
            f"avg={rr(d['sig_avg'])} worst={rr(d['sig_worst'])} | "
            f"NON n={d['non_n']:7d} P(gain)={p(d['non_pgain'])} avg={rr(d['non_avg'])} "
            f"| dP={p(d['delta_pgain'])} davg={rr(d['delta_avg'])} "
            f"| delisting-dropped={d['dropped_delisting']}")
    return "\n".join(lines)


def main():
    p = pd.read_parquet(PANEL)
    close, vol, member = p["close"], p["volume"], p["member"]
    spy = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)["Close"]
    spy = spy[~spy.index.duplicated()].reindex(close.index).ffill()

    all_names = list(close.columns)
    survivors = [t for t in all_names if member[t].iloc[-1]]
    delisted = [t for t in all_names if t not in survivors]
    print(f"Panel: {len(all_names)} names | current members (survivors)="
          f"{len(survivors)} | delisted/removed={len(delisted)}")
    print(f"Span: {close.index[0].date()} .. {close.index[-1].date()}\n")

    print("=" * 100)
    print("TEST A — user's hand-picked mega-cap list (their 'daily with stats' universe)")
    print("         pure survivors + the biggest winners of the last decade")
    print("=" * 100)
    ra = pooled_stats(close, vol, spy, [t for t in HANDPICKED if t in close.columns])
    print(f"  signal-days fired: {ra['n_signal_days']}")
    print(fmt(ra))

    print("\n" + "=" * 100)
    print("TEST B — SURVIVOR-biased S&P 500: only the 502 names still in the index today,")
    print("         full history, NO membership mask  (≈ the iShares-holdings scripts)")
    print("=" * 100)
    rb = pooled_stats(close, vol, spy, survivors)
    print(f"  signal-days fired: {rb['n_signal_days']}")
    print(fmt(rb))

    print("\n" + "=" * 100)
    print("TEST C — PIT-CORRECT S&P 500: all 720 names incl. delisted, membership mask")
    print("         (a name only counts on days it was actually in the index)")
    print("=" * 100)
    rc = pooled_stats(close, vol, spy, all_names, member=member, mask_pit=True)
    print(f"  signal-days fired: {rc['n_signal_days']}")
    print(fmt(rc))

    print("\n" + "=" * 100)
    print("TEST D — PIT-CORRECT but delisting-aware (hold-to-last-price instead of")
    print("         silently dropping signals that delist before the horizon)")
    print("=" * 100)
    rd = pooled_stats(close, vol, spy, all_names, member=member, mask_pit=True,
                      floor_delisting=True)
    print(fmt(rd))

    print("\n" + "=" * 100)
    print("OVERLAPPING-SAMPLE CHECK (PIT universe): raw signal-days vs distinct episodes")
    print("=" * 100)
    raw, eps = count_episodes(close, vol, spy, all_names, member=member, mask_pit=True)
    print(f"  raw signal-days (the 'n' the script reports) = {raw}")
    print(f"  distinct episodes (>=~1 month apart)         = {eps}")
    print(f"  each 5Y stat reuses ~1260 overlapping forward windows per episode;")
    print(f"  the effective independent sample is ~{eps}, not {raw}.")


if __name__ == "__main__":
    main()
