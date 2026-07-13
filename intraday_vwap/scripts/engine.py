"""VWAP Trend Trading (SSRN 4631351) — replication + honest extension on repo
5-minute data (2016-2026, RTH).

Paper's rules (1-min, QQQ, 2018-2023): at 9:31 close go long if close>VWAP else
short; thereafter flip/exit whenever a candle CLOSES on the other side of session
VWAP; flat at 16:00; 100% of capital; $0.0005/share commission; no slippage.

This engine is the 5-minute analogue: first signal at the 9:35 close (bar labeled
09:30), position re-evaluated at every 5-min close, flat at the day's last close.
Session VWAP = cum(bar_vwap*volume)/cum(volume) using the vendor per-bar vwap.

Costs are parameterized per SIDE in dollars/share: commission + half-spread
slippage. The paper's setup = commission 0.0005, slip 0. Honest runs add slippage.

Variants (for the improvement study):
  band_bps    — only flip if |close-VWAP| > band (whipsaw filter); position holds
                its previous side inside the band.
  long_only   — shorts become flat.
  no_midday   — no NEW positions from 12:00-14:55 (existing positions keep running);
                (paper observed midday contributes little — test, don't assume).
  confirm     — require N consecutive closes on the new side before flipping.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

def load(tk):
    df = pd.read_csv(f"{REPO}/data/intraday_5min/{tk}.csv", parse_dates=["ts"])
    df["date"] = df["ts"].dt.normalize()
    # session vwap
    pv = (df["vwap"]*df["volume"]).groupby(df["date"]).cumsum()
    vv = df["volume"].groupby(df["date"]).cumsum()
    df["svwap"] = pv/vv
    df["tmin"] = df["ts"].dt.hour*60 + df["ts"].dt.minute
    return df

def run_day(day, cost_ps, band_bps=0.0, long_only=False, short_only=False,
            no_midday=False, confirm=1):
    """day: frame of one session. Returns (pnl_fraction_list_of_trades, gross_ret,
    net_ret, n_trades, shares_turnover_per_$). Position in {-1,0,1}, changed at bar
    closes; return accrues close->close; all flat at last close."""
    c = day["close"].values; v = day["svwap"].values; t = day["tmin"].values
    n = len(c)
    if n < 10: return [], 0.0, 0.0, 0
    pos = np.zeros(n, dtype=float)      # position held FROM bar i close TO i+1 close
    band = band_bps*1e-4
    streak_side, streak = 0, 0
    cur = 0.0
    for i in range(n-1):                 # last bar: flatten (pos stays whatever until last close, then flat)
        dev = c[i]/v[i]-1.0
        side = 1 if dev > band else (-1 if dev < -band else 0)
        if side != 0:
            streak = streak+1 if side == streak_side else 1
            streak_side = side
        want = cur
        if side != 0 and side != np.sign(cur):
            if streak >= confirm:
                want = float(side)
        elif side != 0 and cur == 0.0 and streak >= confirm:
            want = float(side)
        if band > 0 and side == 0 and cur != 0.0:
            want = cur                   # inside band: hold
        if long_only and want < 0: want = 0.0
        if short_only and want > 0: want = 0.0
        if no_midday and 720 <= t[i] <= 895 and want != cur and abs(want) > 0:
            want = cur if cur != 0 else 0.0   # no new entries midday; exits still allowed via band=0 path
        pos[i] = want; cur = want
    # bar-to-bar returns while holding
    ret = np.zeros(n); ret[1:] = c[1:]/c[:-1]-1.0
    gross = float(np.sum(pos[:-1]*ret[1:]))
    # costs: per position CHANGE, |delta_pos| units traded at that bar's close
    dpos = np.diff(np.concatenate([[0.0], pos[:-1], [0.0]]))   # includes entry + final flatten
    trade_px = np.concatenate([c[:-1], c[-1:]])
    cost = float(np.sum(np.abs(dpos)*cost_ps/trade_px))
    # per-trade pnl for win-rate stats: a "trade" = maximal run of constant nonzero pos
    trades = []
    i = 0; posv = np.concatenate([pos[:-1], [0.0]])
    while i < n:
        if posv[i] != 0:
            j = i
            while j < n and posv[j] == posv[i]: j += 1
            entry = c[i]; exitp = c[min(j, n-1)]
            pnl = posv[i]*(exitp/entry-1.0) - 2*cost_ps/entry
            trades.append(pnl); i = j
        else: i += 1
    return trades, gross, gross-cost, len(trades)

def backtest(tk, start=None, end=None, cost_ps=0.0005, slip_ps=0.0, **kw):
    df = load(tk)
    if start: df = df[df["date"] >= pd.Timestamp(start)]
    if end: df = df[df["date"] <= pd.Timestamp(end)]
    out = []; all_trades = []
    for d, day in df.groupby("date"):
        trades, gross, net, ntr = run_day(day, cost_ps+slip_ps, **kw)
        out.append((d, gross, net, ntr, day["close"].iloc[-1]))
        all_trades += trades
    r = pd.DataFrame(out, columns=["date", "gross", "net", "ntr", "close"]).set_index("date")
    return r, np.array(all_trades)

def stats(r, trades, label=""):
    net = r["net"]
    eq = (1+net).cumprod()
    yrs = len(net)/252
    cagr = eq.iloc[-1]**(1/yrs)-1
    sh = net.mean()/net.std()*np.sqrt(252) if net.std() > 0 else 0
    dd = (eq/eq.cummax()-1).min()
    wins = trades[trades > 0]; losses = trades[trades <= 0]
    wr = len(wins)/len(trades) if len(trades) else 0
    wl = (wins.mean()/abs(losses.mean())) if len(wins) and len(losses) else np.nan
    # buy & hold same window
    bh = r["close"].iloc[-1]/r["close"].iloc[0]
    bhr = r["close"].pct_change().dropna()
    bh_sh = bhr.mean()/bhr.std()*np.sqrt(252)
    bh_eq = r["close"]/r["close"].iloc[0]
    bh_dd = (bh_eq/bh_eq.cummax()-1).min()
    return dict(label=label, mult=float(eq.iloc[-1]), cagr=float(cagr), sharpe=float(sh),
                maxdd=float(dd), ntrades=int(len(trades)), winrate=float(wr),
                wl_ratio=float(wl), bh_mult=float(bh), bh_sharpe=float(bh_sh),
                bh_dd=float(bh_dd), years=float(yrs))

def fmt(s):
    return (f"{s['label']:24} {s['mult']:7.2f}x  CAGR {s['cagr']:6.1%}  Shp {s['sharpe']:5.2f}  "
            f"DD {s['maxdd']:6.1%}  trades {s['ntrades']:6d}  WR {s['winrate']:4.0%}  "
            f"W/L {s['wl_ratio']:4.1f}  [B&H {s['bh_mult']:5.2f}x Shp {s['bh_sharpe']:4.2f} DD {s['bh_dd']:5.1%}]")

if __name__ == "__main__":
    # 1) REPLICATION on the paper's window (2018-2023), paper costs (no slippage)
    print("=== REPLICATION (paper window 2018-01..2023-12, commission only, no slip) ===")
    for tk in ["QQQ", "SPY"]:
        r, tr = backtest(tk, "2018-01-01", "2023-12-31", cost_ps=0.0005)
        print(fmt(stats(r, tr, f"{tk} 5-min VWAP-TT")))
    # 2) full sample + true OOS
    print("\n=== FULL SAMPLE 2016-2026 + OOS windows (commission only) ===")
    for tk in ["QQQ", "SPY"]:
        for a, b, lab in [("2016-01-01", "2026-04-30", "full 16-26"),
                          ("2016-01-01", "2017-12-31", "pre 16-17"),
                          ("2024-01-01", "2026-04-30", "OOS 24-26")]:
            r, tr = backtest(tk, a, b, cost_ps=0.0005)
            print(fmt(stats(r, tr, f"{tk} {lab}")))
    # 3) cross-section, all 7 tickers, full sample
    print("\n=== CROSS-SECTION all tickers (full sample, commission only) ===")
    for tk in ["QQQ", "SPY", "DIA", "IWM", "XLF", "GLD", "TLT"]:
        r, tr = backtest(tk, cost_ps=0.0005)
        print(fmt(stats(r, tr, tk)))
    # 4) cost honesty: add half-spread slippage (1 cent spread -> 0.005/side; also 1c/side)
    print("\n=== COST STRESS on QQQ full sample ===")
    for slip, lab in [(0.0, "slip $0"), (0.005, "slip 0.5c/sh"), (0.01, "slip 1c/sh")]:
        r, tr = backtest("QQQ", cost_ps=0.0005, slip_ps=slip)
        print(fmt(stats(r, tr, f"QQQ {lab}")))
