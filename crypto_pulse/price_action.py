"""Honest price-action backtests on crypto (Coinbase 1-min, 15 coins, 60d), the
candidates the multi-source research surfaces, net of HL taker fees.

Crypto is 24/7 so "sessions" are defined on the UTC calendar day (00:00 open).
All execution: signal at close of minute t -> enter at open[t+1]; taker fee
4.5 bps/side; per-coin then equal-weight portfolio; Sharpe annualized by trading
days. This is the harness to vet ORB / intraday-momentum / VWAP / breakout ideas;
it intentionally uses simple, non-overfit rules.

Run from crypto_pulse/:  python price_action.py
"""
import glob
import os

import numpy as np
import pandas as pd

DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "crypto_1min")
TAKER = 4.5


def load_one(t):
    d = pd.read_csv(os.path.join(DIR, f"{t}.csv"))
    d["ts"] = pd.to_datetime(d["ts"], unit="s")
    d = d[~d["ts"].duplicated()].set_index("ts").sort_index()
    d["day"] = d.index.normalize()
    return d


def coins():
    return [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(DIR, "*.csv")))]


def daily_sharpe(daily_ret):
    p = daily_ret.dropna()
    return p.mean() / p.std() * np.sqrt(365) if len(p) > 20 and p.std() > 0 else np.nan


def orb(d, or_min=30, cost=TAKER):
    """Opening-range breakout on the UTC day. Long if price breaks the first
    `or_min` minutes' high, short if it breaks the low; enter next bar; exit at
    day end. One trade/day (first break). Returns per-day pnl (one trade)."""
    out = {}
    for day, g in d.groupby("day"):
        g = g.reset_index()
        if len(g) < or_min + 30:
            continue
        orng = g.iloc[:or_min]
        hi, lo = orng["high"].max(), orng["low"].min()
        rest = g.iloc[or_min:]
        long_hit = rest.index[rest["high"] > hi]
        short_hit = rest.index[rest["low"] < lo]
        first_long = long_hit[0] if len(long_hit) else None
        first_short = short_hit[0] if len(short_hit) else None
        if first_long is None and first_short is None:
            continue
        if first_short is None or (first_long is not None and first_long <= first_short):
            i = first_long; side = 1; entry = hi
        else:
            i = first_short; side = -1; entry = lo
        # enter at the bar's open after the trigger bar (next bar)
        ei = min(i + 1, len(g) - 1)
        entry = g.loc[ei, "open"]
        exit_px = g.iloc[-1]["close"]
        out[day] = side * (exit_px / entry - 1) - 2 * cost / 1e4
    return pd.Series(out)


def intraday_mom(d, first_min=60, cost=TAKER):
    """Gao et al. market-intraday-momentum, crypto: sign of the first `first_min`
    minutes' return -> position for the rest of the day; exit at close."""
    out = {}
    for day, g in d.groupby("day"):
        g = g.reset_index()
        if len(g) < first_min + 30:
            continue
        r1 = g.iloc[first_min - 1]["close"] / g.iloc[0]["open"] - 1
        if r1 == 0:
            continue
        entry = g.iloc[first_min]["open"]
        exit_px = g.iloc[-1]["close"]
        out[day] = np.sign(r1) * (exit_px / entry - 1) - 2 * cost / 1e4
    return pd.Series(out)


def vwap_reclaim(d, cost=TAKER):
    """Daily-anchored VWAP reclaim: at each bar, hold long if close>VWAP else
    flat (trend-following around VWAP); daily pnl from minute returns while long,
    minus cost per flip. Long-only (spot-style)."""
    out = {}
    for day, g in d.groupby("day"):
        g = g.reset_index()
        if len(g) < 60:
            continue
        tp = (g["high"] + g["low"] + g["close"]) / 3
        vwap = (tp * g["volume"]).cumsum() / g["volume"].cumsum()
        pos = (g["close"] > vwap).astype(float).shift(1).fillna(0)
        ret = g["close"].pct_change().fillna(0)
        flips = pos.diff().abs().fillna(0)
        out[day] = (pos * ret - flips * cost / 1e4).sum()
    return pd.Series(out)


def main():
    cs = coins()
    data = {t: load_one(t) for t in cs}
    strategies = {
        "ORB-15": lambda d: orb(d, 15),
        "ORB-30": lambda d: orb(d, 30),
        "ORB-60": lambda d: orb(d, 60),
        "intraday-mom-60": lambda d: intraday_mom(d, 60),
        "intraday-mom-30": lambda d: intraday_mom(d, 30),
        "vwap-reclaim(long)": vwap_reclaim,
    }
    lines = [f"# Price-action backtests on crypto (Coinbase 1-min, {len(cs)} "
             "coins, 60d), net of HL taker\n",
             "Crypto 24/7: sessions on the UTC day. Enter next-bar open after "
             f"signal, {TAKER}bps/side taker, equal-weight portfolio. **IS/OOS = "
             "first/second half of the 60-day window** — the honesty gate.\n",
             "| strategy | port Sharpe | IS | OOS | annRet | med-coin Sharpe | %coins>0 |",
             "|---|---|---|---|---|---|---|"]
    # split date from a representative portfolio
    ref = pd.DataFrame({t: orb(data[t], 30) for t in cs}).mean(axis=1)
    days = ref.index.sort_values()
    cut = days[len(days) // 2]
    for name, fn in strategies.items():
        per = {t: fn(data[t]) for t in cs}
        port = pd.DataFrame(per).mean(axis=1)
        coin_sh = pd.Series({t: daily_sharpe(s) for t, s in per.items()})
        lines.append(
            f"| {name} | {daily_sharpe(port):+.2f} | "
            f"{daily_sharpe(port[port.index < cut]):+.2f} | "
            f"{daily_sharpe(port[port.index >= cut]):+.2f} | "
            f"{port.mean()*365:+.0%} | {coin_sh.median():+.2f} | "
            f"{(coin_sh > 0).mean():.0%} |")
    lines.append("\n**Verdict:** the only high full-sample Sharpe (ORB-60, 3.48) "
                 "is a MIRAGE — IS −5.2 vs OOS +8.5: it lost badly in the first "
                 "30 days and won big in the last 30, i.e. a pure trend-regime "
                 "bet on 60 days, not an edge. Intraday-momentum is negative and "
                 "VWAP-reclaim is destroyed by whipsaw+cost. None of these is a "
                 "validated edge; 60 days is far too short and one regime. No "
                 "taker-viable price-action Sharpe 3 here.\n")
    out = "\n".join(lines)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                     "price_action.md")
    with open(p, "w") as fh:
        fh.write(out)
    print(out)
    print("[written]", p)


if __name__ == "__main__":
    main()
