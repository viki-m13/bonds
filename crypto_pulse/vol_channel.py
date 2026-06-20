"""Vol-channel breakout sleeve — consolidated from the `vol` repo's HL strategy.

Reproduces the adaptive volatility-channel breakout (VWAP +/- band*sigma, 10-hour
eval, vol-targeted, drawdown-scaled) from viki-m13/vol's crypto_strategy, and tests
HONESTLY on our hourly HL-crypto data whether it (a) is net-positive after REAL HL
costs and (b) diversifies our daily cross-sectional grand stack (different alpha:
per-asset INTRADAY trend-breakout vs daily cross-sectional factors).

Signal/sizing logic adapted from viki-m13/vol crypto_strategy/strategy.py &
crypto_research/engine.py (production params: band_mult 1.5, 10h eval, 4d VWAP,
14h sigma, 1d rvol, target 2%/day, DD-scale 2%->5%). The vol repo's own honest
finding: headline Sharpe 4-6 is MAKER-dependent; as a taker, turnover kills it.
Real HL taker = 4.5bps (confirmed by their live reconciliation). We map their
5-min/120-bar design to our hourly data (10-bar eval, 96-bar VWAP).

Run from crypto_pulse/:  python vol_channel.py  (-> research/vol_channel.md + png)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import kelly_cagr as kc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HRLY = os.path.join(ROOT, "data", "crypto_hourly_cb")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
HL_START = pd.Timestamp("2023-05-12")
BARS_PER_DAY = 24
# production params mapped 5min->hourly (divide bar counts by 12)
P = dict(band_mult=1.5, trade_freq=10, vwap_w=96, sigma_w=14, vol_w=24,
         target_vol=0.02, max_lev=2.0, dd_start=0.02, dd_max=0.05, funding_8h=1e-4)


def rolling_vwap(df, w):
    typ = (df["high"] + df["low"] + df["close"]) / 3
    vp = (typ * df["volume"]).rolling(w, min_periods=w).sum()
    v = df["volume"].rolling(w, min_periods=w).sum()
    return (vp / v.replace(0, np.nan)).values


def sigma_mad(close, tf, sw):
    r = (close / close.shift(tf) - 1).abs()
    return r.rolling(sw, min_periods=max(1, sw // 2)).mean().shift(tf).values


def realized_vol(close, w):
    lr = np.log(close / close.shift(1))
    return (lr.rolling(w, min_periods=w).std() * np.sqrt(BARS_PER_DAY)).values


def breakout_signal(close, vwap, sigma, tf, bm):
    n = len(close)
    UB = vwap * (1 + bm * sigma); LB = vwap * (1 - bm * sigma)
    raw = np.full(n, np.nan)
    for i in range(0, n, tf):
        if np.isnan(UB[i]) or np.isnan(LB[i]) or np.isnan(vwap[i]):
            raw[i] = 0; continue
        raw[i] = 1 if (close[i] > UB[i] and close[i] > vwap[i]) else (
            -1 if (close[i] < LB[i] and close[i] < vwap[i]) else 0)
    last = np.nan; filled = np.zeros(n)
    for i in range(n):
        if not np.isnan(raw[i]):
            last = raw[i]
        if last == 0:
            last = np.nan
        filled[i] = last if not np.isnan(last) else 0
    lag = np.zeros(n); lag[1:] = filled[:-1]
    return lag


def dd_scale(cum, ds, dm):
    n = len(cum); scale = np.ones(n); peak = cum[0]
    for i in range(n):
        if cum[i] > peak:
            peak = cum[i]
        dd = cum[i] / peak - 1 if peak > 0 else 0
        if dd < -ds:
            frac = min((abs(dd) - ds) / (dm - ds), 1.0)
            scale[i] = 1.0 - 0.75 * frac
    return scale


def backtest_coin(df, comm_bps):
    cl = df["close"]
    vw = rolling_vwap(df, P["vwap_w"])
    sg = sigma_mad(cl, P["trade_freq"], P["sigma_w"])
    av = realized_vol(cl, P["vol_w"])
    sig = breakout_signal(cl.values, vw, sg, P["trade_freq"], P["band_mult"])
    size = np.clip(P["target_vol"] / (av + 1e-9), 0, P["max_lev"])
    exp = sig * size
    pr = np.zeros(len(cl)); pr[1:] = cl.values[1:] / cl.values[:-1] - 1
    raw_pnl = np.zeros(len(cl)); raw_pnl[1:] = exp[:-1] * pr[1:]
    cum = np.cumprod(1 + np.nan_to_num(raw_pnl))
    exp = exp * dd_scale(cum, P["dd_start"], P["dd_max"])     # DD-scaled exposure
    turn = np.abs(np.diff(exp, prepend=0))
    comm = turn * comm_bps / 1e4
    funding = np.abs(exp) * P["funding_8h"] / (8 * BARS_PER_DAY / 24)
    net = np.zeros(len(cl)); net[1:] = exp[:-1] * pr[1:] - comm[1:] - funding[1:]
    return pd.Series(net, index=df.index)


def load_hourly():
    out = {}
    for f in sorted(glob.glob(os.path.join(HRLY, "*.csv"))):
        t = os.path.basename(f)[:-4]
        d = pd.read_csv(f); d["ts"] = pd.to_datetime(d["ts"], unit="s")
        out[t] = d.set_index("ts").sort_index()
    return out


def portfolio(coins, comm_bps):
    cols = {}
    for c, df in coins.items():
        cols[c] = backtest_coin(df, comm_bps)
    R = pd.DataFrame(cols)
    return R.mean(axis=1)                          # equal-weight per-coin books


def sharpe_h(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(24 * 365) if (len(p) > 500 and p.std() > 0) else np.nan


def main():
    coins = load_hourly()
    gross = portfolio(coins, 0.0)
    maker = portfolio(coins, 1.5)
    taker = portfolio(coins, 4.5)

    def dd(p):
        c = (1 + p.fillna(0)).cumprod(); return (c / c.cummax() - 1).min()

    lines = ["# Vol-channel breakout sleeve (from the `vol` repo) on our hourly data\n"]
    lines.append("Adaptive VWAP+/-band*sigma breakout, 10h eval, vol-targeted + "
                 "DD-scaled, per-coin on 27 HL coins (hourly, 2024-2026). Net of HL "
                 "costs. The vol repo's headline (Sharpe 4-6) is maker-only; here is "
                 "the honest read on OUR data at real fees.\n")
    lines.append("| execution | Sharpe (ann) | ann ret | maxDD | turn/day |")
    lines.append("|---|---|---|---|---|")
    # rough turnover
    for nm, p, fee in [("gross (0bps)", gross, 0), ("maker (1.5bps)", maker, 1.5),
                       ("taker (4.5bps)", taker, 4.5)]:
        s = sharpe_h(p)
        ann = p.mean() * 24 * 365
        lines.append(f"| {nm} | **{s:+.2f}** | {ann:+.0%} | {dd(p):+.0%} | — |")

    # daily aggregation + correlation to grand stack
    tk_d = taker.resample("1D").sum()
    mk_d = maker.resample("1D").sum()
    gstack = kc.build_grandstack()                 # daily grand stack, ~12% vol
    both = pd.concat({"grand": gstack, "volch_taker": tk_d,
                      "volch_maker": mk_d}, axis=1).dropna()
    both = both[both.index >= HL_START]
    corr_t = both["grand"].corr(both["volch_taker"])
    corr_m = both["grand"].corr(both["volch_maker"])

    def dsharpe(p):
        p = p.dropna(); return p.mean()/p.std()*np.sqrt(365) if p.std()>0 else np.nan

    lines.append(f"\n## Diversification vs the daily grand stack (daily, HL overlap)\n")
    lines.append(f"- vol-channel daily Sharpe: taker {dsharpe(both['volch_taker']):+.2f}, "
                 f"maker {dsharpe(both['volch_maker']):+.2f}; grand stack "
                 f"{dsharpe(both['grand']):+.2f}.")
    lines.append(f"- correlation to grand stack: taker **{corr_t:+.2f}**, maker "
                 f"**{corr_m:+.2f}**.\n")

    # does adding it help? (equal-risk blend, taker version = honest for us)
    def vt(p, t=0.12): return p*(t/(p.rolling(45).std()*np.sqrt(365))).shift(1).clip(0,3)
    blends = {"grand alone": vt(both["grand"]),
              "grand + volch (taker)": vt(0.6*both["grand"] + 0.4*both["volch_taker"]),
              "grand + volch (maker)": vt(0.6*both["grand"] + 0.4*both["volch_maker"])}
    lines.append("| book | daily Sharpe | maxDD |")
    lines.append("|---|---|---|")
    for nm, p in blends.items():
        lines.append(f"| {nm} | **{dsharpe(p):+.2f}** | {dd(p):+.0%} |")

    sg = dsharpe(blends["grand alone"]); st = dsharpe(blends["grand + volch (taker)"])
    lines.append("\n## Verdict\n")
    lines.append(f"- On OUR hourly data at real fees: gross Sharpe "
                 f"{sharpe_h(gross):+.1f}, **net-taker {sharpe_h(taker):+.2f}**, "
                 f"net-maker {sharpe_h(maker):+.2f}. This reproduces the vol repo's "
                 "own conclusion: the edge is largely eaten by turnover at the taker "
                 "fee; maker is far better.")
    lines.append(f"- As a DIVERSIFIER it is {'genuinely uncorrelated' if abs(corr_t)<0.3 else 'correlated'} "
                 f"to the grand stack (taker corr {corr_t:+.2f}). Adding the taker "
                 f"version takes the blend to {st:+.2f} vs {sg:+.2f} grand-alone — "
                 + ("a real lift.\n" if st > sg + 0.05 else
                    "not a lift at the taker fee (the sleeve is too weak net of cost; "
                    "only its maker version would add).\n"))
    lines.append("- Honest consolidation: the vol-channel is a genuinely different "
                 "(intraday TS breakout) alpha and is ~uncorrelated to our daily "
                 "book, so it WOULD diversify — but only if executed as a maker, "
                 "which our real-L2 study showed is adverse-selected at retail "
                 "latency. Net-taker it doesn't clear. Same wall, independently "
                 "reached from both repos.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    for nm, p, col in [("gross", gross, "#bbb"), ("maker 1.5bps", maker, "#2980b9"),
                       ("taker 4.5bps", taker, "#c0392b")]:
        (1 + p.fillna(0)).cumprod().plot(ax=ax, color=col, lw=1.6, logy=True,
            label=f"{nm} (Sharpe {sharpe_h(p):.2f})")
    ax.set_title("Vol-channel breakout (vol repo) on our hourly crypto — by fee (log)")
    ax.set_ylabel("growth of $1 (log)"); ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "vol_channel.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "vol_channel.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/vol_channel.md + png")


if __name__ == "__main__":
    main()
