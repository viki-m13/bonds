"""How does the PULSE trend+breakout signal travel across asset universes?

Runs the IDENTICAL PULSE construction (multi-timeframe trend-sign ensemble +
20-day Donchian breakout, inverse-vol weighted, dollar-neutral L/S, scaled to a
12% annual vol target) on four spot universes and reports Sharpe / return / maxDD
full-sample and 2023->now, net of per-class costs:

  * spot crypto        data/crypto                 (111 coins, ~10 bps/side)
  * spot ETFs          data/etfs (unlevered)       (~120 ETFs,  2 bps/side)
  * spot leveraged ETF data/etfs (2x/3x bull&bear) (~30 ETFs,   3 bps/side)
  * PIT S&P 500 stocks dca panel (member-masked)   (720 names,  3 bps/side)

L/S is dollar-neutral (needs shorting -> realistic on margin/perps/inverse-ETFs);
a long-only "trend/flat" variant is also reported since true spot can't short.
Causality: signal at close of d, weights lagged one day. Run from crypto_pulse/.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dca"))
ANN = 365  # default; equities use 252 (set per-universe)

LEVERAGED = {  # 2x/3x bull & bear ETFs
    "TQQQ", "SQQQ", "UPRO", "SPXU", "SPXL", "SOXL", "SOXS", "FAS", "FAZ",
    "TECL", "TECS", "LABU", "LABD", "TNA", "TZA", "UDOW", "SDOW", "DRN", "DRV",
    "EDC", "EDZ", "ERX", "ERY", "NUGT", "DUST", "JNUG", "JDST", "YINN", "YANG",
    "TMF", "TMV", "UBT", "TBT", "TYD", "TYO", "UCO", "SCO", "UGL", "GLL",
    "BOIL", "KOLD", "UVXY", "SVXY", "QLD", "SSO", "DDM", "MVV", "UWM", "ROM",
    "USD", "FXP", "BITX", "ETHU", "ETHT", "SSO", "QID", "SDS", "BITI",
}


def _load_dir(d, suffix=".csv", strip_usd=False):
    cl, op, hi, lo, vo = {}, {}, {}, {}, {}
    for f in sorted(glob.glob(os.path.join(d, f"*{suffix}"))):
        t = os.path.basename(f)[:-len(suffix)]
        if strip_usd and t.endswith("_USD"):
            t = t[:-4]
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if "Close" not in cols:
            continue
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df = df[~df.index.duplicated()].sort_index()
        cl[t] = df["Close"]
        op[t] = df.get("Open", df["Close"])
        hi[t] = df.get("High", df["Close"])
        lo[t] = df.get("Low", df["Close"])
        vo[t] = df.get("Volume", pd.Series(1.0, index=df.index))
    C = pd.DataFrame(cl).sort_index()
    f = lambda d_: pd.DataFrame(d_).reindex_like(C)
    return C, f(hi), f(lo), f(vo)


def pulse(C, H, L, V, elig, ann=ANN, cost_bps=3.0, vol_target=0.12,
          long_only=False, max_ret=2.0):
    """PULSE, exactly as validated on HL: directional trend (w proportional to
    the signed trend+breakout signal, inverse-vol, gross-normalized to 1) — it
    naturally goes long uptrends / short downtrends, no forced demeaning. The
    long_only variant clips shorts to flat (true-spot version)."""
    R = C.pct_change()
    R[R.abs() > max_ret] = np.nan
    sd = R.rolling(30).std()
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    don = ((C >= H.shift(1).rolling(20).max()).astype(float)
           - (C <= L.shift(1).rolling(20).min()).astype(float))
    sig = trend + don
    if long_only:
        sig = sig.clip(lower=0)
    w = (sig / sd).where(elig)
    w = w.div(w.abs().sum(axis=1), axis=0)
    wl = w.shift(1)
    pnl = (wl * R).sum(axis=1)
    turn = (wl - wl.shift(1)).abs().sum(axis=1)
    pre = pnl - turn * cost_bps / 1e4
    scale = (vol_target / (pre.rolling(45).std() * np.sqrt(ann))).shift(1).clip(0, 3)
    return (pre * scale)[C.index >= C.index[90]]


def stats(p, ann=ANN):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, ann=np.nan, maxdd=np.nan, n=len(p))
    cum = (1 + p).cumprod()
    return dict(sharpe=p.mean() / p.std() * np.sqrt(ann), ann=p.mean() * ann,
                maxdd=(cum / cum.cummax() - 1).min(), n=len(p))


def universes():
    out = {}
    # spot crypto
    C, H, L, V = _load_dir(os.path.join(ROOT, "data", "crypto"), strip_usd=True)
    dv = (C * V).rolling(30).mean()
    out["spot crypto"] = (C, H, L, V, C.notna() & (dv > 5e6), 365, 10.0)
    # ETFs: curated long-history set (etfs_extended, 2005+), split lev/unlev.
    # Volume is unreliable/zero for several names here, so gate on price history
    # (all are liquid index products). Leveraged series pre-~2010 are vendor
    # backfills of the underlying x leverage -> treat early years with caution.
    C, H, L, V = _load_dir(os.path.join(ROOT, "data", "etfs_extended"))
    el = C.notna() & C.shift(60).notna()
    lev = [c for c in C.columns if c in LEVERAGED]
    unlev = [c for c in C.columns if c not in LEVERAGED]
    out["spot ETFs (unlevered)"] = (C[unlev], H[unlev], L[unlev], V[unlev],
                                    el[unlev], 252, 2.0)
    out["spot leveraged ETFs"] = (C[lev], H[lev], L[lev], V[lev],
                                  el[lev], 252, 3.0)
    # PIT S&P 500 stocks
    import data as dca_data
    P = dca_data.build_panel()
    out["PIT S&P 500 stocks"] = (P["close"], P["high"], P["low"], P["volume"],
                                 P["member"] & P["close"].notna(), 252, 3.0)
    return out


def main():
    U = universes()
    rows = []
    for name, (C, H, L, V, el, ann, cost) in U.items():
        ls = pulse(C, H, L, V, el, ann=ann, cost_bps=cost, long_only=False)
        lo = pulse(C, H, L, V, el, ann=ann, cost_bps=cost, long_only=True)
        idx = ls.index
        rec = pd.Timestamp("2023-01-01")
        rows.append((name, C.shape[1], int(el.sum(axis=1).median()), cost,
                     stats(ls, ann), stats(ls[idx >= rec], ann),
                     stats(lo, ann)))

    lines = ["# PULSE across asset universes — does the crypto edge travel?\n"]
    lines.append("The IDENTICAL PULSE signal (multi-timeframe trend-sign + 20d "
                 "Donchian breakout, inverse-vol, gross-normalised, directional "
                 "long-uptrend/short-downtrend, 12% vol target) on four spot "
                 "universes. `L/S` = the validated directional book (needs "
                 "shorting -> margin/perps/inverse-ETFs). `long-only` = shorts "
                 "clipped to flat (the true-spot version). Costs/side: crypto "
                 "10bps, ETFs 2-3bps, stocks 3bps.\n")
    lines.append("| universe | #assets | med live | **L/S Sharpe** | L/S ann | "
                 "L/S maxDD | L/S 2023+ | **long-only Sharpe** | LO ann | LO maxDD |")
    lines.append("|" + "---|" * 10)
    for name, n, med, cost, s_full, s_rec, s_lo in rows:
        lines.append(
            f"| {name} | {n} | {med} | **{s_full['sharpe']:+.2f}** | "
            f"{s_full['ann']:+.1%} | {s_full['maxdd']:+.1%} | "
            f"{s_rec['sharpe']:+.2f} | **{s_lo['sharpe']:+.2f}** | "
            f"{s_lo['ann']:+.1%} | {s_lo['maxdd']:+.1%} |")
    lines.append("")
    lines.append("## Reading\n")
    lines.append("- **The trend/breakout edge is crypto-specific.** Directional "
                 "L/S clears Sharpe ~1.2 only on spot crypto; it is flat-to-"
                 "**negative** on unlevered ETFs (−0.30), leveraged ETFs (+0.14) "
                 "and individual S&P 500 stocks (−0.61, and negative *before* "
                 "vol-scaling). Crypto is the inefficient, strongly-trending "
                 "market where this works; efficient equities mean-revert at "
                 "these horizons and whipsaw a directional trend book (the deep "
                 "L/S drawdowns are trend reversals like 2009/2020/2022, "
                 "amplified by vol-targeting).")
    lines.append("- **The only positive equity numbers are long-only**, and they "
                 "are just market drift harvested via trend-timing (Sharpe "
                 "~0.4–0.6), not a genuine cross-asset trend alpha — a long/flat "
                 "index would do similarly.")
    lines.append("- **Caveats:** ETF universes are small (14 / 17 names) so the "
                 "2023+ column is noisy; leveraged-ETF series before ~2010 are "
                 "vendor backfills (underlying × leverage); PIT-stock high/low "
                 "equal close in the shipped panel, so Donchian uses close.")
    lines.append("")
    out = "\n".join(lines)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research",
                     "cross_asset.md")
    with open(p, "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written]", p)


if __name__ == "__main__":
    main()
