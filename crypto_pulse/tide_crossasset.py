"""TIDE across asset classes, universes, timeframes, and leverage — the hardest overfit test.

The same FROZEN TIDE rule (x-sectional 20d breakout x trend-intensity regime, hold 3, vol-
targeted) applied unchanged to:
  - CRYPTO-57  : the HL-funded universe (baseline).
  - CRYPTO-112 : the FULL crypto daily universe (all coins we have).
  - STOCKS-96  : large-cap US equities (proxy for HL HIP-3 equity perps, which track the name).
  - STOCKS-430 : the extended equity universe.
  - ETFs       : liquid ETFs.
If a rule invented on 57 crypto coins also works on hundreds of stocks, it is a real cross-
sectional-breakout/trend effect, not a crypto fit. Plus: hourly & weekly timeframes, and the
leverage profile. Costs per class (crypto 4.5bps+funding; equities 2bps, no funding).

Run from crypto_pulse/:  python tide_crossasset.py  (-> research/tide_crossasset.md + png)
"""
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v

ANN = 365
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def sh(p, ann=ANN):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ann) if len(p) > 30 and p.std() > 0 else np.nan


def cagr(p, ann=ANN):
    p = p.dropna()
    return (1 + p).prod() ** (ann / len(p)) - 1 if len(p) > 30 else np.nan


def maxdd(p):
    cum = (1 + p.dropna()).cumprod()
    return (cum / cum.cummax() - 1).min()


def vtf(p, t=0.12, win=45, ann=ANN):
    return p * (t / (p.rolling(win).std() * np.sqrt(ann))).shift(1).clip(0, 3)


def load_dir(path, suffix=""):
    cl, vo, hi, lo = {}, {}, {}, {}
    for f in glob.glob(os.path.join(path, f"*{suffix}.csv")):
        name = os.path.basename(f).replace(suffix + ".csv", "").replace(".csv", "")
        try:
            d = pd.read_csv(f, parse_dates=["Date"]).set_index("Date")
        except Exception:
            continue
        if "Close" not in d.columns:
            continue
        d = d[~d.index.duplicated()].sort_index()
        cl[name], vo[name] = d["Close"], d.get("Volume", pd.Series(1.0, index=d.index))
        hi[name], lo[name] = d.get("High", d["Close"]), d.get("Low", d["Close"])
    C = pd.DataFrame(cl).sort_index()
    return C, pd.DataFrame(vo).reindex_like(C), pd.DataFrame(hi).reindex_like(C), pd.DataFrame(lo).reindex_like(C)


def tide(C, V, H, L, F=None, cost_bps=4.5, win=20, reg=50, hold=3, ann=ANN, vt_t=0.12):
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    has_vol = (V.fillna(0) > 1).any().any()
    dv = (C * V).rolling(30).mean()
    el = (C.notna() & (dv > 3e6)) if has_vol else C.notna()   # price-only elig if no volume data
    sd = R.rolling(30).std()
    nm = lambda x: x.div(x.abs().sum(axis=1), axis=0)
    dmf = lambda x: x.sub(x.mean(axis=1), axis=0)
    breakout = dmf(((C - C.rolling(win).mean()) / (C.rolling(win).std() + 1e-9)).where(el))
    ts = ((((C > C.rolling(reg).mean()).where(el)).mean(axis=1) - 0.5).abs() * 2).clip(0, 1)
    w = nm(breakout / sd).mul(ts.shift(1), axis=0)
    rebw = pd.Series(np.arange(len(C)) % hold == 0, index=C.index)
    w = w.where(rebw, axis=0).ffill(limit=hold); wl = w.shift(1)
    pnl = (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * cost_bps / 1e4
    if F is not None:
        pnl = pnl - (wl * F.reindex_like(wl).fillna(0.0)).sum(axis=1)
    return vtf(pnl, t=vt_t, ann=ann), wl


def main():
    L_ = ["# TIDE across asset classes, universes, timeframes, leverage (honest)\n",
          "Same FROZEN rule (20d breakout x trend-intensity, hold3, vol-targeted) everywhere. "
          "The cross-ASSET test is the real one: a crypto-invented rule working on stocks = real "
          "effect, not a fit.\n",
          "| universe | N names | period | Sharpe | CAGR | maxDD |", "|---|---|---|---|---|---|"]

    results = {}
    # crypto 57 (HL-funded)
    coins57 = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins57)
    F = v.load_daily_funding(coins57, C.index)
    p, _ = tide(C, V, H, L, F, cost_bps=4.5); results["CRYPTO-57 (HL-funded)"] = (p, C.shape[1])
    # crypto full (112)
    Cf, Vf, Hf, Lf = load_dir(v.CRYPTO, suffix="_USD")
    pf, wlf = tide(Cf, Vf, Hf, Lf, None, cost_bps=4.5); results["CRYPTO-112 (full daily)"] = (pf, Cf.shape[1])
    # stocks 96
    Cs, Vs, Hs, Ls = load_dir(os.path.join(DATA, "stocks"))
    ps, _ = tide(Cs, Vs, Hs, Ls, None, cost_bps=2.0, ann=252); results["STOCKS-96 (large-cap)"] = (ps, Cs.shape[1], 252)
    # stocks extended 430
    Cx, Vx, Hx, Lx = load_dir(os.path.join(DATA, "stocks_extended"))
    px, _ = tide(Cx, Vx, Hx, Lx, None, cost_bps=2.0, ann=252); results["STOCKS-430 (extended)"] = (px, Cx.shape[1], 252)
    # etfs
    Ce, Ve, He, Le = load_dir(os.path.join(DATA, "etfs_extended"))
    pe, _ = tide(Ce, Ve, He, Le, None, cost_bps=2.0, ann=252); results["ETFs"] = (pe, Ce.shape[1], 252)

    for k, tup in results.items():
        p = tup[0]; n = tup[1]; ann = tup[2] if len(tup) > 2 else ANN
        pp = p.dropna()
        per = f"{pp.index[0].date()}..{pp.index[-1].date()}" if len(pp) else "n/a"
        L_.append(f"| {k} | {n} | {per} | **{sh(p, ann):+.2f}** | {cagr(p, ann):+.0%} | {maxdd(p):+.0%} |")

    # ---- timeframes (crypto hourly + weekly) ----
    L_ += ["\n## Timeframes (crypto)\n", "| timeframe | Sharpe | CAGR |", "|---|---|---|"]
    # hourly
    Ch, Vh, Hh, Lh = {}, {}, {}, {}
    for f in glob.glob(os.path.join(DATA, "crypto_hourly", "*.csv")):
        nm0 = os.path.basename(f).replace("_USD", "").replace(".csv", "")
        d = pd.read_csv(f); d["ts"] = pd.to_datetime(d["ts"], unit="ms")
        d = d.set_index("ts")[~d.set_index("ts").index.duplicated()].sort_index()
        Ch[nm0], Vh[nm0], Hh[nm0], Lh[nm0] = d["close"], d["volume"], d["high"], d["low"]
    Ch = pd.DataFrame(Ch).sort_index()
    annh = 365 * 24
    ph, _ = tide(Ch, pd.DataFrame(Vh).reindex_like(Ch), pd.DataFrame(Hh).reindex_like(Ch),
                 pd.DataFrame(Lh).reindex_like(Ch), None, cost_bps=4.5,
                 win=24, reg=60, hold=6, ann=annh)
    L_.append(f"| hourly (20 coins, scaled windows) | {sh(ph, annh):+.2f} | {cagr(ph, annh):+.0%} |")
    # weekly (resample the 57 daily)
    Cw = C.resample("1W").last(); Vw = V.resample("1W").sum()
    Hw = H.resample("1W").max(); Lw = L.resample("1W").min()
    pw, _ = tide(Cw, Vw, Hw, Lw, None, cost_bps=4.5, win=4, reg=10, hold=1, ann=52)
    L_.append(f"| weekly (57 coins, scaled windows) | {sh(pw, 52):+.2f} | {cagr(pw, 52):+.0%} |")

    # ---- leverage profile (crypto-57 book) ----
    base = results["CRYPTO-57 (HL-funded)"][0]
    rawvol = base.rolling(45).std() * np.sqrt(ANN)
    lev = (0.12 / rawvol).shift(1).clip(0, 3)
    L_ += ["\n## HL leverage profile (CRYPTO-57 book, vol-targeted to 12%)\n",
           f"- Implied gross leverage: average **{lev.mean():.2f}x**, 95th pct {lev.quantile(0.95):.2f}x, "
           f"cap 3.0x. Well within HL limits (majors allow 20-50x).",
           "- Same Sharpe at any leverage; scaling the vol target trades return for drawdown:",
           "| vol target | CAGR | maxDD | ~gross lev |", "|---|---|---|---|"]
    for tt in [0.12, 0.20, 0.30, 0.50]:
        pL = base * (tt / 0.12)                               # leverage = linear scaling of the 12% book
        L_.append(f"| {tt:.0%} | {cagr(pL):+.0%} | {maxdd(pL):+.0%} | ~{lev.mean()*tt/0.12:.1f}x |")

    # ---- verdict ----
    cs = sh(results["CRYPTO-57 (HL-funded)"][0])
    ss = sh(results["STOCKS-96 (large-cap)"][0], 252)
    sx = sh(results["STOCKS-430 (extended)"][0], 252)
    L_ += ["\n## Verdict (honest — and it sharpens what TIDE is)\n",
           f"- **Cross-asset:** crypto-57 {cs:+.2f}, crypto-112 "
           f"{sh(results['CRYPTO-112 (full daily)'][0]):+.2f}; stocks-96 **{ss:+.2f}**, stocks-430 "
           f"**{sx:+.2f}**, ETFs {sh(results['ETFs'][0], 252):+.2f}.",
           "- **TIDE is CRYPTO-SPECIFIC — it does NOT generalize to equities; it INVERTS.** The "
           "same rule that earns ~2 in crypto LOSES (-0.8 to -1.3, -75% DD) on hundreds of stocks. "
           "That is economically coherent: short-horizon cross-sectional moves CONTINUE in crypto "
           "(momentum/breakout) but REVERSE in equities (the well-known equity short-term reversal). "
           "The sign of the edge flips with asset class.",
           "- **Implication for HL HIP-3 equity perps (TSLA, etc.): do NOT run TIDE on them** — it "
           "would lose. TIDE is a crypto-daily strategy, full stop.",
           f"- **Full crypto universe (112):** {sh(results['CRYPTO-112 (full daily)'][0]):+.2f} vs "
           f"{cs:+.2f} on the liquid-57 — adding smaller coins slightly DILUTES it; keep to the "
           "liquid subset.",
           f"- **Timeframes:** daily is the sweet spot; weekly {sh(pw, 52):+.2f} (weaker), hourly "
           f"{sh(ph, annh):+.2f} (fails — costs/noise dominate intraday).",
           f"- **Leverage:** avg gross {lev.mean():.1f}x, cap 3x — trivially within HL limits "
           "(20-50x on majors). Leverage scales CAGR and drawdown linearly, Sharpe unchanged; at "
           "30% vol target it's ~+65% CAGR / -37% DD at ~2.6x gross.",
           "- **Net:** TIDE is robust WITHIN crypto-daily (its design domain) but is NOT a "
           "universal anomaly. Honest scope: a crypto-daily ~2.0 Sharpe book, not a cross-asset one.\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for k, tup in results.items():
        p = tup[0]; (1 + p.fillna(0)).cumprod().plot(ax=ax, lw=1.6, label=f"{k} ({sh(p, tup[2] if len(tup)>2 else ANN):+.2f})")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("TIDE — same rule across asset classes (net, log)"); ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(os.path.join(HERE, "tide_crossasset.png"), dpi=110)
    with open(os.path.join(HERE, "tide_crossasset.md"), "w") as fh:
        fh.write("\n".join(L_))
    print("\n".join(L_)); print("\n[written] research/tide_crossasset.md + png")


if __name__ == "__main__":
    main()
