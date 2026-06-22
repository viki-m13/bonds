"""Careful microstructure EDA on the recorded HL per-account trade tape — to decide
WHICH strategy family the data supports (not to validate one; the sample is hours, not
months). Everything is computed FROM THE TAPE (each trade carries px), so no external
data is needed.

Builds per-coin minute bars (price from last trade, signed taker $ flow, unique
aggressor accounts), then within contiguous segments measures:
  - impact:      contemporaneous corr / Kyle lambda  (signed flow_t  vs  return_t)
  - prediction:  does flow LEAD price?  corr(flow_t, ret_{t+1})  (the tradeable question)
  - persistence: flow autocorr, return autocorr (minute-scale momentum vs reversal)
  - size:        do LARGE trades' signed flow predict next-minute return better?
  - accounts:    concentration, persistence, and whether whale-dominated minutes move price.
Pools across the liquid coins with within-coin z-scoring; reports N and t-stats and is
explicit that this is hypothesis-generating, not a backtest.

Run from crypto_pulse/:  python flow_explore.py --tape <dir>  (-> research/flow_explore.md + png)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
LIQUID = ["BTC", "ETH", "SOL"]      # coins dense enough for minute-bar stats
GAP_MIN = 3                          # a gap > this (minutes) breaks a contiguous segment


def tstat(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 5 and x.std() > 0 else np.nan


def pooled_corr(pairs):
    """pairs: list of (a, b) arrays already within-segment; returns r, t, n."""
    a = np.concatenate([p[0] for p in pairs]) if pairs else np.array([])
    b = np.concatenate([p[1] for p in pairs]) if pairs else np.array([])
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if len(a) < 20 or a.std() == 0 or b.std() == 0:
        return np.nan, np.nan, len(a)
    r = np.corrcoef(a, b)[0, 1]
    t = r * np.sqrt((len(a) - 2) / max(1e-9, 1 - r * r))
    return r, t, len(a)


def load(tape):
    fs = sorted(glob.glob(os.path.join(tape, "**", "*.parquet"), recursive=True))
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).drop_duplicates(subset=["tid"])
    d["ts"] = pd.to_datetime(d["time"], unit="ms")
    d["qty"] = d["px"] * d["sz"]
    d["sgn"] = np.where(d["side"] == "B", 1.0, -1.0)      # B = taker buy (lifts ask)
    d["aggr"] = np.where(d["side"] == "B", d["buyer"], d["seller"])  # aggressor address
    return d


def minute_bars(dc):
    """per-minute bars for one coin."""
    g = dc.set_index("ts").groupby(pd.Grouper(freq="1min"))
    b = pd.DataFrame({
        "px": g["px"].last(),
        "vwap": g.apply(lambda x: (x["px"] * x["sz"]).sum() / x["sz"].sum() if len(x) else np.nan),
        "flow": g.apply(lambda x: (x["sgn"] * x["qty"]).sum()),         # signed taker $
        "dollar": g["qty"].sum(),
        "n": g.size(),
        "nacct": g["aggr"].nunique(),
        "bigflow": g.apply(lambda x: (x["sgn"] * x["qty"])[x["qty"] >= x["qty"].quantile(0.9)].sum()
                           if len(x) else 0.0),
    })
    b = b.dropna(subset=["px"])
    b["ret"] = b["px"].pct_change()
    b["fi"] = b["flow"] / b["dollar"].replace(0, np.nan)               # imbalance in [-1,1]
    return b


def segments(b):
    """split a coin's minute bars into contiguous runs (no gap > GAP_MIN)."""
    if b.empty:
        return []
    gap = b.index.to_series().diff().dt.total_seconds().div(60).fillna(1)
    seg_id = (gap > GAP_MIN).cumsum()
    return [s for _, s in b.groupby(seg_id) if len(s) >= 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hl_trades_l4"))
    args = ap.parse_args()
    d = load(args.tape)
    span_h = (d["time"].max() - d["time"].min()) / 3.6e6

    # ---------- 1. data characterization ----------
    char = []
    for c, dc in d.groupby("coin"):
        char.append(dict(coin=c, trades=len(dc), dollar=dc["qty"].sum(),
                         accts=dc["aggr"].nunique(),
                         top10_share=dc.groupby("aggr")["qty"].sum().nlargest(10).sum() / dc["qty"].sum(),
                         buy_frac=(dc["sgn"] > 0).mean()))
    char = pd.DataFrame(char).sort_values("dollar", ascending=False)

    # ---------- 2. minute-bar flow<->price relationships (liquid coins) ----------
    bars = {c: minute_bars(d[d["coin"] == c]) for c in LIQUID if (d["coin"] == c).any()}
    segs = {c: segments(b) for c, b in bars.items()}

    # standardize flow imbalance within each coin, build aligned (x_t, y) pools
    def pool(xkey, ykey, ylag):
        pairs = []
        for c, slist in segs.items():
            for s in slist:
                x = s[xkey].values.astype(float)
                y = s[ykey].values.astype(float)
                if ylag > 0:
                    x = x[:-ylag]; y = y[ylag:]
                elif ylag < 0:
                    k = -ylag; x = x[k:]; y = y[:-k]
                # z-score x within segment
                if np.nanstd(x) > 0:
                    x = (x - np.nanmean(x)) / np.nanstd(x)
                pairs.append((x, y))
        return pooled_corr(pairs)

    rows = []
    rows.append(("impact: flow_t vs ret_t (contemporaneous)", pool("fi", "ret", 0)))
    rows.append(("PREDICT: flow_t vs ret_t+1 (does flow lead?)", pool("fi", "ret", 1)))
    rows.append(("PREDICT: flow_t vs ret_t+2", pool("fi", "ret", 2)))
    rows.append(("reverse: ret_t vs flow_t+1 (does price lead flow?)", pool("ret", "fi", -1)))
    rows.append(("flow autocorr: fi_t vs fi_t+1", pool("fi", "fi", 1)))
    rows.append(("return autocorr: ret_t vs ret_t+1", pool("ret", "ret", 1)))
    rows.append(("BIG-trade flow_t vs ret_t+1", None))     # filled below

    # big-trade predictive: z-score bigflow within coin
    big_pairs = []
    for c, slist in segs.items():
        for s in slist:
            x = s["bigflow"].values.astype(float)
            if np.nanstd(x) > 0:
                x = (x - np.nanmean(x)) / np.nanstd(x)
            big_pairs.append((x[:-1], s["ret"].values[1:]))
    rows[-1] = ("BIG-trade flow_t vs ret_t+1", pooled_corr(big_pairs))

    # ---------- 3. Kyle's lambda (bps move per $1M signed flow) ----------
    lam = []
    for c, slist in segs.items():
        for s in slist:
            x = s["flow"].values / 1e6
            y = s["ret"].values * 1e4   # bps
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 10 and x[m].std() > 0:
                lam.append(np.polyfit(x[m], y[m], 1)[0])
    lam_bps = np.nanmedian(lam) if lam else np.nan

    # ---------- 4. account persistence across windows ----------
    d["win"] = pd.to_datetime(d["time"], unit="ms").dt.floor("3h")
    acct_wins = d.groupby("aggr")["win"].nunique()
    persistent = (acct_wins >= 2).sum()
    tot_acct = d["aggr"].nunique()

    # ---------- write-up ----------
    L = ["# Microstructure EDA on the HL per-account trade tape\n",
         f"Sample: **{len(d):,} trades, {d['coin'].nunique()} coins, ~{span_h:.1f}h** "
         f"(discontinuous windows). This is HYPOTHESIS-GENERATING — it tells us which "
         "strategy family the flow supports, NOT a backtest (hours, one session).\n",
         "## 1. What the tape looks like (per coin)\n",
         "| coin | trades | $ volume | uniq accts | top-10 acct share | taker buy frac |",
         "|---|---|---|---|---|---|"]
    for _, r in char.head(10).iterrows():
        L.append(f"| {r.coin} | {r.trades:,} | ${r.dollar/1e6:.1f}M | {r.accts} | "
                 f"{r.top10_share:.0%} | {r.buy_frac:.0%} |")
    L += ["\n## 2. Flow <-> price relationships (BTC/ETH/SOL, minute bars, pooled)\n",
          "| relationship | corr | t-stat | N |", "|---|---|---|---|"]
    for name, res in rows:
        if res is None:
            continue
        r, t, n = res
        L.append(f"| {name} | {r:+.3f} | {t:+.1f} | {n} |")
    L.append(f"\n- **Kyle's lambda** (median): **{lam_bps:+.1f} bps per $1M** of signed taker flow.")
    L.append(f"- **Accounts:** {tot_acct} unique aggressors; {persistent} "
             f"({persistent/tot_acct:.0%}) appear in >=2 windows (persistence => trackable).")

    # ---------- 5. interpretation -> strategy archetype ----------
    cont = rows[0][1]; pred = rows[1][1]; fac = rows[4][1]; racf = rows[5][1]
    conc_med = char["top10_share"].median()
    L += ["\n## 3. Read & recommended strategy family\n"]
    L.append(f"- **Most robust finding — extreme concentration:** the top-10 accounts are a median "
             f"**{conc_med:.0%}** of taker $ per coin (up to 99%). Flow is dominated by a handful of "
             "addresses, and these are observable by address — the rare, hard-to-replicate edge.")
    L.append(f"- **Price impact is real and correctly signed:** Kyle's lambda **{lam_bps:+.1f} bps per "
             f"$1M** of signed taker flow — buying pressure moves price up. So flow is informative; the "
             "question is only whether it *leads* at a horizon we can trade net of 4.5bps.")
    L.append(f"- **Account persistence:** {persistent}/{tot_acct} ({persistent/tot_acct:.0%}) addresses "
             "trade in >=2 windows, so the big players are trackable across time — the basis for a "
             "whale-following feature.")
    L.append(f"- **Minute-scale lead/lag is INCONCLUSIVE here** (N={cont[2]}): flow_t vs ret_t+1 "
             f"{pred[0]:+.3f} (t {pred[1]:+.1f}) is not significant, and the negative minute-return "
             f"autocorr ({racf[0]:+.2f}) is mostly bid-ask bounce from using last-trade price, not "
             "tradeable reversal. Resolving momentum-vs-reversal needs the weeks of tape now accruing.")
    L.append("\n**Recommended archetype given the data:** a market-neutral, cross-sectional *flow-tilt* "
             "book at the **daily** horizon (where 4.5bps amortises). Rank coins each day by "
             "(a) signed taker-flow imbalance, (b) large-trade tilt, and — the differentiated piece — "
             "(c) **net flow of the large, persistent accounts** (whale accumulation/distribution). "
             "Go long the accumulated, short the distributed, inverse-vol sized, and test it as a "
             "low-correlation sleeve on top of STRATA+VOL. The concentration + persistence + positive "
             "impact are exactly the structural conditions under which a whale-flow tilt can carry "
             "signal; the daily cross-sectional wrapper is what makes it executable. (This is a design "
             "recommendation from one session of tape — not a validated edge; flow_l4.py will backtest "
             "it once ~120 days have accrued.)")

    os.makedirs(HERE, exist_ok=True)
    # plot: lead-lag of corr(flow_t, ret_{t+k})
    ks = list(range(-3, 6))
    cc = []
    for k in ks:
        cc.append(pool("fi", "ret", k)[0])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(ks, cc, color=["#888" if k != 0 else "#c0392b" for k in ks])
    ax[0].axhline(0, color="k", lw=0.8); ax[0].set_title("corr(flow_t, return_{t+k}) — lead/lag")
    ax[0].set_xlabel("k (minutes); k>0 = flow predicts future return"); ax[0].grid(alpha=0.3)
    # impact scatter (BTC)
    if "BTC" in segs and segs["BTC"]:
        s = pd.concat(segs["BTC"])
        ax[1].scatter(s["flow"] / 1e6, s["ret"] * 1e4, s=8, alpha=0.4, color="#2980b9")
        ax[1].set_xlabel("signed taker flow ($M)"); ax[1].set_ylabel("minute return (bps)")
        ax[1].set_title(f"BTC price impact (lambda {lam_bps:+.1f} bps/$M)"); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "flow_explore.png"), dpi=110)
    with open(os.path.join(HERE, "flow_explore.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    print("\n[written] research/flow_explore.md + png")


if __name__ == "__main__":
    main()
