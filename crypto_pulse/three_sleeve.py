"""Three-sleeve crypto book: TREND + CARRY + ORDER-FLOW — does OF add value?

Adds an order-flow sleeve to the validated trend+carry book and tests honestly
whether it improves the blend (it must be both positive AND uncorrelated to help).
Order flow here is the OHLC PROXY (close-location-value x volume = net taker
buying pressure), computed on the real-OHLC daily crypto data for the 57 HL coins;
it predicts CONTINUATION (the peer-reviewed direction). A cleaner version uses
real signed taker volume recorded forward via record_orderflow.py.

HL era, real funding + 4.5 bps taker, IS=first60/OOS=last40, vol-targeted. Each
signal causal (info through close of d, traded next day). Run from crypto_pulse/:
    python three_sleeve.py   (-> research/three_sleeve.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
TAKER = 4.5
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, ann=np.nan, maxdd=np.nan, calmar=np.nan)
    cum = (1 + p).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    sh = p.mean() / p.std() * np.sqrt(ANN)
    return dict(sharpe=sh, ann=p.mean() * ANN, maxdd=dd,
                calmar=(p.mean() * ANN) / abs(dd) if dd < 0 else np.nan)


def sleeve_weights(C, V, H, L, F):
    """Per-coin gross-1 daily weights for each sleeve (causal: info through close
    of day d, traded next day after the .shift(1) in pnl). Shared by the backtest
    and the live signal so deployment is backtest-identical."""
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    trend = sum(np.sign(C / C.shift(k) - 1) for k in (10, 20, 40, 80)) / 4.0
    don = ((C >= H.shift(1).rolling(20).max()).astype(float)
           - (C <= L.shift(1).rolling(20).min()).astype(float))
    wt = ((trend + don) / sd).where(elig); wt = wt.div(wt.abs().sum(axis=1), axis=0)

    fsm = F.rolling(3).mean(); craw = (-fsm).sub((-fsm).mean(axis=1), axis=0)
    keep = ((craw > 0) & (np.sign(trend) >= 0)) | ((craw < 0) & (np.sign(trend) <= 0))
    wc = craw.where(keep & elig); wc = wc.sub(wc.mean(axis=1), axis=0)
    wc = wc.div(wc.abs().sum(axis=1), axis=0)

    # order-flow CONTINUATION (CLV x volume = net taker buying, 5d sum), demeaned
    clv = ((C - L) - (H - C)) / (H - L).replace(0, np.nan)
    of = (clv * V).rolling(5).sum()
    oz = of.where(elig); oz = oz.sub(oz.mean(axis=1), axis=0)
    wo = (oz / sd); wo = wo.div(wo.abs().sum(axis=1), axis=0)
    # hold 5d to keep turnover (and cost) down
    reb = pd.Series(np.arange(len(wo)) % 5 == 0, index=wo.index)
    wo = wo.where(reb, axis=0).ffill(limit=4)
    return {"TREND": wt, "CARRY": wc, "ORDERFLOW": wo}, R


def sleeves(C, V, H, L, F):
    W, R = sleeve_weights(C, V, H, L, F)

    def pnl(w):
        wl = w.shift(1)
        return ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * TAKER / 1e4
                - (wl * F).sum(axis=1))
    return {nm: pnl(w) for nm, w in W.items()}


def book_weights(C, V, H, L, F, vol_target=0.12, lookback_cut=None):
    """Combined risk-weighted, vol-targeted per-coin TARGET WEIGHTS for the
    deployable 3-sleeve book. Returns (W_target, scale, risk_weights):
      * inverse-vol risk weights across sleeves (estimated on history < lookback_cut,
        or full history if None — for live use the last available estimate);
      * combined gross weight per day = sum_s rw_s * w_sleeve_s;
      * vol-target scale on the combined book's realized PnL (causal, shifted).
    Target position weight on day d = scale[d] * W_target.loc[d]; trade next day."""
    W, R = sleeve_weights(C, V, H, L, F)
    pnls = sleeves(C, V, H, L, F)
    P = pd.DataFrame(pnls).dropna()
    base = P if lookback_cut is None else P[P.index < lookback_cut]
    isv = base.std()
    rw = (1 / isv) / (1 / isv).sum()
    W_target = sum(rw[nm] * W[nm].fillna(0.0) for nm in W)        # gross weight/day
    combined_pnl = (P * rw).sum(axis=1).reindex(C.index)
    scale = (vol_target / (combined_pnl.rolling(45).std() * np.sqrt(ANN))
             ).shift(1).clip(0, 3)
    return W_target, scale, rw


def vt(p, target=0.12):
    return p * (target / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def main():
    coins = [c for c in v.OVERLAP if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    sl = sleeves(C, V, H, L, F)
    hl = C.index >= HL_START
    idxhl = C.index[hl]
    cut = idxhl[int(len(idxhl) * 0.6)]

    def rep(p):
        s = stats(p[hl])
        return s, stats(p[(p.index < cut) & hl])["sharpe"], stats(p[(p.index >= cut) & hl])["sharpe"]

    lines = ["# Three-sleeve book: TREND + CARRY + ORDER-FLOW (HL, net)\n"]
    lines.append(f"Real funding + {TAKER}bps taker, IS=first60/OOS=last40. Order "
                 "flow = OHLC proxy (close-location-value x volume), continuation, "
                 "5d hold.\n")
    lines.append("## Standalone sleeves\n")
    lines.append("| sleeve | Sharpe | IS | OOS |")
    lines.append("|---|---|---|---|")
    for nm, p in sl.items():
        s, i, o = rep(p)
        lines.append(f"| {nm} | {s['sharpe']:+.2f} | {i:+.2f} | {o:+.2f} |")
    P = pd.DataFrame({k: p[hl] for k, p in sl.items()}).dropna()
    corr = P.corr()
    lines.append("\nSleeve correlations: " + ", ".join(
        f"{a}-{b}={corr.loc[a, b]:+.2f}" for i, a in enumerate(corr.columns)
        for b in corr.columns[i + 1:]) + "\n")

    base = vt(0.5 * sl["TREND"] + 0.5 * sl["CARRY"])
    three_eq = vt((sl["TREND"] + sl["CARRY"] + sl["ORDERFLOW"]) / 3)
    # inverse-vol risk weights (IS only)
    isv = P[P.index < cut].std(); rw = (1 / isv) / (1 / isv).sum()
    three_rw = vt((P * rw).sum(axis=1).reindex(C.index))
    lines.append("## Blends — does order flow add value?\n")
    lines.append("| book | Sharpe | IS | OOS | ann | maxDD | Calmar |")
    lines.append("|---|---|---|---|---|---|---|")
    for nm, p in [("trend+carry (2-sleeve base)", base),
                  ("trend+carry+OF equal (3-sleeve)", three_eq),
                  ("trend+carry+OF risk-weighted", three_rw)]:
        s, i, o = rep(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | "
                     f"{s['ann']:+.1%} | {s['maxdd']:+.1%} | {s['calmar']:.2f} |")
    sb, st = stats(base[hl]), stats(three_rw[hl])
    lines.append("")
    lines.append("## Verdict\n")
    verdict = ("ADDS value" if st["sharpe"] > sb["sharpe"] + 0.03 else
               "does NOT meaningfully add")
    lines.append(f"- The order-flow sleeve **{verdict}**: 3-sleeve Sharpe "
                 f"{st['sharpe']:+.2f} vs 2-sleeve {sb['sharpe']:+.2f} (maxDD "
                 f"{st['maxdd']:+.1%} vs {sb['maxdd']:+.1%}). OF correlation to "
                 f"trend = {corr.loc['TREND','ORDERFLOW']:+.2f}. This uses the OHLC "
                 "PROXY; real signed taker volume (record_orderflow.py, forward) "
                 "should be cleaner. Honest expectation: a modest lift, not a "
                 "Sharpe jump — the deployable book is ~1.1-1.4.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + base[hl].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.5,
        label=f"trend+carry (Sharpe {sb['sharpe']:.2f})")
    (1 + three_rw[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.2,
        label=f"+ order-flow 3-sleeve (Sharpe {st['sharpe']:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Three-sleeve crypto book on HL (real funding+fees)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "three_sleeve.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "three_sleeve.md"), "w") as fh:
        fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
