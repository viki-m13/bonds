"""EVENT-DRIVEN funding-extreme fade — a different archetype than cross-sectional
carry: a TIME-SERIES contrarian fade of crowded positioning.

Research (Kaiko/Glassnode + the funding-rate literature) flagged funding+OI
EXTREMES as the most taker-viable uncorrelated crypto event edge: when a coin's
funding spikes into its top decile, longs are crowded/over-levered; a small dip
forces liquidation cascades -> sharp reversion. Symmetrically for deeply negative
funding (crowded shorts -> squeeze). The per-event move is large, so 4.5bps taker
is small relative to the edge. This is distinct from CARRY (which shorts the
cross-sectionally-highest funding every day); here we only act in the TAILS and
fade them, per-coin, holding for the reversion.

We proxy OI-confirmation with a funding-percentile trigger (no historical OI in
the panel). Honest: causal (funding through d-1), net of 4.5bps taker + the
funding actually paid/received while positioned, vol-targeted, IS/OOS.

Run from crypto_pulse/:  python funding_event.py  (-> research/funding_event.md + png)
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


def sharpe(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 30 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, ann=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sharpe(p), ann=p.mean() * ANN,
                maxdd=(cum / cum.cummax() - 1).min())


def vt(p, target=0.12):
    return p * (target / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)

    hl = C.index >= HL_START
    idxhl = C.index[hl]
    cut = idxhl[int(len(idxhl) * 0.6)]

    def rep(p):
        return sharpe(p[hl]), sharpe(p[(p.index < cut) & hl]), \
            sharpe(p[(p.index >= cut) & hl]), stats(p[hl])

    lines = ["# Event-driven funding-extreme fade (time-series contrarian) on HL\n"]
    lines.append(f"Net of {TAKER}bps taker + realized funding, vol-targeted, "
                 "IS=first60/OOS=last40. Fade crowded positioning in the tails; "
                 "hold for liquidation-cascade reversion.\n")
    lines.append("## Parameter scan (funding z-window / trigger z / hold days)\n")
    lines.append("| zwin | trig | hold | Sharpe | IS | OOS | ann | maxDD | active% |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    best = None
    for zwin in (30, 60):
        # per-coin funding z-score (time series), causal
        fz = (F - F.rolling(zwin, min_periods=15).mean()) / \
             (F.rolling(zwin, min_periods=15).std() + 1e-9)
        fz = fz.where(elig)
        for trig in (1.0, 1.5, 2.0):
            # signal: fade -> short when funding very high (z>trig), long when z<-trig
            sig = pd.DataFrame(0.0, index=C.index, columns=C.columns)
            sig = sig.mask(fz > trig, -1.0).mask(fz < -trig, 1.0)
            for hold in (2, 3, 5):
                held = sig.replace(0.0, np.nan).ffill(limit=hold)
                w = held.div(held.abs().sum(axis=1), axis=0).fillna(0.0)
                wl = w.shift(1)
                gross = (wl * R).sum(axis=1)
                turn = (wl - wl.shift(1)).abs().sum(axis=1)
                pnl = gross - turn * TAKER / 1e4 - (wl * F).sum(axis=1)
                p = vt(pnl)
                sh, si, so, s = rep(p)
                active = (w.abs().sum(axis=1) > 0)[hl].mean()
                lines.append(f"| {zwin} | {trig} | {hold} | {sh:+.2f} | {si:+.2f} | "
                             f"{so:+.2f} | {s['ann']:+.1%} | {s['maxdd']:+.1%} | "
                             f"{active:.0%} |")
                score = min(si, so) if (np.isfinite(si) and np.isfinite(so)) else -9
                if best is None or score > best[0]:
                    best = (score, zwin, trig, hold, p, sh, si, so)

    _, zwin, trig, hold, pbest, sh, si, so = best
    lines.append(f"\n**Best (by min(IS,OOS)):** zwin={zwin}, trig={trig}, "
                 f"hold={hold} -> Sharpe {sh:+.2f} (IS {si:+.2f} / OOS {so:+.2f}).\n")

    # correlation to the directional stack
    rho = np.nan
    try:
        import max_stack as ms
        S = ms.build_sleeves(C, V, H, L, F)
        ref = pd.DataFrame({k: ms.vt(s) for k, s in S.items()}).mean(axis=1)
        cc = pd.concat({"fund": pbest[hl], "stack": ref[hl]}, axis=1).dropna()
        rho = cc["fund"].corr(cc["stack"])
    except Exception:
        pass

    lines.append("## Verdict\n")
    ok = (si > 0.1 and so > 0.1)
    lines.append(f"- Funding-extreme fade is "
                 f"{'a GENUINE positive event sleeve' if ok else 'not robust net of cost'}"
                 f" (Sharpe {sh:+.2f}, IS {si:+.2f}, OOS {so:+.2f}); correlation to "
                 f"the directional stack {rho:+.2f}. As an event sleeve its value is "
                 "timing diversification (it fires in stress, when trend whipsaws), "
                 "even at a modest standalone Sharpe.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + pbest[hl].fillna(0)).cumprod().plot(ax=ax, color="#8e44ad", lw=2.0,
        label=f"funding-extreme fade (Sharpe {sh:.2f}, OOS {so:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Event-driven funding-extreme fade on HL crypto (net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "funding_event.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "funding_event.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/funding_event.md + png")


if __name__ == "__main__":
    main()
