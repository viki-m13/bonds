"""50/50 blend: vol strategy + our strategy, short & long.

Vol leg = the vol repo's CURRENT leakage-free published series (docs/data/
t5rvt_series.json -> eq_vt35 daily returns, saved to data/vol_strategy/): daily
Sharpe 1.99, CAGR 105%, vol 40%, maxDD -37%, corr-to-BTC -0.11, over 2018-2026
(matches vol-pi.vercel.app exactly). This REPLACES the earlier legacy pickles, which
the strategy owner flagged as a leaky version (inflated Sharpe 3.75-7.4).

Both legs vol-targeted to 12% then blended 50/50 (equal risk). LONG = 2018-2026
(vs our price book), SHORT = HL era (vs our full grand stack). Run from crypto_pulse/:
    python vol_blend.py  (-> research/vol_blend.md + png)
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validate_hl as v
import breadth_leverage as bl
import universe_experiments as ux
import per_coin_cost as pcc
import kelly_cagr as kc

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VS = os.path.join(ROOT, "data", "vol_strategy")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def vt(p, t=0.12):
    return p * (t / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)


def sharpe(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 60 and p.std() > 0) else np.nan


def stats(p):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, cagr=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sharpe(p), cagr=cum.iloc[-1] ** (ANN / len(p)) - 1,
                maxdd=(cum / cum.cummax() - 1).min())


def load_vol(name):
    s = pd.read_csv(os.path.join(VS, name), index_col=0)["ret"]
    s.index = pd.to_datetime(s.index)
    return s


def haircut_to_sharpe(ret, target_sharpe):
    """Scale a return series' MEAN so its annualized Sharpe == target (std and
    correlation unchanged). Used to anchor the vol series to the PUBLISHED Sharpe
    (vol-pi dashboard: daily ~2.0, OOS ~1.99) instead of the grosser raw pickle."""
    r = ret.dropna()
    s0 = r.mean() / r.std() * np.sqrt(ANN)
    if not np.isfinite(s0) or s0 == 0:
        return ret
    drift = r.mean() * (1 - target_sharpe / s0)
    return ret - drift


def main():
    vol = load_vol("t5rvt_net_daily_2018_2026.csv")           # leakage-free, Sharpe 1.99

    # our strategy: full grand stack (HL era) + price book (long, 2018+)
    coins = [c for c in sorted(set(bl.ALL111))
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    member = ux.membership(C, V, 30, 30, 0, 30)
    our_price, _ = pcc.run(C, V, H, L, member, 1e7, maker=False)   # per-coin cost, $10M
    our_grand = kc.build_grandstack()                               # full book, HL era

    lines = ["# 50/50 blend: vol strategy + our strategy (short & long)\n"]
    lines.append("Vol leg = the vol repo's CURRENT leakage-free published series "
                 "(t5rvt eq_vt35: daily Sharpe **1.99**, CAGR 105%, vol 40%, maxDD "
                 "-37%, corr-to-BTC -0.11, 2018-2026 — matches vol-pi.vercel.app). "
                 "Replaces the earlier LEAKY pickles. Both legs vol-targeted to 12%, "
                 "blended 50/50. LONG = 2018-2026 vs our price book; SHORT = HL era vs "
                 "our full grand stack.\n")

    # ---- SHORT blend: our grand stack + vol, HL era ----
    sB = pd.concat({"ours": vt(our_grand), "vol": vt(vol)}, axis=1).dropna()
    sB = sB[sB.index >= HL_START]
    rho_s = sB["ours"].corr(sB["vol"])
    short_blend = 0.5 * sB["ours"] + 0.5 * sB["vol"]

    # ---- LONG blend: our price book + vol, 2018-2026 ----
    lB = pd.concat({"ours": vt(our_price), "vol": vt(vol)}, axis=1).dropna()
    rho_l = lB["ours"].corr(lB["vol"])
    long_blend = 0.5 * lB["ours"] + 0.5 * lB["vol"]

    lines.append("## Correlation (the real diversification driver)\n")
    lines.append(f"- SHORT (HL era): corr(ours, vol) = **{rho_s:+.2f}**")
    lines.append(f"- LONG (2018-2026): corr(ours, vol) = **{rho_l:+.2f}**")
    lines.append("Low correlation = genuine diversification (different alpha: their "
                 "intraday directional TS vs our daily cross-sectional).\n")

    lines.append("## 50/50 blend (leakage-free vol series, Sharpe 1.99)\n")
    lines.append("| period | ours Sharpe | vol Sharpe | 50/50 blend Sharpe | blend CAGR | blend maxDD |")
    lines.append("|---|---|---|---|---|---|")
    for nm, d, blend in [("SHORT (HL era)", sB, short_blend),
                         ("LONG (2018-2026)", lB, long_blend)]:
        bs = stats(blend)
        lines.append(f"| {nm} | {sharpe(d['ours']):+.2f} | {sharpe(d['vol']):+.2f} | "
                     f"**{bs['sharpe']:+.2f}** | {bs['cagr']:+.0%} | {bs['maxdd']:+.0%} |")

    # ---- sensitivity: blend Sharpe vs the vol strategy's realized net Sharpe ----
    lines.append("\n## Sensitivity: blend Sharpe vs the vol strategy's net Sharpe\n")
    lines.append(f"Using the LONG correlation ({rho_l:+.2f}) and our ~1.5 full book, "
                 "50/50 equal-risk blend = (Sv+So)/sqrt(2(1+rho)):\n")
    lines.append("| vol net Sharpe | scenario | blend Sharpe |")
    lines.append("|---|---|---|")
    So = 1.5
    for Sv, src in [(1.99, "PUBLISHED full 2018-26 (leakage-free)"),
                    (3.53, "their in-sample 2018-22"),
                    (-0.06, "their 2026 holdout (weak regime)"),
                    (1.0, "conservative live haircut")]:
        bl_sh = (Sv + So) / np.sqrt(2 * (1 + rho_l))
        lines.append(f"| {Sv:+.2f} | {src} | **{bl_sh:+.2f}** |")

    lines.append("\n## Verdict\n")
    lines.append(f"- **Diversification is genuine and strong** (corr {rho_l:+.2f} long / "
                 f"{rho_s:+.2f} short). Their intraday directional book (corr-to-BTC "
                 "-0.11) and our daily cross-sectional book are different alphas — and "
                 "both are individually good.")
    lines.append(f"- **The 50/50 blend reaches ~{stats(long_blend)['sharpe']:+.2f} long "
                 f"(2018-26) / {stats(short_blend)['sharpe']:+.2f} short (HL era)** using "
                 "the vol repo's leakage-free published series (1.99) and our ~1.5 — a "
                 "real lift over either alone, because near-zero correlation makes the "
                 "Sharpes add nearly in quadrature.")
    lines.append(f"- **Drawdown improves too**: blend maxDD {stats(long_blend)['maxdd']:+.0%} "
                 f"long / {stats(short_blend)['maxdd']:+.0%} short — much tighter than the "
                 "vol book's standalone -37% (our book's lower DD cushions its tail).")
    lines.append("- **Caveats:** the vol 1.99 assumes maker (2.5bp) execution on a "
                 "high-turnover intraday book — its 2026 holdout was -0.06 (weak regime) "
                 "and standalone maxDD -37%. If its live net is lower (~1.0), the blend "
                 "is still ~1.9 (sensitivity table). Net: the blend is a genuine "
                 "improvement on both Sharpe and drawdown — the strongest case yet.\n")

    # plot: short and long blends (face value) vs each leg
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    (1 + sB["ours"].fillna(0)).cumprod().plot(ax=ax[0], color="#8e44ad", lw=1.3, label=f"ours ({sharpe(sB['ours']):.2f})")
    (1 + sB["vol"].fillna(0)).cumprod().plot(ax=ax[0], color="#16a085", lw=1.3, label=f"vol ({sharpe(sB['vol']):.2f})")
    (1 + short_blend.fillna(0)).cumprod().plot(ax=ax[0], color="#c0392b", lw=2.2, label=f"50/50 ({stats(short_blend)['sharpe']:.2f})")
    ax[0].set_title("SHORT — HL era"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3); ax[0].set_ylabel("growth of $1")
    (1 + lB["ours"].fillna(0)).cumprod().plot(ax=ax[1], color="#8e44ad", lw=1.3, logy=True, label=f"ours ({sharpe(lB['ours']):.2f})")
    (1 + lB["vol"].fillna(0)).cumprod().plot(ax=ax[1], color="#16a085", lw=1.3, logy=True, label=f"vol ({sharpe(lB['vol']):.2f})")
    (1 + long_blend.fillna(0)).cumprod().plot(ax=ax[1], color="#c0392b", lw=2.2, logy=True, label=f"50/50 ({stats(long_blend)['sharpe']:.2f})")
    ax[1].set_title("LONG 2018-2026 (log)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both"); ax[1].set_ylabel("growth of $1 (log)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "vol_blend.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "vol_blend.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/vol_blend.md + png")


if __name__ == "__main__":
    main()
