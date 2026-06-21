"""STRATA through PIT stock data — survivorship-free S&P panel (720 names, 2004-2026,
real volume + PIT membership). Applies STRATA's cross-sectional sleeves to the equity
panel (point-in-time members, liquidity-filtered), then combines with crypto STRATA +
crypto VOL toward Sharpe 3 OOS. Honest: survivorship-free, net of 2bps.

Sleeves (cross-sectional, market-neutral, PIT-member-eligible):
  MOM 12-1m | STREV 1w reversal | LOWVOL | BAB (low beta to equal-wt mkt) |
  ACCEL mom-of-mom | VOLSHOCK volume expansion x trend | TREND multi-TF sign
shrunk-MV combine. Run from crypto_pulse/:  python pit_equity_strata.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "dca"))
import data as dca  # noqa: E402
import vol_blend as vb  # noqa: E402
import dynamic_blend as db  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
ANN = 252


def sh(p, ann=ANN):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ann) if (len(p) > 60 and p.std() > 0) else np.nan


def stats(p, ann=ANN):
    p = p.dropna()
    if len(p) < 60:
        return dict(sharpe=np.nan, maxdd=np.nan)
    cum = (1 + p).cumprod()
    return dict(sharpe=sh(p, ann), maxdd=(cum / cum.cummax() - 1).min())


def vt(p, t=0.12, ann=ANN):
    return p * (t / (p.rolling(45).std() * np.sqrt(ann))).shift(1).clip(0, 3)


def pit_equity_strata():
    P = dca.build_panel()
    C, Vol, member = P["close"], P["volume"], P["member"]
    R = C.pct_change(); R[R.abs() > 0.5] = np.nan
    dv = (C * Vol).rolling(30).mean()
    # PIT members, liquidity-filtered: top-150 most liquid members each month
    elig = member.astype(bool) & C.notna() & (dv > dv.quantile(0.5, axis=1).values[:, None] * 0 + 1)
    # simpler robust filter: member AND dollar-vol in top half of members that day
    rank = dv.where(member.astype(bool)).rank(axis=1, ascending=False)
    elig = member.astype(bool) & C.notna() & (rank <= 150)
    sd = R.rolling(60).std(); n = len(C)
    mkt = R.where(elig).mean(axis=1)
    rebw = pd.Series(np.arange(n) % 5 == 0, index=C.index)

    def norm(w): return w.div(w.abs().sum(axis=1), axis=0)
    def dm(x): return x.sub(x.mean(axis=1), axis=0)
    def pnl(w, hold=None, cost=2.0):
        if hold is not None:
            w = w.where(hold, axis=0).ffill(limit=4)
        wl = w.shift(1)
        return (wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * cost / 1e4

    trend = sum(np.sign(C / C.shift(k) - 1) for k in (20, 60, 120, 250)) / 4.0
    mom = (C.shift(21) / C.shift(252) - 1)
    strev = -(C / C.shift(5) - 1)
    lowvol = -R.rolling(60).std()
    beta = R.rolling(120).cov(mkt).div(mkt.rolling(120).var(), axis=0)
    accel = (C / C.shift(20) - 1) - (C.shift(20) / C.shift(40) - 1)
    vsh = (Vol.rolling(5).mean() / Vol.rolling(60).mean())
    sl = {}
    sl["TREND"] = vt(pnl(norm((dm(trend.where(elig))) / sd), hold=rebw))
    sl["MOM"] = vt(pnl(norm(dm(mom.where(elig)) / sd), hold=rebw))
    sl["STREV"] = vt(pnl(norm(dm(strev.where(elig)) / sd), hold=rebw))
    sl["LOWVOL"] = vt(pnl(norm(dm(lowvol.where(elig)) / sd), hold=rebw))
    sl["BAB"] = vt(pnl(norm(dm((-beta).where(elig))), hold=rebw))
    sl["ACCEL"] = vt(pnl(norm(dm(accel.where(elig)) / sd), hold=rebw))
    sl["VOLSHOCK"] = vt(pnl(norm((dm(vsh.where(elig)) * np.sign(trend)) / sd), hold=rebw))
    Pf = pd.DataFrame(sl).dropna()
    mu = Pf.mean().values * ANN; S = Pf.cov().values * ANN
    Ss = 0.6 * np.diag(np.diag(S)) + 0.4 * S
    w = np.clip(np.linalg.solve(Ss + 1e-6 * np.eye(len(mu)), mu), 0, None)
    w = pd.Series(w / w.sum(), index=Pf.columns)
    return Pf, w, vt((Pf * w).sum(axis=1).reindex(C.index))


def wk(p):
    return p.fillna(0.0).resample("W-FRI").sum()


def main():
    Pf, w, eqs = pit_equity_strata()
    eqs = eqs[eqs.notna()]
    warm = eqs.index[60]; eqs = eqs[eqs.index >= warm]
    cutE = eqs.index[int(len(eqs) * 0.6)]

    lines = ["# STRATA through PIT stock data (survivorship-free S&P)\n"]
    lines.append("Cross-sectional STRATA sleeves on the PIT S&P panel (720 names, real "
                 "volume + PIT membership, top-150 liquid members), net 2bps, shrunk-MV.\n")
    lines.append("## Equity-STRATA sleeves (full history)\n")
    lines.append("| sleeve | Sharpe | weight |")
    lines.append("|---|---|---|")
    for c in Pf.columns:
        lines.append(f"| {c} | {sh(Pf[c]):+.2f} | {w[c]:.0%} |")
    se = stats(eqs)
    lines.append(f"\n**PIT equity-STRATA: Sharpe {se['sharpe']:+.2f} "
                 f"(IS {sh(eqs[eqs.index<cutE]):+.2f}/OOS {sh(eqs[eqs.index>=cutE]):+.2f}), "
                 f"maxDD {se['maxdd']:+.0%}** ({eqs.index.min().date()}->{eqs.index.max().date()})\n")

    # combine with crypto books (weekly)
    vol = vb.load_vol("t5rvt_net_daily_2018_2026.csv")
    strata_c = db.strata_v2()
    series = {"EQ-STRATA": wk(eqs), "VOL-CRYPTO": wk(vol), "STRATA-CRYPTO": wk(strata_c)}
    W = pd.DataFrame({k: s * (0.12 / np.sqrt(52)) / s.std() for k, s in series.items()}).dropna()
    corr = W.corr()
    cut = W.index[int(len(W) * 0.6)]
    def io(p): return sh(p[p.index < cut], 52), sh(p[p.index >= cut], 52)
    Wis = W[W.index < cut]; isr = {c: sh(Wis[c], 52) for c in W.columns}
    keep = [c for c in W.columns if isr[c] > 0.1]; rk = {c: max(isr[c], 0) for c in keep}
    tot = sum(rk.values()) or 1; opt = sum((rk[c] / tot) * W[c] for c in keep)
    eq = W.mean(axis=1)

    lines.append("## Multi-asset combine (weekly)\n")
    lines.append("Per-book Sharpe: " + ", ".join(f"{c} {sh(W[c],52):+.2f}" for c in W.columns) + "\n")
    lines.append(f"Mean pairwise correlation: **{corr.values[np.triu_indices(len(corr),1)].mean():+.2f}**\n")
    lines.append("| combiner | Sharpe | IS | OOS | maxDD |")
    lines.append("|---|---|---|---|---|")
    for nm, p in [("equal-risk", eq), ("Sharpe-opt (IS)", opt)]:
        s = stats(p, 52); i, o = io(p)
        lines.append(f"| {nm} | **{s['sharpe']:+.2f}** | {i:+.2f} | {o:+.2f} | {s['maxdd']:+.0%} |")
    so = io(opt)
    lines.append("\n## Verdict\n")
    lines.append(f"- PIT equity-STRATA Sharpe {se['sharpe']:+.2f}; multi-asset combined "
                 f"OOS **{so[1]:+.2f}**. " + (
                 "**Reaches ~3 OOS** with the survivorship-free equity book." if so[1] >= 2.9
                 else f"Survivorship-free equity book is "
                 + ("strong and lifts the stack." if se['sharpe'] > 1.0 else
                    "still modest after cost; stack ~2-2.5, not 3.")))
    lines.append("\n")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    (1 + eq.fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.4, label=f"equal-risk ({sh(eq,52):.2f})")
    (1 + opt.fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.4, label=f"Sharpe-opt ({sh(opt,52):.2f}, OOS {so[1]:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("PIT equity-STRATA + crypto books toward Sharpe 3 (weekly, net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "pit_equity_strata.png"), dpi=120)
    with open(os.path.join(HERE, "pit_equity_strata.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/pit_equity_strata.md + png")


if __name__ == "__main__":
    main()
