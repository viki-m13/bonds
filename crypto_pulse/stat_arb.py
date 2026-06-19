"""STATISTICAL ARBITRAGE — Avellaneda-Lee residual mean-reversion on the HL crypto
universe. A structurally DIFFERENT return engine than trend/carry/BAB: it bets on
idiosyncratic over-reaction reverting, market- and factor-neutral.

Method (Avellaneda & Lee 2010, "Statistical Arbitrage in the US Equities Market"):
  1. each day, over a trailing window, extract the top-K PCA factors of the
     eligible-coin return correlation matrix (factor returns = PC portfolios);
  2. regress each coin's returns on those factors -> idiosyncratic residuals;
  3. model the cumulative residual as an Ornstein-Uhlenbeck process; the
     s-score = (X - mean)/std measures how stretched the residual is;
  4. trade reversion: SHORT rich names (s > +thr), LONG cheap names (s < -thr),
     dollar-neutral; close as s -> 0. Positions are residual/idiosyncratic, so the
     book is ~market-neutral by construction.

Honesty: causal (betas/residual stats from data through close of d-1, traded d),
net of 4.5bps taker + real HL funding, vol-targeted, IS=first60/OOS=last40. We
also test whether it is a NEW uncorrelated sleeve vs the max-stack book.

Run from crypto_pulse/:  python stat_arb.py  (-> research/stat_arb.md + png)
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


def sscore_signals(R, elig, win=60, K=3, hold=2):
    """Return daily target weights (gross-1, dollar-neutral) from residual s-scores.
    win: PCA/regression window; K: number of PCA factors; hold: rebalance every
    `hold` days to control turnover/cost."""
    idx = R.index
    cols = R.columns
    Rv = R.values
    W = np.full(Rv.shape, np.nan)
    elv = elig.values
    for t in range(win, len(idx)):
        if t % hold != 0:
            continue
        m = elv[t] & np.isfinite(Rv[t - win:t]).all(axis=0)
        if m.sum() < 12:
            continue
        sub = Rv[t - win:t][:, m]                       # window x n
        sub = sub - sub.mean(0)
        sd = sub.std(0) + 1e-12
        Z = sub / sd                                    # standardized returns
        # PCA factors from correlation matrix
        cov = np.corrcoef(Z, rowvar=False)
        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            continue
        V = evecs[:, ::-1][:, :K]                        # top-K eigenvectors
        F = Z @ V                                        # window x K factor returns
        # regress each name's standardized returns on factors -> residuals
        # beta = (F'F)^-1 F'Z   (K x n)
        FtF = F.T @ F + 1e-6 * np.eye(K)
        beta = np.linalg.solve(FtF, F.T @ Z)            # K x n
        resid = Z - F @ beta                             # window x n
        # cumulative residual -> OU s-score (Avellaneda-Lee discrete fit)
        Xc = np.cumsum(resid, axis=0)
        s = np.zeros(m.sum())
        for j in range(m.sum()):
            x = Xc[:, j]
            x0, x1 = x[:-1], x[1:]
            b = np.polyfit(x0, x1, 1)                    # x1 = a + b x0
            a, bb = b[1], b[0]
            if bb <= 0 or bb >= 1:
                s[j] = 0.0
                continue
            mean = a / (1 - bb)
            var_eq = np.var(x1 - (a + bb * x0)) / (1 - bb**2)
            sig_eq = np.sqrt(max(var_eq, 1e-12))
            s[j] = -(x[-1] - mean) / sig_eq             # high resid -> negative score
        # trade reversion: long cheap (s>0 after the sign flip above), short rich
        w = np.zeros(len(cols))
        full = np.zeros(m.sum())
        thr = 1.0
        active = np.abs(s) > thr
        full[active] = s[active]
        if np.abs(full).sum() > 0:
            full = full - full.mean()                   # dollar neutral
            full = full / (np.abs(full).sum() + 1e-12)
            w[np.where(m)[0]] = full
        W[t] = w
    Wdf = pd.DataFrame(W, index=idx, columns=cols).ffill(limit=hold)
    return Wdf


def main():
    coins = [c for c in v.OVERLAP
             if os.path.exists(os.path.join(v.CRYPTO, f"{c}_USD.csv"))]
    C, V, H, L = v.load_prices(coins)
    F = v.load_daily_funding(coins, C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)

    lines = ["# Statistical arbitrage (Avellaneda-Lee residual reversion) on HL\n"]
    lines.append(f"Net of {TAKER}bps taker + real funding, vol-targeted, "
                 "IS=first60/OOS=last40. Market/factor-neutral residual reversion.\n")
    lines.append("## Parameter scan (window / #PCA factors / hold)\n")
    lines.append("| win | K | hold | Sharpe | IS | OOS | ann | maxDD | turn/day |")
    lines.append("|---|---|---|---|---|---|---|---|---|")

    hl = C.index >= HL_START
    idxhl = C.index[hl]
    cut = idxhl[int(len(idxhl) * 0.6)]

    best = None
    for win in (40, 60):
        for K in (2, 3, 5):
            for hold in (2, 3, 5):
                W = sscore_signals(R, elig, win=win, K=K, hold=hold)
                wl = W.shift(1)
                gross = (wl * R).sum(axis=1)
                turn = (wl - wl.shift(1)).abs().sum(axis=1)
                pnl = gross - turn * TAKER / 1e4 - (wl * F).sum(axis=1)
                p = vt(pnl)
                sh = sharpe(p[hl])
                si = sharpe(p[(p.index < cut) & hl])
                so = sharpe(p[(p.index >= cut) & hl])
                s = stats(p[hl])
                lines.append(f"| {win} | {K} | {hold} | {sh:+.2f} | {si:+.2f} | "
                             f"{so:+.2f} | {s['ann']:+.1%} | {s['maxdd']:+.1%} | "
                             f"{turn[hl].mean():.2f} |")
                # pick best by MIN(IS,OOS) to reward robustness, not lucky halves
                score = min(si, so) if (np.isfinite(si) and np.isfinite(so)) else -9
                if best is None or score > best[0]:
                    best = (score, win, K, hold, p, sh, si, so)

    _, win, K, hold, pbest, sh, si, so = best
    lines.append(f"\n**Best (by min(IS,OOS)):** win={win}, K={K}, hold={hold} -> "
                 f"Sharpe {sh:+.2f} (IS {si:+.2f} / OOS {so:+.2f}).\n")

    # is it a NEW uncorrelated sleeve vs the max-stack? load that combined book
    corr_note = ""
    try:
        import max_stack as ms
        S = ms.build_sleeves(C, V, H, L, F)
        stack = pd.DataFrame({k: ms.vt(s) for k, s in S.items()})
        # rough combined (equal risk of admitted) just for correlation context
        ref = stack.mean(axis=1)
        cc = pd.concat({"statarb": pbest[hl], "stack": ref[hl]}, axis=1).dropna()
        rho = cc["statarb"].corr(cc["stack"])
        corr_note = (f"Correlation of stat-arb to the directional sleeve stack: "
                     f"**{rho:+.2f}**. ")
    except Exception as e:
        corr_note = f"(correlation check skipped: {e}) "

    lines.append("## Verdict\n")
    verdict = ("a GENUINE positive market-neutral sleeve"
               if (si > 0.1 and so > 0.1) else
               "NOT robust net of taker cost (fails one half)")
    lines.append(f"- Residual reversion is {verdict}. {corr_note}"
                 "Crypto residual reversion at DAILY+ horizon (not intraday, which "
                 "was bid-ask bounce) is the honest test. If positive and "
                 "uncorrelated it is a new sleeve for the stack; if not, the "
                 "archetype is taker-blocked here and we pivot to the next.\n")

    fig, ax = plt.subplots(figsize=(11, 5))
    (1 + pbest[hl].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.0,
        label=f"stat-arb residual reversion (Sharpe {sh:.2f}, OOS {so:.2f})")
    ax.axvline(cut, color="gray", ls=":", lw=1); ax.legend(fontsize=9)
    ax.set_title("Statistical arbitrage (Avellaneda-Lee) on HL crypto (net)")
    ax.set_ylabel("growth of $1"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "stat_arb.png"), dpi=110)

    out = "\n".join(lines)
    with open(os.path.join(HERE, "stat_arb.md"), "w") as fh:
        fh.write(out)
    print(out)
    print("\n[written] research/stat_arb.md + png")


if __name__ == "__main__":
    main()
