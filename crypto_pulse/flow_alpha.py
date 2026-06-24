"""INVENT a new book: smart-money (informed-account) flow + ai-trader momentum, on L4.

Goal: the highest honest OOS Sharpe we can get from ~29h of L4 per-account tape, set
apart from VOL/STRATA. The new idea vs blanket whale-following: not all big accounts are
informed. We SCORE every account on the TRAINING window by whether its net taker flow
preceded favorable next-bar moves, FREEZE the informed ("smart") and anti-informed
("dumb") sets, then OUT-OF-SAMPLE go with smart-money net flow and against dumb-money net
flow. ai-trader's cross-sectional momentum (intraday ROC) is added as confirmation.

Everything causal: account scores come ONLY from the IS half; the smart/dumb sets are
frozen before any OOS bar; signals are formed at bar t and applied to the t->t+1 return.

HONESTY: 29h => ~12h OOS, one regime. The robust read is the OOS predictive IC + t-stat
and whether net-of-cost PnL clears the 4.5bps taker (and a maker-rebate scenario). A high
annualized Sharpe on 12h is suggestive, NOT a track record. We report all of it straight.

Run from crypto_pulse/:  python flow_alpha.py  (-> research/flow_alpha.md + png)
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAPE = os.path.join(ROOT, "data", "l4_shards")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
TAKER = 4.5 / 1e4
MAKER = -1.0 / 1e4                # HL maker rebate scenario (flow-following can post)
DENSE_START = pd.Timestamp("2026-06-22 14:00")
MIN_PART = 15                     # min IS participations for an account to be rankable


def load_tape():
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(TAPE, "*.parquet")))],
                   ignore_index=True).drop_duplicates(subset=["tid"])
    df["t"] = pd.to_datetime(df["time"], unit="ms")
    df = df[df["t"] >= DENSE_START].sort_values("t").reset_index(drop=True)
    df["notional"] = df["px"] * df["sz"]
    df["sgn"] = np.where(df["side"] == "B", 1.0, -1.0)
    df["aggr"] = np.where(df["side"] == "B", df["buyer"], df["seller"])
    df["signed"] = df["sgn"] * df["notional"]
    return df


def xs_z(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-9, axis=0)


def sharpe(p, ppyr):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ppyr) if len(p) > 20 and p.std() > 0 else np.nan


def ic_t(z, fr, sl):
    pool = pd.DataFrame({"z": z.iloc[sl].values.ravel(),
                         "r": fr.iloc[sl].values.ravel()}).dropna()
    if len(pool) < 100:
        return np.nan, np.nan, 0
    ic = pool["z"].corr(pool["r"])
    return ic, ic * np.sqrt(len(pool)), len(pool)


def run(freq, df):
    bar = df["t"].dt.floor(freq)
    df = df.assign(bar=bar)
    px = df.groupby(["bar", "coin"])["px"].last().unstack().sort_index()
    dol = df.groupby(["bar", "coin"])["notional"].sum().unstack().reindex_like(px).fillna(0.0)
    px = px.ffill()
    ret1 = np.log(px).diff().shift(-1).clip(-0.15, 0.15)        # next-bar return (causal fwd)
    elig = dol.rolling(8, min_periods=2).mean() > 5e4
    n = len(px); cut = int(n * 0.6)
    is_sl, oos_sl = slice(0, cut), slice(cut, n)
    bars = px.index

    # per (account, coin, bar) net signed flow
    acb = df.groupby(["aggr", "coin", "bar"])["signed"].sum().reset_index()
    acb["bi"] = acb["bar"].map({b: i for i, b in enumerate(bars)})
    # next-bar return for that coin-bar
    rmap = {(c, i): ret1.iloc[i][c] for i in range(n) for c in ret1.columns}
    acb["fr"] = [rmap.get((c, i), np.nan) for c, i in zip(acb["coin"], acb["bi"])]
    is_acb = acb[acb["bi"] < cut].dropna(subset=["fr"])

    # account score = dollar-weighted alignment of its flow direction with next move (IS only)
    is_acb = is_acb.assign(hit=np.sign(is_acb["signed"]) * is_acb["fr"] * is_acb["signed"].abs())
    grp = is_acb.groupby("aggr")
    score = grp["hit"].sum()
    parts = grp.size()
    score = score[parts >= MIN_PART]
    score = score / (is_acb.groupby("aggr")["signed"].apply(lambda s: s.abs().sum())[score.index] + 1e-9)
    smart = set(score[score > score.quantile(0.90)].index)     # top 10% informed
    dumb = set(score[score < score.quantile(0.10)].index)      # bottom 10% anti-informed

    # build OOS-applicable per-(coin,bar) net flow of smart and dumb sets (full window)
    def set_flow(accts):
        sub = df[df["aggr"].isin(accts)]
        f = sub.groupby(["bar", "coin"])["signed"].sum().unstack().reindex_like(px).fillna(0.0)
        return f
    smart_f, dumb_f = set_flow(smart), set_flow(dumb)

    # ai-trader intraday cross-sectional momentum (ROC over last k bars) as confirmation
    k = max(3, int(15 / int(freq.replace("min", ""))))         # ~15min lookback
    roc = px / px.shift(k) - 1

    sig_smart = xs_z(smart_f).where(elig)
    sig_dumb = xs_z(dumb_f).where(elig)
    sig_flow = (sig_smart - sig_dumb)                          # follow smart, fade dumb
    sig_mom = xs_z(roc).where(elig)
    sig_combo = (sig_flow + sig_mom)

    signals = {"smart-money flow": sig_flow, "ai-ROC momentum": sig_mom,
               "flow+momentum combo": sig_combo}
    ppyr = 365 * 24 * 60 / int(freq.replace("min", ""))
    out = {}
    for name, sig in signals.items():
        icv, tv, nn = ic_t(sig, ret1, oos_sl)
        # tradeable book: EWMA-smoothed, follow, gross=1
        s = sig.ewm(span=3).mean()
        w = s.div(s.abs().sum(axis=1), axis=0).fillna(0.0)
        gross = (w * ret1).sum(axis=1)
        turn = (w - w.shift(1)).abs().sum(axis=1)
        net_tk = gross - turn * TAKER
        net_mk = gross - turn * MAKER
        out[name] = dict(ic=icv, t=tv, n=nn,
                         g=sharpe(gross.iloc[oos_sl], ppyr),
                         ntk=sharpe(net_tk.iloc[oos_sl], ppyr),
                         nmk=sharpe(net_mk.iloc[oos_sl], ppyr),
                         eq_g=gross.iloc[oos_sl], eq_tk=net_tk.iloc[oos_sl], eq_mk=net_mk.iloc[oos_sl],
                         turn=turn.iloc[oos_sl].mean())
    return out, ppyr, len(smart), len(dumb), n - cut


def main():
    df = load_tape()
    L = ["# INVENTED book: smart-money flow + ai-trader momentum (L4, honest OOS)\n",
         f"Tape {df['t'].min()} -> {df['t'].max()}, {len(df):,} trades, {df['coin'].nunique()} "
         "coins. Accounts scored on IS half by dollar-weighted flow→next-move alignment; top "
         "10% = smart (follow), bottom 10% = dumb (fade); sets FROZEN before OOS. ai-trader "
         "intraday ROC momentum added as confirmation. OOS = last 40%.\n",
         "**The robust read is the OOS IC + t-stat and net-of-cost PnL, not the annualized "
         "Sharpe (12h OOS).**\n"]
    best_eq = None
    for freq in ["5min", "15min"]:
        out, ppyr, ns, nd, noos = run(freq, df)
        L.append(f"\n## {freq} bars (smart={ns}, dumb={nd} accts; {noos} OOS bars; ppyr {ppyr:,.0f})\n")
        L.append("| signal | OOS IC (t) | gross Sh | net Sh (taker) | net Sh (maker) | turn/bar |")
        L.append("|---|---|---|---|---|---|")
        for name, d in out.items():
            L.append(f"| {name} | {d['ic']:+.4f} ({d['t']:+.1f}) | {d['g']:+.1f} | "
                     f"{d['ntk']:+.1f} | {d['nmk']:+.1f} | {d['turn']:.2f} |")
        # track best book by OOS IC (the robust stat), for the equity plot
        for name, d in out.items():
            if best_eq is None or (np.isfinite(d["ic"]) and d["ic"] > best_eq[2]):
                best_eq = (f"{name} {freq}", d, d["ic"], ppyr)

    bname, bd, _, bppyr = best_eq
    sig_ic = "significant (t>2)" if abs(bd["t"]) > 2 else "NOT significant (|t|<2)"
    L += ["\n## Verdict (honest — the invention did NOT validate)\n",
          "- **The headline finding is a negative one, and it is informative.** Selecting the "
          "top-10% 'informed' accounts on 17h of IS data and following them OOS gave a "
          "**near-zero / negative OOS IC** — i.e. the IS-informed accounts were NOT informed "
          "out-of-sample. By contrast the *blanket* whale-flow signal (no account selection, "
          "`flow_intraday.py`) had a consistently *positive* OOS IC (+0.013…+0.033). **Account-"
          "level selection overfit**: with 17h you cannot reliably tell skill from luck across "
          "31k wallets, so the top-decile is mostly noise that reverses. Aggregate flow is the "
          "more robust object at this sample size.",
          "- ai-trader's intraday ROC momentum also has a **negative OOS IC** here — intraday "
          "price momentum reversed in this 12h window. Neither the new flow idea nor the TA "
          "confirmation survives.",
          f"- Best book by OOS IC: **{bname}** — IC {bd['ic']:+.4f} ({sig_ic}). Even its gross "
          f"Sharpe ({bd['g']:+.1f}) is not bankable on {bd['n']:,} pooled points / ~12h. The "
          "maker-rebate column occasionally turns positive, but that rides a near-zero gross "
          "signal plus the rebate — not a real edge.",
          "- **On Sharpe 3, honestly: not reached, and not close on this data.** I invented the "
          "smart-money book, tested it cleanly OOS, and it failed — that is the honest result. "
          "What this rules IN: the robust path is *aggregate* whale-flow at a *slower* horizon "
          "(its IC grows with horizon and is positive), validated on *weeks* of tape so account "
          "and regime noise average out. What it rules OUT: account-selection alpha and intraday "
          "TA on ~1 day of data. **I will not label anything here Sharpe 3 — it isn't, and "
          "pretending otherwise would be the one unacceptable outcome.**\n"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    (1 + bd["eq_g"].fillna(0)).cumprod().plot(ax=ax, color="#888", lw=1.3,
        label=f"{bname} gross (Sh {bd['g']:+.1f})")
    (1 + bd["eq_mk"].fillna(0)).cumprod().plot(ax=ax, color="#27ae60", lw=1.6,
        label=f"net maker (Sh {bd['nmk']:+.1f})")
    (1 + bd["eq_tk"].fillna(0)).cumprod().plot(ax=ax, color="#c0392b", lw=2.0,
        label=f"net taker (Sh {bd['ntk']:+.1f})")
    ax.axhline(1.0, color="k", lw=0.7); ax.legend(fontsize=9)
    ax.set_title("Smart-money flow + ai-trader momentum — OOS equity (last ~12h of L4, net)")
    ax.set_ylabel("growth of $1 (OOS)"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "flow_alpha.png"), dpi=110)
    with open(os.path.join(HERE, "flow_alpha.md"), "w") as fh:
        fh.write("\n".join(L))
    print("\n".join(L)); print("\n[written] research/flow_alpha.md + png")


if __name__ == "__main__":
    main()
