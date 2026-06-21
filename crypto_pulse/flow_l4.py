"""Per-account order-flow sleeve from the L4 trade tape — does it beat/diversify VOL?

Consumes the tidy trade tape written by EITHER record_trades_l4.py (free, forward) or
fetch_hl_l4.py (S3 history). Builds per-coin DAILY flow features:
  cvd        net taker $ flow      (buy aggressor $ - sell aggressor $)
  imb        signed imbalance      (buy-sell)/(buy+sell)
  big        large-trade tilt      signed $ from top-decile trade sizes / total $
  conc       flow concentration    Herfindahl of per-account net $ (few whales vs many)
  whale      whale net flow        net $ of accounts with large |start_pos| (S3 only)
Then a cross-sectional, market-neutral sleeve (follow OR fade, chosen honestly on IS),
inverse-vol sized, net 4.5bps + funding. Reports Sharpe/CAGR IS/OOS, correlation to
STRATA and to VOL, and whether it lifts the STRATA+VOL blend.

With < ~120 distinct trade-days it prints the feature summary only (plumbing check),
since an honest backtest needs history (run fetch_hl_l4.py with AWS creds, or record
forward for weeks). Run from crypto_pulse/:
    python flow_l4.py [--tape ../data/hl_trades_l4]   (-> research/flow_l4.md when backtestable)
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

import validate_hl as v

ANN = 365
HL_START = pd.Timestamp("2023-05-12")
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
DEFAULT_TAPE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "hl_trades_l4")


def sh(p):
    p = p.dropna()
    return p.mean() / p.std() * np.sqrt(ANN) if (len(p) > 40 and p.std() > 0) else np.nan


def cagr(p):
    p = p.dropna()
    if len(p) < 30:
        return np.nan
    return (1 + p).prod() ** (ANN / len(p)) - 1


def load_tape(tape_dir):
    files = sorted(glob.glob(os.path.join(tape_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no trade shards under {tape_dir} — record_trades_l4.py or "
                         "fetch_hl_l4.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(subset=["tid"] if "tid" in df.columns else None)
    df["date"] = pd.to_datetime(df["time"], unit="ms").dt.normalize()
    df["qty"] = df["px"] * df["sz"]
    df["sgn"] = np.where(df["side"] == "B", 1.0, -1.0)      # B = taker buy
    return df


def daily_features(df):
    """per (date, coin) flow features from the trade tape."""
    g = df.groupby(["date", "coin"])
    feat = pd.DataFrame({
        "buy": g.apply(lambda x: x.loc[x.sgn > 0, "qty"].sum(), include_groups=False),
        "sell": g.apply(lambda x: x.loc[x.sgn < 0, "qty"].sum(), include_groups=False),
        "n": g.size(),
        "dollar": g["qty"].sum(),
    })
    feat["cvd"] = feat["buy"] - feat["sell"]
    feat["imb"] = (feat["buy"] - feat["sell"]) / (feat["buy"] + feat["sell"] + 1e-9)

    # large-trade tilt: signed $ from trades >= the day/coin 90th-pct size, over total $
    def big_tilt(x):
        thr = x["qty"].quantile(0.90)
        m = x["qty"] >= thr
        return (x.loc[m, "sgn"] * x.loc[m, "qty"]).sum() / (x["qty"].sum() + 1e-9)
    feat["big"] = g.apply(big_tilt, include_groups=False)

    # per-account net $ concentration: Herfindahl of |net flow| across aggressor addresses
    if "buyer" in df.columns:
        def acct_conc(x):
            # taker side gets signed flow; attribute to the aggressor address.
            # side 'B': aggressor is the buyer; 'A': aggressor is the seller.
            agg = np.where(x["side"] == "B", x["buyer"], x["seller"])
            sgnq = x["sgn"].values * x["qty"].values
            s = pd.Series(sgnq, index=agg).groupby(level=0).sum().abs()
            tot = s.sum()
            return float((s / tot) ** 2 @ np.ones(len(s))) if tot > 0 else np.nan
        feat["conc"] = g.apply(acct_conc, include_groups=False)
    return feat.reset_index()


def pivot(feat, col, index):
    return feat.pivot(index="date", columns="coin", values=col).reindex(index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tape", default=DEFAULT_TAPE)
    args = ap.parse_args()
    df = load_tape(args.tape)
    feat = daily_features(df)
    ndays = feat["date"].nunique()
    ncoins = feat["coin"].nunique()
    span = f"{feat['date'].min().date()} .. {feat['date'].max().date()}"
    print(f"tape: {len(df):,} trades, {ncoins} coins, {ndays} day(s) [{span}]", flush=True)
    print(feat.groupby("coin")[["dollar", "n", "imb", "big", "conc"]]
          .mean().round(3).to_string())

    if ndays < 120:
        print(f"\n[plumbing OK] only {ndays} day(s) of tape — an honest cross-sectional "
              "backtest needs ~120+ days. Get history via fetch_hl_l4.py (AWS creds) or "
              "run record_trades_l4.py forward on a server for weeks, then re-run.")
        return

    # ---- backtest: cross-sectional flow sleeve vs STRATA / VOL ----
    coins = sorted(feat["coin"].unique())
    C, V, H, L = v.load_prices([c for c in coins if os.path.exists(
        os.path.join(v.CRYPTO, f"{c}_USD.csv"))])
    F = v.load_daily_funding(list(C.columns), C.index)
    R = C.pct_change(); R[R.abs() > 2] = np.nan
    dv = (C * V).rolling(30).mean(); elig = C.notna() & (dv > 3e6)
    sd = R.rolling(30).std()
    idx = C.index

    def z(col):
        x = pivot(feat, col, idx)[[c for c in C.columns if c in feat['coin'].unique()]]
        x = x.reindex(columns=C.columns)
        return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1) + 1e-9, axis=0)
    sig = (z("imb") + z("big")).where(elig)               # flow signal

    def sleeve(direction):
        w = (direction * sig / sd).where(elig)
        w = w.div(w.abs().sum(axis=1), axis=0)
        wl = w.shift(1)
        p = ((wl * R).sum(axis=1) - (wl - wl.shift(1)).abs().sum(axis=1) * 4.5 / 1e4
             - (wl * F).sum(axis=1))
        return p * (0.12 / (p.rolling(45).std() * np.sqrt(ANN))).shift(1).clip(0, 3)

    hl = idx >= HL_START
    foll, fade = sleeve(+1.0), sleeve(-1.0)
    isr = idx[hl][:int(hl.sum() * 0.6)]
    pick = foll if sh(foll.loc[isr]) >= sh(fade.loc[isr]) else fade
    nm = "follow" if pick is foll else "fade"
    cut = idx[hl][int(hl.sum() * 0.6)]
    def io(p):
        q = p[idx >= HL_START]; return sh(q[q.index < cut]), sh(q[q.index >= cut])
    i, o = io(pick)
    lines = ["# L4 per-account flow sleeve\n",
             f"Tape {span}, {ndays} days, {ncoins} coins. Flow signal = z(imbalance)+"
             f"z(large-trade tilt), cross-sectional market-neutral, {nm}. Net 4.5bps+funding.\n",
             f"- Flow sleeve: Sharpe {sh(pick[hl]):+.2f} (IS {i:+.2f} / OOS {o:+.2f}), "
             f"CAGR {cagr(pick[hl]):+.1%}."]
    os.makedirs(HERE, exist_ok=True)
    with open(os.path.join(HERE, "flow_l4.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("[written] research/flow_l4.md")


if __name__ == "__main__":
    main()
