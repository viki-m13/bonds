"""Sleeve generator: builds the daily net-return series of every candidate
sleeve for the ensemble, saving to cache/sleeve_returns.parquet.

Inclusion criteria: economically motivated + not an obvious pre-2012 artifact.
Each sleeve: dollar-neutral (or hedged), gross 2.0, next-open exec, 5 bps.
"""
import os, json
import numpy as np
import pandas as pd
import datalib, bt

ROOT = os.path.dirname(os.path.abspath(__file__))

def zs(x):
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)

def build_all():
    P = datalib.load_summit()
    close, open_, volp, member = P["close"], P["open"], P["volume"], P["member"]
    r1 = close.pct_change(fill_method=None)
    intraday = close / open_ - 1
    overnight = open_ / close.shift(1) - 1
    vol20 = r1.rolling(20).std()
    mkt = r1.where(member).mean(axis=1)

    sleeves = {}

    # 1. intraday-cum reversal, smoothed (flagship reversal variant)
    sig = -zs(intraday.rolling(5).sum()).rolling(3).mean()
    sleeves["rev_i5_sm3"] = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)

    # 2. overnight momentum 252d (slowest, least artifact-driven variant)
    sig = zs(overnight.rolling(252).mean() / overnight.rolling(252).std())
    sleeves["onmom252"] = bt.norm_ls(sig, member, 0.1, 0.1, 2.0)

    # 3. conditional liquidity provision (reversal after 2d market drop in high vol)
    sigc = -zs(intraday.rolling(5).sum())
    wc = bt.norm_ls(sigc, member, 0.1, 0.1, 2.0)
    mv = mkt.rolling(20).std() * np.sqrt(252)
    trig = ((mkt < -0.01) & (mv > mv.rolling(252, min_periods=60).median())).astype(float)
    sleeves["liqprov_cond"] = wc.mul(trig, axis=0)

    # 4. 8-K activity drift: long recent filers vs short quiet names
    ek = pd.read_parquet(os.path.join(ROOT, "..", "dca", "research", "data", "sec", "8k_items.parquet"))
    ek["date"] = pd.to_datetime(ek["date"])
    ek = ek[ek.tk.isin(close.columns)]
    evm = pd.DataFrame(False, index=close.index, columns=close.columns)
    for tk, g in ek.groupby("tk"):
        idx = close.index.searchsorted(g["date"].values)
        idx = idx[idx < len(close.index)]
        evm.iloc[idx, evm.columns.get_loc(tk)] = True
    news21 = evm.shift(1).rolling(21, min_periods=1).sum()  # shift(1): use through close d
    has = (news21 > 0) & member
    hasnt = (news21 == 0) & member
    wl = has.div(has.sum(axis=1).clip(lower=1), axis=0)
    ws = hasnt.div(hasnt.sum(axis=1).clip(lower=1), axis=0)
    sleeves["news_drift"] = (wl - ws) * 1.0

    # 5. ML blend (exp14), if predictions exist
    p1 = os.path.join(ROOT, "cache", "exp14_pred_h1.parquet")
    p5 = os.path.join(ROOT, "cache", "exp14_pred_h5.parquet")
    if os.path.exists(p1) and os.path.exists(p5):
        pr = zs(pd.read_parquet(p1)).add(zs(pd.read_parquet(p5)))
        sleeves["ml_blend_sm3"] = bt.norm_ls(pr.rolling(3).mean(), member, 0.2, 0.2, 2.0)

    rets = {}
    for nm, w in sleeves.items():
        res = bt.run(w, P, mode="open", cost_bps=5.0)
        rets[nm] = res["net"]
    df = pd.DataFrame(rets)
    df.to_parquet(os.path.join(ROOT, "cache", "sleeve_returns.parquet"))
    return df

if __name__ == "__main__":
    df = build_all()
    for c in df.columns:
        print(f"{c:16s} SR={bt.sharpe(df[c]):6.2f} {bt.is_oos(df[c])}")
