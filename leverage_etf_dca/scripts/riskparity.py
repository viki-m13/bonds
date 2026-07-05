"""LEVERAGED RISK PARITY — the one construction that can OUT-RETURN dot-com, not
just protect. Blend 3x NASDAQ (TQQQ) with 3x long Treasury (TMF, reconstructed
back to 2000 from FRED yields, validated 0.95 corr vs real TLT), inverse-vol
weighted so each contributes equal risk, then vol-target the whole basket.

Mechanism: in deflationary busts (dot-com, 2008, 2020) long-Treasury rallies hard
and its 3x amplifies it, offsetting the tech collapse. Known failure: 2022, when
BOTH stocks and bonds fell -> risk parity gets hit. We test that honestly.

Monthly DCA $1000, 10bps/side, no look-ahead (weights from data through dt,
deployed at dt, earn dt->dt+1). Post-2004 a gold sleeve (GLD/UGL) can be added.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl")
close = P["close"].sort_index().copy(); kind = dict(P["kind"])
BR = pd.read_pickle(f"{HERE}/_bond_recon.pkl")
# splice reconstructed long-Treasury + TMF into the panel (recon throughout for
# a consistent long history; validated vs real TLT post-2005)
close["TLT_R"] = BR["TLT_R"].reindex(close.index).ffill()
close["TMF_R"] = BR["TMF_R"].reindex(close.index).ffill()
# reconstruct UGL (2x gold) from GLD where GLD exists (2004+), for the gold sleeve
if "GLD" in close:
    rg = close["GLD"].pct_change()
    r2 = 2*rg - (0.0095 + 0.03)/252.0
    ugl = (1+r2.fillna(0)).cumprod(); close["UGL_R"] = ugl/ugl.dropna().iloc[0]*100.0
retd = close.pct_change()
valid_start = {}
for t in close.columns:
    nz = retd[t].ne(0) & retd[t].notna()
    valid_start[t] = nz.index[nz.argmax()] if nz.any() else close.index[-1]

def month_grid(nth=None):
    g = []
    for _, idx in close.groupby(close.index.to_period("M")).groups.items():
        d = close.loc[idx].index
        if nth is None: g.append(d[-1])
        elif len(d) > nth: g.append(d[nth])
    return pd.DatetimeIndex(sorted(g))

def rp_weights(mg, assets, target=0.14, cap=1.0, volwin=63, mode="invvol",
               fixed=None):
    """return DataFrame [mg x assets] of weights (fraction of portfolio), plus the
    residual goes to cash. target = basket annualized vol target."""
    vol = {t: (retd[t].rolling(volwin, min_periods=int(volwin*0.7)).std()*np.sqrt(252)
               ).reindex(mg, method="ffill") for t in assets}
    vol = pd.DataFrame(vol)
    if mode == "invvol":
        raw = 1.0/vol
    elif mode == "momtilt":
        # continuous, slow tilt toward the better-trending asset (dual-momentum,
        # smooth): inv-vol base * exp(tilt_k * cross-sectional z of 12m trend).
        mom = pd.DataFrame({t: (close[t]/close[t].shift(252)-1).reindex(mg, method="ffill")
                            for t in assets})
        z = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1).replace(0, np.nan), axis=0).fillna(0)
        raw = (1.0/vol) * np.exp(fixed.get("tilt_k", 1.0) * z)
    else:  # fixed target risk weights
        raw = pd.DataFrame({t: pd.Series(fixed[t], index=mg) for t in assets})
    # zero out not-yet-tradeable
    for t in assets:
        raw.loc[mg < valid_start[t], t] = np.nan
    rw = raw.div(raw.sum(axis=1), axis=0)                     # normalized risk weights
    # basket vol proxy (ignoring correlation) then scale to target
    bvol = (rw * vol).sum(axis=1)
    scale = (target / bvol).clip(0, cap)
    W = rw.mul(scale, axis=0).fillna(0.0)
    return W                                                  # no shift: forward-buy loop

def run(start, end, W, nth=None, contrib=1000.0, cost=0.001):
    mg = month_grid(nth); grid = mg[(mg >= start) & (mg <= end)]
    px = close.reindex(mg, method="ffill")
    assets = list(W.columns)
    pos = {}; contributed = 0.0; rows = []
    for dt in grid:
        cash = 0.0
        for t in list(pos.keys()):
            if t == "_CASH_": cash += pos.pop(t)
            elif np.isfinite(px.loc[dt, t]): cash += pos.pop(t)*px.loc[dt, t]*(1-cost)
            else: pos.pop(t)
        contributed += contrib; cash += contrib
        wrow = W.loc[dt] if dt in W.index else pd.Series(0.0, index=assets)
        spent = 0.0
        for t in assets:
            wt = float(wrow.get(t, 0.0))
            if wt > 0 and dt >= valid_start[t] and np.isfinite(px.loc[dt, t]):
                pos[t] = wt*cash*(1-cost)/px.loc[dt, t]; spent += wt*cash
        if cash - spent > 0: pos["_CASH_"] = pos.get("_CASH_", 0)+(cash-spent)
        V = sum((sh if t == "_CASH_" else sh*px.loc[dt, t])
                for t, sh in pos.items() if t == "_CASH_" or np.isfinite(px.loc[dt, t]))
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

def qqq_dca(start, end, nth=None, contrib=1000.0):
    g = month_grid(nth); g = g[(g >= start) & (g <= end)]
    r = close.reindex(month_grid(nth))["QQQ"].pct_change().reindex(g).fillna(0)
    V = 0; c = 0; rows = []
    for dt, x in r.items(): V = V*(1+x)+contrib; c += contrib; rows.append((dt, V, c))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")

def maxdd(eq):
    V = eq["V"]; C = eq["contributed"].diff().fillna(eq["contributed"])
    r = ((V-C)/V.shift(1)-1).dropna(); cum = (1+r).cumprod()
    return (cum/cum.cummax()-1).min()

ERAS = [("2000-01","2003-12"),("2000-01","2010-12"),("2006-01","2009-12"),
        ("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),
        ("2000-01","2026-06"),("2006-01","2026-06")]

def evalcfg(name, assets, nth=None, **kw):
    out = []
    for st, en in ERAS:
        s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
        W = rp_weights(month_grid(nth), assets, **kw)
        eq = run(s, e, W, nth=nth); b = qqq_dca(s, e, nth=nth)
        out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
    print(f"{name:24}" + "".join(f"{v:>7.2f}" for v in out))
    return out

if __name__ == "__main__":
    print("RATIO vs QQQ-DCA (leveraged risk parity)")
    print(f"{'config':24}" + "".join(f"{a[:7]:>7}" for a, _ in ERAS))
    evalcfg("RP tqqq+tmf t14", ["TQQQ","TMF_R"], target=0.14)
    evalcfg("RP tqqq+tmf t18", ["TQQQ","TMF_R"], target=0.18)
    evalcfg("RP tqqq+tmf t22", ["TQQQ","TMF_R"], target=0.22)
    evalcfg("RP +gold t18", ["TQQQ","TMF_R","UGL_R"], target=0.18)
    evalcfg("HFEA 55/45 t18", ["TQQQ","TMF_R"], target=0.18, mode="fixed",
            fixed={"TQQQ":0.55,"TMF_R":0.45})
    print("\nPHASE ROBUSTNESS (RP tqqq+tmf t18, full 2000-2026, ratio/maxDD):")
    for nth in [None,4,9,14]:
        s,e=pd.Timestamp("2000-01-01"),pd.Timestamp("2026-07-01")
        W=rp_weights(month_grid(nth),["TQQQ","TMF_R"],target=0.18)
        eq=run(s,e,W,nth=nth); b=qqq_dca(s,e,nth=nth)
        print(f"  {'ME' if nth is None else 'd'+str(nth):4}: {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x  maxDD {maxdd(eq):.0%}")
    # 2022 honesty check
    s,e=pd.Timestamp("2022-01-01"),pd.Timestamp("2022-12-31")
    W=rp_weights(month_grid(None),["TQQQ","TMF_R"],target=0.18)
    eq=run(s,e,W); b=qqq_dca(s,e)
    print(f"\n2022 (both-fell): RP ratio {eq['V'].iloc[-1]/b['V'].iloc[-1]:.2f}x  RP maxDD {maxdd(eq):.0%}  QQQ-DCA maxDD {maxdd(b):.0%}")
