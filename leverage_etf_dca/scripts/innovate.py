"""INNOVATION LAB — attack QQQ-DCA from angles that NEVER sell the compounder.

Rotation fails because it sells the winner. These ideas instead time HOW MUCH
leverage and WHEN to buy, always on the best-drift core (leveraged NASDAQ):

  (1) REVERSAL-AUGMENTED LEVERAGE DIAL.  Pure vol-targeting de-levers into
      capitulation bottoms (vol peaks exactly when price is cheapest) -> it sells
      low. Add a mean-reversion term: when tech is oversold BUT its secular trend
      is intact, lean the dial back toward full leverage to catch the V-rebound,
      while a broken secular trend (dot-com) keeps it de-levered.
  (2) OVERSOLD-SCALED CONTRIBUTIONS (self-funded value-averaging). Bank part of
      the buy in expensive months; deploy the reserve when the core is below trend.

Everything monthly, DCA $1000/mo baseline, 10bps/side. No look-ahead: every signal
is sampled to the monthly grid and .shift(1) to the PRIOR month-end.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
P = pd.read_pickle(f"{HERE}/_etf_panel.pkl")
close = P["close"].sort_index(); kind = P["kind"]
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

def core_weight(mg, target=0.30, cap=1.0, volwin=63, fastwin=20, accel_k=2.0,
                rev_k=0.0, rev_cap=1.0, sec_win=200, os_win=50, sec_mode="ma"):
    """TQQQ weight in [0,cap]. rev_k>0 adds the reversal lean.
    sec_mode: 'ma' secular gate = QQQ>sec_win MA; 'ret' = 12m QQQ return>0."""
    def av(w):
        return (retd["TQQQ"].rolling(w, min_periods=int(w*0.7)).std()*np.sqrt(252)
                ).reindex(mg, method="ffill")
    vs = av(volwin)
    if accel_k:
        vs = vs * (av(fastwin)/vs).clip(lower=1.0)**accel_k
    w = (target/vs).clip(0, cap)
    if rev_k:
        q = close["QQQ"]
        px = q.reindex(mg, method="ffill")
        if sec_mode == "ma":
            sec = (q > q.rolling(sec_win, min_periods=120).mean()).reindex(mg, method="ffill")
        else:
            sec = (px / px.shift(12) - 1) > 0
        ma_os = q.rolling(os_win, min_periods=os_win//2).mean().reindex(mg, method="ffill")
        oversold = (ma_os/px - 1.0).clip(lower=0.0)          # how far below short MA
        boost = 1.0 + rev_k * oversold.where(sec.fillna(False), 0.0)
        w = (w * boost.clip(1.0, rev_cap)).clip(0, cap)
    # NB: no .shift(1) here. run() is a forward-buy loop (deploy at month-end dt,
    # earn dt->dt+1), so the raw weight decided from data through dt is already
    # look-ahead-free. Shifting here would double-lag (de-lever a month late).
    return w

def run(start, end, wser, nth=None, contrib=1000.0, cost=0.001,
        defense=("GLD", "TLT"), contrib_os_k=0.0):
    mg = month_grid(nth)
    grid = mg[(mg >= start) & (mg <= end)]
    px = close.reindex(mg, method="ffill")
    dtk_all = list(defense)
    # optional contribution scaling reserve
    q = close["QQQ"]; ma = q.rolling(50, min_periods=25).mean()
    os_ser = (ma/q - 1.0).clip(lower=0.0).reindex(mg, method="ffill").shift(1).fillna(0)
    reserve = 0.0
    pos = {}; contributed = 0.0; rows = []
    for dt in grid:
        # liquidate
        cash = 0.0
        for t in list(pos.keys()):
            if t == "_CASH_": cash += pos.pop(t)
            elif np.isfinite(px.loc[dt, t]): cash += pos.pop(t)*px.loc[dt, t]*(1-cost)
            else: pos.pop(t)
        # contribution (optionally scaled: bank in rich months, deploy reserve when cheap)
        add = contrib
        if contrib_os_k:
            o = float(os_ser.loc[dt])
            if o > 0:                      # cheap: deploy reserve, up to 100% extra
                deploy = min(reserve, contrib*contrib_os_k*o*10)
                add = contrib + deploy; reserve -= deploy
            else:                          # rich: bank 20% of contribution
                bank = contrib*0.20; add = contrib - bank; reserve += bank
        contributed += contrib
        cash += add
        w = float(wser.loc[dt]) if (dt in wser.index and np.isfinite(wser.loc[dt])) else 0.0
        w = min(max(w, 0.0), 1.0)
        buy = cash*w
        if buy > 0 and np.isfinite(px.loc[dt, "TQQQ"]):
            pos["TQQQ"] = buy*(1-cost)/px.loc[dt, "TQQQ"]
        rem = cash - buy
        dtk = [t for t in dtk_all if dt >= valid_start[t] and np.isfinite(px.loc[dt, t])]
        if rem > 0 and dtk:
            for t in dtk: pos[t] = pos.get(t, 0)+(rem/len(dtk))*(1-cost)/px.loc[dt, t]
        elif rem > 0:
            pos["_CASH_"] = pos.get("_CASH_", 0)+rem
        V = sum((sh if t == "_CASH_" else sh*px.loc[dt, t])
                for t, sh in pos.items() if t == "_CASH_" or np.isfinite(px.loc[dt, t]))
        rows.append((dt, V+reserve, contributed))
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

ERAS = [("1999-03","2003-12"),("2000-01","2010-12"),("2006-01","2009-12"),
        ("2010-01","2014-12"),("2015-01","2019-12"),("2020-01","2026-06"),
        ("2006-01","2026-06")]

def evalcfg(name, nth=None, **wkw):
    contrib_k = wkw.pop("contrib_os_k", 0.0)
    out = []
    for st, en in ERAS:
        s, e = pd.Timestamp(st+"-01"), pd.Timestamp(en+"-01")
        w = core_weight(month_grid(nth), **wkw)
        eq = run(s, e, w, nth=nth, contrib_os_k=contrib_k)
        b = qqq_dca(s, e, nth=nth)
        out.append(eq["V"].iloc[-1]/b["V"].iloc[-1])
    print(f"{name:26}" + "".join(f"{v:>7.2f}" for v in out))
    return out

if __name__ == "__main__":
    print("RATIO vs QQQ-DCA   " + "".join(f"{a[:7]:>7}" for a, _ in ERAS))
    evalcfg("VOLT baseline")
    evalcfg("VOLT rev_k=3 ma", rev_k=3.0, rev_cap=2.0, sec_mode="ma")
    evalcfg("VOLT rev_k=6 ma", rev_k=6.0, rev_cap=2.5, sec_mode="ma")
    evalcfg("VOLT rev_k=6 ret", rev_k=6.0, rev_cap=2.5, sec_mode="ret")
    evalcfg("VOLT rev_k=10 ret", rev_k=10.0, rev_cap=3.0, sec_mode="ret")
    evalcfg("VOLT contribOS k=1", contrib_os_k=1.0)
    evalcfg("VOLT rev6+contribOS", rev_k=6.0, rev_cap=2.5, sec_mode="ret", contrib_os_k=1.0)
