"""INTERNATIONAL EXTENSION: do Japanese / international stocks offer what the
US cross-section could not?

Data honesty note: Tiingo (this key) serves US-listed, USD-denominated
securities only — no native Tokyo/London/Frankfurt listings. What IS testable
is the international exposure a US investor can actually trade: US-listed ADRs
of foreign companies + country ETFs. For Japan and Asia this is arguably the
*more* interesting structure, because the home market trades while the US is
closed — exactly the overnight/intraday split that killed the US books (Wall 1).

UNIVERSE BIAS (logged): the ticker list is compiled by hand from names known
today, plus several since-delisted ones (CHL, PTR, SNP, TOT, CS). It is
survivorship-flavored, which INFLATES results — so negative results are strong.

Tests:
  A. Session decomposition by region: overnight vs intraday share of return,
     and the gap->intraday relationship (underreaction = continuation and is
     tradeable at the open; overreaction = reversal, also tradeable).
  B. Gap books: after a big overnight gap, go with it / fade it during the US
     session. Entry at open (OPTIMISTIC bound — labeled), exit at close.
  C. Cross-sectional weekly reversal among country ETFs (honest t+1 close exec).
  D. Cross-sectional residual reversal on the ADR universe (honest t+1 close),
     compared against the US-stock result of exp11.
  E. ADR vs home-country-ETF spread reversion (economic-anchor pairs).
Splits: DEV 1996-2014, VAL 2015-2019, TEST 2020+ (opened only if VAL survives).
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
D = "/tmp/sharpe3_work/intl"
GROUPS = {
 "Japan":   ["TM","SONY","HMC","MUFG","SMFG","MFG","NMR","TAK","IX"],
 "AsiaXJP": ["TSM","UMC","ASX","CHT","KB","SHG","PKX","LPL","BABA","BIDU","JD","NTES","CHL","PTR","SNP"],
 "Europe":  ["SAP","ASML","NVS","AZN","GSK","UL","DEO","BTI","SNY","HSBC","BCS","DB","UBS","ING","PHG",
             "STM","E","RIO","BHP","NGG","VOD","LYG","TOT","CS","ORAN"],
 "LatAmIN": ["VALE","PBR","ITUB","BBD","ABEV","AMX","CX","INFY","WIT","HDB","IBN","RDY","TTM","TEVA"],
 "CtryETF": ["EWJ","EWG","EWU","EWQ","EWI","EWP","EWA","EWC","EWY","EWT","EWZ","EWM","EWH","EWS","EWW",
             "EWL","EWD","EWN","EWO","EPOL","THD","INDA","FXI","EEM","EFA"],
}
HOME_ETF = {**{t: "EWJ" for t in GROUPS["Japan"]},
            **{t: "FXI" for t in ["BABA","BIDU","JD","NTES","CHL","PTR","SNP"]},
            **{t: "EWT" for t in ["TSM","UMC","ASX","CHT"]},
            **{t: "EWY" for t in ["KB","SHG","PKX","LPL"]},
            **{t: "EWZ" for t in ["VALE","PBR","ITUB","BBD","ABEV"]},
            **{t: "INDA" for t in ["INFY","WIT","HDB","IBN","RDY","TTM"]}}

ao, ac = {}, {}
for g, ts in list(GROUPS.items()) + [("mkt", ["SPY", "QQQ"])]:
    for t in ts:
        p = f"{D}/{t}.csv"
        if not os.path.exists(p) or os.path.getsize(p) < 10000: continue
        try:
            d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            if "adjOpen" not in d.columns: continue
            ao[t] = d["adjOpen"]; ac[t] = d["adjClose"]
        except Exception: continue
AO = pd.DataFrame(ao).sort_index(); AC = pd.DataFrame(ac).sort_index()
AO, AC = AO.align(AC, join="inner")
idx = AC.index
print(f"panel: {AC.shape[1]} securities, {len(idx)} days, {idx[0].date()}..{idx[-1].date()}  t={time.time()-t0:.0f}s", flush=True)

R_cc = AC.pct_change(fill_method=None)
R_on = (AO / AC.shift(1) - 1)          # overnight: home-market session for Asia
R_id = (AC / AO - 1)                   # US intraday session
for X in (R_cc, R_on, R_id):
    X[np.abs(X) > 1.0] = np.nan

DEV = (idx >= pd.Timestamp("1996-01-01")) & (idx <= pd.Timestamp("2014-12-31"))
VAL = (idx >= pd.Timestamp("2015-01-01")) & (idx <= pd.Timestamp("2019-12-31"))

# ---------------- A) session decomposition ----------------
print("\n=== A) session decomposition (bps/day, full sample) ===")
print(f"{'group':10} {'overnight':>10} {'intraday':>10} {'gap->intraday corr':>20} {'n':>5}")
for g, ts in GROUPS.items():
    cols = [t for t in ts if t in AC.columns]
    if not cols: continue
    on = R_on[cols].mean().mean()*1e4; ind = R_id[cols].mean().mean()*1e4
    cs = []
    for t in cols:
        s = pd.concat([R_on[t], R_id[t]], axis=1).dropna()
        if len(s) > 500: cs.append(s.iloc[:, 0].corr(s.iloc[:, 1]))
    print(f"{g:10} {on:10.1f} {ind:10.1f} {np.mean(cs):20.3f} {len(cols):5}", flush=True)

# ---------------- B) gap books ----------------
print("\n=== B) gap books: entry at OPEN (optimistic bound), exit same close ===")
vol = R_cc.rolling(63, min_periods=40).std().shift(1)
gz = (R_on / vol)
def gap_book(cols, sign, th, mask, fee=10, label=""):
    g = gz[cols].values; r = R_id[cols].values
    pnl, dates = [], []
    for ti in range(len(idx)):
        if not mask[ti]: continue
        row = g[ti]
        sel_up = np.where(row > th)[0]; sel_dn = np.where(row < -th)[0]
        rets = [sign*r[ti, k] for k in sel_up if np.isfinite(r[ti, k])]
        rets += [-sign*r[ti, k] for k in sel_dn if np.isfinite(r[ti, k])]
        pnl.append(np.mean(rets) - 2*fee/1e4 if rets else 0.0); dates.append(idx[ti])
    s = pd.Series(pnl, index=pd.DatetimeIndex(dates))
    act = s[s != 0]
    print(f"  {label:34} Sharpe {G.sharpe(s):5.2f}  active {len(act)/max(len(s),1):3.0%}  "
          f"mean/trade {act.mean()*1e4:+6.0f}bp  (gross/trade {(act.mean()+2*fee/1e4)*1e4:+5.0f}bp)", flush=True)
    return G.sharpe(s)

for g in ("Japan", "AsiaXJP", "Europe", "CtryETF"):
    cols = [t for t in GROUPS[g] if t in AC.columns]
    for sign, nm in ((+1, "momentum"), (-1, "fade")):
        gap_book(cols, sign, 1.5, DEV, label=f"{g} gap-{nm} |z|>1.5 DEV")

# ---------------- C) country-ETF cross-sectional reversal ----------------
print("\n=== C) country-ETF weekly cross-sectional reversal (honest t+1 close exec) ===")
etf = [t for t in GROUPS["CtryETF"] if t in AC.columns]
Rw = R_cc[etf]
L = np.log1p(Rw)
def xs_book(sig, mask_name, fee):
    z = sig.sub(sig.mean(axis=1), axis=0).div(sig.std(axis=1)+1e-12, axis=0).clip(-3, 3)
    pos = z.clip(lower=0); neg = (-z).clip(lower=0)
    W = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
    wk = pd.Series(idx.to_period("W"), index=idx)
    W = W.where(wk != wk.shift(-1)).ffill().fillna(0.0)   # trade weekly, hold
    net, gross, tno = G.run(W, Rw, fee_bps=fee)
    return net, gross, tno
for hz, nm in ((5, "r5"), (21, "r21")):
    sig = -np.expm1(L.rolling(hz).sum())
    for fee in (5, 10):
        net, gross, tno = xs_book(sig, nm, fee)
        print(f"  ctryETF rev{nm} {fee}bp: DEV {G.sharpe(net['1996':'2014']):5.2f}  VAL {G.sharpe(net['2015':'2019']):5.2f}  "
              f"(gross {G.sharpe(gross['1996':'2014']):5.2f}, tno {tno['1996':'2014'].mean():.3f})", flush=True)

# ---------------- D) ADR cross-sectional residual reversal ----------------
print("\n=== D) ADR universe residual reversal (honest t+1 close exec) ===")
adr = [t for t in sum([GROUPS[g] for g in ("Japan", "AsiaXJP", "Europe", "LatAmIN")], []) if t in AC.columns]
Ra = R_cc[adr]
mkt = Ra.mean(axis=1)
xs = Ra.sub(mkt, axis=0)
rv = xs.rolling(63, min_periods=40).std().shift(1)
for hz in (1, 5, 21):
    z = -(xs.rolling(hz).sum()/(rv*np.sqrt(hz)))
    zz = z.sub(z.mean(axis=1), axis=0).div(z.std(axis=1)+1e-12, axis=0).clip(-3, 3)
    pos = zz.clip(lower=0); neg = (-zz).clip(lower=0)
    W = 0.5*pos.div(pos.sum(axis=1), axis=0).fillna(0.0) - 0.5*neg.div(neg.sum(axis=1), axis=0).fillna(0.0)
    if hz > 1:
        wk = pd.Series(idx.to_period("W"), index=idx)
        W = W.where(wk != wk.shift(-1)).ffill().fillna(0.0)
    for fee in (5, 10):
        net, gross, tno = G.run(W, Ra, fee_bps=fee)
        print(f"  ADR resrev h{hz:2} {fee}bp: DEV {G.sharpe(net['1996':'2014']):5.2f}  VAL {G.sharpe(net['2015':'2019']):5.2f}  "
              f"(gross {G.sharpe(gross['1996':'2014']):5.2f}, tno {tno['1996':'2014'].mean():.3f})", flush=True)

# ---------------- E) ADR vs home-ETF spread reversion ----------------
print("\n=== E) ADR vs home-country-ETF spread reversion (weekly, t+1 close) ===")
pairs = [(t, HOME_ETF[t]) for t in adr if t in HOME_ETF and HOME_ETF[t] in AC.columns]
print(f"  {len(pairs)} anchored pairs", flush=True)
lp = np.log(AC)
rows = []
wkends = [d for d in G.week_ends(idx)]
for d in wkends:
    ti = idx.get_loc(d)
    if ti < 60: continue
    w = {}
    for a, b in pairs:
        sp = (lp[a] - lp[b]).iloc[ti-40:ti+1]
        if sp.isna().any(): continue
        z = (sp.iloc[-1] - sp.mean())/(sp.std()+1e-12)
        if abs(z) < 1.5: continue
        w[a] = w.get(a, 0) - np.sign(z); w[b] = w.get(b, 0) + np.sign(z)
    if not w: continue
    s = pd.Series(w, dtype=float); s = s/s.abs().sum(); s.name = d
    rows.append(s)
Wp = pd.DataFrame(rows).reindex(columns=AC.columns).fillna(0.0)
for fee in (5, 10):
    net, gross, tno = G.run(Wp, R_cc, fee_bps=fee)
    print(f"  ADR-vs-ETF pairs {fee}bp: DEV {G.sharpe(net['1996':'2014']):5.2f}  VAL {G.sharpe(net['2015':'2019']):5.2f}  "
          f"(gross {G.sharpe(gross['1996':'2014']):5.2f}, tno {tno['1996':'2014'].mean():.3f})", flush=True)
print(f"\nexp20 done t={time.time()-t0:.0f}s", flush=True)
