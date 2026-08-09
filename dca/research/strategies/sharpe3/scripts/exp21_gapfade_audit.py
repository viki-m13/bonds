"""AUDIT of the country-ETF gap-fade (exp20 §B: Sharpe 1.90, +24bp/trade net).

The suspicion: signal and payoff SHARE the opening price.
  signal  z = (O_t / C_{t-1} - 1) / sigma
  payoff  r = C_t / O_t - 1
If O_t is measured with noise e (opening-auction bid-ask bounce, stale prints,
wide spreads on thin international ETFs), then e enters the signal with a +
sign and the payoff with a - sign. A fade book then "earns" the noise —
and it is NOT tradeable, because you must transact AT that same noisy price.

Decisive tests — same signal, payoffs that share NO price with it:
  T1  payoff C_t -> C_{t+1}     (execute at close of the gap day)
  T2  payoff O_{t+1} -> C_{t+1} (skip a full day, then take the US session)
  T3  payoff O_t -> C_t         (the original, contaminated)
  T4  payoff C_t -> C_{t+2}     (two-day hold from close)
Plus a liquidity-signature test: if the effect is open-price noise, it must be
strongest in the THINNEST ETFs and weakest in the most liquid.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
D = "/tmp/sharpe3_work/intl"
ETFS = ["EWJ","EWG","EWU","EWQ","EWI","EWP","EWA","EWC","EWY","EWT","EWZ","EWM","EWH","EWS","EWW",
        "EWL","EWD","EWN","EWO","EPOL","THD","INDA","FXI","EEM","EFA"]
ao, ac, av = {}, {}, {}
for t in ETFS:
    p = f"{D}/{t}.csv"
    if not os.path.exists(p) or os.path.getsize(p) < 10000: continue
    d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    ao[t] = d["adjOpen"]; ac[t] = d["adjClose"]; av[t] = d["adjClose"]*d["adjVolume"]
AO = pd.DataFrame(ao).sort_index(); AC = pd.DataFrame(ac).sort_index(); DV = pd.DataFrame(av).sort_index()
AO, AC = AO.align(AC, join="inner"); DV = DV.reindex(AC.index)
idx = AC.index
R_cc = AC.pct_change(fill_method=None)
R_on = AO/AC.shift(1) - 1
R_id = AC/AO - 1
for X in (R_cc, R_on, R_id):
    X[np.abs(X) > 0.5] = np.nan
vol = R_cc.rolling(63, min_periods=40).std().shift(1)
gz = (R_on/vol)
DEV = (idx >= pd.Timestamp("1996-01-01")) & (idx <= pd.Timestamp("2014-12-31"))
VAL = (idx >= pd.Timestamp("2015-01-01")) & (idx <= pd.Timestamp("2019-12-31"))
print(f"{AC.shape[1]} country ETFs, {len(idx)} days  t={time.time()-t0:.0f}s", flush=True)

PAYOFFS = {
 "T3 O_t->C_t  (SHARES the open: contaminated)": R_id,
 "T1 C_t->C_t+1 (clean)":                        R_cc.shift(-1),
 "T2 O_t+1->C_t+1 (clean, skips a day)":         R_id.shift(-1),
 "T4 C_t->C_t+2 (clean, 2-day hold)":            (AC.shift(-2)/AC - 1),
}
def book(pay, mask, th=1.5, fee=10, cols=None):
    c = cols if cols is not None else list(AC.columns)
    g = gz[c].values; r = pay[c].values
    pnl = []
    for ti in np.where(mask)[0]:
        row = g[ti]
        up = np.where(row > th)[0]; dn = np.where(row < -th)[0]
        rets = [-r[ti, k] for k in up if np.isfinite(r[ti, k])]
        rets += [r[ti, k] for k in dn if np.isfinite(r[ti, k])]
        pnl.append(np.mean(rets) - 2*fee/1e4 if rets else 0.0)
    s = pd.Series(pnl, index=idx[mask])
    act = s[s != 0]
    return G.sharpe(s), act.mean()*1e4, (act.mean()+2*fee/1e4)*1e4, len(act)/max(len(s), 1)

print("\n=== gap-FADE with payoffs that do/don't share the open price (DEV, 10bps) ===")
for nm, pay in PAYOFFS.items():
    sh, net_bp, gross_bp, actv = book(pay, DEV)
    print(f"  {nm:46} Sharpe {sh:6.2f}  net/trade {net_bp:+6.0f}bp  gross/trade {gross_bp:+6.0f}bp  active {actv:3.0%}", flush=True)

print("\n=== liquidity signature: does the effect live in the THIN names? (T3 contaminated, DEV) ===")
med = DV[DEV].median()
thin = list(med.sort_values().head(8).index); thick = list(med.sort_values().tail(8).index)
for nm, cols in (("8 THINNEST ETFs", thin), ("8 MOST LIQUID ETFs", thick)):
    sh, net_bp, gross_bp, actv = book(R_id, DEV, cols=cols)
    print(f"  {nm:22} Sharpe {sh:6.2f}  gross/trade {gross_bp:+6.0f}bp   [{', '.join(cols[:5])}...]", flush=True)

print("\n=== same split, CLEAN payoff T1 (C_t->C_t+1) ===")
for nm, cols in (("8 THINNEST ETFs", thin), ("8 MOST LIQUID ETFs", thick)):
    sh, net_bp, gross_bp, actv = book(R_cc.shift(-1), DEV, cols=cols)
    print(f"  {nm:22} Sharpe {sh:6.2f}  gross/trade {gross_bp:+6.0f}bp", flush=True)

print("\n=== does ANY clean version survive into VAL 2015-2019? ===")
for nm, pay in PAYOFFS.items():
    sh, net_bp, gross_bp, actv = book(pay, VAL)
    print(f"  {nm:46} VAL Sharpe {sh:6.2f}  net/trade {net_bp:+6.0f}bp", flush=True)
print(f"\nexp21 done t={time.time()-t0:.0f}s", flush=True)
