"""Invention track H: ONE netted book from a composite alpha.

Composite daily z (PIT):  zC = w_rev * z_resrev5 + w_tail * z_tail1 + w_ll * z_leadlag
 - z_resrev5: negative 5d residual sum, vol-scaled (from cached resid matrix)
 - z_tail1:  negative 1d residual z clipped beyond |2| (tail kicker)
 - z_ll:     leader-catchup (mega-cap composite 5d minus own 5d, z-scored)
Book: patient hysteresis (enter |z|>zin, exit |z|<zout, partial-adjust lam),
top-K cap per side, EMA smoothing of the composite, twice-weekly trade batching.
Sweep the small grid, DEV only. All configs logged.
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import engine as G

t0 = time.time()
PX, DV, ELIG, R = G.load()
DEV_END = pd.Timestamp("2014-12-31")
idx = R.index
resid = pd.read_pickle("/tmp/sharpe3_work/_resid_daily.pkl")
E_d = ELIG.reindex(idx).ffill().fillna(False)[R.columns].astype(bool)
rvol = resid.rolling(63, min_periods=40).std().shift(1)
z1 = (resid / rvol).astype(np.float32).where(E_d)
z5 = (resid.rolling(5).sum() / (rvol * np.sqrt(5))).astype(np.float32).where(E_d)

# leader catchup
L = np.log1p(R)
r5 = np.expm1(L.rolling(5).sum())
DV63 = DV.rolling(63, min_periods=40).median()
lead100 = DV63.rank(axis=1, ascending=False) <= 100
lead_ret = (r5.where(lead100)).mean(axis=1)
zll_raw = -(r5.sub(lead_ret, axis=0))            # own below leaders -> positive
zll = (zll_raw.sub(zll_raw.mean(axis=1), axis=0)).div(zll_raw.std(axis=1) + 1e-12, axis=0)
zll = zll.astype(np.float32).where(E_d)
print(f"signals ready t={time.time()-t0:.0f}s", flush=True)

dev_mask = (idx >= pd.Timestamp("1996-06-01")) & (idx <= DEV_END)

def run_cfg(w_rev, w_tail, w_ll, ema, zin, zout, lam, K, batch=("Tue", "Fri"), name=""):
    zC = (-z5 * w_rev) + (-(z1.clip(-6, 6).where(z1.abs() > 2).fillna(0.0)) * w_tail) + (zll * w_ll)
    if ema > 1:
        zC = zC.ewm(span=ema, min_periods=1).mean()
    Zv = zC.values
    days = pd.Series(idx.day_name(), index=idx).str[:3]
    trade_day = days.isin(batch).values
    Wv = np.zeros_like(Zv, dtype=np.float64)
    cur = np.zeros(Zv.shape[1])
    for ti in range(len(idx)):
        if dev_mask[ti] and trade_day[ti]:
            z = Zv[ti].copy()
            z[~np.isfinite(z)] = 0.0
            tgt = np.zeros_like(cur)
            # keep current names until they fall below zout; enter above zin
            enter = np.abs(z) >= zin
            keep = (np.abs(z) >= zout) & (cur != 0) & (np.sign(z) == np.sign(cur))
            active = enter | keep
            if active.any():
                za = np.where(active, z, 0.0)
                # cap K per side by |z|
                order = np.argsort(-np.abs(za))
                sel = np.zeros_like(active)
                nl = ns = 0
                for ni in order:
                    if za[ni] == 0: break
                    if za[ni] > 0 and nl < K: sel[ni] = True; nl += 1
                    elif za[ni] < 0 and ns < K: sel[ni] = True; ns += 1
                za = np.where(sel, za, 0.0)
                pos = za.clip(0, None); neg = (-za).clip(0, None)
                t_ = np.zeros_like(za)
                if pos.sum() > 0: t_ += 0.5 * pos / pos.sum()
                if neg.sum() > 0: t_ -= 0.5 * neg / neg.sum()
                tgt = t_
            cur = cur + lam * (tgt - cur)
            g = np.abs(cur).sum()
            if g > 1e-9: cur = cur / g
        Wv[ti] = cur
    W = pd.DataFrame(Wv, index=idx, columns=R.columns)
    net, gross, tno = G.run(W, R, lag=2)
    rep = G.report(net[:DEV_END], gross[:DEV_END], tno[:DEV_END], name)
    print(G.fmt(rep), flush=True)
    return rep, net[:DEV_END]

cfgs = [
    (1.0, 0.0, 0.0, 1, 1.5, 0.5, 0.4, 60, "rev only"),
    (1.0, 0.5, 0.0, 1, 1.5, 0.5, 0.4, 60, "rev+tail"),
    (1.0, 0.5, 0.5, 1, 1.5, 0.5, 0.4, 60, "rev+tail+ll"),
    (1.0, 0.5, 0.5, 3, 1.5, 0.5, 0.4, 60, "composite ema3"),
    (1.0, 0.5, 0.5, 3, 2.0, 0.5, 0.4, 40, "composite deep k40"),
    (1.0, 0.5, 0.5, 3, 1.2, 0.3, 0.25, 80, "composite wide slow"),
]
best = None
for w_rev, w_tail, w_ll, ema, zin, zout, lam, K, nm in cfgs:
    rep, net = run_cfg(w_rev, w_tail, w_ll, ema, zin, zout, lam, K, name=f"cmp {nm}")
    if best is None or rep["sharpe_net"] > best[0]["sharpe_net"]:
        best = (rep, net, nm)
best[1].to_pickle("/tmp/sharpe3_work/sleeve_composite.pkl")
print("best:", best[2], flush=True)
print(f"exp09 done t={time.time()-t0:.0f}s", flush=True)
