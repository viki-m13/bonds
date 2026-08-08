"""SHARPE3 shared engine: honest daily long-short backtester + nulls.

Conventions (see README honesty contract):
- Signals use data through close t. Execution at close t+1. Position earns the
  t+1 -> t+2 return. Implemented as W.shift(2) on the daily grid.
- Weights: gross exposure normalized to 1 (|W| sums to 1). Sharpe is therefore
  per unit of gross capital, comparable across strategies. No cash yield.
- Costs: FEE_BPS per side on traded notional; 25 bps/yr borrow on short book.
- Delisting: held name with NaN return -> 0 return that day, position freed.
"""
import os, time
import numpy as np, pandas as pd

HERE = os.environ.get("SHARPE3_WORK", "/tmp/sharpe3_work")
FEE_BPS = 10.0
BORROW_BPS_Y = 25.0

_cache = {}
def _clean_returns(R):
    """Remove data glitches, not outcomes:
    - |r| > 500% in one day on a $10M-ADV name is treated as a bad print
    - spike-reversal pairs (huge move immediately ~fully reversed) are bad prints
    Both the spike day and the day after (computed off the bad price) are voided.
    """
    Rv = R.values
    nxt = np.vstack([Rv[1:], np.full((1, Rv.shape[1]), np.nan)])
    with np.errstate(invalid="ignore"):
        bad_abs = np.abs(Rv) > 5.0
        rt = (1 + Rv) * (1 + nxt) - 1
        spike_up = (Rv > 1.0) & (nxt < -0.4) & (np.abs(rt) < 0.25)
        spike_dn = (Rv < -0.5) & (nxt > 0.8) & (np.abs(rt) < 0.25)
    bad = bad_abs | spike_up | spike_dn
    badnext = np.vstack([np.zeros((1, Rv.shape[1]), dtype=bool), bad[:-1]])
    n = int(bad.sum() + (badnext & ~bad).sum())
    Rv = Rv.copy()
    Rv[bad | badnext] = np.nan
    print(f"[engine] cleaned {n} glitch return cells ({n/np.isfinite(Rv).sum()*1e4:.1f} per 10k)", flush=True)
    return pd.DataFrame(Rv, index=R.index, columns=R.columns)

def load():
    if "PX" not in _cache:
        _cache["PX"] = pd.read_pickle(f"{HERE}/_px_daily.pkl")
        _cache["DV"] = pd.read_pickle(f"{HERE}/_dv_daily.pkl")
        # eligibility rebuilt WITHOUT the adjusted-price filter (adjusted price
        # thresholds are not PIT-pure); pure raw dollar-volume floor instead.
        p2 = f"{HERE}/_elig2.pkl"
        if os.path.exists(p2):
            _cache["ELIG"] = pd.read_pickle(p2)
        else:
            med = _cache["DV"].rolling(63, min_periods=40).median()
            _cache["ELIG"] = (med >= 1e7).resample("ME").last().fillna(False)
            _cache["ELIG"].to_pickle(p2)
            print(f"[engine] rebuilt ELIG (DV-only): avg eligible/mo {float(_cache['ELIG'].sum(axis=1).mean()):.0f}", flush=True)
        p = f"{HERE}/_r_clean.pkl"
        if os.path.exists(p):
            _cache["R"] = pd.read_pickle(p)
        else:
            _cache["R"] = _clean_returns(_cache["PX"].pct_change(fill_method=None))
            _cache["R"].to_pickle(p)
    return _cache["PX"], _cache["DV"], _cache["ELIG"], _cache["R"]

def run(W, R, fee_bps=FEE_BPS, borrow_bps_y=BORROW_BPS_Y, lag=2):
    """W: DataFrame of signal-date weights (already gross-normalized).
    Returns (daily net returns, daily gross returns, daily turnover)."""
    W = W.reindex(R.index).shift(lag)
    Wv = W.fillna(0.0).values
    Rv = R[W.columns].values.copy()
    held_nan = (~np.isfinite(Rv)) & (Wv != 0)
    Rv[~np.isfinite(Rv)] = 0.0
    gross = (Wv * Rv).sum(axis=1)
    dW = np.abs(np.diff(Wv, axis=0, prepend=np.zeros((1, Wv.shape[1])))).sum(axis=1)
    short = np.clip(-Wv, 0, None).sum(axis=1)
    net = gross - dW * fee_bps / 1e4 - short * borrow_bps_y / 1e4 / 252
    idx = W.index
    return (pd.Series(net, idx), pd.Series(gross, idx), pd.Series(dW, idx))

def sharpe(r):
    r = r.dropna()
    r = r[r.index >= r[r != 0].index.min()] if (r != 0).any() else r
    if len(r) < 60 or r.std() == 0: return np.nan
    return float(r.mean() / r.std() * np.sqrt(252))

def report(net, gross, tno, name="", splits=(("DEV", "1995", "2014"), ("VAL", "2015", "2019"), ("TEST", "2020", "2027"))):
    out = {"name": name, "sharpe_net": sharpe(net), "sharpe_gross": sharpe(gross),
           "turnover_day": float(tno.mean()), "ann_ret_net": float(net.mean()*252),
           "maxdd": float((net.cumsum() - net.cumsum().cummax()).min())}
    for nm, a, b in splits:
        out[f"sharpe_{nm}"] = sharpe(net[a:b])
    yr = net.groupby(net.index.year).apply(sharpe)
    out["worst_year_sharpe"] = float(yr.min()) if len(yr) else np.nan
    out["years_pos"] = f"{int((net.groupby(net.index.year).sum()>0).sum())}/{net.index.year.nunique()}"
    return out

def fmt(rep):
    return (f"{rep['name']:34} Snet {rep['sharpe_net']:5.2f}  Sgross {rep['sharpe_gross']:5.2f}  "
            f"DEV {rep.get('sharpe_DEV', float('nan')):5.2f}  VAL {rep.get('sharpe_VAL', float('nan')):5.2f}  "
            f"tno {rep['turnover_day']:.3f}/d  ret {rep['ann_ret_net']*100:5.1f}%  wyS {rep['worst_year_sharpe']:5.2f}  yrs+ {rep['years_pos']}")

def month_ends(index):
    s = pd.Series(1, index=index)
    return s.groupby(index.to_period("M")).apply(lambda x: x.index[-1]).values

def week_ends(index):
    s = pd.Series(1, index=index)
    return s.groupby(index.to_period("W")).apply(lambda x: x.index[-1]).values

def elig_on(dates, ELIG):
    """PIT eligibility as of each rebalance date: last month-end ELIG row at/before date."""
    E = ELIG.reindex(ELIG.index.union(pd.DatetimeIndex(dates))).ffill().loc[pd.DatetimeIndex(dates)]
    return E.fillna(False)

def normalize_ls(sig, topq=0.1, botq=0.1):
    """Rank signal row -> dollar-neutral weights, gross 1. sig: Series (one date)."""
    s = sig.dropna()
    if len(s) < 50: return {}
    lo, hi = s.quantile([botq, 1-topq])
    longs = s[s >= hi].index; shorts = s[s <= lo].index
    w = {}
    for t in longs: w[t] = 0.5/len(longs)
    for t in shorts: w[t] = -0.5/len(shorts)
    return w

def build_W(dates, weight_fn, columns):
    """weight_fn(date) -> {ticker: weight}. Returns sparse weight DataFrame."""
    rows = []
    for d in dates:
        w = weight_fn(d)
        rows.append(pd.Series(w, name=d))
    W = pd.DataFrame(rows).reindex(columns=columns).fillna(0.0)
    W.index = pd.DatetimeIndex([r.name for r in rows] if rows else [])
    return W

def null_sharpes(dates, ELIG, R, n_long, n_short, K=50, seed=0, fee_bps=FEE_BPS):
    """Random strategies matched on universe/position count/rebalance grid."""
    rng = np.random.default_rng(seed)
    E = elig_on(dates, ELIG)
    out = []
    cols = R.columns
    for k in range(K):
        rows = []
        for d in dates:
            e = E.loc[d]
            pool = e[e].index.intersection(cols)
            if len(pool) < n_long + n_short: rows.append(pd.Series({}, name=d)); continue
            pick = rng.choice(pool, n_long + n_short, replace=False)
            w = {t: 0.5/n_long for t in pick[:n_long]}
            w.update({t: -0.5/n_short for t in pick[n_long:]})
            rows.append(pd.Series(w, name=d))
        W = pd.DataFrame(rows).reindex(columns=cols).fillna(0.0)
        net, _, _ = run(W, R, fee_bps=fee_bps)
        out.append(sharpe(net))
    return np.array(out)
