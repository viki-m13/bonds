"""KRAKEN v5: Risk-parity multi-asset trend-following.

Multiple uncorrelated streams:
  - Stocks: TQQQ/UPRO when SPY trends up + breadth + macro OK
  - Bonds: TMF/UBT/TYD when TLT/IEF trends up
  - Gold: UGL/NUGT when GLD trends up
  - Oil: UCO when USO trends up
  - EM: EDC/YINN when EEM trends up

Each asset has its OWN trend filter (not tied to broad regime).
Weight = equal risk across active positions using inverse 63d vol.

The diversification should lift Sharpe meaningfully over pure equity LEV.
"""
import os, json, itertools
import numpy as np
import pandas as pd

REPO   = "/home/user/bonds"
ETF_D  = os.path.join(REPO, "data", "etfs")
FRED_D = os.path.join(REPO, "data", "fred")

SECTORS = ["XLB","XLE","XLF","XLI","XLK","XLP","XLU","XLV","XLY"]

PAIRS = {
    "TQQQ": ("QQQ", 3), "UPRO": ("SPY", 3),
    "QLD":  ("QQQ", 2), "SSO":  ("SPY", 2),
    "SOXL": ("SMH", 3), "TECL": ("XLK", 3),
    "FAS":  ("XLF", 3),
    "ERX":  ("XLE", 2), "DRN":  ("VNQ", 3),
    "EDC":  ("EEM", 3), "YINN": ("FXI", 3),
    "NUGT": ("GLD", 2), "UGL":  ("GLD", 2),
    "UCO":  ("USO", 2), "TMF":  ("TLT", 3),
    "UBT":  ("TLT", 2), "TYD":  ("IEF", 3),
}
LEV_ALL = list(PAIRS.keys())
# Must have >=10 leveraged — this has 17

START = pd.Timestamp("2010-03-11")
IS_END= pd.Timestamp("2018-12-31")
OOS_S = pd.Timestamp("2019-01-01")
OOS_E = pd.Timestamp("2026-04-02")

def load_etf(sym):
    df = pd.read_csv(os.path.join(ETF_D, f"{sym}.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    return df[["Open","Close"]].astype(float)

def load_fred(name):
    df = pd.read_csv(os.path.join(FRED_D, f"{name}.csv"), parse_dates=["Date"]).set_index("Date").sort_index()
    return df[name].astype(float)

underlying_syms = set(u for u,_ in PAIRS.values())
syms = set(LEV_ALL + SECTORS + list(underlying_syms) + ["BIL","SPY","IEF","TLT","GLD","USO","EEM","FXI","SMH","VNQ"])
opens={}; closes={}
for s in syms:
    try:
        df = load_etf(s); opens[s]=df["Open"]; closes[s]=df["Close"]
    except FileNotFoundError: pass
opens = pd.DataFrame(opens); closes = pd.DataFrame(closes)
cal = closes["SPY"].dropna().index
cal = cal[(cal>=START) & (cal<=OOS_E)]
opens = opens.reindex(cal); closes = closes.reindex(cal)
vix = load_fred("VIXCLS").reindex(cal).ffill()
hy  = load_fred("BAMLH0A0HYM2").reindex(cal).ffill()

open_ret = (opens.shift(-1) / opens - 1.0).fillna(0.0)

def metrics(r):
    r = r.dropna()
    if len(r)==0: return dict(sharpe=np.nan,cagr=np.nan,mdd=np.nan,vol=np.nan,navx=np.nan)
    ann=252
    mu=r.mean()*ann; sd=r.std(ddof=0)*np.sqrt(ann)
    sr=mu/sd if sd>0 else float("nan")
    nav=(1+r).cumprod()
    n_years=len(r)/ann
    cagr=nav.iloc[-1]**(1/n_years)-1 if n_years>0 else 0
    dd=nav/nav.cummax()-1
    return dict(sharpe=sr,cagr=cagr,mdd=dd.min(),vol=sd,navx=nav.iloc[-1])

def breadth_sig(ma=200):
    sub = closes[SECTORS]
    mvg = sub.rolling(ma, min_periods=ma).mean()
    above = (sub>mvg).astype(float)
    valid = sub.notna() & mvg.notna()
    num = above.where(valid,0).sum(axis=1)
    den = valid.sum(axis=1).replace(0,np.nan)
    return num/den

def make_state(breadth_hi, vix_max, confirm=3, hy_win=60, ma=200):
    br = breadth_sig(ma=ma)
    hy_ema = hy.ewm(span=hy_win, adjust=False, min_periods=hy_win).mean()
    hy_ok  = (hy<hy_ema)
    vix_ok = (vix<vix_max)
    filt = (br>=breadth_hi) & hy_ok & vix_ok
    raw = filt.astype(int).values
    if confirm<=1: out = raw
    else:
        out = np.empty_like(raw); state=0; streak=0
        for i in range(len(raw)):
            r=raw[i]
            if r==state: streak=0
            else:
                streak+=1
                if streak>=confirm: state=r; streak=0
            out[i]=state
    return pd.Series(out, index=closes.index, dtype=int).shift(1).fillna(0).astype(int)

def build_trend_matrix(universe, use_under, win):
    out = pd.DataFrame(index=closes.index, columns=universe, dtype=float)
    for a in universe:
        src = a
        if use_under and a in PAIRS and PAIRS[a][0] in closes.columns:
            src = PAIRS[a][0]
        p = closes[src]
        ma = p.rolling(win, min_periods=win).mean()
        out[a] = (p > ma).astype(float)
    return out.shift(1).fillna(0.0)

def risk_parity_backtest(universe, trend_mat, regime_mask_per_asset,
                         vol_win=63, rebal_days=21, tc_bps=10, target_vol=None, gate_series=None):
    """
    regime_mask_per_asset: dict {asset: binary series} — asset allowed only if this is 1
    gate_series: optional global state (0/1) that kills all equity-risk positions when 0
    """
    T = len(closes)
    cols = universe + ["BIL"]
    W = np.zeros((T, len(cols)))
    last_w = np.zeros(len(cols))
    rebal_mask = np.zeros(T, dtype=bool)
    for i in range(0, T, rebal_days): rebal_mask[i]=True

    trend_arr = trend_mat.reindex(columns=universe).fillna(0.0).values
    close_arr = closes.reindex(columns=universe).values
    or_mat = open_ret.reindex(columns=cols).values

    # Per-asset vol (use underlier)
    underlier_vol = {}
    for a in universe:
        src = a
        if a in PAIRS and PAIRS[a][0] in closes.columns:
            src = PAIRS[a][0]
        rr = closes[src].pct_change()
        underlier_vol[a] = rr.rolling(vol_win, min_periods=vol_win).std().shift(1).reindex(closes.index).values * PAIRS[a][1]

    regime_arr = None
    if regime_mask_per_asset is not None:
        regime_arr = np.zeros((T, len(universe)))
        for j, a in enumerate(universe):
            regime_arr[:, j] = regime_mask_per_asset.get(a, pd.Series(1.0, index=closes.index)).reindex(closes.index).fillna(0).values

    gate_arr = gate_series.values if gate_series is not None else np.ones(T)

    pnl_out = np.zeros(T); dW_out = np.zeros(T)

    for i in range(T):
        if rebal_mask[i]:
            t_row = trend_arr[i]
            # For equity-group assets, also require global gate on
            valid = np.isfinite(close_arr[i]) & (t_row > 0)
            if regime_arr is not None:
                valid = valid & (regime_arr[i] > 0)
            if valid.any():
                # Inverse-vol within active set
                vols = np.array([underlier_vol[universe[j]][i] if valid[j] else np.nan for j in range(len(universe))])
                inv = np.where(np.isfinite(vols) & (vols>0), 1.0/vols, 0.0)
                s = inv.sum()
                if s > 0:
                    w = inv/s
                    # If a target portfolio vol is set, scale weights
                    if target_vol is not None:
                        # expected portfolio vol ≈ sum(w_i * vol_i) — conservative
                        port_vol = (w * np.where(np.isfinite(vols),vols,0)).sum()
                        if port_vol > 0:
                            scale = min(target_vol / port_vol, 1.0)  # no leverage beyond 1
                            w = w * scale
                    target = np.zeros(len(cols))
                    target[:-1] = w
                    target[-1] = 1.0 - w.sum()
                else:
                    target = np.zeros(len(cols)); target[-1]=1.0
            else:
                target = np.zeros(len(cols)); target[-1]=1.0
        else:
            target = last_w.copy()
            # apply daily trend-exit for held names
            t_row = trend_arr[i]
            held = target[:-1] > 0
            bad = held & ((t_row <= 0) | (~np.isfinite(close_arr[i])))
            if regime_arr is not None:
                bad = bad | (held & (regime_arr[i] <= 0))
            if bad.any():
                target[:-1][bad] = 0.0
                target[-1] = 1.0 - target[:-1].sum()

        dW = np.abs(target - last_w).sum()
        tc = dW * (tc_bps/1e4)
        pnl = (target * or_mat[i]).sum() - tc
        W[i] = target
        last_w = target
        pnl_out[i] = pnl
        dW_out[i] = dW

    r = pd.Series(pnl_out, index=closes.index)
    turn = pd.Series(dW_out, index=closes.index)
    return r, turn, W

def eval_cfg(univ, bh, vxm, cf, tw, rb, eq_gate=True, vol_win=63, target_vol=None, tc_bps=10):
    gate = make_state(bh, vxm, cf) if eq_gate else None
    # per-asset regime mask: equity-risk assets gated by state; bonds/gold/oil are NOT gated by broad regime
    eq_assets = set(["TQQQ","UPRO","QLD","SSO","SOXL","TECL","FAS","ERX","DRN","EDC","YINN"])
    regime_mask = {}
    if gate is not None:
        for a in univ:
            regime_mask[a] = gate if a in eq_assets else pd.Series(1.0, index=closes.index)
    tr = build_trend_matrix(univ, True, tw)
    r, turn, W = risk_parity_backtest(univ, tr, regime_mask if eq_gate else None,
                                       vol_win=vol_win, rebal_days=rb, tc_bps=tc_bps,
                                       target_vol=target_vol)
    is_m = metrics(r[r.index<=IS_END])
    oos_m = metrics(r[r.index>=OOS_S])
    full_m = metrics(r)
    return dict(
        un=len(univ), bh=bh, vxm=vxm, cf=cf, tw=tw, rb=rb, eqg=eq_gate, vw=vol_win, tv=target_vol,
        full_sr=full_m["sharpe"], full_cagr=full_m["cagr"], full_mdd=full_m["mdd"], full_vol=full_m["vol"], full_navx=full_m["navx"],
        is_sr=is_m["sharpe"], is_cagr=is_m["cagr"], is_mdd=is_m["mdd"],
        oos_sr=oos_m["sharpe"], oos_cagr=oos_m["cagr"], oos_mdd=oos_m["mdd"],
        gap=abs(is_m["sharpe"]-oos_m["sharpe"]),
        turn_ann=turn.mean()*252,
    )

if __name__ == "__main__":
    rows = []
    # Broad universe with bonds/gold/oil mixed in
    u1 = LEV_ALL   # 17 names

    grid = list(itertools.product(
        [200, 150, 100],          # trend win
        [5, 21, 10],              # rebal
        [0.5, 0.6],               # bh
        [25, 28],                 # vxm
        [3, 5],                   # confirm
        [True, False],            # equity gate
        [63, 126],                # vol win
        [None, 0.20, 0.40],       # target vol
    ))
    print(f"configs: {len(grid)}")
    for i, (tw, rb, bh, vxm, cf, eqg, vw, tv) in enumerate(grid):
        res = eval_cfg(u1, bh, vxm, cf, tw, rb, eqg, vw, tv)
        rows.append(res)
        if i%50==0: print(f"  {i}/{len(grid)}")

    df = pd.DataFrame(rows)
    df.to_csv("data/results/kraken_grid5.csv", index=False)

    print("\n--- Top 30 by full_sr ---")
    print(df.sort_values("full_sr", ascending=False).head(30).to_string(index=False))

    mask = (df.full_sr>=2.0)&(df.full_cagr>=0.20)&(df.is_sr>=1.5)&(df.oos_sr>=1.5)&(df.gap<=0.5)
    qual = df[mask].sort_values("full_sr", ascending=False)
    print(f"\n--- Qualifying: {len(qual)} ---")
    print(qual.head(30).to_string(index=False))

    mask2 = (df.full_sr>=1.5)&(df.is_sr>=1.2)&(df.oos_sr>=1.2)
    print(f"\n--- sr>=1.5: {len(df[mask2])} ---")
    print(df[mask2].sort_values("full_sr", ascending=False).head(20).to_string(index=False))
