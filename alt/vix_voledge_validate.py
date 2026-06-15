"""
Independent validation backtest of the Concretum "Volatility Edge" VIX-ETN strategy
(Strategy 4: eVRP + Backwardation/Contango + VIX-level sizing), exactly as encoded
in the automation notebook the user supplied.

Rules being validated (from the article + the notebook code):
  eRV30  = std(last 10 SPY daily returns, ddof=1) * sqrt(252) * 100      (annualized %)
  eVRP   = VIX - eRV30
  contango      = VIX < VIX3M
  backwardation = VIX > VIX3M
  allocation    = VIX/100                                  ("VIX%")

  R1: eVRP>0  & VIX<VIX3M  -> SHORT vol, full   : SVXY weight = 2 * VIX%   (-0.5x -> -1.0x exposure)
  R2: eVRP<=0 & VIX<VIX3M  -> SHORT vol, half   : SVXY weight = 1 * VIX%   (-> -0.5x exposure)
  R3: eVRP<=0 & VIX>VIX3M  -> LONG  vol, full   : VXX  weight = 1 * VIX%   (+1.0x exposure)
  R4: eVRP>0  & VIX>VIX3M  -> CASH

Timing (no look-ahead): signal computed from data through close of day t; position is
established at the close of day t (the notebook submits MOC orders ~15:45) and earns the
instrument return from close_t -> close_{t+1}.

Because actual SVXY was -1x before ~2018-02-27 and -0.5x after, the literal "2x SVXY"
sizing is only correct post-2018. To validate the *strategy* cleanly over the full
history we also express vol exposure through the underlying short-term VIX-futures index
(VIXY=+1x), where short-vol exposure = -(VIX%) regardless of the SVXY leverage regime.
"""
import numpy as np
import pandas as pd

NY = "America/New_York"

def load():
    spy = pd.read_csv("data/etfs_extended/SPY.csv", parse_dates=["Date"]).set_index("Date")["Close"].sort_index()
    vix = pd.read_csv("data/fred/VIXCLS.csv", parse_dates=["Date"]).set_index("Date")["VIXCLS"].sort_index()
    vix = pd.to_numeric(vix, errors="coerce").dropna()
    v3m = pd.read_csv("/tmp/vix3m.csv", parse_dates=["DATE"]).rename(columns={"DATE": "Date"}).set_index("Date")["CLOSE"].sort_index()
    vixy = pd.read_csv("data/etfs/VIXY.csv", parse_dates=["Date"]).set_index("Date")["Close"].sort_index()  # +1x long-vol (VXX proxy)
    svxy = pd.read_csv("data/etfs/SVXY.csv", parse_dates=["Date"]).set_index("Date")["Close"].sort_index()  # inverse-vol
    rf = pd.read_csv("data/fred/DGS3MO.csv", parse_dates=["Date"]).set_index("Date").iloc[:,0].sort_index()  # 3M T-bill, annualized %
    rf = pd.to_numeric(rf, errors="coerce").dropna()
    df = pd.DataFrame({"spy": spy, "vix": vix, "vix3m": v3m, "vixy": vixy, "svxy": svxy}).dropna(subset=["spy","vix","vix3m"])
    df["rf_daily"] = (rf.reindex(df.index).ffill()/100.0)/252.0
    return df

def signals(df):
    d = df.copy()
    d["spy_ret"] = d["spy"].pct_change()
    # 10-day realized vol of SPY returns (annualized, %), matching notebook (ddof=1)
    d["erv30"] = d["spy_ret"].rolling(10).std(ddof=1) * np.sqrt(252) * 100
    d["evrp"] = d["vix"] - d["erv30"]
    d["contango"] = d["vix"] < d["vix3m"]
    d["backward"] = d["vix"] > d["vix3m"]
    vixpct = d["vix"] / 100.0

    # exposure to the SHORT-TERM VIX FUTURES INDEX (signed): +long vol / -short vol
    # R1 short full -> -VIX% ; R2 short half -> -0.5*VIX% ; R3 long -> +VIX% ; R4 cash 0
    idx_exp = pd.Series(0.0, index=d.index)
    r1 = (d["evrp"] > 0) & d["contango"]
    r2 = (d["evrp"] <= 0) & d["contango"]
    r3 = (d["evrp"] <= 0) & d["backward"]
    idx_exp[r1] = -vixpct[r1]
    idx_exp[r2] = -0.5 * vixpct[r2]
    idx_exp[r3] = +vixpct[r3]
    d["idx_exp"] = idx_exp

    # literal notebook weights (dollar weights in the traded ETFs)
    d["svxy_w"] = np.where(r1, 2*vixpct, np.where(r2, 1*vixpct, 0.0))
    d["vxx_w"]  = np.where(r3, vixpct, 0.0)
    # dollar actually deployed into ETFs; the rest of the account earns the risk-free rate
    d["deployed"] = d["svxy_w"] + d["vxx_w"]

    regime = pd.Series("R4_cash", index=d.index)
    regime[r1] = "R1_short_full"; regime[r2] = "R2_short_half"; regime[r3] = "R3_long"
    d["regime"] = regime
    return d

def metrics(ret, name):
    ret = ret.dropna()
    if len(ret) == 0: return {}
    eq = (1+ret).cumprod()
    yrs = len(ret)/252
    cagr = eq.iloc[-1]**(1/yrs)-1
    vol = ret.std()*np.sqrt(252)
    sharpe = (ret.mean()*252)/vol if vol>0 else np.nan
    dd = (eq/eq.cummax()-1).min()
    downside = ret[ret<0].std()*np.sqrt(252)
    sortino = (ret.mean()*252)/downside if downside>0 else np.nan
    return dict(name=name, start=str(ret.index[0].date()), end=str(ret.index[-1].date()),
                n=len(ret), CAGR=cagr, Vol=vol, Sharpe=sharpe, Sortino=sortino,
                MaxDD=dd, total=eq.iloc[-1]-1)

def run(d, instr_ret_index, label, cost_bps=0.0, cash=True):
    """instr_ret_index: forward return of the +1x vol index (vixy returns), aligned.
    strategy day-t+1 return = idx_exp_t * idx_ret_{t->t+1} + idle_cash_t * rf_{t+1} - costs."""
    fwd = instr_ret_index.shift(-1)               # close_t -> close_{t+1}
    gross = (d["idx_exp"] * fwd)
    if cash:
        idle = (1.0 - d["deployed"]).clip(lower=0.0)
        gross = gross + idle * d["rf_daily"].shift(-1)
    # transaction cost: charge on change in absolute index-equivalent exposure
    turn = d["idx_exp"].diff().abs().fillna(d["idx_exp"].abs())
    cost = turn * (cost_bps/1e4)
    net = (gross - cost).rename(label)
    return net

if __name__ == "__main__":
    pd.set_option("display.width", 160, "display.max_columns", 30)
    df = load()
    print("Loaded ranges:")
    for c in df.columns:
        s = df[c].dropna(); print(f"  {c:6s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}")

    d = signals(df)
    d = d.dropna(subset=["erv30","evrp"])

    # regime distribution
    print("\nRegime distribution (full period with VIX3M, from {} ):".format(d.index.min().date()))
    print((d["regime"].value_counts(normalize=True)*100).round(1).to_string())

    vixy_ret = df["vixy"].pct_change()
    svxy_ret = df["svxy"].pct_change()

    # ---- Primary: index-equivalent exposure (valid across SVXY leverage regimes) ----
    rows = []
    for bps in (0, 5, 10):
        net = run(d, vixy_ret, f"VolEdge_{bps}bps", cost_bps=bps)
        rows.append(metrics(net, f"VolEdge (idx-equiv, {bps}bps/side)"))
    # SPY buy&hold over same window
    spy_bh = df["spy"].pct_change().reindex(d.index)
    rows.append(metrics(spy_bh, "SPY buy&hold (same window)"))

    res = pd.DataFrame(rows).set_index("name")
    pd.options.display.float_format = lambda x: f"{x:,.3f}"
    print("\n================ FULL-PERIOD RESULTS (index-equivalent vol exposure) ================")
    print(res.to_string())

    # ---- Literal notebook sizing on ACTUAL SVXY/VIXY, post-2018-02-27 (SVXY=-0.5x) ----
    d2 = d[d.index >= "2018-03-01"].copy()
    fwd_vixy = vixy_ret.shift(-1)
    fwd_svxy = svxy_ret.shift(-1)
    idle2 = (1.0 - d2["deployed"]).clip(lower=0.0)
    lit = (d2["svxy_w"]*fwd_svxy.reindex(d2.index) + d2["vxx_w"]*fwd_vixy.reindex(d2.index)
           + idle2*d2["rf_daily"].shift(-1)).rename("literal")
    print("\n========= LITERAL NOTEBOOK SIZING on ACTUAL ETFs (2018-03 -> SVXY end) =========")
    print(pd.DataFrame([
        metrics(lit, "VolEdge literal (real SVXY/VIXY, 2x SVXY)"),
        metrics(run(d2, vixy_ret, "x", 0), "VolEdge idx-equiv (same window)"),
        metrics(df["spy"].pct_change().reindex(d2.index), "SPY buy&hold (same window)"),
    ]).set_index("name").to_string())

    # ---- Per-year returns (index-equivalent, 5bps) ----
    net5 = run(d, vixy_ret, "n", cost_bps=5).dropna()
    yr = (1+net5).groupby(net5.index.year).prod()-1
    spyyr = (1+spy_bh.dropna()).groupby(spy_bh.dropna().index.year).prod()-1
    comp = pd.DataFrame({"VolEdge_5bps": yr, "SPY": spyyr}).dropna(how="all")
    print("\n================ CALENDAR-YEAR RETURNS ================")
    print((comp*100).round(1).to_string())

    # sanity: worst single days of the strategy
    print("\nWorst 6 strategy days (idx-equiv, 5bps):")
    print((net5.sort_values().head(6)*100).round(2).to_string())
