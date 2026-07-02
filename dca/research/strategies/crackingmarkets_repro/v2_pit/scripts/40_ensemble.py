"""Final ensemble — risk-parity across the honest, costed sleeves, with
inter-sleeve rebalance costs, leverage financed at FEDFUNDS+150bp, and
dev/holdout evaluation. Answers: what CAGR/Sharpe is honestly reachable?

Sleeves (all survivorship-clean / PIT, all costed, best-honest execution):
  A  NDX-100 PIT Connors MR, limit entries          (2015-01+)
  B  clean-universe momentum, dev-picked variant    (2001-02+)
  C  clean-universe buy-the-dip RSI(5), MOC         (2000-02+)
  D  crisis-alpha TLT/IEF/GLD trend                 (2005-12+)
  E  SPY+QQQ RSI(2) ETF mean reversion, MOC         (2005-10+)
  F  15-ETF time-series momentum, long-flat         (2008-02+)

Weights: inverse trailing-126d vol, recomputed monthly, applied NEXT day
(PIT). Inter-sleeve rebalance turnover charged 5bp. Leverage sweep with
financing; vol-target variant reported too.
"""
import sys, os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_etf, riskfree_daily, financing_daily, stats, fmt, OUT

CH = os.path.join(os.path.dirname(OUT), "charts")
os.makedirs(CH, exist_ok=True)

S = {}
S["A_mr"] = pd.read_parquet(os.path.join(OUT, "sleeveA_mr_modes.parquet"))["limit"]
S["B_mom"] = pd.read_parquet(os.path.join(OUT, "sleeveB_momentum_best.parquet"))["mom_best"]
S["C_dip"] = pd.read_parquet(os.path.join(OUT, "sleeveC_dip.parquet"))["dip_moc"]
S["D_crisis"] = pd.read_parquet(os.path.join(OUT, "sleeveD_crisis.parquet"))["crisis"]
S["E_etfmr"] = pd.read_parquet(os.path.join(OUT, "sleeveE_etfmr.parquet"))["etfmr_moc"]
S["F_tsmom"] = pd.read_parquet(os.path.join(OUT, "sleeveF_tsmom.parquet"))["tsmom"]
R = pd.DataFrame(S)

REB_COST = 5.0 / 1e4


def ensemble(cols, start, end=None, lev=1.0, voltarget=None, fin_spread=150):
    r = R[cols].loc[start:end].dropna(how="all").fillna(0.0)
    vol126 = r.rolling(126, min_periods=60).std()
    me = r.groupby(r.index.to_period("M")).tail(1).index
    w = pd.DataFrame(index=r.index, columns=cols, dtype=float)
    for d in me:
        v = vol126.loc[d]
        iv = 1.0 / v.replace(0, np.nan)
        iv = iv / iv.sum()
        w.loc[d] = iv.values
    w = w.shift(1).ffill()
    port = (r * w).sum(axis=1)
    # inter-sleeve rebalance cost
    turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0)
    port -= turn * REB_COST
    port = port.iloc[126:]                     # weight warmup
    if voltarget:
        rv = port.rolling(63, min_periods=40).std() * np.sqrt(252)
        scale = (voltarget / rv).clip(upper=3.0).shift(1).fillna(1.0)
    else:
        scale = pd.Series(lev, index=port.index)
    fin = financing_daily(port.index, fin_spread)
    excess_lev = (scale - 1.0).clip(lower=0.0)
    port_lev = port * scale - excess_lev * fin
    return port_lev, scale


if __name__ == "__main__":
    t0 = time.time()
    qqq = load_etf("QQQ")["Close"].pct_change().dropna()
    spy = load_etf("SPY")["Close"].pct_change().dropna()

    print("=== sleeve correlations (2015-07 .. 2026-06, daily) ===")
    common = R.loc["2015-07-01":].dropna(how="any")
    print(common.corr().round(2).to_string())
    print()

    for name, cols, start, dev_end in [
        ("LONG 5-sleeve (no MR) 2008+", ["B_mom", "C_dip", "D_crisis",
                                         "E_etfmr", "F_tsmom"],
         "2008-02-01", "2017-12-31"),
        ("FULL 6-sleeve 2015+", list(R.columns), "2015-01-02", "2020-12-31"),
    ]:
        print(f"=== {name} ===")
        port, _ = ensemble(cols, start)
        rf = riskfree_daily(port.index)
        print(fmt(stats(port, rf, "unlevered")))
        print(fmt(stats(port.loc[:dev_end], rf, "  dev half")))
        print(fmt(stats(port.loc[dev_end:], rf, "  holdout half")))
        for lev in [1.5, 2.0, 2.5, 3.0]:
            pl, _ = ensemble(cols, start, lev=lev)
            print(fmt(stats(pl, rf, f"levered {lev:.1f}x (fin FF+150)")))
        pv, sc = ensemble(cols, start, voltarget=0.15)
        print(fmt(stats(pv, rf, f"vol-target 15% (avg lev {sc.mean():.2f})")))
        print(fmt(stats(qqq.loc[port.index[0]:], rf, "QQQ same window")))
        print(fmt(stats(spy.loc[port.index[0]:], rf, "SPY same window")))
        print()

    # ---- charts for the FULL 6-sleeve system --------------------------------
    port, _ = ensemble(list(R.columns), "2015-01-02")
    p2, _ = ensemble(list(R.columns), "2015-01-02", lev=2.0)
    rf = riskfree_daily(port.index)
    eq = (1 + port).cumprod()
    eq2 = (1 + p2).cumprod()
    eqq = (1 + qqq.loc[port.index[0]:]).cumprod()
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    ax[0].semilogy(eq, label="6-sleeve 1x")
    ax[0].semilogy(eq2, label="6-sleeve 2x (financed)")
    ax[0].semilogy(eqq, label="QQQ")
    ax[0].legend(); ax[0].set_title("Honest PIT ensemble vs QQQ (2015-2026)")
    dd = eq / eq.cummax() - 1
    dd2 = eq2 / eq2.cummax() - 1
    ddq = eqq / eqq.cummax() - 1
    ax[1].fill_between(dd.index, dd, 0, alpha=.6, label="1x")
    ax[1].fill_between(dd2.index, dd2, 0, alpha=.4, label="2x")
    ax[1].plot(ddq, lw=.8, label="QQQ")
    ax[1].legend(); ax[1].set_title("drawdown")
    rs = port.rolling(252).mean() / port.rolling(252).std() * np.sqrt(252)
    ax[2].plot(rs); ax[2].axhline(2, ls="--", c="r")
    ax[2].set_title("rolling 12m Sharpe (1x)")
    plt.tight_layout()
    plt.savefig(os.path.join(CH, "v2_ensemble.png"), dpi=110)

    # leverage frontier
    rows = []
    for lev in np.arange(1.0, 3.51, 0.25):
        pl, _ = ensemble(list(R.columns), "2015-01-02", lev=lev)
        st = stats(pl, rf, f"{lev}x")
        rows.append((lev, st["CAGR"], st["Sharpe_d"], st["maxDD"]))
    fr = pd.DataFrame(rows, columns=["lev", "CAGR", "Sharpe", "maxDD"])
    print(fr.round(3).to_string(index=False))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(fr.lev, fr.CAGR * 100, "o-", label="CAGR %")
    ax1.axhline(20, ls="--", c="g")
    ax2 = ax1.twinx()
    ax2.plot(fr.lev, fr.Sharpe, "s-", c="orange", label="Sharpe(d)")
    ax2.axhline(2, ls="--", c="r")
    ax1.set_xlabel("leverage (financed at FF+150bp)")
    ax1.set_ylabel("CAGR %"); ax2.set_ylabel("daily Sharpe")
    fig.legend(loc="upper left")
    plt.title("Leverage frontier — honest 6-sleeve ensemble 2015-2026")
    plt.tight_layout()
    plt.savefig(os.path.join(CH, "v2_leverage_frontier.png"), dpi=110)
    port.to_frame("ens1x").to_parquet(os.path.join(OUT, "ensemble_final.parquet"))
    print(f"charts saved. t={time.time()-t0:.0f}s")
