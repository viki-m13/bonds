"""Final ensemble v2 — exposure-aware capital model.

The v1 ensemble (40_ensemble.py) was doubly conservative: sleeve cash
earned nothing AND leverage was financed on notional weights. Here the
account is modeled properly:

  gross_t = k * sum_i w_it * expo_it          (true invested fraction)
  r_t     = k * sum_i w_it * r_it             (trading P&L)
          + max(0, 1-gross_t) * tbill_t       (idle cash in T-bills)
          - max(0, gross_t-1) * (FF+150bp)_t  (margin debit financed)
          - rebalance costs on d|w|/dt (5bp)

Weight schemes (chosen on DEV only):
  ivol   inverse trailing-126d-vol, sum to 1 (v1 scheme)
  evol   each sleeve pre-scaled to 10% vol (cap 3x), then equal weight
k = overall multiplier sweep. Windows: FULL 6-sleeve 2015+, LONG 5-sleeve
2008+ (no NDX-MR). Dev/holdout reported for every config.
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

R = pd.DataFrame({
    "A_mr": pd.read_parquet(os.path.join(OUT, "sleeveA_mr_modes.parquet"))["limit"],
    "B_mom": pd.read_parquet(os.path.join(OUT, "sleeveB_momentum_best.parquet"))["mom_best"],
    "C_dip": pd.read_parquet(os.path.join(OUT, "sleeveC_dip.parquet"))["dip_moc"],
    "D_crisis": pd.read_parquet(os.path.join(OUT, "sleeveD_crisis.parquet"))["crisis"],
    "E_etfmr": pd.read_parquet(os.path.join(OUT, "sleeveE_etfmr.parquet"))["etfmr_moc"],
    "F_tsmom": pd.read_parquet(os.path.join(OUT, "sleeveF_tsmom.parquet"))["tsmom"],
})
E = pd.DataFrame({
    "A_mr": pd.read_parquet(os.path.join(OUT, "sleeveA_mr_expo.parquet"))["limit"],
    "B_mom": pd.read_parquet(os.path.join(OUT, "sleeveB_momentum_expo.parquet"))["mom_best"],
    "C_dip": pd.read_parquet(os.path.join(OUT, "sleeveC_dip_expo.parquet"))["dip_moc"],
    "D_crisis": pd.read_parquet(os.path.join(OUT, "sleeveD_crisis_expo.parquet"))["crisis"],
    "E_etfmr": pd.read_parquet(os.path.join(OUT, "sleeveE_etfmr_expo.parquet"))["etfmr_moc"],
    "F_tsmom": pd.read_parquet(os.path.join(OUT, "sleeveF_tsmom_expo.parquet"))["tsmom"],
})
REB_COST = 5.0 / 1e4


def ensemble(cols, start, scheme="ivol", k=1.0, end=None):
    r = R[cols].loc[start:end].dropna(how="all").fillna(0.0)
    e = E[cols].reindex(r.index).ffill().fillna(0.0)
    vol126 = r.rolling(126, min_periods=60).std() * np.sqrt(252)
    me = r.groupby(r.index.to_period("M")).tail(1).index
    w = pd.DataFrame(index=r.index, columns=cols, dtype=float)
    for d in me:
        v = vol126.loc[d]
        if scheme == "ivol":
            iv = 1.0 / v.replace(0, np.nan)
            w.loc[d] = (iv / iv.sum()).values
        elif scheme == "evol":
            s = (0.10 / v.replace(0, np.nan)).clip(upper=3.0)
            w.loc[d] = (s / len(cols)).values
    w = w.shift(1).ffill()
    trading = (r * w).sum(axis=1) * k
    gross = (e * w).sum(axis=1) * k
    turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0) * k
    rf_d = riskfree_daily(r.index)
    fin_d = financing_daily(r.index, 150)
    port = (trading
            + (1 - gross).clip(lower=0) * rf_d
            - (gross - 1).clip(lower=0) * fin_d
            - turn * REB_COST)
    port = port.iloc[126:]
    return port, gross.iloc[126:]


if __name__ == "__main__":
    t0 = time.time()
    qqq = load_etf("QQQ")["Close"].pct_change().dropna()
    spy = load_etf("SPY")["Close"].pct_change().dropna()

    for name, cols, start, dev_end in [
        ("LONG 5-sleeve (no MR) 2008+", ["B_mom", "C_dip", "D_crisis",
                                         "E_etfmr", "F_tsmom"],
         "2008-02-01", "2017-12-31"),
        ("FULL 6-sleeve 2015+", list(R.columns), "2015-01-02", "2020-12-31"),
    ]:
        print(f"=== {name} ===")
        rf = None
        for scheme in ["ivol", "evol"]:
            for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
                port, gross = ensemble(cols, start, scheme, k)
                if rf is None:
                    rf = riskfree_daily(port.index)
                st = stats(port, rf, f"{scheme} k={k:.1f} (gross avg "
                           f"{gross.mean():.2f})")
                dev = stats(port.loc[:dev_end], rf, "")
                hold = stats(port.loc[dev_end:], rf, "")
                print(fmt(st) + f" | dev {dev['Sharpe_d']:.2f} hold "
                      f"{hold['Sharpe_d']:.2f}")
            print()
        print(fmt(stats(qqq.loc[port.index[0]:], rf, "QQQ same window")))
        print(fmt(stats(spy.loc[port.index[0]:], rf, "SPY same window")))
        print()

    # headline chart: FULL 6-sleeve evol k=2 vs QQQ
    port1, g1 = ensemble(list(R.columns), "2015-01-02", "evol", 1.0)
    port2, g2 = ensemble(list(R.columns), "2015-01-02", "evol", 2.0)
    rf = riskfree_daily(port1.index)
    eq1, eq2 = (1 + port1).cumprod(), (1 + port2).cumprod()
    eqq = (1 + qqq.loc[port1.index[0]:]).cumprod()
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    ax[0].semilogy(eq1, label="6-sleeve evol 1x")
    ax[0].semilogy(eq2, label="6-sleeve evol 2x")
    ax[0].semilogy(eqq, label="QQQ")
    ax[0].legend()
    ax[0].set_title("Honest PIT 6-sleeve ensemble (exposure-aware) vs QQQ")
    for s, lab, a in [(eq1, "1x", .6), (eq2, "2x", .4)]:
        dd = s / s.cummax() - 1
        ax[1].fill_between(dd.index, dd, 0, alpha=a, label=lab)
    ax[1].plot(eqq / eqq.cummax() - 1, lw=.8, label="QQQ")
    ax[1].legend(); ax[1].set_title("drawdown")
    ax[2].plot(g2, lw=.7)
    ax[2].set_title("gross exposure (k=2)")
    plt.tight_layout()
    plt.savefig(os.path.join(CH, "v2_ensemble_final.png"), dpi=110)
    port2.to_frame("evol_k2").to_parquet(os.path.join(OUT, "ensemble_v2.parquet"))
    print(f"chart saved -> charts/v2_ensemble_final.png  t={time.time()-t0:.0f}s")
