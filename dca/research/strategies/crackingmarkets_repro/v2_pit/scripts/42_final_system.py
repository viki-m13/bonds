"""FINAL SYSTEM — honest PIT sleeves with dev-picked configs, exposure-aware
ensemble, leverage frontier, and the correlation ceiling math.

Dev-picked sleeve upgrades vs the baseline scripts (selection on dev only):
  A  MR limit pf30 cash-capped   (dev 1.02 / hold 1.09; beats pf20 all axes)
  C  dip hold=3, maxpos=20, pf5  (dev 0.93 / hold 0.32; beats hold5 both)
Everything else as in 30/23/31/32.
"""
import sys, os, time, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lib import load_etf, riskfree_daily, financing_daily, stats, fmt, OUT

CH = os.path.join(os.path.dirname(OUT), "charts")


def load_mod(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


t0 = time.time()
mr = load_mod("mr", "10_mr_pit_execution.py")
eqA, trA, expoA = mr.run("limit", pos_frac=0.30)
rA = eqA.pct_change().dropna()
dip = load_mod("dip", "22_buythedip_clean.py")
eqC, trC, expoC = dip.run(lag=0, hold=3, maxpos=20, pos_frac=0.05)
rC = eqC.pct_change().dropna()
print(f"sleeves A/C rebuilt t={time.time()-t0:.0f}s")

R = pd.DataFrame({
    "A_mr": rA,
    "B_mom": pd.read_parquet(os.path.join(OUT, "sleeveB_momentum_best.parquet"))["mom_best"],
    "C_dip": rC,
    "D_crisis": pd.read_parquet(os.path.join(OUT, "sleeveD_crisis.parquet"))["crisis"],
    "E_etfmr": pd.read_parquet(os.path.join(OUT, "sleeveE_etfmr.parquet"))["etfmr_moc"],
    "F_tsmom": pd.read_parquet(os.path.join(OUT, "sleeveF_tsmom.parquet"))["tsmom"],
})
E = pd.DataFrame({
    "A_mr": expoA,
    "B_mom": pd.read_parquet(os.path.join(OUT, "sleeveB_momentum_expo.parquet"))["mom_best"],
    "C_dip": expoC,
    "D_crisis": pd.read_parquet(os.path.join(OUT, "sleeveD_crisis_expo.parquet"))["crisis"],
    "E_etfmr": pd.read_parquet(os.path.join(OUT, "sleeveE_etfmr_expo.parquet"))["etfmr_moc"],
    "F_tsmom": pd.read_parquet(os.path.join(OUT, "sleeveF_tsmom_expo.parquet"))["tsmom"],
})
R.to_parquet(os.path.join(OUT, "final_sleeve_returns.parquet"))
E.to_parquet(os.path.join(OUT, "final_sleeve_expo.parquet"))
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
        else:
            s = (0.10 / v.replace(0, np.nan)).clip(upper=3.0)
            w.loc[d] = (s / len(cols)).values
    w = w.shift(1).ffill()
    trading = (r * w).sum(axis=1) * k
    gross = (e * w).sum(axis=1) * k
    turn = (w - w.shift(1)).abs().sum(axis=1).fillna(0) * k
    rf_d = riskfree_daily(r.index)
    fin_d = financing_daily(r.index, 150)
    port = (trading + (1 - gross).clip(lower=0) * rf_d
            - (gross - 1).clip(lower=0) * fin_d - turn * REB_COST)
    return port.iloc[126:], gross.iloc[126:]


if __name__ == "__main__":
    qqq = load_etf("QQQ")["Close"].pct_change().dropna()

    print("\n=== per-sleeve honest stats (full own windows) ===")
    for c in R.columns:
        r = R[c].dropna()
        rf = riskfree_daily(r.index)
        print(fmt(stats(r, rf, c)) + f"  avg expo "
              f"{E[c].reindex(r.index).mean()*100:3.0f}%")

    common = R.loc["2015-07-01":].dropna()
    corr = common.corr()
    print("\n=== daily correlations 2015-07+ ===")
    print(corr.round(2).to_string())
    off = corr.values[np.triu_indices(len(corr), 1)]
    rho = off.mean()
    shs = [stats(R[c].dropna(), riskfree_daily(R[c].dropna().index), "")["Sharpe_d"]
           for c in R.columns]
    sbar = np.mean(shs)
    print(f"\navg pairwise corr {rho:.2f}; avg sleeve Sharpe {sbar:.2f}; "
          f"N->inf ensemble Sharpe ceiling = s/sqrt(rho) = {sbar/np.sqrt(rho):.2f}")

    for name, cols, start, dev_end in [
        ("LONG 5-sleeve 2008+", ["B_mom", "C_dip", "D_crisis", "E_etfmr",
                                 "F_tsmom"], "2008-02-01", "2017-12-31"),
        ("FULL 6-sleeve 2015+", list(R.columns), "2015-01-02", "2020-12-31"),
    ]:
        print(f"\n=== {name} (exposure-aware, costed, financed) ===")
        rf = None
        for scheme in ["ivol", "evol"]:
            for k in [1.0, 2.0, 3.0, 4.0]:
                port, gross = ensemble(cols, start, scheme, k)
                if rf is None:
                    rf = riskfree_daily(port.index)
                st = stats(port, rf, f"{scheme} k={k:.1f} gross~{gross.mean():.2f}")
                dev = stats(port.loc[:dev_end], rf, "")
                hold = stats(port.loc[dev_end:], rf, "")
                print(fmt(st) + f" | dev {dev['Sharpe_d']:.2f} hold "
                      f"{hold['Sharpe_d']:.2f}")
        print(fmt(stats(qqq.loc[port.index[0]:], rf, "QQQ same window")))

    # ---- charts -------------------------------------------------------------
    port1, _ = ensemble(list(R.columns), "2015-01-02", "ivol", 1.0)
    port2, g2 = ensemble(list(R.columns), "2015-01-02", "ivol", 2.0)
    rf = riskfree_daily(port1.index)
    eq1, eq2 = (1 + port1).cumprod(), (1 + port2).cumprod()
    eqq = (1 + qqq.loc[port1.index[0]:]).cumprod()
    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    ax[0].semilogy(eq1, label=f"6-sleeve ivol 1x")
    ax[0].semilogy(eq2, label=f"6-sleeve ivol 2x financed")
    ax[0].semilogy(eqq, label="QQQ")
    ax[0].legend(); ax[0].set_title(
        "FINAL honest PIT system vs QQQ, 2015-2026 (all costs, PIT universe)")
    for s, lab, a in [(eq1, "1x", .6), (eq2, "2x", .4)]:
        dd = s / s.cummax() - 1
        ax[1].fill_between(dd.index, dd, 0, alpha=a, label=lab)
    ax[1].plot(eqq / eqq.cummax() - 1, lw=.8, label="QQQ", c="g")
    ax[1].legend(); ax[1].set_title("drawdown")
    rs = port1.rolling(252).apply(lambda x: x.mean() / x.std() * np.sqrt(252))
    ax[2].plot(rs); ax[2].axhline(2, ls="--", c="r"); ax[2].axhline(1, ls=":")
    ax[2].set_title("rolling 12m Sharpe (1x)")
    plt.tight_layout()
    plt.savefig(os.path.join(CH, "v2_final_system.png"), dpi=110)

    rows = []
    for k in np.arange(1.0, 4.51, 0.25):
        pl, g = ensemble(list(R.columns), "2015-01-02", "ivol", k)
        st = stats(pl, rf, "")
        rows.append((k, g.mean(), st["CAGR"], st["Sharpe_d"], st["Sharpe_m"],
                     st["maxDD"]))
    fr = pd.DataFrame(rows, columns=["k", "gross", "CAGR", "Sh_d", "Sh_m",
                                     "maxDD"])
    print("\n=== leverage frontier, FULL 6-sleeve ivol 2015+ ===")
    print(fr.round(3).to_string(index=False))
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(fr.k, fr.CAGR * 100, "o-", label="CAGR %")
    ax1.axhline(20, ls="--", c="g", lw=.8)
    ax2 = ax1.twinx()
    ax2.plot(fr.k, fr.Sh_d, "s-", c="orange", label="Sharpe (daily)")
    ax2.axhline(2, ls="--", c="r", lw=.8)
    ax1.set_xlabel("k (multiplier; financed at FF+150bp)")
    ax1.set_ylabel("CAGR %"); ax2.set_ylabel("Sharpe")
    fig.legend(loc="upper left")
    plt.title("Leverage frontier — the honest CAGR/Sharpe trade-off")
    plt.tight_layout()
    plt.savefig(os.path.join(CH, "v2_leverage_frontier.png"), dpi=110)
    print(f"\ncharts saved t={time.time()-t0:.0f}s")
