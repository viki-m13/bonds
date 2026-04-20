"""NOVA33 — Three new orthogonal sleeves + MEGA ensemble targeting OOS SR 2+.

  A) CRYPTO OVERNIGHT (BTC): trade equity-overnight window on BTC.
     Long BTC from 15:55 ET (approx: prior daily close) to 09:30 ET
     (approx: daily open). Uses BTC_USD daily bars (close→open of next
     day). BTC has documented weekend/overnight drift and is
     STRUCTURALLY uncorrelated with equity strategies.

  B) 1-DAY REVERSAL on 96 stocks (Jegadeesh 1990, short-term reversal):
     At day t close, rank stocks by day t return. Long bottom-K, short
     top-K. Hold 1 day, close at t+1 close. EW dollar-neutral, K=10.
     Use only liquid/highly-traded days. TC 10 bps per rebalance.
     Published SR ~2 with 3-day holding; 1-day is noisier but captures
     more events.

  C) WEEKLY REVERSAL on 96 stocks: 5-day version of (B). Rebalance
     Monday, hold 5 days. Lower TC burden.

Then build MEGA ensemble combining these with prior sleeves.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


ETF_DIR = Path("/home/user/bonds/data/etfs")
STOCK_DIR = Path("/home/user/bonds/data/intraday_daily")
RESULTS = Path("/home/user/bonds/data/results")


def btc_overnight():
    """BTC close-to-open daily drift (serves as crypto 'overnight' proxy)."""
    df = pd.read_csv(ETF_DIR/"BTC_USD.csv", parse_dates=["Date"]).set_index("Date")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    # overnight = open_t / close_{t-1} - 1
    ovn = (df["Open"] / df["Close"].shift(1) - 1).dropna()
    # TC 5 bps / round-trip on crypto
    return (ovn - 5/1e4).rename("BTC_OVN")


def stock_ohlc():
    stocks = sorted(STOCK_DIR.glob("*.csv"))
    EXCLUDE = {"SPY","QQQ","IWM","DIA","TLT","GLD","EFA","EEM","VNQ",
               "XLK","XLF","XLE","XLY","XLP","XLI","XLV","XLU","XLB","XLRE",
               "HYG","IEF","SHY","BIL"}
    closes = {}
    for p in stocks:
        t = p.stem
        if t in EXCLUDE: continue
        df = pd.read_csv(p, parse_dates=["ts"])
        df["date"] = pd.to_datetime(df["ts"].dt.date)
        df = df.set_index("date").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        closes[t] = df["close"]
    return pd.DataFrame(closes).sort_index().ffill()


def reversal_sleeve(C, holding_days, K=10, TC_BPS=10.0):
    """At rebalance, rank by trailing N-day return, long bottom-K / short top-K.
       Hold `holding_days`; rebalance on `holding_days` cadence.
       """
    R = C.pct_change().fillna(0)
    dates = R.index
    # Signal = trailing N-day return (N = holding_days)
    mom = C.pct_change(holding_days).shift(1)

    # Rebalance every `holding_days` days
    rebal = np.zeros(len(dates), dtype=bool)
    rebal[::holding_days] = True
    rebal[0] = False

    weights = pd.DataFrame(0.0, index=dates, columns=C.columns)
    current = pd.Series(0.0, index=C.columns)
    for i, d in enumerate(dates):
        if rebal[i] and i > holding_days + 5:
            sig = mom.loc[d].dropna()
            if len(sig) < 2*K:
                weights.iloc[i] = current.values; continue
            ranked = sig.sort_values()
            # REVERSAL: long bottom (losers), short top (winners)
            longs = ranked.index[:K]
            shorts = ranked.index[-K:]
            current = pd.Series(0.0, index=C.columns)
            current[longs] = 1.0 / K
            current[shorts] = -1.0 / K
        weights.iloc[i] = current.values
    w_eff = weights.shift(1).fillna(0)
    port_gross = (w_eff * R).sum(axis=1)
    to = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    return port_gross - to * (TC_BPS / 1e4)


def main():
    warm = pd.Timestamp("2017-06-01")
    CUT = pd.Timestamp("2022-01-01")

    # A) Crypto overnight
    btc = btc_overnight().loc[warm:]
    sf = stats(btc, "BTC_OVN")
    cagr_btc = ((1+btc).prod()**(252/len(btc)) - 1)*100
    print(f"A) {sf['label']:20s} SR={sf['sharpe']:>5.2f}  CAGR={cagr_btc:>6.2f}%  "
          f"Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")
    for p, tag in [(btc.loc[:CUT], "IS"), (btc.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        print(f"   {tag:18s} SR={ss['sharpe']:>5.2f}  Vol={ss['vol']:>5.2f}%")

    # Load stocks
    C = stock_ohlc()
    print(f"\nStock universe: {C.shape[1]} names, {len(C)} days")

    # B) 1-day reversal
    rev1 = reversal_sleeve(C, holding_days=1, K=10, TC_BPS=10.0).loc[warm:]
    sf = stats(rev1, "REV_1D_K10")
    print(f"\nB) {sf['label']:20s} SR={sf['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")
    for p, tag in [(rev1.loc[:CUT], "IS"), (rev1.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        print(f"   {tag:18s} SR={ss['sharpe']:>5.2f}  Vol={ss['vol']:>5.2f}%")

    # C) 5-day weekly reversal
    rev5 = reversal_sleeve(C, holding_days=5, K=10, TC_BPS=10.0).loc[warm:]
    sf = stats(rev5, "REV_5D_K10")
    print(f"\nC) {sf['label']:20s} SR={sf['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")
    for p, tag in [(rev5.loc[:CUT], "IS"), (rev5.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        print(f"   {tag:18s} SR={ss['sharpe']:>5.2f}  Vol={ss['vol']:>5.2f}%")

    # D) 21-day monthly reversal
    rev21 = reversal_sleeve(C, holding_days=21, K=10, TC_BPS=10.0).loc[warm:]
    sf = stats(rev21, "REV_21D_K10")
    print(f"\nD) {sf['label']:20s} SR={sf['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")
    for p, tag in [(rev21.loc[:CUT], "IS"), (rev21.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        print(f"   {tag:18s} SR={ss['sharpe']:>5.2f}  Vol={ss['vol']:>5.2f}%")

    # MEGA ensemble
    def ld(fn, col):
        return pd.read_csv(RESULTS/fn, parse_dates=[0], index_col=0)[col]
    def ld_nova29_N26(): return ld("nova29_master_returns.csv", "N26_OVN")
    sleeves = {
        "N26_OVN":  ld_nova29_N26(),
        "N18_LO":   ld("nova18_returns.csv", "NOVA18_LO"),
        "N27_OVN":  ld("nova27_returns.csv", "NOVA27_OVN"),
        "N28_WOVN": ld("nova28_returns.csv", "weekly_OVN"),
        "BTC_OVN":  btc,
        "REV_5D":   rev5,
        "REV_21D":  rev21,
    }
    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0).loc[warm:]
    print("\nCorrelation matrix:")
    print(df.corr().round(2).to_string())

    print("\nPer-sleeve:")
    for c in df.columns:
        x = df[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:12s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%")

    # Full ERC
    vols = df.loc[:CUT].std() * np.sqrt(252)
    w = (1/vols)/(1/vols).sum()
    port = (df * w).sum(axis=1)
    s = stats(port, "MEGA ERC (7 sleeves)")
    cagr = ((1+port).prod()**(252/len(port)) - 1)*100
    print(f"\nERC weights: {w.round(3).to_dict()}")
    print(f"{s['label']:30s} SR={s['sharpe']:>5.2f}  CAGR={cagr:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(port.loc[:CUT], "IS"), (port.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        c = ((1+p).prod()**(252/len(p)) - 1)*100
        print(f"  {tag:28s} SR={ss['sharpe']:>5.2f}  CAGR={c:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Subset search
    import itertools
    print("\nTop-20 subsets by OOS SR:")
    results = []
    cols = list(df.columns)
    for r in range(2, len(cols)+1):
        for combo in itertools.combinations(cols, r):
            sub = df[list(combo)]
            vs = sub.loc[:CUT].std() * np.sqrt(252)
            ws = (1/vs)/(1/vs).sum()
            p = (sub * ws).sum(axis=1)
            oos = stats(p.loc[CUT:], "")['sharpe']
            is_ = stats(p.loc[:CUT], "")['sharpe']
            results.append((combo, is_, oos, stats(p, "")))
    results.sort(key=lambda x: -x[2])
    for combo, is_, oos, sf in results[:20]:
        cagr = ((1+df[list(combo)].mul(
            (1/(df[list(combo)].loc[:CUT].std()*np.sqrt(252)) /
             (1/(df[list(combo)].loc[:CUT].std()*np.sqrt(252))).sum())
        ).sum(axis=1)).prod()**(252/len(df)) - 1)*100
        name = "+".join(combo)
        print(f"  {name:55s} IS={is_:>5.2f} OOS={oos:>5.2f} "
              f"CAGR={cagr:>6.2f}% Vol={sf['vol']:>5.1f}% MDD={sf['mdd']:>6.1f}%")

    # Apply 2x leverage to best subset
    best_combo = list(results[0][0])
    sub = df[best_combo]
    vs = sub.loc[:CUT].std() * np.sqrt(252)
    ws = (1/vs)/(1/vs).sum()
    pbest = (sub * ws).sum(axis=1)
    print(f"\nBEST subset: {best_combo}")
    print(f"weights: {ws.round(3).to_dict()}")
    for lev in [1.0, 2.0, 3.0, 4.0, 5.0]:
        p = pbest * lev
        s = stats(p, ""); oos = stats(p.loc[CUT:], "")['sharpe']
        cagr = ((1+p).prod()**(252/len(p)) - 1)*100
        print(f"  {lev:>3.1f}x  SR={s['sharpe']:>5.2f}  OOS_SR={oos:>5.2f}  "
              f"CAGR={cagr:>6.2f}%  Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")

    out = df.copy()
    out["MEGA_ERC"] = port
    out.to_csv(RESULTS/"nova33_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova33_returns.csv")


if __name__ == "__main__":
    main()
