"""NOVA32 — LEVERAGED-ETF overnight drift, gated by their siblings' 5-min RV.

Key insight (from leveraged-ETF mechanics):
  The daily reset of 2x/3x ETFs borrows intraday and pays back at
  close.  Overnight drift is captured at ~2-3x the unleveraged version
  WITHOUT the volatility-decay drag (which is an INTRADAY phenomenon).
  So same alpha source, much higher per-$ vol and return.

Pairs tested (leveraged → RV-gate source):
  TQQQ (3x NDX)  gated by QQQ 5-min RV
  UPRO (3x SPX)  gated by SPY 5-min RV
  SSO  (2x SPX)  gated by SPY 5-min RV
  TECL (3x tech) gated by QQQ 5-min RV   (proxy — no XLK 5min)
  SOXL (3x semi) gated by QQQ 5-min RV   (proxy — no SMH 5min)
  FAS  (3x fin)  gated by XLF 5-min RV

Strategy: same as NOVA26 but on the leveraged sibling.  RV<0.15 → hold
overnight (15:55→09:30 captured via close-to-open daily); else BIL.
TC 2 bps per active night.  Daily data from data/etfs/ used for the
leveraged-ETF overnight returns (since we lack their 5-min bars).
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
ETF_DIR = Path("/home/user/bonds/data/etfs")
TC_BPS = 2.0
RV_CUT = 0.15


def five_min_rv(t):
    df = pd.read_csv(INTRA/f"{t}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["logret"] = np.log(df["close"]).diff()
    fod = df["date"] != df["date"].shift(1)
    df.loc[fod, "logret"] = 0.0
    rv = df.groupby("date")["logret"].apply(lambda x: np.sqrt(np.sum(x**2)))
    rv.index = pd.to_datetime(rv.index)
    return rv * np.sqrt(252)


def ovn_ret_daily(ticker):
    """Daily overnight return (O_t / C_{t-1} - 1) from etfs/ directory."""
    df = pd.read_csv(ETF_DIR/f"{ticker}.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index = pd.to_datetime(df.index)
    return (df["Open"] / df["Close"].shift(1) - 1).dropna()


def ovn_sleeve_lev(lev_ticker, rv_source, bil_ret):
    rv = five_min_rv(rv_source)
    ovn = ovn_ret_daily(lev_ticker)
    c = rv.index.intersection(ovn.index).intersection(bil_ret.index)
    rv, ovn, b = rv.loc[c], ovn.loc[c], bil_ret.loc[c]
    rv20 = rv.rolling(20).mean().shift(1)
    gate = rv20 < RV_CUT
    r = pd.Series(0.0, index=c)
    r[gate] = ovn[gate] - TC_BPS / 1e4
    r[~gate] = b[~gate]
    return r


def main():
    bil = pd.read_csv(ETF_DIR/"BIL.csv", parse_dates=["Date"]).set_index("Date")
    bil.index = pd.to_datetime(bil.index)
    bil_ret = bil["Close"].pct_change().fillna(0)
    bil_ret = bil_ret[~bil_ret.index.duplicated(keep="first")].sort_index()

    pairs = [
        ("TQQQ", "QQQ"), ("UPRO", "SPY"), ("SSO", "SPY"),
        ("TECL", "QQQ"), ("SOXL", "QQQ"), ("FAS", "XLF"),
    ]

    warm = pd.Timestamp("2017-06-01")
    CUT = pd.Timestamp("2022-01-01")

    sleeves = {}
    print(f"{'LevETF':8s} {'GateSrc':8s} {'SR':>5s} {'IS':>5s} {'OOS':>5s} "
          f"{'Vol':>6s} {'MDD':>8s}")
    for lev, src in pairs:
        r = ovn_sleeve_lev(lev, src, bil_ret).loc[warm:]
        sleeves[lev] = r
        sf = stats(r, ""); si = stats(r.loc[:CUT], "")
        so = stats(r.loc[CUT:], "")
        print(f"  {lev:8s} {src:8s} {sf['sharpe']:>5.2f} {si['sharpe']:>5.2f} "
              f"{so['sharpe']:>5.2f}  {sf['vol']:>5.2f}%  {sf['mdd']:>7.2f}%")

    df = pd.DataFrame(sleeves).dropna(how="all").fillna(0)

    # ERC basket
    vols = df.loc[:CUT].std() * np.sqrt(252)
    w = (1/vols) / (1/vols).sum()
    erc = (df * w).sum(axis=1)
    s = stats(erc, "ERC lev-ovn")
    cagr = ((1+erc).prod()**(252/len(erc)) - 1)*100
    print(f"\nERC weights: {w.round(3).to_dict()}")
    print(f"{'ERC 6-lev':30s} SR={s['sharpe']:>5.2f}  CAGR={cagr:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(erc.loc[:CUT], "IS"), (erc.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        c = ((1+p).prod()**(252/len(p)) - 1)*100
        print(f"  {tag:30s} SR={ss['sharpe']:>5.2f}  CAGR={c:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # EW equal-weight basket (no ERC dampening)
    ew = df.mean(axis=1)
    s = stats(ew, "EW lev-ovn")
    cagr = ((1+ew).prod()**(252/len(ew)) - 1)*100
    print(f"\n{'EW 6-lev':30s} SR={s['sharpe']:>5.2f}  CAGR={cagr:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    for p, tag in [(ew.loc[:CUT], "IS"), (ew.loc[CUT:], "OOS")]:
        ss = stats(p, tag)
        c = ((1+p).prod()**(252/len(p)) - 1)*100
        print(f"  {tag:30s} SR={ss['sharpe']:>5.2f}  CAGR={c:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Stack lev-overnight with NOVA29 daytime sleeves (time-disjoint)
    n18 = pd.read_csv("/home/user/bonds/data/results/nova18_returns.csv",
                      parse_dates=[0], index_col=0)["NOVA18_LO"].loc[warm:]
    idx = erc.index.intersection(n18.index)
    stack_erc = erc.loc[idx] + n18.loc[idx]
    stack_ew = ew.loc[idx] + n18.loc[idx]

    for name, st in [("STACK ERC+N18", stack_erc), ("STACK EW+N18", stack_ew)]:
        s = stats(st, "")
        cagr = ((1+st).prod()**(252/len(st)) - 1)*100
        is_ = stats(st.loc[:CUT], "")['sharpe']
        oos = stats(st.loc[CUT:], "")['sharpe']
        print(f"\n{name:30s} SR={s['sharpe']:>5.2f}  CAGR={cagr:>6.2f}%  "
              f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  IS={is_:.2f}  OOS={oos:.2f}")

    # Save
    out = df.copy()
    out["ERC_lev_ovn"] = erc
    out["EW_lev_ovn"] = ew
    out["STACK_ERC_N18"] = stack_erc
    out.to_csv("/home/user/bonds/data/results/nova32_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova32_returns.csv")

    # Annual for stack_erc
    p = stack_erc
    ann = p.groupby(p.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean()*252)/(x.std()*np.sqrt(252)) if x.std()>0 else 0,
            "MDD%": ((1+x).cumprod()/(1+x).cumprod().cummax()-1).min()*100,
        })
    ).round(2)
    print("\nAnnual (STACK ERC+N18):")
    print(ann.to_string())


if __name__ == "__main__":
    main()
