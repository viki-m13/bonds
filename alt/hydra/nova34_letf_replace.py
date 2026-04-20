"""NOVA34 — Replace broker margin with leveraged ETFs.

Goal: match 4x-leveraged NOVA29's risk/return profile without using
margin.  Substitute LETFs for the overnight-ETF sleeves:

  SPY (1x) → UPRO (3x SPY daily)       — available
  QQQ (1x) → TQQQ (3x QQQ daily)       — available
  GLD (1x) → UGL  (2x GLD daily)       — available
  IWM (1x) → keep IWM (no liquid 3x)   — partial leverage only
  DIA (1x) → keep DIA (no liquid 3x)   — partial leverage only

For daytime TSMOM (N18_LO) we keep 1x ETFs (TSMOM on 12 ETFs is vol ~6-7%,
not worth LETF complexity for modest vol bump).

Stock L/S sleeve (N27_OVN) has no LETF substitute — keep as-is.

Three configs compared:
  A) NOVA29_1x        — pure 1x, no leverage, no LETFs  (baseline)
  B) NOVA29_4x margin — static 4x broker leverage
  C) NOVA34_LETF      — LETF substitution (mostly no margin)

Reports vol, SR, CAGR, and estimated interest drag.
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


INTRA = Path("/home/user/bonds/data/intraday_5min")
ETF_DIR = Path("/home/user/bonds/data/etfs")
RESULTS = Path("/home/user/bonds/data/results")

TC = 2.0
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


def ovn_from_intraday(t):
    df = pd.read_csv(INTRA/f"{t}.csv", parse_dates=["ts"])
    df["date"] = pd.to_datetime(df["ts"].dt.date)
    df["time"] = df["ts"].dt.time
    px = df[df["time"]==pd.to_datetime("15:55").time()].set_index("date")["close"]
    op = df.groupby("date")["open"].first()
    px.index = pd.to_datetime(px.index); op.index = pd.to_datetime(op.index)
    px = px[~px.index.duplicated()].sort_index()
    op = op[~op.index.duplicated()].sort_index()
    c = px.index.intersection(op.index)
    return ((op.loc[c].shift(-1)/px.loc[c]) - 1).dropna()


def ovn_from_daily(ticker):
    df = pd.read_csv(ETF_DIR/f"{ticker}.csv", parse_dates=["Date"]).set_index("Date")
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated()].sort_index()
    return (df["Open"] / df["Close"].shift(1) - 1).dropna()


def ovn_sleeve(ovn_ticker, rv_source, bil, use_daily=False):
    rv = five_min_rv(rv_source)
    ovn = ovn_from_daily(ovn_ticker) if use_daily else ovn_from_intraday(ovn_ticker)
    c = rv.index.intersection(ovn.index).intersection(bil.index)
    rv, ovn, b = rv.loc[c], ovn.loc[c], bil.loc[c]
    gate = rv.rolling(20).mean().shift(1) < RV_CUT
    r = pd.Series(0.0, index=c)
    r[gate] = ovn[gate] - TC / 1e4
    r[~gate] = b[~gate]
    return r


def main():
    bil = pd.read_csv(ETF_DIR/"BIL.csv", parse_dates=["Date"]).set_index("Date")
    bil.index = pd.to_datetime(bil.index)
    bil_ret = bil["Close"].pct_change().fillna(0)
    bil_ret = bil_ret[~bil_ret.index.duplicated()].sort_index()

    warm = pd.Timestamp("2017-06-01")
    CUT = pd.Timestamp("2022-01-01")

    # --- Build N26_OVN 1x (from intraday 5-min bars) ---
    pairs_1x = [("SPY","SPY"), ("QQQ","QQQ"), ("IWM","IWM"),
                ("DIA","DIA"), ("GLD","GLD")]
    sl_1x = {t: ovn_sleeve(t, src, bil_ret, False) for t, src in pairs_1x}
    df_1x = pd.DataFrame(sl_1x).fillna(0).loc[warm:]
    vs = df_1x.loc[:CUT].std() * np.sqrt(252)
    w = (1/vs) / (1/vs).sum()
    N26_1x = (df_1x * w).sum(axis=1)

    # --- Build N26_OVN LETF (substituted) at EQUAL dollar weight ---
    pairs_letf = [("UPRO","SPY"), ("TQQQ","QQQ"), ("IWM","IWM"),
                  ("DIA","DIA"), ("UGL","GLD")]
    sl_letf = {}
    for tk, src in pairs_letf:
        use_daily = tk in ("UPRO","TQQQ","UGL")
        sl_letf[tk] = ovn_sleeve(tk, src, bil_ret, use_daily=use_daily)
    df_letf = pd.DataFrame(sl_letf).fillna(0).loc[warm:]
    # EQUAL dollar weights (not ERC) — this is how LETFs actually provide leverage.
    w2 = pd.Series(1/5, index=df_letf.columns)
    N26_LETF = (df_letf * w2).sum(axis=1)

    # Other sleeves (unchanged)
    def ld(fn, col):
        return pd.read_csv(RESULTS/fn, parse_dates=[0], index_col=0)[col]
    N18 = ld("nova18_returns.csv", "NOVA18_LO").loc[warm:]
    N27 = ld("nova27_returns.csv", "NOVA27_OVN").loc[warm:]
    N28 = ld("nova28_returns.csv", "weekly_OVN").loc[warm:]

    # --- Version A: 1x NOVA29 (baseline) ---
    sleeves_A = {"N26_OVN": N26_1x, "N18_LO": N18, "N27_OVN": N27, "N28_WOVN": N28}
    A = pd.DataFrame(sleeves_A).dropna(how="all").fillna(0).loc[warm:]
    vA = A.loc[:CUT].std() * np.sqrt(252)
    wA = (1/vA) / (1/vA).sum()
    portA = (A * wA).sum(axis=1)

    # --- Version B: 4x margin NOVA29 ---
    portB = portA * 4.0
    # Interest drag: 5% APR on borrowed 3x of equity.  Not ALL positions are
    # always on — overnight sleeves are active roughly half of nights, daytime
    # and stock-L/S roughly always.  Rough weighted-deployed fraction ≈ 0.55,
    # and overnight vs day duration split ≈ 15.5 / 8.5 hours.  Effective
    # continuous-leverage charge ≈ 5% × 3 × 0.55 ≈ 8.25% / yr.
    interest_annual = 0.0825
    portB_net = portB - interest_annual / 252

    # --- Version C: LETF substitution with EQUAL-dollar sleeve weights ---
    # (no ERC dampening on the LETFs — otherwise the leverage gets weighted-out)
    sleeves_C = {"N26_OVN_LETF": N26_LETF, "N18_LO": N18, "N27_OVN": N27,
                 "N28_WOVN": N28}
    C = pd.DataFrame(sleeves_C).dropna(how="all").fillna(0).loc[warm:]
    # Full-notional stacking (time windows disjoint): sum at 1.0 each
    portC = C.sum(axis=1)
    # LETF cost drag already baked in (UPRO 0.93% ER, TQQQ 0.84%, UGL 0.95% + swap ≈ 1%)
    # So portC is the REALIZED net return already.

    # --- Report ---
    def fmt(x, name):
        s = stats(x, "")
        cagr = ((1+x).prod()**(252/len(x)) - 1) * 100
        is_ = stats(x.loc[:CUT], "")['sharpe']
        oos = stats(x.loc[CUT:], "")['sharpe']
        return (f"{name:25s}  CAGR={cagr:>6.2f}%  Vol={s['vol']:>5.2f}%  "
                f"SR={s['sharpe']:>5.2f}  IS={is_:>5.2f}  OOS={oos:>5.2f}  "
                f"MDD={s['mdd']:>7.2f}%")

    print("=" * 100)
    print("COMPARISON: 4x-margin leverage vs LETF substitution")
    print("=" * 100)
    print(fmt(portA, "A) NOVA29 1x (base)"))
    print(fmt(portB, "B) NOVA29 4x GROSS"))
    print(fmt(portB_net, "B') NOVA29 4x NET (int.)"))
    print(fmt(portC, "C) NOVA34 LETF"))

    print("\nPer-sleeve in LETF basket:")
    for c in df_letf.columns:
        x = df_letf[c]
        sf = stats(x, c); si = stats(x.loc[:CUT], "")
        so = stats(x.loc[CUT:], "")
        print(f"  {c:8s} SR={sf['sharpe']:>5.2f}  IS={si['sharpe']:>5.2f}  "
              f"OOS={so['sharpe']:>5.2f}  Vol={sf['vol']:>5.2f}%  MDD={sf['mdd']:>7.2f}%")

    print(f"\nLETF-basket ERC weights: {w2.round(3).to_dict()}")
    print(f"LETF-basket vol: {(N26_LETF.std()*np.sqrt(252))*100:.2f}%  "
          f"vs 1x basket vol: {(N26_1x.std()*np.sqrt(252))*100:.2f}%  "
          f"(effective leverage on overnight sleeve: "
          f"{(N26_LETF.std()/N26_1x.std()):.2f}x)")

    # Annual comparison
    print("\nAnnual returns:")
    print(f"{'Year':>6s} {'1x':>8s} {'4x margin':>11s} {'4x net':>9s} {'LETF':>8s} {'SPY':>8s}")
    spy = pd.read_csv(ETF_DIR/"SPY.csv", parse_dates=["Date"]).set_index("Date")
    spy.index = pd.to_datetime(spy.index)
    spy_ret = spy["Close"].pct_change().fillna(0)
    for yr in sorted(portA.index.year.unique()):
        a = portA[portA.index.year==yr]
        b = portB[portB.index.year==yr]
        bn = portB_net[portB_net.index.year==yr]
        c = portC[portC.index.year==yr]
        s = spy_ret[spy_ret.index.year==yr]
        print(f"{yr:>6d} {((1+a).prod()-1)*100:>7.2f}% "
              f"{((1+b).prod()-1)*100:>10.2f}% "
              f"{((1+bn).prod()-1)*100:>8.2f}% "
              f"{((1+c).prod()-1)*100:>7.2f}% "
              f"{((1+s).prod()-1)*100:>7.2f}%")

    # Save
    out = pd.DataFrame({
        "NOVA29_1x": portA, "NOVA29_4x": portB, "NOVA29_4x_net": portB_net,
        "NOVA34_LETF": portC,
    })
    out.to_csv(RESULTS/"nova34_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova34_returns.csv")


if __name__ == "__main__":
    main()
