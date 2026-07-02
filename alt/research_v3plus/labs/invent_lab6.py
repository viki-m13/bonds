"""Iteration 6 — NEW data channels (High/Low/Volume) + construction changes.
IS ONLY (2010-2018). Sub-period halves reported for anything promising."""
import sys
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import backtest_weights, sharpe, cagr, max_dd, ann_vol

ETF = "/home/user/bonds/data/etfs/"
IS_END = pd.Timestamp("2018-12-31")

def load(t, cols=("Open", "Close", "High", "Low", "Volume")):
    df = pd.read_csv(ETF + f"{t}.csv", parse_dates=["Date"]).sort_values("Date")
    df = df.drop_duplicates(subset=["Date"]).set_index("Date")
    return df[list(cols)].apply(pd.to_numeric, errors="coerce")

SPY = load("SPY")
DATES = SPY["Open"].dropna().index
DATES = DATES[DATES >= "2010-03-11"]

def rep(name, r, halves=False):
    ri = r.loc[:IS_END]
    extra = ""
    if halves:
        h1, h2 = ri.loc[:"2014-12-31"], ri.loc["2015-01-02":]
        extra = f"  halves {sharpe(h1):.2f}/{sharpe(h2):.2f}"
    print(f"{name:40s} IS: SR {sharpe(ri):5.2f}  CAGR {cagr(ri)*100:6.1f}%  "
          f"Vol {ann_vol(ri)*100:5.1f}%  MDD {max_dd(ri)*100:6.1f}%{extra}")

# ---------- A. Close-location-value (CLV) accumulation ----------
def clv_sleeve(k_days=10, th=0.6, hold=5, pairs=None):
    """CLV = (close-low)/(high-low); rolling mean of CLV > th while above
    200dma -> hold the LETF `hold` days. Buying-pressure proxy from a data
    channel (H/L) the system has never used."""
    pairs = pairs or {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL"}
    W_parts = {}
    for u, l in pairs.items():
        d = load(u)
        clv = ((d["Close"] - d["Low"]) / (d["High"] - d["Low"]).replace(0, np.nan)).reindex(DATES)
        sig = clv.rolling(k_days).mean()
        above = (d["Close"] > d["Close"].rolling(200).mean()).reindex(DATES)
        trig = ((sig > th) & above).shift(1).fillna(False)
        W_parts[l] = trig.astype(float).rolling(hold, min_periods=1).max()
    W = pd.DataFrame(W_parts)
    tot = W.sum(axis=1)
    W = W.div(tot.where(tot > 0), axis=0).mul(tot.clip(upper=1.0), axis=0).fillna(0.0)
    opens = pd.DataFrame({l: load(l)["Open"].reindex(DATES) for l in pairs.values()})
    return backtest_weights(W, opens, 10.0)["ret"]

print("== A. CLV accumulation ==")
for k, th in [(10, 0.60), (10, 0.55), (21, 0.58), (5, 0.65)]:
    rep(f"clv k={k} th={th}", clv_sleeve(k, th))

# ---------- B. Range compression breakout (NR-N) ----------
def nrn_breakout(n=7, hold=3, vehicle="TQQQ", sig_t="QQQ", trend=True):
    d = load(sig_t)
    rng = ((d["High"] - d["Low"]) / d["Close"]).reindex(DATES)
    isnr = (rng == rng.rolling(n).min())
    up = (d["Close"] > d["Close"].rolling(200).mean()).reindex(DATES) if trend else True
    trig = (isnr & up).shift(1).fillna(False) if trend else isnr.shift(1).fillna(False)
    held = trig.astype(float).rolling(hold, min_periods=1).max()
    W = pd.DataFrame({vehicle: held.clip(upper=1.0)})
    opens = pd.DataFrame({vehicle: load(vehicle)["Open"].reindex(DATES)})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== B. NR-N compression breakout ==")
for n, hold in [(7, 3), (7, 5), (4, 3)]:
    rep(f"nr{n} hold={hold} trend", nrn_breakout(n, hold))
rep("nr7 hold=3 no-trend", nrn_breakout(7, 3, trend=False))

# ---------- C. Volume confirmation on the momentum book ----------
def vol_confirm_book(mode="obv", dow=2):
    """ORION-ish top-4 momentum book; volume filter: hold a name only if its
    OBV (or dollar-volume trend) confirms (OBV 63d slope > 0)."""
    risk = {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL", "XLK": "TECL",
            "XLF": "FAS", "XLE": "ERX", "EEM": "EDC", "FXI": "YINN",
            "VNQ": "DRN", "USO": "UCO"}
    closes, obv_ok = {}, {}
    for u in risk:
        d = load(u)
        closes[u] = d["Close"].reindex(DATES).ffill(limit=3)
        direction = np.sign(d["Close"].diff())
        obv = (direction * d["Volume"]).cumsum()
        if mode == "obv":
            ok = obv.diff(63) > 0
        else:  # dollar volume regime: rising participation
            dv = (d["Close"] * d["Volume"]).rolling(21).mean()
            ok = dv.diff(63) > 0
        obv_ok[u] = ok.reindex(DATES)
    closes = pd.DataFrame(closes); okf = pd.DataFrame(obv_ok)
    mom = closes.pct_change(252).shift(1)
    above = (closes > closes.rolling(200).mean()).shift(1).fillna(False)
    conf = okf.shift(1).fillna(False)
    score = mom.where(above)
    score_c = mom.where(above & conf)
    def book(sc):
        ranks = sc.rank(axis=1, ascending=False, method="first")
        Wu = (ranks <= 4).astype(float) / 4.0
        W = Wu.rename(columns=risk)
        reb = (pd.Series(DATES.dayofweek, index=DATES) == dow); reb.iloc[0] = True
        mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape), index=W.index, columns=W.columns)
        W = W.where(mask, np.nan).ffill().fillna(0.0)
        opens = pd.DataFrame({l: load(l)["Open"].reindex(DATES) for l in risk.values()})
        return backtest_weights(W, opens, 10.0)["ret"]
    return book(score), book(score_c)

print("\n== C. volume confirmation ==")
base, conf = vol_confirm_book("obv")
rep("mom book (no volume filter)", base)
rep("mom book + OBV-63 confirm", conf)
_, conf2 = vol_confirm_book("dv")
rep("mom book + $vol-trend confirm", conf2)

# ---------- D. Parkinson H/L vol for the gate ----------
def parkinson_gate_test():
    """Blend raw with vol gate driven by Parkinson (H/L) 20d vol instead of
    close-based 60d std — faster, less lagged crisis detection."""
    prod_raw = pd.read_csv("/home/user/bonds/data/results/phoenix_production_returns.csv",
                           parse_dates=["Date"]).set_index("Date")["raw_ret"]
    d = SPY
    pk = np.sqrt((np.log(d["High"] / d["Low"]) ** 2 / (4 * np.log(2))).rolling(20).mean() * 252)
    pk = pk.reindex(prod_raw.index).ffill()
    thr = pk.rolling(252, min_periods=60).quantile(0.99)
    gate = pd.Series(np.where(pk <= thr, 1.0, 0.5), index=prod_raw.index)
    # existing close-based gate for reference
    sv = prod_raw.rolling(60).std()
    thr2 = sv.rolling(252, min_periods=60).quantile(0.99)
    gate2 = pd.Series(np.where(sv <= thr2, 1.0, 0.5), index=prod_raw.index)
    for name, g in [("close-vol gate (current)", gate2), ("Parkinson-20 gate", gate),
                    ("both (min)", pd.concat([gate, gate2], axis=1).min(axis=1))]:
        tot = g.shift(2).fillna(1.0)
        tc = tot.diff().abs().fillna(0) * 10 / 1e4
        net = prod_raw * tot - tc
        rep(f"D. {name}", net.loc["2014-01-02":])

print("\n== D. Parkinson gate (on 2014-2018 segment of WF blend) ==")
parkinson_gate_test()

# ---------- E. Continuous softmax + lookback ensemble ORION ----------
def soft_orion(lbs=(126, 189, 252), temp=1.0, dow=2):
    risk = {"QQQ": "TQQQ", "SPY": "UPRO", "SMH": "SOXL", "XLK": "TECL",
            "XLF": "FAS", "XLE": "ERX", "EEM": "EDC", "FXI": "YINN",
            "VNQ": "DRN", "USO": "UCO"}
    closes = pd.DataFrame({u: load(u)["Close"].reindex(DATES).ffill(limit=3) for u in risk})
    z = 0
    for lb in lbs:
        m = closes.pct_change(lb)
        z = z + (m.sub(m.mean(axis=1), axis=0)).div(m.std(axis=1).replace(0, np.nan), axis=0)
    z = (z / len(lbs)).shift(1)
    above = (closes > closes.rolling(200).mean()).shift(1).fillna(False)
    ez = np.exp(z.where(above) / temp)
    W = ez.div(ez.sum(axis=1), axis=0).fillna(0.0)
    # scale gross by fraction of names in uptrend (breathes with breadth)
    W = W.mul(above.mean(axis=1), axis=0)
    W = W.rename(columns=risk)
    reb = (pd.Series(DATES.dayofweek, index=DATES) == dow); reb.iloc[0] = True
    mask = pd.DataFrame(np.broadcast_to(reb.values[:, None], W.shape), index=W.index, columns=W.columns)
    W = W.where(mask, np.nan).ffill().fillna(0.0)
    opens = pd.DataFrame({l: load(l)["Open"].reindex(DATES) for l in risk.values()})
    return backtest_weights(W, opens, 10.0)["ret"]

print("\n== E. softmax lookback-ensemble book ==")
for temp in [0.5, 1.0, 2.0]:
    rep(f"softmax T={temp} ens(126,189,252)", soft_orion(temp=temp), halves=True)
rep("softmax T=1 single lb 252", soft_orion(lbs=(252,)), halves=True)
print("\ndone (IS only)")
