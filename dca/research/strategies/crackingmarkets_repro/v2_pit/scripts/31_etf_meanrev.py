"""New sleeve E — index-ETF mean reversion (Connors RSI(2)), long history,
real OHLC, execution-timing study.

Spec (canonical, published for two decades — not fitted here):
  long SPY (and QQQ, run as separate 50/50 books) when RSI(2) < 10 and
  close > 200DMA; exit when close > 5-day MA. 100% of the book per signal.

Execution modes:
  moc     signal ~3:55pm, trade MOC at close t
  moo     trade next open t+1
  limit   scale-in limit at close - 0.25*ATR(5) working day t+1; if unfilled
          and signal persists, MOC at close t+1
Costs: 1bp auction/half-spread + $0.005/sh commission (ETF tier).
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import load_etf, wilder_rsi, atr_simple, riskfree_daily, stats, fmt, OUT

COST = 1.5 / 1e4     # per side, ETFs (auction 1bp + commission ~0.5bp)


def sleeve(ticker, mode, start="2005-01-03"):
    df = load_etf(ticker).loc[start:]
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    rsi2 = wilder_rsi(c, 2)
    dma200 = c.rolling(200, min_periods=200).mean()
    ma5 = c.rolling(5).mean()
    atr = atr_simple(h, l, c, 5)
    entry_sig = (rsi2 < 10) & (c > dma200)
    exit_sig = c > ma5
    n = len(df)
    ret = np.zeros(n)
    inpos_v = np.zeros(n)
    in_pos, entry_px = False, np.nan
    pend_limit = pend_moc = False
    lim = np.nan
    cv, ov, lv = c.values, o.values, l.values
    es, xs, atrv = entry_sig.values, exit_sig.values, atr.values
    for i in range(1, n):
        if in_pos:
            ret[i] = cv[i] / cv[i - 1] - 1
        # exits (signal at close i, executed per-mode)
        if in_pos and xs[i]:
            if mode in ("moc", "limit"):
                ret[i] -= COST
                in_pos = False
            elif mode == "moo" and i + 1 < n and np.isfinite(ov[i + 1]):
                # hold to next open: overwrite next-day return leg
                ret[i + 1] = ov[i + 1] / cv[i] - 1 - COST
                in_pos = False
                # skip entry logic today; position closes at next open
        # pending entries from yesterday
        if not in_pos:
            if pend_limit:
                if np.isfinite(lv[i]) and lv[i] <= lim:
                    fill = min(ov[i], lim) if np.isfinite(ov[i]) else lim
                    ret[i] = cv[i] / fill - 1 - COST
                    in_pos = True
                elif es[i]:      # unfilled but signal persists -> MOC now
                    ret[i] -= 0  # enter at close i
                    in_pos = True
                    ret[i] -= COST
                pend_limit = False
            elif pend_moc:       # moo mode: enter at open i
                if np.isfinite(ov[i]):
                    ret[i] = cv[i] / ov[i] - 1 - COST
                    in_pos = True
                pend_moc = False
        # new signals at close i
        if not in_pos:
            if es[i]:
                if mode == "moc":
                    in_pos = True
                    ret[i] -= COST
                elif mode == "moo":
                    pend_moc = True
                elif mode == "limit":
                    pend_limit = True
                    lim = cv[i] - 0.25 * atrv[i] if np.isfinite(atrv[i]) else cv[i] * 0.997
        inpos_v[i] = 1.0 if in_pos else 0.0
    return pd.Series(ret, index=df.index), pd.Series(inpos_v, index=df.index)


if __name__ == "__main__":
    t0 = time.time()
    idx = load_etf("SPY").index
    rf = riskfree_daily(idx)
    print("Sleeve E: SPY+QQQ RSI(2) mean reversion (2005-2026), costed")
    res, expos = {}, {}
    for mode in ["moc", "moo", "limit"]:
        rs, es = sleeve("SPY", mode)
        rq, eq_ = sleeve("QQQ", mode)
        r = pd.concat([rs, rq], axis=1).fillna(0).mean(axis=1)
        e = pd.concat([es, eq_], axis=1).fillna(0).mean(axis=1)
        st = stats(r, rf, f"SPY+QQQ rsi2 {mode}")
        print(fmt(st) + f"  exposure {e.mean()*100:.0f}%")
        res[mode] = r
        expos[mode] = e
    pd.DataFrame({"etfmr_moc": res["moc"], "etfmr_moo": res["moo"],
                  "etfmr_limit": res["limit"]}) \
        .to_parquet(os.path.join(OUT, "sleeveE_etfmr.parquet"))
    pd.DataFrame({"etfmr_moc": expos["moc"]}) \
        .to_parquet(os.path.join(OUT, "sleeveE_etfmr_expo.parquet"))
    print(f"saved -> out/sleeveE_etfmr.parquet  t={time.time()-t0:.0f}s")
