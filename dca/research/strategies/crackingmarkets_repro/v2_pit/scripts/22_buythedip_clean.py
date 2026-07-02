"""Sleeve C — Buy-the-dip (RSI(5)<20, close>200DMA, 5-day hold) as a real
portfolio on the survivorship-clean PIT liquidity universe.

Portfolio: max 10 concurrent positions, 10% of equity each, candidates
ranked by lowest RSI(5). Universe = previous month-end top-500 dollar-ADV.

Execution modes:
  moc        signal ~3:55pm, enter MOC close t, exit MOC close t+5
  nextclose  run after close, enter close t+1, exit close t+6
Costs: auction fills + impact vs per-name dollar-ADV ($1M book).
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import wilder_rsi, fill_cost_bps, riskfree_daily, stats, fmt, OUT

t0 = time.time()
C = pd.read_parquet(os.path.join(OUT, "clean_close.parquet")).astype("float64")
DADV = pd.read_parquet(os.path.join(OUT, "clean_dadv.parquet")).astype("float64")
UNIV = pd.read_parquet(os.path.join(OUT, "univ_mask_monthly.parquet"))

rsi5 = wilder_rsi(C, 5)
dma200 = C.rolling(200, min_periods=200).mean()
# previous month-end universe, forward-filled to daily, then LAGGED one day
univ_daily = UNIV.reindex(C.index, method="ffill").shift(1).fillna(False)
sig = (rsi5 < 20) & (C > dma200) & univ_daily

dates = C.index
i_start = dates.searchsorted(pd.Timestamp("2000-01-31"))
Cv, Rv, Av, Sv = C.values, rsi5.values, DADV.values, sig.values
tk = list(C.columns)


def run(lag=0, hold=5, maxpos=10, pos_frac=0.10, costs=True, E0=1e6,
        start_i=i_start, end_i=None):
    cash, pos, eq, trades = E0, {}, [], []
    queue = []                      # (col, rsi) signals awaiting execution
    end_i = end_i or len(dates)
    for i in range(start_i, end_i):
        # exits first (frees slots)
        for j in list(pos):
            p = pos[j]
            px = Cv[i, j]
            if np.isfinite(px):
                p["last"] = px
            p["age"] += 1
            if p["age"] >= hold + lag:
                fillpx = p["last"]
                notional = p["sh"] * fillpx
                fee = fill_cost_bps("auction", fillpx, notional, p["adv"]) / 1e4 \
                    if costs else 0
                cash += notional * (1 - fee)
                trades.append(fillpx / p["ep"] - 1)
                del pos[j]
        # entries: signals from day i-lag (lag=0 -> same close)
        isig = i - lag
        if isig >= 0:
            cand = [(Rv[isig, j], j) for j in np.where(Sv[isig])[0]]
            cand.sort()
            for r, j in cand:
                if j in pos or len(pos) >= maxpos:
                    continue
                px = Cv[i, j]
                adv = Av[isig, j]
                if not np.isfinite(px) or px <= 0:
                    continue
                mv = sum(pp["sh"] * pp["last"] for pp in pos.values())
                notional = min(pos_frac * (cash + mv), max(cash - 1, 0))
                if notional < 1000:
                    continue
                sh = notional / px
                fee = fill_cost_bps("auction", px, notional, adv) / 1e4 \
                    if costs else 0
                cash -= notional * (1 + fee)
                pos[j] = {"sh": sh, "ep": px, "age": 0, "adv": adv, "last": px}
        mv = sum(pp["sh"] * pp["last"] for pp in pos.values())
        eq.append((cash + mv, mv / (cash + mv) if cash + mv > 0 else 0.0))
    tot = pd.Series([e[0] for e in eq], index=dates[start_i:end_i])
    expo = pd.Series([e[1] for e in eq], index=dates[start_i:end_i])
    return tot, np.array(trades), expo


if __name__ == "__main__":
    rf = riskfree_daily(dates)
    print("Sleeve C: clean-universe buy-the-dip (2000-2026), $1M book")
    res, expos = {}, {}
    for label, kw in [
        ("dip MOC t, costed",       dict(lag=0)),
        ("dip close t+1, costed",   dict(lag=1)),
        ("dip MOC t, FREE",         dict(lag=0, costs=False)),
        ("dip t, 3d hold, costed",  dict(lag=0, hold=3)),
        ("dip t, 7d hold, costed",  dict(lag=0, hold=7)),
    ]:
        eq, tr, expo = run(**kw)
        r = eq.pct_change().dropna()
        st = stats(r, rf, label)
        wr = (tr > 0).mean()
        print(fmt(st) + f"  win {wr*100:4.0f}%  n={len(tr):5d} "
              f"avg {tr.mean()*100:+.2f}%  expo {expo.mean()*100:.0f}%")
        res[label] = r
        expos[label] = expo
    keep = pd.DataFrame({"dip_moc": res["dip MOC t, costed"],
                         "dip_lag": res["dip close t+1, costed"]})
    keep.to_parquet(os.path.join(OUT, "sleeveC_dip.parquet"))
    pd.DataFrame({"dip_moc": expos["dip MOC t, costed"]}) \
        .to_parquet(os.path.join(OUT, "sleeveC_dip_expo.parquet"))
    print(f"saved -> out/sleeveC_dip.parquet  t={time.time()-t0:.0f}s")
