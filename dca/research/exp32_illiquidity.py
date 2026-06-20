"""Exp 32 — HARVEST the illiquidity premium (the one signal that passed exp31),
net of realistic costs. Universe: S&P400 midcaps (where illiquidity varies and
the premium lives). Monthly: sort by Amihud illiquidity, go long the most-
illiquid quintile equal-weight. Compare GROSS vs NET (20/40 bps round-trip on
turnover) to: liquid quintile, equal-weight universe, RANDOM quintile (control),
QQQ, MDY (midcap ETF). The crux: does the premium survive the cost of trading
illiquid names?"""
import warnings, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
t0 = time.time()
names = sorted(set(l.strip() for l in open("/tmp/wave/xuniverse_sp400.txt") if l.strip()))
bench = ["QQQ", "MDY", "IWM", "SPY"]
raw = yf.download(names + bench, start="2009-01-01", auto_adjust=True, progress=False)
C, V = raw["Close"], raw["Volume"]
names = [t for t in names if t in C.columns and C[t].notna().sum() > 1200]
C, V = C[names], V[names]
print(f"{len(names)} midcaps {C.index[0].date()}->{C.index[-1].date()}  t={time.time()-t0:.0f}s",
      flush=True)
r = C.pct_change()
dvol = C * V
amihud = (r.abs() / dvol.replace(0, np.nan)).rolling(21).mean() * 1e6   # illiquidity
me = C.resample("ME").last()
me.index = me.index.to_period("M").to_timestamp()
mret = me.pct_change()
am_m = amihud.resample("ME").last(); am_m.index = am_m.index.to_period("M").to_timestamp()
qq = raw["Close"]["QQQ"].resample("ME").last(); qq.index = qq.index.to_period("M").to_timestamp()
mdy = raw["Close"]["MDY"].resample("ME").last(); mdy.index = mdy.index.to_period("M").to_timestamp()


def quintile_bt(which, rt_cost=0.0, seed=None):
    """which: 'illiq' (top Amihud), 'liq' (bottom), 'ew' (all), 'rand'."""
    rng = np.random.default_rng(seed or 0)
    prev = set(); rets = []
    dates = [d for d in am_m.index if d >= pd.Timestamp("2010-01-01")]
    for i, d in enumerate(dates[:-1]):
        a = am_m.loc[d].replace([np.inf, -np.inf], np.nan).dropna()
        a = a[[n for n in a.index if np.isfinite(mret.loc[dates[i + 1], n])]] if dates[i+1] in mret.index else a
        if len(a) < 50:
            continue
        if which == "illiq":
            sel = list(a[a >= a.quantile(0.8)].index)
        elif which == "liq":
            sel = list(a[a <= a.quantile(0.2)].index)
        elif which == "ew":
            sel = list(a.index)
        else:                                            # random same-size as quintile
            sel = list(rng.choice(a.index, max(5, len(a) // 5), replace=False))
        nxt = dates[i + 1]
        rr = mret.loc[nxt, sel].mean()
        turn = 1.0 - len(prev & set(sel)) / max(len(sel), 1)   # fraction changed
        rets.append((nxt, rr - turn * rt_cost))
        prev = set(sel)
    return pd.Series(dict(rets)).dropna()


def stats(s, name):
    s = s.dropna(); eq = (1 + s).cumprod(); yrs = len(s) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1; mdd = float((eq / eq.cummax() - 1).min())
    sh = float(s.mean() / (s.std() + 1e-12) * np.sqrt(12))
    print(f"   {name:34s} CAGR {cagr*100:5.1f}%  Sharpe {sh:.2f}  maxDD {mdd*100:4.0f}%", flush=True)
    return cagr, sh


print("\nIlliquidity-premium harvest, S&P400 midcaps, 2010-2026 (monthly rebal):", flush=True)
illiq_g = quintile_bt("illiq", 0.0)
idx = illiq_g.index
stats(qq.pct_change().reindex(idx).dropna(), "QQQ (benchmark)")
stats(mdy.pct_change().reindex(idx).dropna(), "MDY midcap ETF (benchmark)")
stats(quintile_bt("ew", 0.0), "midcap universe equal-weight")
print("   --- illiquidity quintile (the premium) ---", flush=True)
stats(illiq_g, "ILLIQUID quintile  GROSS")
stats(quintile_bt("illiq", 0.0020), "ILLIQUID quintile  NET 20bps")
stats(quintile_bt("illiq", 0.0040), "ILLIQUID quintile  NET 40bps")
stats(quintile_bt("liq", 0.0020), "liquid quintile   NET 20bps")
rnd = [quintile_bt("rand", 0.0020, seed=s) for s in range(8)]
rc = np.mean([stats.__wrapped__ if False else (1+x).cumprod().iloc[-1]**(12/len(x))-1 for x in rnd])
rsh = np.mean([x.mean()/(x.std()+1e-12)*np.sqrt(12) for x in rnd])
print(f"   {'RANDOM quintile (avg8) NET 20bps':34s} CAGR {rc*100:5.1f}%  Sharpe {rsh:.2f}", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
