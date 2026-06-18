"""Experiment 28 — NOVEL FREE DATA: FINRA daily short-volume -> short-selling
pressure panel (consolidated from raw daily files nobody sells pre-packaged).
Signal = short-volume ratio SVR = ShortVolume/TotalVolume (Boehmer-Jones-Zhang:
heavily-shorted stocks underperform). Build monthly SVR panel for S&P500
universe, test cross-sectional IC + quintile spread vs forward returns, with
random/equal-weight CONTROLS, TRAIN(2017-21)/TEST(2021-26)."""
import warnings, time, io, urllib.request, ssl, calendar
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from scipy.stats import spearmanr
t0 = time.time()
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "research viktormashalov@gmail.com"}
uni = set(open("/tmp/wave/sp500_universe.txt").read().split())
uni |= set(l.strip() for l in open("/tmp/wave/xuniverse_ndx.txt") if l.strip())
print(f"universe {len(uni)} names", flush=True)


def get(u):
    try:
        return urllib.request.urlopen(urllib.request.Request(u, headers=UA),
                                      timeout=30, context=ctx).read()
    except Exception:
        return None


# ---- download sampled FINRA daily short-vol files, build monthly SVR ----
recs = []
for yr in range(2017, 2027):
    for mo in range(1, 13):
        if (yr, mo) > (2026, 5):
            break
        for day in (6, 13, 20, 27):
            d = pd.Timestamp(yr, mo, day)
            if d.weekday() >= 5:
                d += pd.Timedelta(days=7 - d.weekday())
            ds = d.strftime("%Y%m%d")
            raw = get(f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{ds}.txt")
            if not raw:
                continue
            try:
                df = pd.read_csv(io.StringIO(raw.decode()), sep="|").iloc[:-1]
            except Exception:
                continue
            df = df[df.Symbol.isin(uni)]
            tot = df.groupby("Symbol")[["ShortVolume", "TotalVolume"]].sum()
            svr = (tot.ShortVolume / tot.TotalVolume.replace(0, np.nan))
            for sym, v in svr.items():
                if np.isfinite(v):
                    recs.append((pd.Timestamp(yr, mo, 1), sym, v))
    print(f"  through {yr}: {len(recs)} records  t={time.time()-t0:.0f}s", flush=True)

P = pd.DataFrame(recs, columns=["ym", "sym", "svr"])
SVR = P.groupby(["ym", "sym"]).svr.mean().unstack()        # month x symbol
SVR.to_pickle("/tmp/wave/_shortvol.pkl")
print(f"\nSVR panel {SVR.shape}  t={time.time()-t0:.0f}s", flush=True)

# ---- prices for forward returns ----
names = [s for s in SVR.columns]
px = yf.download(names + ["QQQ"], start="2016-06-01", auto_adjust=True,
                 progress=False)["Close"]
me = px.resample("ME").last()
me.index = me.index.to_period("M").to_timestamp()          # align to month-start
q = me["QQQ"]
def fwd(h): return (me.shift(-h) / me - 1).sub(q.shift(-h) / q - 1, axis=0)

# ---- IC + quintile spread, with controls, train/test ----
def analyze(lo, hi, tag):
    print(f"\n{tag}:", flush=True)
    for h in (1, 3):
        f = fwd(h)
        ics, qspread, randspread = [], [], []
        rng = np.random.default_rng(0)
        for d in SVR.index:
            if not (pd.Timestamp(lo) <= d < pd.Timestamp(hi)):
                continue
            s = SVR.loc[d].dropna()
            common = [c for c in s.index if c in f.columns and np.isfinite(f.loc[d, c])]
            if len(common) < 50:
                continue
            x = s[common]; y = f.loc[d, common]
            ics.append(spearmanr(x, y).correlation)
            # quintiles: LOW short-vol (bullish) minus HIGH short-vol
            lowq = y[x <= x.quantile(0.2)].mean(); highq = y[x >= x.quantile(0.8)].mean()
            qspread.append(lowq - highq)
            # random control: two random groups, same sizes
            n5 = max(5, len(common) // 5)
            randspread.append(np.mean([y.sample(n5, random_state=int(rng.integers(1e6))).mean()
                                       - y.sample(n5, random_state=int(rng.integers(1e6))).mean()
                                       for _ in range(10)]))
        ics = np.array(ics); qs = np.array(qspread); rs = np.array(randspread)
        print(f"  fwd{h}m: IC {ics.mean():+.4f} (t {ics.mean()/(ics.std()+1e-12)*np.sqrt(len(ics)):+.1f})"
              f"  low-minus-high-SVR {qs.mean()*100:+.2f}%/mo (t {qs.mean()/(qs.std()+1e-12)*np.sqrt(len(qs)):+.1f})"
              f"  | RANDOM control {rs.mean()*100:+.2f}%  n={len(ics)}", flush=True)

analyze("2017-01-01", "2021-01-01", "TRAIN 2017-2020")
analyze("2021-01-01", "2026-06-01", "TEST 2021-2025")
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
