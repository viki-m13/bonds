"""LOOP iter 6a — precision-by-conviction-bucket curve, and the 'shoulder'
basket. If tail crowding poisons the top of the ML score, the best-precision
bucket is interior. Map P(true fwd-12m top decile | score bucket), then
backtest baskets drawn from the best bucket through the mandate harness.
Guard: bucket choice uses 2015-2020 only (dev); 2021-2026 is the honest test.
"""
import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
sys.path.insert(0, HERE)
from engine import dca_benchmark, stats

t0 = time.time()
def p(*a): print(*a, flush=True)
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
FEAT, liq, me, dv, bench, cols = D["FEAT"], D["liq"], D["me"], D["dv"], D["bench"], D["cols"]
M = me.index
ma10 = me.rolling(10, min_periods=10).mean()
qqq = bench["QQQ"].reindex(M)
PROB = pd.read_pickle(f"{HERE}/_mlprob.pkl").reindex(M).reindex(columns=cols)
dvr = dv.rank(axis=1, ascending=False)
top1500 = dvr <= 1500
fwd12 = (me.shift(-12) / me - 1)
FR = fwd12.where(liq & top1500).rank(axis=1, pct=True)
SCR = PROB.where(liq & top1500).rank(axis=1, pct=True)

def curve(lo, hi, label):
    buckets = np.linspace(0.5, 1.0, 11)
    p(f"\nprecision curve {label}: P(true top-decile | score bucket) and mean fwd12 rank")
    for b0, b1 in zip(buckets[:-1], buckets[1:]):
        hits = tot = 0; mr = []
        for dt in M[(M >= lo) & (M <= hi)]:
            el = liq.loc[dt] & top1500.loc[dt] & (me.loc[dt] >= 3) & (me.loc[dt] > ma10.loc[dt])
            s = SCR.loc[dt].where(el)
            sel = s[(s >= b0) & (s < b1)].index
            f = FR.loc[dt].reindex(sel).dropna()
            hits += (f >= 0.9).sum(); tot += len(f); mr.extend(f.tolist())
        p(f"  score [{b0:.2f},{b1:.2f}): precision {hits/max(tot,1):5.1%}  mean fwd rank {np.mean(mr) if mr else np.nan:.3f}  n={tot}")

curve(pd.Timestamp("2015-01-01"), pd.Timestamp("2020-12-01"), "DEV 2015-2020")
curve(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-06-01"), "TEST 2021-2025")

# shoulder basket: pick k names randomly-but-reproducibly (by secondary $vol rank)
# from the chosen bucket, mandate mechanics
def run_bucket(b0, b1, start, end, k=8, contrib=1000.0, cost=0.002, trail=-0.30):
    dates = M[(M >= start) & (M <= end)]
    pos = {}; qqq_units = 0.0; qqq_entry = None; cash = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        prow = me.loc[dt]; qp = qqq.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]; cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]; e["last_px"] = cp; e["peak_px"] = max(e["peak_px"], cp)
            else:
                e["val"] *= 0.75; cash += e["val"] * (1 - cost); pos.pop(tk)
        el = liq.loc[dt] & top1500.loc[dt] & (me.loc[dt] >= 3) & (me.loc[dt] > ma10.loc[dt])
        s = SCR.loc[dt].where(el)
        bucket = s[(s >= b0) & (s < b1)]
        marow = ma10.loc[dt]
        for tk in list(pos.keys()):
            e = pos[tk]
            if (dt - e["entry_date"]).days < 30: continue
            cp = e["last_px"]
            if (cp / e["peak_px"] - 1) <= trail or (np.isfinite(marow.get(tk, np.nan)) and cp < marow[tk]):
                cash += e["val"] * (1 - cost); pos.pop(tk)
        cash += contrib; contributed += contrib
        cand = dv.loc[dt].reindex(bucket.index).dropna().sort_values(ascending=False)
        cand = cand[~cand.index.isin(pos)]
        need = k - len(pos)
        if need > 0 and cash > 1e-9 and len(cand):
            if qqq_units > 0 and qqq_entry is not None and (dt - qqq_entry).days >= 30:
                cash += qqq_units * qp * (1 - 0.0005); qqq_units = 0.0; qqq_entry = None
            picks = list(cand.index[:need]); amt = cash / len(picks)
            for tk in picks:
                pos[tk] = {"val": amt * (1 - cost), "last_px": prow[tk], "peak_px": prow[tk], "entry_date": dt}
            cash = 0.0
        elif cash > 1e-9 and len(pos):
            for tk in list(pos)[:3]:
                pos[tk]["val"] += (cash / min(3, len(pos))) * (1 - cost)
            cash = 0.0
        if cash > 1e-9 and np.isfinite(qp):
            qqq_units += cash * (1 - 0.0005) / qp
            if qqq_entry is None: qqq_entry = dt
            cash = 0.0
        V = cash + sum(e["val"] for e in pos.values()) + (qqq_units * qp if np.isfinite(qp) else 0)
        rows.append((dt, V, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date"), dates

p("\nBucket baskets (k=8, $vol tiebreak) — DEV chooses, TEST judges:")
for b0, b1 in [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.70, 0.80)]:
    line = f"  bucket [{b0:.2f},{b1:.2f}):"
    for st, en in [("2015-01", "2020-12"), ("2021-01", "2026-06")]:
        eq, dts = run_bucket(b0, b1, pd.Timestamp(st + "-01"), pd.Timestamp(en + "-01"))
        b = dca_benchmark(bench["QQQ"], dts)
        line += f"  {st[:4]}-{en[:4]} {stats(eq)['final']/b['V'].iloc[-1]:5.2f}x"
    p(line)
p(f"\nDONE t={time.time()-t0:.0f}s")
