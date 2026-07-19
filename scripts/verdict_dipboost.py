"""Provenance for verdict.html §7b: (1) the dip-boost DCA strategy — optimal
extra-contribution threshold and its measured lift; (2) reserve-based dip
buying at equal dollars (the honest control); (3) selling-when-expensive
modeled. All on QQQ monthly, 1999-2026, from the committed panel.

Run:  python3 scripts/verdict_dipboost.py
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_df = pd.read_csv(f"{ROOT}/data/etfs/QQQ.csv", parse_dates=["Date"]).set_index("Date")
_ser = _df["Adj Close"] if ("Adj Close" in _df.columns and _df["Adj Close"].notna().sum() > 100) else _df["Close"]
q = _ser.resample("ME").last().dropna()
r = q.pct_change().fillna(0.0)
idx = q.index
dd = q/q.cummax() - 1.0

# ---------- 1) forward returns by drawdown bucket (fixed horizon, no time conflation)
print("=== forward 3y annualized return of a dollar invested, by QQQ drawdown at purchase ===")
f36 = (q.shift(-36)/q)**(12/36) - 1
buckets = [("at/near ATH (dd 0..-5%)", (dd >= -0.05)),
           ("-5..-15%", (dd < -0.05) & (dd >= -0.15)),
           ("-15..-25%", (dd < -0.15) & (dd >= -0.25)),
           ("-25..-40%", (dd < -0.25) & (dd >= -0.40)),
           ("deeper than -40%", (dd < -0.40))]
fwd_by_bucket = []
for nm, m in buckets:
    v = f36[m].dropna()
    fwd_by_bucket.append((nm, round(float(v.mean())*100, 1), int(m.sum()), round(float((v > 0).mean())*100)))
    print(f"  {nm:26} mean fwd-3y {v.mean()*100:+5.1f}%/yr  (months: {int(m.sum())}, positive {float((v>0).mean())*100:.0f}%)")

# ---------- 2) dip-boost sweep: base $1k/mo + EXTRA $1k when dd <= -X%
def dca_wealth(contrib):
    v = 0.0
    for c, x in zip(contrib, r.values):
        v = (v + c)*(1 + x)
    return v
base_contrib = np.full(len(idx), 1000.0)
W0 = dca_wealth(base_contrib); C0 = base_contrib.sum()
print(f"\nbaseline DCA: ${W0:,.0f} on ${C0:,.0f}  ({W0/C0:.2f}x per $)")
print("=== dip-boost: +$1,000 extra in months with drawdown <= -X% ===")
print(f"{'X':>4} {'months':>7} {'extra$':>9} {'final':>12} {'lift':>9} {'per-extra-$ mult':>17} {'vs base mult':>13}")
boost_rows = []
for X in [5, 10, 15, 20, 25, 30, 40]:
    m = (dd.values <= -X/100)
    contrib = base_contrib + 1000.0*m
    W = dca_wealth(contrib); extra = 1000.0*m.sum()
    per_extra = (W - W0)/extra if extra > 0 else np.nan
    boost_rows.append((X, int(m.sum()), extra, W, per_extra))
    print(f"{X:>4} {int(m.sum()):>7} {extra:>9,.0f} {W:>12,.0f} {(W-W0)/W0*100:>8.1f}% {per_extra:>17.2f}x {W0/C0:>12.2f}x")

# ---------- 3) the honest control: reserve part of the SAME money for dips
print("\n=== reserve control (same total $): hold back f% of each contribution, deploy reserve at dd <= -X% ===")
print(f"{'f':>4} {'X':>4} {'final':>12} {'vs plain DCA':>13}")
for f in [0.25, 0.5]:
    for X in [10, 20, 30]:
        v = 0.0; res = 0.0
        for i in range(len(idx)):
            res += 1000.0*f
            c = 1000.0*(1 - f)
            if dd.values[i] <= -X/100:
                c += res; res = 0.0
            v = (v + c)*(1 + r.values[i])
        v += res
        print(f"{int(f*100):>3}% {X:>4} {v:>12,.0f} {(v-W0)/W0*100:>+12.1f}%")

# ---------- 4) selling when "expensive": sell after big run-ups, re-enter on dips
print("\n=== 'market is too expensive' seller: sell all when trailing-24m return > +T%, re-enter at -X% dip; contributions continue ===")
r24 = q/q.shift(24) - 1
print(f"{'T':>5} {'X':>4} {'final':>12} {'vs DCA-hold':>12} {'months in cash':>15}")
for T in [60, 80, 100]:
    for X in [15, 20, 30]:
        v_eq = 0.0; v_cash = 0.0; in_mkt = True; peak_since_exit = None; months_cash = 0
        for i in range(len(idx)):
            if in_mkt:
                v_eq = (v_eq + 1000.0)*(1 + r.values[i])
                t24 = r24.values[i]
                if np.isfinite(t24) and t24 > T/100:
                    v_cash = v_eq; v_eq = 0.0; in_mkt = False; peak_since_exit = q.values[i]
            else:
                months_cash += 1
                v_cash += 1000.0
                peak_since_exit = max(peak_since_exit, q.values[i])
                if q.values[i]/peak_since_exit - 1 <= -X/100:
                    v_eq = v_cash; v_cash = 0.0; in_mkt = True
        W = v_eq + v_cash
        print(f"{T:>4}% {X:>4} {W:>12,.0f} {(W-W0)/W0*100:>+11.1f}% {months_cash:>15}")

# ---------- 5) the ATH fear check
ath = dd >= -0.001
f12 = q.shift(-12)/q - 1
print("\n=== is an all-time high a dangerous time to buy? ===")
print(f"  fwd-1y from ATH months:   mean {float(f12[ath].dropna().mean())*100:+.1f}%, positive {float((f12[ath].dropna()>0).mean())*100:.0f}%  (n={int(ath.sum())})")
print(f"  fwd-1y from ALL months:   mean {float(f12.dropna().mean())*100:+.1f}%, positive {float((f12.dropna()>0).mean())*100:.0f}%")
