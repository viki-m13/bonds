"""Fast research engine for the corporate panel.

Three pieces, built for many-loop experimentation without sacrificing honesty:

1. Compact cache — per-bond numpy arrays (float32) pickled to
   corps/data/cache_bonds.pkl. Loads in ~30-60s vs ~8min for the pandas path.
   Build once:  python corps/research/engine2.py build

2. Fast event engine — pure-numpy port of munis/research/backtest.py
   run_signal/matched_random_control with IDENTICAL honesty rules:
   signal on day t uses data through t only; entry at the first customer-ask
   print strictly AFTER t (<=7d); exit at the first customer-bid print after
   min_hold, hard stop at max_hold using the last bid (stale flagged);
   trailing-liquidity eligibility gate excludes today; per-bond cooldown.

3. Honest mark-to-market — daily portfolio NAV marked at each bond's actual
   mid prints (stale marks stay flat — disclosed), fills at bid/ask, coupon
   accrued daily, equal-weight across open positions, cash earns the 3M T-bill.
   Sharpe is computed at MONTHLY frequency vs T-bill (bond marks are stale at
   daily frequency, which would overstate daily Sharpe), with daily and annual
   variants reported for robustness.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache_bonds.pkl"
MIN_ACTIVE_DAYS_90 = 8


# ---------------------------------------------------------------- cache build

def build_cache():
    sys.path.insert(0, str(ROOT / "research"))
    from panel_io import load_full
    cols = ["six", "date", "mid", "s_px", "p_px", "ytw", "cs", "mat", "qvolume"]
    p = load_full(columns=cols)
    p["mat_yr"] = p["mat"].astype("int64").clip(0, 200).astype(np.int16)
    p = p.drop(columns=["mat"])
    p["day"] = p["date"].values.astype("datetime64[D]").astype(np.int32)
    p = p.drop(columns=["date"])
    coup = p.groupby("six")["ytw"].median().clip(1, 12)
    print(f"{p['six'].nunique()} bonds, {len(p)} rows; packing ...", flush=True)
    bonds = {}
    for six, g in p.groupby("six"):
        day = g["day"].to_numpy()
        order = np.argsort(day, kind="stable")
        day = day[order].astype(np.int32)
        # trailing-90d active-day count EXCLUDING today (same as bt.prepare)
        active = (np.searchsorted(day, day) -
                  np.searchsorted(day, day - 90)).astype(np.int16)
        bonds[six] = {
            "day": day,
            "mid": g["mid"].to_numpy(np.float32)[order],
            "s_px": g["s_px"].to_numpy(np.float32)[order],
            "p_px": g["p_px"].to_numpy(np.float32)[order],
            "cs": g["cs"].to_numpy(np.float32)[order],
            "qv": g["qvolume"].to_numpy(np.float32)[order],
            "mat": g["mat_yr"].to_numpy(np.int16)[order],
            "elig": active >= MIN_ACTIVE_DAYS_90,
            "coupon": float(coup.get(six, 4.5)),
        }
    with open(CACHE, "wb") as f:
        pickle.dump(bonds, f, protocol=5)
    print(f"wrote {CACHE} ({CACHE.stat().st_size/1e9:.2f} GB)", flush=True)


_BONDS = None


def load_cache():
    global _BONDS
    if _BONDS is None:
        with open(CACHE, "rb") as f:
            _BONDS = pickle.load(f)
    return _BONDS


# ---------------------------------------------------------------- event engine

@dataclass
class Fill:
    six: str
    entry_day: int      # days since epoch
    entry_px: float
    exit_day: int
    exit_px: float
    coupon: float
    stale: bool
    hold: int = field(init=False)
    ret: float = field(init=False)

    def __post_init__(self):
        self.hold = int(self.exit_day - self.entry_day)
        acc = self.coupon / 100.0 / 365.0 * self.hold * 100.0
        self.ret = (self.exit_px - self.entry_px + acc) / self.entry_px


def D(ts: str) -> int:
    return int(np.datetime64(ts, "D").astype(np.int64))


def run_events(bonds, signal_fn, min_hold=365, max_hold=455,
               date_lo=None, date_hi=None, cooldown=30, extra_gate=None):
    """signal_fn(b) -> bool array over b's rows (True = enter at next ask).
    extra_gate(b) -> bool array further restricting eligibility (point-in-time
    fields only). Entry/exit honesty rules identical to backtest.run_signal."""
    fills = []
    for six, b in bonds.items():
        sig = signal_fn(b)
        if sig is None:
            continue
        gate = b["elig"]
        if extra_gate is not None:
            g2 = extra_gate(b)
            if g2 is None:
                continue
            gate = gate & g2
        idx = np.flatnonzero(sig & gate)
        if not len(idx):
            continue
        day = b["day"]
        smask = ~np.isnan(b["s_px"]); s_day = day[smask]; s_px = b["s_px"][smask]
        pmask = ~np.isnan(b["p_px"]); p_day = day[pmask]; p_px = b["p_px"][pmask]
        if not len(s_day) or not len(p_day):
            continue
        last_exit = -10**9
        for i in idx:
            sd = day[i]
            j = np.searchsorted(s_day, sd, side="right")
            if j >= len(s_day) or s_day[j] - sd > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            if date_lo is not None and ed < date_lo:
                continue
            if date_hi is not None and ed > date_hi:
                continue
            if ed - last_exit < cooldown:
                continue
            lo = ed + min_hold; hi = ed + max_hold
            k = np.searchsorted(p_day, lo, side="left")
            if k < len(p_day) and p_day[k] <= hi:
                xd, xp, st = int(p_day[k]), float(p_px[k]), False
            else:
                k2 = np.searchsorted(p_day, hi, side="right") - 1
                if k2 < 0:
                    continue
                xd, xp, st = hi, float(p_px[k2]), True
            fills.append(Fill(six, ed, ep, xd, xp, b["coupon"], st))
            last_exit = xd
    return fills


def matched_control(bonds, fills, min_hold=365, max_hold=455, n_draws=15,
                    extra_gate=None, seed=7):
    """Random-entry control: same bonds, same eligibility gate, same window,
    same exit logic; n_draws random entries per real fill."""
    rng = np.random.default_rng(seed)
    per_bond = {}
    for f in fills:
        per_bond.setdefault(f.six, []).append(f)
    rets = []
    for six, fl in per_bond.items():
        b = bonds[six]
        gate = b["elig"]
        if extra_gate is not None:
            g2 = extra_gate(b)
            gate = gate & g2 if g2 is not None else gate
        day = b["day"]
        smask = ~np.isnan(b["s_px"]); s_day = day[smask]; s_px = b["s_px"][smask]
        pmask = ~np.isnan(b["p_px"]); p_day = day[pmask]; p_px = b["p_px"][pmask]
        lo = min(f.entry_day for f in fl); hi = max(f.entry_day for f in fl)
        cand = np.flatnonzero(gate & (day >= lo - 30) & (day <= hi + 30))
        if not len(cand):
            continue
        picks = rng.choice(cand, size=n_draws * len(fl), replace=True)
        for i in picks:
            sd = day[i]
            j = np.searchsorted(s_day, sd, side="right")
            if j >= len(s_day) or s_day[j] - sd > 7:
                continue
            ed = int(s_day[j]); ep = float(s_px[j])
            lo2 = ed + min_hold; hi2 = ed + max_hold
            k = np.searchsorted(p_day, lo2, side="left")
            if k < len(p_day) and p_day[k] <= hi2:
                xd, xp = int(p_day[k]), float(p_px[k])
            else:
                k2 = np.searchsorted(p_day, hi2, side="right") - 1
                if k2 < 0:
                    continue
                xd, xp = hi2, float(p_px[k2])
            acc = bonds[six]["coupon"] / 100.0 / 365.0 * (xd - ed) * 100.0
            rets.append((xp - ep + acc) / ep)
    return np.array(rets)


def summarize(fills, control=None, n_boot=2000, seed=11):
    if not fills:
        return {"n": 0}
    r = np.array([f.ret for f in fills])
    out = {"n": len(fills), "n_bonds": len({f.six for f in fills}),
           "mean_ret": float(r.mean()), "median_ret": float(np.median(r)),
           "win_rate": float((r > 0).mean()),
           "mean_hold": float(np.mean([f.hold for f in fills])),
           "stale_share": float(np.mean([f.stale for f in fills]))}
    if control is not None and len(control):
        ex = out["mean_ret"] - float(control.mean())
        out["excess_vs_control"] = ex
        # cluster bootstrap by bond
        by = {}
        for f in fills:
            by.setdefault(f.six, []).append(f.ret)
        keys = list(by)
        rng = np.random.default_rng(seed)
        cm = float(control.mean())
        stats = []
        for _ in range(n_boot):
            ks = rng.choice(len(keys), size=len(keys), replace=True)
            vals = np.concatenate([np.asarray(by[keys[k]]) for k in ks])
            stats.append(vals.mean() - cm)
        stats = np.array(stats)
        out["excess_p_boot"] = float((stats <= 0).mean())
    return out


# ------------------------------------------------------- honest mark-to-market

def load_rf(days):
    """Daily risk-free (3M T-bill, decimal/yr) aligned to `days` (int days)."""
    f = ROOT / "data" / "etf" / "DGS3MO.csv"
    if not f.exists():
        return np.zeros(len(days))
    df = pd.read_csv(f)
    df.columns = [c.lower() for c in df.columns]
    dcol = [c for c in df.columns if "date" in c][0]
    vcol = [c for c in df.columns if c != dcol][0]
    d = pd.to_datetime(df[dcol]).values.astype("datetime64[D]").astype(np.int64)
    v = pd.to_numeric(df[vcol], errors="coerce").to_numpy() / 100.0
    ok = ~np.isnan(v)
    d, v = d[ok], v[ok]
    idx = np.clip(np.searchsorted(d, days, side="right") - 1, 0, len(v) - 1)
    return v[idx]


def mtm_nav(bonds, fills, rf_in_cash=True):
    """Daily NAV from real mid marks. Equal-weight daily across open positions.
    Position return on day t = (mark_t + accrual) / mark_{t-1} - 1, where marks
    only change on the bond's actual print days (stale = flat). Entry basis is
    the ask fill; final mark is the bid fill (spread honestly paid).
    Returns (days, nav, daily_ret)."""
    if not fills:
        return None
    d0 = min(f.entry_day for f in fills); d1 = max(f.exit_day for f in fills)
    days = np.arange(d0, d1 + 1)
    n = len(days)
    sret = np.zeros(n); cnt = np.zeros(n)
    for f in fills:
        b = bonds[f.six]
        day = b["day"]; mid = b["mid"]
        i0 = np.searchsorted(day, f.entry_day, side="right")
        i1 = np.searchsorted(day, f.exit_day, side="left")
        # mark path: entry ask -> mids strictly inside (entry, exit) -> exit bid
        mds = [f.entry_day]; mks = [f.entry_px]
        for i in range(i0, i1):
            if not np.isnan(mid[i]):
                mds.append(int(day[i])); mks.append(float(mid[i]))
        mds.append(f.exit_day); mks.append(f.exit_px)
        acc = f.coupon / 100.0 / 365.0 * 100.0   # $ accrual per day per 100 face
        for k in range(1, len(mds)):
            gap = mds[k] - mds[k - 1]
            if gap <= 0:
                continue
            tot = (mks[k] + acc * gap) / mks[k - 1] - 1.0
            dr = (1.0 + tot) ** (1.0 / gap) - 1.0
            a = mds[k - 1] + 1 - d0; z = mds[k] + 1 - d0
            sret[a:z] += dr
            cnt[a:z] += 1
    rf = load_rf(days) if rf_in_cash else np.zeros(n)
    daily = np.where(cnt > 0, sret / np.where(cnt > 0, cnt, 1), rf / 365.0)
    nav = np.cumprod(1.0 + daily)
    return days, nav, daily


def perf_stats(days, nav, daily):
    """Honest performance: CAGR, maxDD, and Sharpe at monthly (headline),
    daily and annual frequencies, all vs the 3M T-bill."""
    yrs = (days[-1] - days[0]) / 365.25
    cagr = nav[-1] ** (1 / yrs) - 1
    peak = np.maximum.accumulate(nav)
    maxdd = float((nav / peak - 1).min())
    rf = load_rf(days)
    ts = pd.Series(daily, index=pd.to_datetime(days, unit="D"))
    rfs = pd.Series(rf / 365.0, index=ts.index)

    def sharpe(freq, ann):
        r = (1 + ts).resample(freq).prod() - 1
        rr = (1 + rfs).resample(freq).prod() - 1
        ex = (r - rr).dropna()
        if len(ex) < 3 or ex.std() == 0:
            return None
        return float(ex.mean() / ex.std() * np.sqrt(ann))

    return {"cagr": float(cagr), "total": float(nav[-1] - 1), "maxdd": maxdd,
            "years": round(float(yrs), 1),
            "sharpe_m": sharpe("ME", 12), "sharpe_d": sharpe("D", 365),
            "sharpe_a": sharpe("YE", 1),
            "vol_m_ann": float(((1 + ts).resample("ME").prod() - 1).std() * np.sqrt(12))}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "build":
        build_cache()
