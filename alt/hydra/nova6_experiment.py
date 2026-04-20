"""NOVA6 — Stat-arb pairs mean-reversion ensemble.

NOVA2-5 all ceilinged around SR 0.83 (even MVO on 26 trend-based
sleeves hits only 1.09 full-sample / 0.66 OOS). The bottleneck is
that trend sleeves are all 0.2-0.3 correlated to each other because
they share SPY/risk-on exposure.

Stat-arb PAIRS mean-reversion is the one family genuinely uncorrelated
with trend. Design:

  - 5 ETF pair spreads (ratio, dollar-neutral):
      SPY/QQQ, TLT/IEF, GLD/SLV, XLY/XLP, EFA/SPY
  - Z-score of log-ratio vs 60d MA (60d std).
  - Enter long-pair when z < -2 (spread stretched low → revert up)
  - Enter short-pair when z > +2 (spread stretched high → revert down)
  - Exit when |z| < 0.5
  - Hold up to 20 trading days max (safety)
  - Dollar-neutral +1 long / -1 short, 15 bps TC each leg

Plus the strong trend sleeves from NOVA5. Equal-weight."""
from pathlib import Path
import numpy as np
import pandas as pd

from hydra_core import load_etf, load_fred, stats


TC_BPS = 15.0


def monthly_first_flag(index):
    out = pd.Series(False, index=index)
    out.iloc[0] = True
    for i in range(1, len(index)):
        if index[i].month != index[i - 1].month:
            out.iloc[i] = True
    return out


def _rebal_to_monthly(raw):
    first = monthly_first_flag(raw.index)
    return raw.where(first, np.nan).ffill().fillna(False).astype(bool)


def binary_long(tic, signal, dates, off_tic=None, tc_bps=TC_BPS):
    L = load_etf(tic).reindex(dates).ffill().pct_change().fillna(0)
    sig = signal.reindex(dates).fillna(False).astype(bool).shift(1).fillna(False)
    if off_tic is None:
        r = pd.Series(np.where(sig, L, 0), index=dates)
    else:
        B = load_etf(off_tic).reindex(dates).ffill().pct_change().fillna(0)
        r = pd.Series(np.where(sig, L, B), index=dates)
    changes = sig.astype(int).diff().abs().fillna(0)
    tc = changes * (tc_bps / 1e4) * (2 if off_tic else 1)
    return r - tc


def ttm_signal(tic, dates, days):
    p = load_etf(tic).reindex(dates).ffill()
    sma = p.rolling(days).mean()
    return _rebal_to_monthly(p > sma)


# ---------- pairs mean-reversion ----------

def pair_mean_reversion(long_tic, short_tic, dates, lookback=60,
                         entry=2.0, exit=0.5, max_hold=20):
    """Trade z-score of log-ratio long_tic/short_tic.
      z = (log(L/S) - rolling_mean) / rolling_std (over lookback days)
      - When z < -entry: go long L, short S (expect spread to rise)
      - When z > +entry: go short L, long S (expect spread to fall)
      - Exit when |z| < exit OR held for max_hold days
    Dollar-neutral, 15 bps TC each leg on entry & exit."""
    L = load_etf(long_tic).reindex(dates).ffill()
    S = load_etf(short_tic).reindex(dates).ffill()
    log_ratio = np.log(L / S)
    m = log_ratio.rolling(lookback).mean()
    v = log_ratio.rolling(lookback).std()
    z = (log_ratio - m) / v

    position = pd.Series(0, index=dates)  # +1 long L-short S, -1 short L-long S
    hold = 0
    pos = 0
    for i, d in enumerate(dates):
        zi = z.iloc[i] if not pd.isna(z.iloc[i]) else 0
        if pos == 0:
            if zi < -entry:
                pos = 1
                hold = 0
            elif zi > entry:
                pos = -1
                hold = 0
        else:
            hold += 1
            if abs(zi) < exit or hold >= max_hold:
                pos = 0
                hold = 0
        position.iloc[i] = pos

    # Returns: signal is position at t-1 applied to (L_ret - S_ret) at t
    rL = L.pct_change().fillna(0)
    rS = S.pct_change().fillna(0)
    sig = position.shift(1).fillna(0)
    r = sig * (rL - rS)
    # TC on every position change (2 legs)
    changes = position.diff().abs().fillna(0)
    tc = changes * (TC_BPS / 1e4) * 2
    return (r - tc).rename(f"pair_{long_tic}_{short_tic}")


PAIRS = [
    ("SPY", "QQQ"), ("TLT", "IEF"), ("GLD", "SLV"),
    ("XLY", "XLP"), ("EFA", "SPY"), ("HYG", "IEF"),
    ("EEM", "SPY"), ("XLK", "XLF"), ("XLE", "XLI"),
]


def make_pair_sleeve(lt, st, lb=60, ent=2.0, ex=0.5):
    name = f"pair_{lt}_{st}_{lb}"
    def _fn(dates):
        s = pair_mean_reversion(lt, st, dates, lookback=lb, entry=ent, exit=ex)
        return s.rename(name)
    _fn.__name__ = name
    return _fn


PAIR_SLEEVES = [make_pair_sleeve(lt, st) for lt, st in PAIRS]


# ---------- trend sleeves (from NOVA5) ----------

TREND_ASSETS = [
    ("SPY", "IEF"), ("QQQ", "IEF"), ("EEM", "IEF"), ("EFA", "IEF"),
    ("VNQ", "IEF"), ("GLD", "BIL"), ("TLT", "BIL"),
]
LOOKBACKS = [126, 252]


def make_trend_sleeve(tic, off, lb):
    name = f"trend_{tic}_{lb}"
    def _fn(dates):
        sig = ttm_signal(tic, dates, lb)
        return binary_long(tic, sig, dates, off_tic=off).rename(name)
    _fn.__name__ = name
    return _fn


TREND_SLEEVES = [make_trend_sleeve(t, o, lb) for t, o in TREND_ASSETS for lb in LOOKBACKS]


# ---------- misc ----------

def s_gem(dates):
    spy = load_etf("SPY").reindex(dates).ffill()
    efa = load_etf("EFA").reindex(dates).ffill()
    bil = load_etf("BIL").reindex(dates).ffill()
    ief = load_etf("IEF").reindex(dates).ffill()
    mom_spy = spy.pct_change(252)
    mom_efa = efa.pct_change(252)
    mom_bil = bil.pct_change(252)
    risk_on = _rebal_to_monthly(mom_spy > mom_bil)
    spy_wins = _rebal_to_monthly(mom_spy > mom_efa)
    sig_spy = (risk_on & spy_wins).shift(1).fillna(False)
    sig_efa = (risk_on & ~spy_wins).shift(1).fillna(False)
    sig_ief = (~risk_on).shift(1).fillna(False)
    r = (sig_spy.astype(float) * spy.pct_change().fillna(0) +
         sig_efa.astype(float) * efa.pct_change().fillna(0) +
         sig_ief.astype(float) * ief.pct_change().fillna(0))
    mat = pd.DataFrame({"spy": sig_spy, "efa": sig_efa, "ief": sig_ief}).astype(int)
    changes = mat.diff().abs().sum(axis=1).fillna(0)
    return (r - changes * (TC_BPS / 1e4)).rename("s_gem")


def s_halloween(dates):
    sig = pd.Series([d.month in [11, 12, 1, 2, 3, 4] for d in dates], index=dates)
    return binary_long("SPY", sig, dates).rename("s_halloween")


def s_gold_crisis(dates):
    gld = load_etf("GLD").reindex(dates).ffill()
    spy = load_etf("SPY").reindex(dates).ffill()
    raw = (gld.pct_change(63) > 0) & (spy.pct_change(63) < 0)
    sig = _rebal_to_monthly(raw)
    return binary_long("GLD", sig, dates).rename("s_gold_crisis")


# ---------- ensemble ----------

def build_pairs_only(dates):
    out = {}
    for fn in PAIR_SLEEVES:
        print(f"  Building {fn.__name__}...", flush=True)
        s = fn(dates)
        out[s.name] = s
    return pd.DataFrame(out).reindex(dates).fillna(0)


def build_combined(dates):
    sleeves = PAIR_SLEEVES + TREND_SLEEVES + [s_gem, s_halloween, s_gold_crisis]
    out = {}
    for fn in sleeves:
        print(f"  Building {fn.__name__}...", flush=True)
        s = fn(dates)
        out[s.name] = s
    return pd.DataFrame(out).reindex(dates).fillna(0)


def summarise_ensemble(df, label):
    valid = (df != 0).sum(axis=1) >= 3
    corr = df[valid].corr()
    tri = corr.values[np.triu_indices_from(corr, k=1)]
    print(f"  |corr| mean={np.mean(np.abs(tri)):.3f}  median={np.median(np.abs(tri)):.3f}  max={np.max(np.abs(tri)):.2f}")

    live = (df != 0).cummax().astype(float)
    w = live.div(live.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    port = (w * df).sum(axis=1)
    nz = (df != 0).any(axis=1)
    port = port[nz]
    s = stats(port, label)
    print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    IS = pd.Timestamp("2018-01-01")
    for p, lbl in [(port.loc[:IS], f"{label} IS"), (port.loc[IS:], f"{label} OOS")]:
        s = stats(p, lbl)
        print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  MDD={s['mdd']:>7.2f}%")
    return port


def main():
    spy = load_etf("SPY")
    dates = spy.index
    print(f"Universe: {dates[0].date()} .. {dates[-1].date()} ({len(dates)/252:.1f}y)")

    print("\n=== PAIRS ONLY ===")
    df_pairs = build_pairs_only(dates)
    print("\nPer-pair stats:")
    for c in df_pairs.columns:
        s = stats(df_pairs[c], c)
        print(f"  {c:25s} SR={s['sharpe']:>5.2f}  Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%")
    port_pairs = summarise_ensemble(df_pairs, "PAIRS eq-wt")

    print("\n=== PAIRS + TREND + GEM + CAL COMBINED ===")
    df_comb = build_combined(dates)
    port_comb = summarise_ensemble(df_comb, "COMBINED eq-wt")

    # Leverage options
    for tag, port in [("PAIRS", port_pairs), ("COMB", port_comb)]:
        nv = port.std() * np.sqrt(252)
        for tgt_str, tgt in [("10%", 0.10), ("20%", 0.20)]:
            lev = tgt / nv
            s = stats(port * lev, f"{tag} {lev:.2f}x → {tgt_str}")
            print(f"  {s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
                  f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")


if __name__ == "__main__":
    main()
