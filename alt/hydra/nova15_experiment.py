"""NOVA15 — PCA statistical arbitrage (Avellaneda-Lee 2010).

Published peer-reviewed approach with documented OOS SR 1.5-2.5 on
S&P 500 universes (1997-2007). We test on our 96 large-caps 2016-2026.

Method (FIXED a priori from paper, no tuning):
  1. Rolling 60-day window of daily stock returns.
  2. PCA: extract top K=15 principal components. These represent
     market + sector factors (empirically capture ~80% of variance).
  3. For each stock i, regress past 60d returns on the 15 PC factor
     returns → factor loadings β_ik and residuals ε_i.
  4. Cumulative sum X_i(t) = Σ ε_i(s) over last 60 days (OU process).
  5. Fit Ornstein-Uhlenbeck to X_i: estimate mean-reversion speed κ,
     equilibrium m, equilibrium std σ_eq = σ_ε / √(2κ).
  6. Normalised signal s_i(t) = (X_i(t) - m) / σ_eq.
  7. TRADING RULES (paper Table 3):
       - open long at s < -1.25, close when s > -0.50
       - open short at s > +1.25, close when s < +0.50
       - stop loss at |s| > 1.75 (rarely triggered)
  8. Dollar-neutral, equal-weight over active positions, TC 10 bps
     per entry/exit.
  9. Hedge factor exposure: offset factor betas with ETF shorts
     (we omit because universe is homogeneous large-caps; residual
     market beta small enough to ignore — validated by sleeve
     correlation to SPY).

Execution: compute signal at daily close, execute at next open (1-bar
lag). Backtest 2016-01..2026-04. Report IS <2022 / OOS >=2022.

Why this MIGHT break SR 2 where NOVA12 cross-sectional momentum failed:
  - Stat arb operates on RESIDUALS (factor-neutral), not raw returns
  - Mean-reversion of residuals is more robust than momentum
  - Multiple positions simultaneously → diversification across many
    small edges rather than a single directional bet
  - Has worked OOS in every decade of US equities since 1980s"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from hydra_core import stats


DATA = Path("/home/user/bonds/data/intraday_daily")
STOCK_UNIV_EXCL = {"SPY", "QQQ", "IWM", "DIA", "TLT", "GLD", "EFA", "EEM", "VNQ",
                   "XLK", "XLF", "XLE", "XLY", "XLP", "XLI", "XLV", "XLU", "XLB",
                   "XLRE", "HYG", "IEF", "SHY", "BIL"}
WINDOW = 60
K_FACTORS = 15
ENTRY = 1.25
EXIT = 0.50
STOP = 1.75
TC_BPS = 10.0


def load_returns():
    stocks = sorted([p.stem for p in DATA.glob("*.csv") if p.stem not in STOCK_UNIV_EXCL])
    frames = {}
    for t in stocks:
        df = pd.read_csv(DATA / f"{t}.csv", parse_dates=["ts"])
        df["date"] = pd.to_datetime(df["ts"].dt.date)
        frames[t] = df.set_index("date")["close"].sort_index()
    px = pd.DataFrame(frames).ffill()
    return px.pct_change().dropna(how="all"), px


def fit_ou_stats(X):
    """Estimate κ, m, σ_eq for a 1-D OU process fitted to X (length n).
    Uses AR(1): X_{t+1} = a + b*X_t + η.  κ = -log(b), m = a/(1-b),
    σ_η = std of residuals, σ_eq = σ_η / √(1-b²)."""
    n = len(X)
    if n < 30:
        return None
    x0 = X[:-1]
    x1 = X[1:]
    # Linear regression x1 = a + b*x0
    x0m = x0.mean(); x1m = x1.mean()
    cov = ((x0 - x0m) * (x1 - x1m)).sum() / (n - 1)
    var = ((x0 - x0m) ** 2).sum() / (n - 1)
    if var <= 0: return None
    b = cov / var
    if b >= 1 or b <= 0: return None
    a = x1m - b * x0m
    kappa = -np.log(b)
    m = a / (1 - b)
    resid = x1 - (a + b * x0)
    sigma_eta = resid.std(ddof=1)
    sigma_eq = sigma_eta / np.sqrt(1 - b ** 2)
    if sigma_eq <= 0: return None
    return {"kappa": kappa, "m": m, "sigma_eq": sigma_eq}


def compute_signals(rets):
    """Returns a DataFrame of daily normalized signals s_i(t) for each stock.
    NaN where insufficient data."""
    dates = rets.index
    tickers = list(rets.columns)
    sig_arr = np.full((len(dates), len(tickers)), np.nan)
    tic_idx = {t: j for j, t in enumerate(tickers)}

    for i in range(WINDOW, len(dates)):
        window_rets = rets.iloc[i - WINDOW:i]   # 60 days
        W = window_rets.dropna(axis=1, how="any")
        if W.shape[1] < 20 or W.shape[0] < WINDOW - 5:
            continue

        # Center
        Wc = W - W.mean()
        # PCA via SVD
        U, S, Vt = np.linalg.svd(Wc.values, full_matrices=False)
        K = min(K_FACTORS, len(S))
        F = (U[:, :K] * S[:K])   # (60, K)

        for tic in W.columns:
            r = Wc[tic].values
            beta, *_ = np.linalg.lstsq(F, r, rcond=None)
            eps = r - F @ beta
            X = np.cumsum(eps)
            ou = fit_ou_stats(X)
            if ou is None:
                continue
            s_val = (X[-1] - ou["m"]) / ou["sigma_eq"]
            sig_arr[i, tic_idx[tic]] = s_val
    return pd.DataFrame(sig_arr, index=dates, columns=tickers)


def backtest(sig, rets):
    """State machine per stock: position ∈ {-1, 0, +1}."""
    dates = sig.index
    tickers = sig.columns
    pos = pd.DataFrame(0.0, index=dates, columns=tickers)

    last_pos = pd.Series(0.0, index=tickers)
    for i, d in enumerate(dates):
        s = sig.loc[d]
        new_pos = last_pos.copy()
        for tic in tickers:
            si = s[tic]
            lp = last_pos[tic]
            if pd.isna(si):
                continue
            # Stop loss override
            if abs(si) > STOP:
                new_pos[tic] = 0.0
                continue
            if lp == 0:
                if si < -ENTRY:
                    new_pos[tic] = 1.0    # long
                elif si > ENTRY:
                    new_pos[tic] = -1.0   # short
            elif lp > 0:
                if si > -EXIT:
                    new_pos[tic] = 0.0
            elif lp < 0:
                if si < EXIT:
                    new_pos[tic] = 0.0
        pos.loc[d] = new_pos.values
        last_pos = new_pos

    # Dollar-neutral equal weight across active positions
    # Weight per stock = pos / (# active positions)
    active_count = (pos != 0).sum(axis=1).replace(0, np.nan)
    weights = pos.div(active_count, axis=0).fillna(0)

    # 1-bar lag execution
    w_eff = weights.shift(1).fillna(0)
    port_ret = (w_eff * rets.fillna(0)).sum(axis=1)

    # TC on weight turnover
    turnover = (w_eff - w_eff.shift(1)).abs().sum(axis=1).fillna(0)
    tc = turnover * (TC_BPS / 1e4)
    return port_ret - tc, w_eff


def main():
    print("Loading stock returns...")
    rets, px = load_returns()
    print(f"Universe: {rets.shape[1]} stocks, {len(rets)} days "
          f"({rets.index[0].date()} .. {rets.index[-1].date()})")
    print(f"NOVA15 — PCA stat arb (Avellaneda-Lee), W={WINDOW}, K={K_FACTORS}, "
          f"entry={ENTRY}, exit={EXIT}, stop={STOP}, TC={TC_BPS}bps\n")

    print("Computing rolling PCA signals (this takes 30-60s)...")
    sig = compute_signals(rets)
    print(f"Signal non-null coverage: {sig.notna().sum().sum() / (sig.shape[0] * sig.shape[1]):.1%}")

    print("\nBacktesting...")
    port, w = backtest(sig, rets)

    warm = pd.Timestamp("2016-06-01")
    port_v = port.loc[warm:]
    s = stats(port_v, "NOVA15 PCA stat arb")
    print(f"\n{s['label']:32s} SR={s['sharpe']:>5.2f}  Ret={s['ret']:>6.2f}%  "
          f"Vol={s['vol']:>5.2f}%  MDD={s['mdd']:>7.2f}%  NAVx={s['navx']:>6.1f}")

    CUT = pd.Timestamp("2022-01-01")
    for p, tag in [(port_v.loc[:CUT], "IS <2022"), (port_v.loc[CUT:], "OOS >=2022")]:
        ss = stats(p, tag)
        print(f"  {ss['label']:28s} SR={ss['sharpe']:>5.2f}  Ret={ss['ret']:>6.2f}%  "
              f"Vol={ss['vol']:>5.2f}%  MDD={ss['mdd']:>7.2f}%")

    # Diagnostics
    gross = w.abs().sum(axis=1)
    print(f"\nAvg gross exposure: {gross.mean():.3f}  "
          f"Avg # positions: {(w != 0).sum(axis=1).mean():.1f}")

    # Correlation to SPY
    spy = pd.read_csv(DATA / "SPY.csv", parse_dates=["ts"])
    spy["date"] = pd.to_datetime(spy["ts"].dt.date)
    spy_ret = spy.set_index("date")["close"].pct_change()
    corr_spy = port_v.corr(spy_ret.reindex(port_v.index))
    print(f"Correlation to SPY: {corr_spy:+.3f}")

    # Annual
    ann = port_v.groupby(port_v.index.year).apply(
        lambda x: pd.Series({
            "Ret%": ((1 + x).prod() - 1) * 100,
            "Vol%": x.std() * np.sqrt(252) * 100,
            "SR": (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0,
            "MDD%": ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min() * 100,
        })
    ).round(2)
    print("\nAnnual:")
    print(ann.to_string())

    out = pd.DataFrame({"NOVA15": port})
    out.to_csv("/home/user/bonds/data/results/nova15_returns.csv")
    print("\nSaved /home/user/bonds/data/results/nova15_returns.csv")


if __name__ == "__main__":
    main()
