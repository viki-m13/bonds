"""Priority 2a — Time-series momentum (Moskowitz, Ooi, Pedersen 2012).

PRE-REGISTERED form (no parameter sweep):
  - Signal: sign of past K-month return on each UNDERLYING (SPY, QQQ, TLT, GLD)
  - Position: +1 if past-K positive, 0 otherwise (no shorts — LETFs only long)
  - Expression: use LETF (UPRO/TQQQ/TMF/UGL) when position = +1
  - Parked in cash (BIL) when position = 0
  - Equal-weight across the 4 assets
  - K = 12 months (published parameter), monthly rebal
  - Also report K=6 and K=3 as pre-registered alternatives

Underlyings: SPY(UPRO), QQQ(TQQQ), TLT(TMF), GLD(UGL). We use the UNDERLYING
price history (which has much longer data) for the signal but the LETF for
the execution — this gives us clean signal without LETF inception noise.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from letf_engine import summarise
from hydra_core import load_etf


OUT = Path("/home/user/bonds/data/results")

PAIRS = [
    ("SPY", "UPRO"),
    ("QQQ", "TQQQ"),
    ("TLT", "TMF"),
    ("GLD", "UGL"),
]
BIL = "BIL"  # T-bill proxy for cash sleeve


def prep():
    data = {}
    for und, letf in PAIRS:
        s_u = load_etf(und)
        s_l = load_etf(letf)
        data[und] = s_u
        data[letf] = s_l
    data[BIL] = load_etf(BIL)
    # Start common window (LETFs post-inception)
    letfs = [l for _, l in PAIRS]
    px = pd.DataFrame({k: v for k, v in data.items()}).sort_index()
    px = px.dropna(subset=letfs, how="any")
    px = px.loc["2011-01-01":]
    return px


def tsmom_backtest(px, K_months=12, rebal_days=21, tc_bps=15):
    """Monthly rebal; sign of past K*21-day underlying return; execute in LETF."""
    lookback = K_months * 21
    idx = px.index
    n = len(idx)
    tickers = list(px.columns)
    # Build returns of each ticker
    rets = px.pct_change().fillna(0)

    # Target weights table
    W = pd.DataFrame(0.0, index=idx, columns=tickers)

    rebal_iloc = list(range(0, n, rebal_days))
    for i in rebal_iloc:
        if i < lookback + 5:
            # Park in BIL until enough history
            if BIL in tickers:
                W.iloc[i, tickers.index(BIL)] = 1.0
            continue
        sig = {}
        for und, letf in PAIRS:
            past = px[und].iloc[i - lookback:i]
            if past.isna().all() or len(past) < lookback // 2:
                sig[letf] = 0
                continue
            ret_k = past.iloc[-1] / past.iloc[0] - 1
            sig[letf] = 1 if ret_k > 0 else 0
        n_on = sum(sig.values())
        if n_on == 0:
            W.iloc[i, tickers.index(BIL)] = 1.0
        else:
            w_each = 1.0 / len(PAIRS)  # equal-weight across all 4 signals
            for und, letf in PAIRS:
                W.iloc[i, tickers.index(letf)] = w_each * sig[letf]
            # The unused capacity goes to BIL (cash)
            cash_w = 1.0 - sum(W.iloc[i].values)
            if cash_w > 1e-6:
                W.iloc[i, tickers.index(BIL)] += cash_w

    W = W.replace(0, np.nan).ffill().fillna(0)
    W_eff = W.shift(1).fillna(0)
    tc = W_eff.diff().abs().sum(axis=1).fillna(0) * (tc_bps / 1e4)
    port_ret = (W_eff * rets).sum(axis=1) - tc
    return port_ret, W_eff


def tsmom_with_vol_target(px, K_months=12, target_vol=0.20,
                           vol_lb=63, rebal_days=21, tc_bps=15):
    """TSMOM with ex-ante vol targeting at portfolio level.

    After computing raw 1/K-th weights per signal, scale portfolio gross
    exposure so that ex-ante annualised vol = target_vol (using realised
    vol of that weight-vector over vol_lb days).
    """
    lookback = K_months * 21
    idx = px.index
    n = len(idx)
    tickers = list(px.columns)
    rets = px.pct_change().fillna(0)
    W = pd.DataFrame(0.0, index=idx, columns=tickers)

    rebal_iloc = list(range(0, n, rebal_days))
    for i in rebal_iloc:
        if i < max(lookback, vol_lb) + 5:
            W.iloc[i, tickers.index(BIL)] = 1.0
            continue
        sig = {}
        for und, letf in PAIRS:
            past = px[und].iloc[i - lookback:i]
            ret_k = past.iloc[-1] / past.iloc[0] - 1
            sig[letf] = 1 if ret_k > 0 else 0
        n_on = sum(sig.values())
        if n_on == 0:
            W.iloc[i, tickers.index(BIL)] = 1.0
            continue
        w_each = 1.0 / len(PAIRS)
        raw = {letf: w_each * sig[letf] for _, letf in PAIRS}
        # Ex-ante portfolio vol using past vol_lb days
        slice_rets = rets.iloc[i - vol_lb:i]
        w_vec = np.array([raw.get(t, 0) for t in tickers])
        port_daily_vol = np.sqrt(w_vec @ slice_rets.cov().values @ w_vec.T)
        port_ann_vol = port_daily_vol * np.sqrt(252)
        k = min(target_vol / port_ann_vol, 3.0) if port_ann_vol > 0 else 0
        for t in tickers:
            W.iloc[i, tickers.index(t)] = raw.get(t, 0) * k
        cash_w = 1.0 - sum(W.iloc[i].values)
        if cash_w > 1e-6:
            W.iloc[i, tickers.index(BIL)] += cash_w

    W = W.replace(0, np.nan).ffill().fillna(0)
    W_eff = W.shift(1).fillna(0)
    tc = W_eff.diff().abs().sum(axis=1).fillna(0) * (tc_bps / 1e4)
    port_ret = (W_eff * rets).sum(axis=1) - tc
    return port_ret, W_eff


def main():
    px = prep()
    print(f"Window: {px.index[0].date()} .. {px.index[-1].date()} "
          f"({len(px)} days); cols = {list(px.columns)}")

    rows = []
    for K in (3, 6, 12):
        r, _ = tsmom_backtest(px, K_months=K)
        s = summarise(r, f"TSMOM K={K}m plain")
        rows.append(s)
        print(f"  TSMOM K={K}m plain: CAGR={s['cagr']}% SR={s['sharpe']} "
              f"MDD={s['mdd']}%")

    for K in (3, 6, 12):
        for tv in (0.15, 0.20, 0.25, 0.30):
            r, _ = tsmom_with_vol_target(px, K_months=K, target_vol=tv)
            s = summarise(r, f"TSMOM K={K}m tv={int(tv*100)}%")
            rows.append(s)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "letf_tsmom.csv", index=False)
    print("\n" + df.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
