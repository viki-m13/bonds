"""ASCENT engine — honest DCA stock-selection backtester.

Mandate: cash arrives every period (monthly or biweekly); buy 1..N stocks;
every position must be held >= MINHOLD calendar days before it can be sold;
sells allowed after that (cut losers / rotate); objective = max terminal
wealth vs QQQ-DCA with identical cash flows.

Realism: per-side transaction costs, delisting haircut, liquidity floor,
signals computed from data through the decision close, execution at that
close (standard monthly-rebalance convention; robustness = +1-period delay).
"""
import numpy as np, pandas as pd


def dca_run(px, score, elig, dates, contrib=1000.0,
            N=12, trail=-0.30, ma=None, minhold_days=30,
            cost=0.0020, delist_ret=-0.25,
            cash_policy="add_top_held", rank_exit=None,
            exec_delay=0, seed_px=None):
    """px: DataFrame period-close prices (period grid = dates ⊂ px.index).
    score: DataFrame same index (higher=better); NaN = not a candidate.
    elig: boolean DataFrame — buyable at that date.
    ma: optional trend-exit MA panel (exit if px < ma), None disables.
    rank_exit: if int R, sell holdings whose score rank falls below R (after minhold).
    cash_policy: 'hold' | 'add_top_held' | 'refill_only'
    exec_delay: execute buys/sells at dates[i+exec_delay] prices (robustness).
    Returns dict of results."""
    didx = list(px.index)
    pos = {}   # tk -> {val, peak_px, entry_date}
    cash = 0.0
    rows = []  # (date, V_end, contributed_so_far)
    contributed = 0.0
    trades = []
    pxi = {d: i for i, d in enumerate(didx)}
    dset = set(dates)

    for i, dt in enumerate(didx):
        if dt < dates[0] or dt > dates[-1]:
            continue
        prow = px.loc[dt]
        # 1) mark to market + delisting
        for tk in list(pos.keys()):
            e = pos[tk]
            cp = prow.get(tk, np.nan)
            if np.isfinite(cp):
                e["val"] *= cp / e["last_px"]
                e["last_px"] = cp
                e["peak_px"] = max(e["peak_px"], cp)
            else:
                # price gone -> delisted/halted: haircut, force to cash
                e["val"] *= (1.0 + delist_ret)
                cash += e["val"] * (1 - cost)
                trades.append((dt, tk, "delist", e["val"], (e["val"] / e["cost_basis"] - 1)))
                pos.pop(tk)
        if dt not in dset:
            rows.append((dt, cash + sum(e["val"] for e in pos.values()), contributed))
            continue

        srow = score.loc[dt] if dt in score.index else pd.Series(dtype=float)
        erow = elig.loc[dt] if dt in elig.index else pd.Series(dtype=float)
        marow = ma.loc[dt] if ma is not None else None

        # 2) exits (respect min-hold)
        if len(srow):
            rk = srow.rank(ascending=False)
        for tk in list(pos.keys()):
            e = pos[tk]
            held_days = (dt - e["entry_date"]).days
            if held_days < minhold_days:
                continue
            cp = e["last_px"]
            hit_trail = (cp / e["peak_px"] - 1) <= trail
            hit_trend = (marow is not None and np.isfinite(marow.get(tk, np.nan))
                         and cp < marow[tk])
            hit_rank = (rank_exit is not None and len(srow)
                        and (not np.isfinite(srow.get(tk, np.nan))
                             or rk.get(tk, 1e9) > rank_exit))
            if hit_trail or hit_trend or hit_rank:
                cash += e["val"] * (1 - cost)
                why = "trail" if hit_trail else ("trend" if hit_trend else "rank")
                trades.append((dt, tk, why, e["val"], (e["val"] / e["cost_basis"] - 1)))
                pos.pop(tk)

        # 3) contribution
        cash += contrib
        contributed += contrib

        # 4) buys
        if len(srow):
            cand = srow[erow.reindex(srow.index).fillna(False).astype(bool)].dropna()
            cand = cand[~cand.index.isin(pos.keys())].sort_values(ascending=False)
            need = N - len(pos)
            if need > 0 and cash > 1e-9 and len(cand):
                picks = list(cand.index[:need])
                amt = cash / len(picks)
                for tk in picks:
                    p0 = prow[tk]
                    v = amt * (1 - cost)
                    pos[tk] = {"val": v, "last_px": p0, "peak_px": p0,
                               "entry_date": dt, "cost_basis": v}
                    trades.append((dt, tk, "buy", v, np.nan))
                cash = 0.0
            elif cash > 1e-9 and cash_policy == "add_top_held" and len(pos):
                held_sc = {tk: srow.get(tk, np.nan) for tk in pos}
                tops = sorted(held_sc, key=lambda t: -(held_sc[t] if np.isfinite(held_sc[t]) else -1))[:3]
                amt = cash / len(tops)
                for tk in tops:
                    v = amt * (1 - cost)
                    pos[tk]["val"] += v
                    pos[tk]["cost_basis"] += v
                cash = 0.0
            # 'hold' & 'refill_only': cash sits idle until a slot opens
        rows.append((dt, cash + sum(e["val"] for e in pos.values()), contributed))

    eqty = pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")
    return {"equity": eqty, "trades": pd.DataFrame(trades, columns=["date", "tk", "why", "val", "ret"]),
            "final_pos": pos, "final_cash": cash}


def dca_benchmark(px_b, dates, contrib=1000.0, cost=0.0005):
    """DCA into a single asset (e.g. QQQ)."""
    units = 0.0; contributed = 0.0; rows = []
    for dt in dates:
        p0 = px_b.get(dt, np.nan)
        if not np.isfinite(p0):
            continue
        units += contrib * (1 - cost) / p0
        contributed += contrib
        rows.append((dt, units * p0, contributed))
    return pd.DataFrame(rows, columns=["date", "V", "contributed"]).set_index("date")


def twr(eq, contrib_dates=None):
    """Time-weighted returns from equity path with known flows.
    V_end(t) includes flow C(t); r_t=(V(t)-C(t))/V(t-1)-1."""
    V = eq["V"]; C = eq["contributed"].diff().fillna(eq["contributed"])
    r = (V - C) / V.shift(1) - 1
    return r.dropna()


def irr_annual(eq, freq=12):
    """Money-weighted annual IRR via bisection on periodic rate."""
    C = eq["contributed"].diff().fillna(eq["contributed"]).values
    V = eq["V"].values
    n = len(C)
    if n < 2 or V[-1] <= 0:
        return np.nan
    flows = list(-C)
    flows[-1] += V[-1]
    def npv(rate):
        d = (1 + rate) ** np.arange(n)
        return float(np.sum(np.array(flows) / d))
    # bracket the periodic rate; cap at 100%/period to keep annualization sane
    lo, hi = -0.75, 1.0
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(100):
        mid = (lo + hi) / 2
        if npv(lo) * npv(mid) <= 0: hi = mid
        else: lo = mid
    return (1 + (lo + hi) / 2) ** freq - 1


def stats(eq, freq=12):
    r = twr(eq)
    if len(r) < 6:
        return {}
    cagr = (1 + r).prod() ** (freq / len(r)) - 1
    sh = r.mean() / r.std() * np.sqrt(freq) if r.std() > 0 else np.nan
    c = (1 + r).cumprod()
    dd = (c / c.cummax() - 1).min()
    return {"twr_cagr": cagr, "sharpe": sh, "maxdd": dd,
            "moic": eq["V"].iloc[-1] / eq["contributed"].iloc[-1],
            "irr": irr_annual(eq, freq), "final": eq["V"].iloc[-1]}
