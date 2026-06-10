"""QUANTUM-WF — walk-forward retrained version of the QUANTUM ML sleeve.

The production QUANTUM model is trained once on 2010-2018 and then frozen for
the entire 2019-2026 OOS window. Seven years of staleness is the most likely
cause of its IS 2.73 -> OOS 0.86 Sharpe collapse.

QUANTUM-WF keeps everything else identical (universe, features, model spec,
N=21 rebalance, top-K=3, 10bp/side TC, close[t-1] signals -> open[t] fills)
but retrains the model every January on an EXPANDING window of all data
available up to that point. The first model (used for 2013) is trained on
2010-2012; the model used for year Y is trained on data through Dec Y-1, with
a 21-day embargo before the training cutoff so no target window overlaps the
live period. This is fully causal — at no point does a prediction use a model
that has seen the future.

Output: phoenix5/results/quantum_wf_returns.csv  (Date, ret)
        phoenix5/results/quantum_wf_metrics.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "alt"))

from quantum_strategy import (  # noqa: E402  (reuse audited pipeline)
    UNIVERSE, TC_BPS, build_features, build_targets, load_all_prices, make_model,
)

RESULTS = ROOT / "phoenix5/results"
RESULTS.mkdir(parents=True, exist_ok=True)

N_REB = 21      # same horizon as production QUANTUM
TOP_K = 3
EMBARGO = 21    # days dropped before each training cutoff
FIRST_TRADE_YEAR = 2013   # first 3 calendar years are warm-up training data
IS_END = "2018-12-31"


def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 60:
        return {}
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    return {
        "sharpe": round(float(mu / sd), 4),
        "cagr": round(float(c.iloc[-1] ** (1 / yrs) - 1), 4),
        "vol": round(float(sd), 4),
        "mdd": round(float(mdd), 4),
        "n": int(len(r)),
    }


def main():
    opens, closes = load_all_prices()
    feats = build_features(opens, closes)
    targets = build_targets(closes, N_REB)
    df = feats.join(targets)
    feature_cols = [c for c in feats.columns if c not in ("Ticker", "Date")]

    dates = pd.DatetimeIndex(sorted(set(df.index.get_level_values("Date"))))
    dates = dates[dates >= "2010-03-11"]
    last_year = dates[-1].year

    # --- annual walk-forward predictions -------------------------------
    preds = []
    for year in range(FIRST_TRADE_YEAR, last_year + 1):
        train_cutoff = pd.Timestamp(f"{year-1}-12-31")
        d_idx = df.index.get_level_values("Date")
        embargo_cut = train_cutoff - pd.tseries.offsets.BDay(EMBARGO)
        tr = df[d_idx <= embargo_cut].dropna(subset=feature_cols + ["fwd_ret"])
        live_mask = (d_idx >= pd.Timestamp(f"{year}-01-01")) & (d_idx <= pd.Timestamp(f"{year}-12-31"))
        live = df[live_mask].dropna(subset=feature_cols)
        if len(tr) < 2000 or len(live) == 0:
            continue
        m = make_model()
        m.fit(tr[feature_cols].values, tr["fwd_ret"].values, verbose=False)
        p = pd.Series(m.predict(live[feature_cols].values), index=live.index, name="pred")
        preds.append(p)
        print(f"  {year}: trained on {len(tr):>7d} rows (through {embargo_cut.date()}), "
              f"predicted {len(p):>5d} rows")
    pred = pd.concat(preds).sort_index()

    # --- portfolio construction (same as production QUANTUM) -----------
    trade_dates = [d for d in dates if d >= pd.Timestamp(f"{FIRST_TRADE_YEAR}-01-01")]
    weights = pd.DataFrame(0.0, index=pd.DatetimeIndex(trade_dates), columns=UNIVERSE)
    current = pd.Series(0.0, index=UNIVERSE)
    for i, d in enumerate(trade_dates):
        if i % N_REB == 0:
            try:
                day_pred = pred.loc[d]
            except KeyError:
                day_pred = None
            if day_pred is not None and len(day_pred) >= TOP_K:
                top = day_pred.nlargest(TOP_K).index
                current = pd.Series(0.0, index=UNIVERSE)
                current[list(top)] = 1.0 / TOP_K
        weights.loc[d] = current

    # open->open execution with 1-day signal lag already embedded in features;
    # weights at date t drive open[t]->open[t+1] returns
    open_ret = opens[UNIVERSE].pct_change().shift(-1)  # ret from open[t] to open[t+1]
    open_ret = open_ret.reindex(weights.index)
    gross = (weights * open_ret).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(0)
    net = gross - turnover * (TC_BPS / 1e4)
    net.name = "ret"

    m_full = metrics(net)
    m_is = metrics(net.loc[:IS_END])
    m_oos = metrics(net.loc["2019":])
    print("\nQUANTUM-WF (annual expanding-window retrain):")
    print(f"  IS  (2013-2018): {m_is}")
    print(f"  OOS (2019-    ): {m_oos}")
    print(f"  full           : {m_full}")

    # frozen-model comparison
    qf = pd.read_csv(ROOT / "data/results/quantum_returns.csv",
                     parse_dates=["Date"]).set_index("Date")["ret"]
    print(f"\n  frozen QUANTUM OOS for reference: {metrics(qf.loc['2019':])}")

    out = {
        "config": {"N": N_REB, "K": TOP_K, "embargo": EMBARGO,
                   "retrain": "annual, expanding window", "tc_bps": TC_BPS},
        "full": m_full, "is": m_is, "oos": m_oos,
        "frozen_quantum_oos": metrics(qf.loc["2019":]),
    }
    (RESULTS / "quantum_wf_metrics.json").write_text(json.dumps(out, indent=2))
    net.dropna().rename("ret").to_csv(RESULTS / "quantum_wf_returns.csv")
    print(f"\nSaved {RESULTS}/quantum_wf_returns.csv")


if __name__ == "__main__":
    main()
