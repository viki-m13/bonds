"""LOCKED-GTAA-v1  —  FORWARD-LOCKED strategy. FROZEN 2026-06-17.
================================================================
This file is a commitment device. Every choice below (universe, rule,
parameters, benchmark, success criteria) is FIXED as of the freeze date and
MUST NOT be changed. The ONLY valid evaluation is on data dated strictly AFTER
the freeze date — anything before is in-sample and contaminated, shown only for
context, and forms NO part of any claim.

Why this design: 22 prior experiments showed every in-sample "beat QQQ" reduced
to selection bias (survivorship / regime / parameter / universe). The only
bias-free test of "can a rule beat QQQ" is to lock the rule with zero remaining
degrees of freedom and judge it forward. This is that test.

UNIVERSE (objective, ex-ante; major investable asset classes via the oldest/
largest liquid ETF in each; deliberately includes assets that may lose —
commodities, EM, long bonds — so it is NOT cherry-picked for returns):
  SPY  US large-cap        QQQ  US tech/growth     IWM  US small-cap
  EFA  developed intl      EEM  emerging markets   VNQ  US REITs
  GLD  gold                DBC  broad commodities  TLT  long Treasuries
  IEF  intermediate Treas (also the 'cash'/defensive anchor)
  Variant +BTC: BTC-USD added as ONE more asset, treated identically by the
  rule (no special weighting) — so the crypto question is itself tested forward.

RULE (textbook TSMOM defaults — NO optimization, all round numbers):
  - Monthly, evaluated on the last trading day of each month.
  - Signal = trailing 12-month total return.
  - Eligible = asset whose 12m return > 0 AND > IEF's 12m return (absolute /
    trend filter; IEF is the cash proxy).
  - Hold = equal-weight the top 3 eligible assets by 12m return. Any unfilled
    slot (fewer than 3 eligible) goes to IEF. None eligible -> 100% IEF.
  - Long-only, no leverage, monthly rebalance.

BENCHMARKS: QQQ buy&hold (primary), SPY buy&hold, 60/40 (60% SPY / 40% IEF).

SUCCESS CRITERIA (judged ONLY on post-freeze data, window >= 3 years):
  PRIMARY (risk-adjusted): maxDD <= 0.7 x QQQ maxDD  AND  Calmar >= QQQ Calmar.
  SECONDARY (honesty): CAGR within 3pp of QQQ (we EXPECT to trail QQQ slightly
    on raw return and win on drawdown — that is the bias-free prior).
  STRETCH (a genuine bias-free beat, NOT expected): CAGR >= QQQ CAGR.

HONEST PRIOR (recorded before any forward data exists): I expect LOCKED-GTAA-v1
to roughly track or modestly TRAIL QQQ on return with materially LOWER drawdown.
I do NOT expect it to beat QQQ on raw return. If it does, that is real, because
nothing here was fit.
"""
import warnings, time, hashlib, json
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf

FREEZE_DATE = "2026-06-17"
LOOKBACK_M = 12
TOP_N = 3
UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "VNQ", "GLD", "DBC", "TLT", "IEF"]
CASH = "IEF"
PARAMS = dict(freeze=FREEZE_DATE, lookback_m=LOOKBACK_M, top_n=TOP_N,
              universe=UNIVERSE, cash=CASH, rule="12m TSMOM, eq-wt top3, "
              "abs-filter >0 and >IEF, monthly")
PARAM_HASH = hashlib.sha256(json.dumps(PARAMS, sort_keys=True).encode()).hexdigest()[:16]


def evaluate(include_btc=False, start=None, end=None, verbose=True):
    uni = UNIVERSE + (["BTC-USD"] if include_btc else [])
    px = yf.download(uni + ["QQQ", "SPY"], start="2002-01-01", auto_adjust=True,
                     progress=False)["Close"]
    me = px.resample("ME").last()
    ret = me.pct_change()
    mom = me / me.shift(LOOKBACK_M) - 1
    idx = me.index
    held = []
    for i in range(LOOKBACK_M, len(idx) - 1):
        d = idx[i]
        m = mom.loc[d, uni].dropna()
        elig = m[(m > 0) & (m > (mom.loc[d, CASH] if pd.notna(mom.loc[d, CASH]) else 0))]
        picks = list(elig.sort_values(ascending=False).index[:TOP_N])
        picks += [CASH] * (TOP_N - len(picks))            # fill with cash
        nxt = idx[i + 1]
        r = np.mean([ret.loc[nxt, p] for p in picks])
        held.append((nxt, r))
    s = pd.Series(dict(held))

    def mets(series):
        series = series.dropna()
        if start:
            series = series[series.index >= start]
        if end:
            series = series[series.index <= end]
        if len(series) < 6:
            return None
        eq = (1 + series).cumprod(); yrs = len(series) / 12
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        mdd = float((eq / eq.cummax() - 1).min())
        return dict(cagr=float(cagr), mdd=mdd,
                    sharpe=float(series.mean() / (series.std() + 1e-12) * np.sqrt(12)),
                    calmar=float(cagr / abs(mdd)) if mdd else np.nan, n=len(series))

    qqq = ret["QQQ"]; spy = ret["SPY"]
    s6040 = 0.6 * ret["SPY"] + 0.4 * ret["IEF"]
    out = {f"LOCKED-GTAA{'+BTC' if include_btc else ''}": mets(s),
           "QQQ buy&hold": mets(qqq), "SPY buy&hold": mets(spy),
           "60/40": mets(s6040)}
    if verbose:
        for nm, m in out.items():
            if m:
                print(f"   {nm:18s} CAGR {m['cagr']*100:6.1f}%  maxDD {m['mdd']*100:5.0f}%"
                      f"  Sharpe {m['sharpe']:.2f}  Calmar {m['calmar']:.2f}  n={m['n']}",
                      flush=True)
    return out


if __name__ == "__main__":
    print(f"LOCKED-GTAA-v1  freeze={FREEZE_DATE}  param_hash={PARAM_HASH}", flush=True)
    print("\n[CONTAMINATED — in-sample context only, NOT a claim] full history:",
          flush=True)
    evaluate(include_btc=False)
    print("\n[CONTAMINATED] +BTC variant, full history:", flush=True)
    evaluate(include_btc=True)
    print(f"\n>>> THE REAL TEST is evaluate(start='{FREEZE_DATE}') run in the "
          "future, on data that did not exist at freeze. <<<", flush=True)
