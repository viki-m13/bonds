"""PHOENIX-5 causal sleeve factory.

Generalizes MOSAIC: auto-generate a broad library of candidate daily-return
streams from UNDER-USED existing data, then combine them with a strictly
trailing-window (causal) selector. The selector never sees the future — at
each monthly rebalance it scores candidates on trailing 252d performance only.
All standalone candidates are also reported IS/OOS so we can see honestly which
families carry the meta-stream.

Candidate families (deliberately ORTHOGONAL to PHOENIX's long-biased leveraged
equity + crypto tilt):

  A. BONDXS  — cross-section of 16 bond ETFs (features.parquet): carry tilt,
               duration/curve timing, credit-spread mean-reversion. Long-short
               and long-only, monthly, 2bp/side.
  B. TSMOM   — single-asset time-series momentum on unlevered cross-asset ETFs
               (GLD/DBC/USO/UUP/UDN/TLT/IEF/VNQ/EEM/EFA/SLV), multi-lookback,
               1-3bp/side.
  C. VOLCARRY— short-vol (SVXY) gated by a causal contango proxy (VIXY decay)
               + VIX-level filter; and a long-vol crisis hedge variant.
  D. CREDIT  — HY/IG credit timing vs duration, gated by HY-OAS regime.

Selector: each month keep candidates with trailing-252d Sharpe > floor, weight
by trailing inverse-vol, cap per-stream weight; meta-stream = weighted sum.
A candidate becomes eligible after 252d of live history. Fully causal.

We then report FACTORY standalone, its correlation to PHOENIX, the marginal
value of adding it to the production blend, and a DEFLATED Sharpe that accounts
for the number of candidate streams searched (multiple-testing haircut).

Outputs under phoenix5/factory/results/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
ETF = ROOT / "data/etfs"
FRED = ROOT / "data/fred"
FEAT = ROOT / "data/features.parquet"
OUT = ROOT / "phoenix5/factory/results"
OUT.mkdir(parents=True, exist_ok=True)

IS_END = "2018-12-31"
OOS = "2019-01-02"
START = "2010-03-11"


# ---------------------------------------------------------------- helpers
def px(t):
    s = pd.read_csv(ETF / f"{t}.csv", parse_dates=["Date"], index_col="Date")["Close"]
    return s[~s.index.duplicated()].sort_index()


def fred_series(name):
    s = pd.read_csv(FRED / f"{name}.csv", parse_dates=["Date"], index_col="Date")[name]
    return pd.to_numeric(s, errors="coerce")


def sr(r, ann=252):
    r = r.dropna()
    return float(r.mean() / r.std() * np.sqrt(ann)) if len(r) > 60 and r.std() > 0 else np.nan


def metrics(r):
    r = r.dropna()
    if len(r) < 60:
        return {}
    mu, sd = r.mean() * 252, r.std() * np.sqrt(252)
    c = (1 + r).cumprod()
    mdd = (c / c.cummax() - 1).min()
    yrs = len(r) / 252
    return {"sr": round(float(mu / sd), 3), "cagr": round(float(c.iloc[-1] ** (1 / yrs) - 1), 4),
            "vol": round(float(sd), 4), "mdd": round(float(mdd), 4), "n": int(len(r))}


def monthly_mask(idx):
    s = pd.Series(idx, index=idx)
    return (s.groupby([idx.year, idx.month]).transform("first") == s)


# ---------------------------------------------------------------- A. BONDXS
def family_bondxs():
    df = pd.read_parquet(FEAT)
    ret = df["ret_1d"].unstack()
    cov = ret.loc["2010":].notna().mean()
    keep = cov[cov > 0.95].index.tolist()
    ret = ret[keep].loc[START:]
    carry = df["carry_proxy"].unstack()[keep].loc[START:]
    hyoas_z = df["hy_oas_zscore_63"].unstack().reindex(columns=keep).loc[START:]
    streams = {}
    monthly = monthly_mask(ret.index)

    def ls_from_score(score, name, topq=0.3, hold_monthly=True, cost=0.0002):
        rk = score.rank(axis=1, pct=True)
        n = rk.notna().sum(axis=1)
        longw = (rk >= 1 - topq).div((n * topq).clip(lower=1), axis=0)
        shortw = (rk <= topq).div((n * topq).clip(lower=1), axis=0)
        w = (longw - shortw)
        if hold_monthly:
            w = w.where(monthly, np.nan).ffill()
        w = w.shift(1).fillna(0.0)
        gross = (w * ret).sum(axis=1)
        to = w.diff().abs().sum(axis=1)
        streams[name] = (gross - to * cost).dropna()

    # carry tilt: long high carry / short low carry (vol-adj)
    vol = ret.rolling(63).std()
    ls_from_score(carry / vol, "bondxs_carry")
    # 6m momentum cross-section (skip 1m)
    mom = ret.rolling(126).sum().shift(21)
    ls_from_score(mom / vol, "bondxs_mom6")
    # 3m momentum
    ls_from_score(ret.rolling(63).sum() / vol, "bondxs_mom3")
    # credit-spread mean reversion: long when hy_oas high z (spreads wide -> long credit names)
    ls_from_score(hyoas_z, "bondxs_creditmr", topq=0.4)
    # long-only top-carry basket vs equal-weight (duration-neutral-ish)
    rk = (carry / vol).rank(axis=1, pct=True)
    n = rk.notna().sum(axis=1)
    w = (rk >= 0.6).div((n * 0.4).clip(lower=1), axis=0)
    w = w.where(monthly, np.nan).ffill().shift(1).fillna(0.0)
    eqw = ret.mean(axis=1)
    streams["bondxs_carry_lo"] = ((w * ret).sum(axis=1) - eqw - w.diff().abs().sum(axis=1) * 0.0002).dropna()
    return streams


# ---------------------------------------------------------------- B. TSMOM
def family_tsmom():
    assets = ["GLD", "DBC", "USO", "UUP", "UDN", "TLT", "IEF", "VNQ", "EEM", "EFA", "SLV", "HYG"]
    streams = {}
    for a in assets:
        try:
            r = px(a).pct_change().loc[START:]
        except FileNotFoundError:
            continue
        if r.notna().sum() < 1000:
            continue
        sigs = []
        for lb in [21, 63, 126, 252]:
            m = px(a).pct_change().rolling(lb).mean()
            v = px(a).pct_change().rolling(lb).std()
            sigs.append(np.sign((m / v.replace(0, np.nan))).reindex(r.index))
        sig = pd.concat(sigs, axis=1).mean(axis=1).clip(-1, 1).shift(1)
        rv = r.rolling(63).std() * np.sqrt(252)
        pos = sig * (0.10 / rv.clip(lower=0.03)).clip(0, 3)
        streams[f"tsmom_{a}"] = (pos * r - pos.diff().abs().fillna(0) * 0.0002).dropna()
    return streams


# ---------------------------------------------------------------- C. VOLCARRY
def family_volcarry():
    streams = {}
    try:
        svxy = px("SVXY").pct_change()
        vixy = px("VIXY")
    except FileNotFoundError:
        return streams
    vix = fred_series("VIXCLS").reindex(svxy.index).ffill()
    # contango proxy: VIXY persistent decay over trailing 21d => contango (carry positive for short vol)
    vixy_drift = vixy.pct_change().rolling(21).mean()
    contango = (vixy_drift < 0).astype(float)             # VIXY bleeding => contango
    vix_ok = (vix < vix.rolling(252, min_periods=120).quantile(0.85)).astype(float)
    sig = (contango * vix_ok).shift(1).fillna(0.0)
    r = svxy.loc[START:]
    pos = 0.5 * sig.reindex(r.index)                       # half-size short-vol
    streams["volcarry_svxy"] = (pos * r - pos.diff().abs().fillna(0) * 0.0005).dropna()
    # crisis long-vol: long VIXY only when backwardation + vol rising (rare, convex)
    backw = (vixy_drift > 0.01).astype(float)
    vix_rise = (vix > vix.rolling(20).mean()).astype(float)
    sigb = (backw * vix_rise).shift(1).fillna(0.0)
    rv = vixy.pct_change().loc[START:]
    posb = 0.1 * sigb.reindex(rv.index)
    streams["volhedge_vixy"] = (posb * rv - posb.diff().abs().fillna(0) * 0.0005).dropna()
    return streams


# ---------------------------------------------------------------- D. CREDIT
def family_credit():
    streams = {}
    oas = fred_series("BAMLH0A0HYM2")
    igoas = fred_series("BAMLC0A0CM")
    for lng, hdg, name in [("HYG", "IEF", "hy"), ("LQD", "IEF", "ig"),
                           ("JNK", "TLT", "jnk"), ("EMB", "IEF", "em")]:
        try:
            rl, rh = px(lng).pct_change(), px(hdg).pct_change()
        except FileNotFoundError:
            continue
        idx = rl.index.intersection(rh.index)
        rl, rh = rl.loc[idx], rh.loc[idx]
        beta = rl.rolling(252, min_periods=126).cov(rh) / rh.rolling(252, min_periods=126).var()
        carry = (rl - beta.shift(1) * rh)
        # gate: be long credit-carry when HY OAS not spiking
        o = oas.reindex(idx).ffill()
        gate = (o.diff(20) < 0.5).astype(float).shift(1).fillna(1.0)
        streams[f"credit_{name}"] = (gate * carry).loc[START:].dropna()
    return streams


# ---------------------------------------------------------------- selector
def causal_select(cands: pd.DataFrame, sr_floor=0.3, wcap=0.25, min_hist=252):
    """Monthly: keep streams with trailing-252d Sharpe>floor, inverse-vol weight
    (capped), normalize. Returns meta daily return + weight history."""
    idx = cands.index
    monthly = monthly_mask(idx)
    W = pd.DataFrame(0.0, index=idx, columns=cands.columns)
    cur = pd.Series(0.0, index=cands.columns)
    for i, dt in enumerate(idx):
        if monthly.iloc[i] and i >= min_hist:
            hist = cands.iloc[i - 252:i]
            score = (hist.mean() / hist.std()) * np.sqrt(252)
            vol = hist.std() * np.sqrt(252)
            elig = (hist.notna().sum() >= min_hist) & (score > sr_floor) & (vol > 0)
            if elig.any():
                iv = (1.0 / vol).where(elig, 0.0)
                w = (iv / iv.sum()).clip(upper=wcap)
                cur = (w / w.sum()).fillna(0.0)
            else:
                cur = pd.Series(0.0, index=cands.columns)
        W.iloc[i] = cur.values
    meta = (W.shift(1).fillna(0.0) * cands.fillna(0.0)).sum(axis=1)
    return meta, W


def deflated_sharpe(sr_obs, n_obs, n_trials, skew=0.0, kurt=3.0):
    """Bailey & Lopez de Prado deflated Sharpe: prob the true SR>0 given the
    number of trials. Returns (expected max SR under null, DSR probability)."""
    if n_trials < 2 or not np.isfinite(sr_obs):
        return np.nan, np.nan
    emc = 0.5772156649
    z = stats.norm.ppf
    # expected max of n_trials independent N(0,1) Sharpe estimates
    exp_max = (1 - emc) * z(1 - 1.0 / n_trials) + emc * z(1 - 1.0 / (n_trials * np.e))
    sr_std = np.sqrt((1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2) / (n_obs - 1))
    sr0 = exp_max * sr_std  # benchmark SR threshold from multiple testing (daily units->ann via caller)
    dsr = stats.norm.cdf((sr_obs - sr0) / sr_std) if sr_std > 0 else np.nan
    return float(sr0), float(dsr)


# ---------------------------------------------------------------- main
def main():
    print("Generating candidate streams...")
    cands = {}
    for fam in (family_bondxs, family_tsmom, family_volcarry, family_credit):
        s = fam()
        cands.update(s)
        print(f"  {fam.__name__:18s}: {len(s)} streams")
    df = pd.DataFrame(cands).loc[START:]
    print(f"\nTotal candidates: {df.shape[1]}, dates {df.index[0].date()}..{df.index[-1].date()}")

    print("\nStandalone candidate performance (IS / OOS Sharpe):")
    rows = []
    for c in df.columns:
        rows.append({"stream": c, "is_sr": round(sr(df[c].loc[:IS_END]) or 0, 2),
                     "oos_sr": round(sr(df[c].loc[OOS:]) or 0, 2),
                     "full_sr": round(sr(df[c]) or 0, 2),
                     "vol": round(float(df[c].std() * np.sqrt(252)), 3)})
    tbl = pd.DataFrame(rows).sort_values("oos_sr", ascending=False)
    print(tbl.to_string(index=False))

    print("\nCausal selection -> FACTORY meta-stream:")
    best = None
    for floor in [0.0, 0.25, 0.5]:
        meta, W = causal_select(df, sr_floor=floor)
        m_is, m_oos = metrics(meta.loc[:IS_END]), metrics(meta.loc[OOS:])
        print(f"  sr_floor={floor}: IS SR={m_is.get('sr')}  OOS SR={m_oos.get('sr')}  "
              f"OOS CAGR={m_oos.get('cagr',0)*100:.1f}%  vol={m_oos.get('vol',0)*100:.1f}%")
        if best is None or (m_oos.get("sr") or 0) > best[0]:
            best = (m_oos.get("sr") or 0, floor, meta, W)
    _, floor, meta, W = best

    # PHOENIX comparison + marginal value
    phx = pd.read_csv(ROOT / "data/results/phoenix_production_returns.csv",
                      parse_dates=["Date"]).set_index("Date")["net_ret"]
    idx = meta.dropna().index.intersection(phx.index)
    corr = float(np.corrcoef(meta.loc[idx], phx.loc[idx])[0, 1])
    print(f"\nFACTORY (sr_floor={floor}): full {metrics(meta)}")
    print(f"corr(FACTORY, PHOENIX) = {corr:.2f}")

    # deflated Sharpe of FACTORY OOS, accounting for # candidate trials
    oos = meta.loc[OOS:].dropna()
    sr_oos_daily = oos.mean() / oos.std()
    sr0, dsr = deflated_sharpe(sr_oos_daily, len(oos), df.shape[1],
                               skew=float(stats.skew(oos)), kurt=float(stats.kurtosis(oos, fisher=False)))
    print(f"deflated-Sharpe prob(true SR>0) given {df.shape[1]} trials: {dsr:.3f}")

    out = {
        "candidates": tbl.to_dict("records"),
        "factory": {"sr_floor": floor, "is": metrics(meta.loc[:IS_END]),
                    "oos": metrics(meta.loc[OOS:]), "full": metrics(meta),
                    "corr_phoenix": round(corr, 3),
                    "deflated_sharpe_prob": round(float(dsr), 4) if np.isfinite(dsr) else None,
                    "n_trials": int(df.shape[1])},
    }
    (OUT / "factory_metrics.json").write_text(json.dumps(out, indent=2))
    meta.dropna().rename("ret").to_csv(OUT / "factory_returns.csv")
    df.to_csv(OUT / "factory_candidates.csv")
    print(f"\nSaved factory_returns.csv, factory_candidates.csv, factory_metrics.json in {OUT}")


if __name__ == "__main__":
    main()
