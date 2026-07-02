"""Allocator-layer experiments for PHOENIX v3.

HONESTY PROTOCOL: all selection metrics printed here are computed on the
2014-01-02 .. 2018-12-31 segment ONLY. Full-period series are built (weights
are walk-forward) but post-2018 numbers are never computed or printed.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/bonds/alt")
import phoenix_production as prod

SEG_START = "2014-01-02"
SEG_END = "2018-12-31"
BLEND_START = prod.BLEND_START
TC_W_BPS = 10.0          # 10 bps on sleeve-weight L1 turnover (uniform, incl. baseline)

SLEEVES = None  # set in main


# ---------------------------------------------------------------- weights ---

def _cap_renorm(wa: pd.Series, cap: float = prod.W_CAP) -> pd.Series:
    wa = wa.copy()
    for _ in range(10):
        over = wa[wa > cap]
        if over.empty:
            break
        excess = float((over - cap).sum())
        wa[wa > cap] = cap
        under = wa[wa < cap]
        if under.empty or excess <= 0:
            break
        wa[under.index] = wa[under.index] + excess * wa[under.index] / wa[under.index].sum()
    return wa / wa.sum()


def compute_w(sleeve_df: pd.DataFrame, end: pd.Timestamp,
              fit_years: int = 4, expanding: bool = False,
              shrink: float | None = None, mom_k: float = 0.0,
              downside: bool = False, fam_cap: float | None = None) -> pd.Series:
    """Generalized version of prod.compute_year_weights (identical defaults)."""
    if expanding:
        start = sleeve_df.index[0]
    else:
        start = end - pd.DateOffset(years=fit_years)
    win = sleeve_df.loc[start:end]
    w = pd.Series(0.0, index=sleeve_df.columns)
    active = [c for c in win.columns
              if win[c].std() * np.sqrt(252) >= prod.MIN_ACTIVE_VOL]
    if not active:
        return w
    srs = {k: prod._sharpe(win[k]) for k in active}
    corr = win[active].corr()
    rho_bar = (corr.sum(axis=1) - 1) / max(len(corr) - 1, 1)
    budgets = pd.Series({k: max(srs[k], prod.SR_FLOOR) * max(1 - rho_bar[k], 0.05)
                         for k in active})
    if mom_k > 0:
        tr12 = (1 + win[active].iloc[-252:]).prod() - 1
        top = tr12.rank(pct=True) > 0.5
        budgets = budgets * np.where(top.reindex(budgets.index), 1 + mom_k, 1 - mom_k)
        budgets = budgets.clip(lower=1e-6)
    if downside:
        m = win[active].clip(upper=0.0)
        cov = pd.DataFrame(m.values.T @ m.values / max(len(m), 1),
                           index=active, columns=active)
        cov += np.eye(len(active)) * 1e-10
    else:
        cov = win[active].cov()
    if shrink is not None:
        diag = pd.DataFrame(np.diag(np.diag(cov.values)), index=cov.index, columns=cov.columns)
        cov = (1 - shrink) * cov + shrink * diag
    wa = _budget_erc_safe(cov, budgets)
    wa = _cap_renorm(wa)
    if fam_cap is not None:
        fam = [c for c in ("VAN", "ORI", "HEL") if c in wa.index]
        s = float(wa[fam].sum()) if fam else 0.0
        if s > fam_cap:
            wa[fam] *= fam_cap / s
            others = wa.index.difference(fam)
            if len(others) and wa[others].sum() > 0:
                wa[others] *= (1 - fam_cap) / wa[others].sum()
            wa = _cap_renorm(wa)
    w[active] = wa.reindex(active).values
    return w


def _budget_erc_safe(cov, budgets):
    return prod._budget_erc(cov, budgets)


def build_W(sleeve_df: pd.DataFrame, cadence: str = "A",
            band: float = 0.0, **kw) -> pd.DataFrame:
    """Piecewise-constant walk-forward weight frame from BLEND_START.
    cadence: 'A' annual (Jan 1), 'Q' quarterly, 'M' monthly refits.
    band: no-trade band — at a refit, only adopt sleeves whose |dw| > band,
    then renormalize."""
    idx = sleeve_df.loc[BLEND_START:].index
    per = idx.to_period({"A": "Y", "Q": "Q", "M": "M"}[cadence])
    W = pd.DataFrame(0.0, index=idx, columns=sleeve_df.columns)
    prev = None
    for p in per.unique():
        end = p.start_time - pd.Timedelta(days=1)
        w = compute_w(sleeve_df, end, **kw)
        if band > 0 and prev is not None:
            keep = (w - prev).abs() <= band
            w = w.where(~keep, prev)
            if w.sum() > 0:
                w = w / w.sum()
        W.loc[per == p] = w.values
        prev = w
    return W


# ---------------------------------------------------------------- overlay ---

def apply_overlay_v(raw: pd.Series, dd_floor: float = prod.DD_FLOOR,
                    gate_pct: float = prod.GATE_PCT,
                    gate_mult: float = 0.5) -> tuple[pd.Series, pd.Series]:
    """prod.apply_overlay with TARGET_VOL=None and parameterized tail knobs."""
    vol_mult = pd.Series(1.0, index=raw.index)
    scaled = raw * vol_mult.shift(2).fillna(1.0)
    cum = (1 + scaled).cumprod()
    hwm = cum.rolling(prod.DD_WIN, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = (1 + dd / dd_floor).clip(0.0, 1.0)
    sv = scaled.rolling(prod.GATE_VOL_WIN).std()
    sv_thr = sv.rolling(prod.GATE_LOOKBACK, min_periods=60).quantile(gate_pct)
    g = pd.Series(np.where(sv <= sv_thr, 1.0, gate_mult), index=raw.index)
    total_dec = vol_mult * dd_mult * g
    total_app = total_dec.shift(2).fillna(1.0)
    tc = total_app.diff().abs().fillna(0.0) * (prod.TC_BPS_PER_LEV_CHG / 1e4)
    net = raw * total_app - tc
    return net, total_app


def run_variant(sleeve_df: pd.DataFrame, W: pd.DataFrame,
                dd_floor: float = prod.DD_FLOOR, gate_pct: float = prod.GATE_PCT,
                gate_mult: float = 0.5) -> tuple[pd.Series, pd.DataFrame]:
    raw = (sleeve_df.loc[W.index] * W).sum(axis=1)
    net, mult = apply_overlay_v(raw, dd_floor, gate_pct, gate_mult)
    # sleeve-weight turnover cost, uniform across variants (incl. baseline)
    dw = W.diff().abs().sum(axis=1).fillna(0.0)
    net = net - dw * mult * (TC_W_BPS / 1e4)
    return net, W


def seg_metrics(net: pd.Series, W: pd.DataFrame, name: str) -> dict:
    r = net.loc[SEG_START:SEG_END]
    m = prod.metrics(r)
    dw = W.diff().abs().sum(axis=1).fillna(0.0).loc[SEG_START:SEG_END]
    ann_to = float(dw.sum() / (len(r) / 252))
    return {"variant": name, "SR": m["sharpe"], "CAGR": m["cagr"],
            "vol": m["vol"], "MDD": m["mdd"], "ann_TO": round(ann_to, 3)}


def main():
    sleeve_df = prod.load_sleeve_returns()
    rows = []

    def add(name, W, **ov):
        net, _ = run_variant(sleeve_df, W, **ov)
        rows.append(seg_metrics(net, W, name))
        return net

    # --- 0. baseline: reconstruct via generalized code AND via prod, sanity check
    W_base = build_W(sleeve_df, "A", fit_years=4)
    W_prod = prod.build_wf_weight_frame(sleeve_df)
    diff = (W_base - W_prod).abs().to_numpy().max()
    print(f"# sanity: max |W_base - W_prod| = {diff:.2e}")
    add("base_A4 (baseline)", W_base)

    # --- 1. refit cadence
    add("cad_Q4", build_W(sleeve_df, "Q", fit_years=4))
    add("cad_M4", build_W(sleeve_df, "M", fit_years=4))

    # --- 2. fitting window
    add("win_A3", build_W(sleeve_df, "A", fit_years=3))
    add("win_A5", build_W(sleeve_df, "A", fit_years=5))
    W_exp = build_W(sleeve_df, "A", expanding=True)
    add("win_Aexp", W_exp)

    # --- 3. sleeve-momentum tilt
    add("momo_k15", build_W(sleeve_df, "A", fit_years=4, mom_k=0.15))
    add("momo_k30", build_W(sleeve_df, "A", fit_years=4, mom_k=0.30))

    # --- 4. shrinkage covariance
    add("shrink_d30", build_W(sleeve_df, "A", fit_years=4, shrink=0.3))
    add("shrink_d60", build_W(sleeve_df, "A", fit_years=4, shrink=0.6))

    # --- 5. ensemble of specifications
    W_q3 = build_W(sleeve_df, "Q", fit_years=3)
    W_ens = (W_base + W_q3 + W_exp) / 3.0
    add("ensemble(A4,Q3,Aexp)", W_ens)

    # --- 6. no-trade band 2%
    add("band2_A4", build_W(sleeve_df, "A", fit_years=4, band=0.02))
    add("band2_Q4", build_W(sleeve_df, "Q", fit_years=4, band=0.02))
    add("band2_M4", build_W(sleeve_df, "M", fit_years=4, band=0.02))

    # --- 8. inventions
    add("semicov_A4", build_W(sleeve_df, "A", fit_years=4, downside=True))
    add("famcap60_A4", build_W(sleeve_df, "A", fit_years=4, fam_cap=0.60))
    add("famcap50_A4", build_W(sleeve_df, "A", fit_years=4, fam_cap=0.50))

    df = pd.DataFrame(rows)
    print("\n=== ALLOCATOR VARIANTS (2014-01-02 .. 2018-12-31 ONLY) ===")
    print(df.to_string(index=False))

    # --- 7. overlay grid on the BASELINE allocator
    print("\n=== OVERLAY GRID on base_A4 (2014-2018 ONLY) ===")
    orows = []
    for ddf in (-0.08, -0.10, -0.12):
        for gp in (0.97, 0.99):
            for gm in (0.5, 0.25):
                net, _ = run_variant(sleeve_df, W_base, dd_floor=ddf,
                                     gate_pct=gp, gate_mult=gm)
                m = seg_metrics(net, W_base, f"dd{ddf} gp{gp} gm{gm}")
                orows.append(m)
    odf = pd.DataFrame(orows)
    print(odf.to_string(index=False))


if __name__ == "__main__":
    main()
