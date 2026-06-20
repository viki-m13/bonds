"""EDA: what empirically precedes large medium-term single-stock runs
("parabolic moves") in the PIT S&P 500 panel.

Causality: every characteristic uses information through the close of day d
only (trailing windows, no shift(-k), no full-sample stats). Forward outcomes
use close[d] -> close[d+126]/close[d+252].

Run:  python research/eda_parabolic.py   (from /home/user/bonds/dca)
Prints markdown tables consumed by research/eda_parabolic.md.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import data

pd.set_option("display.width", 220)

H6, H12 = 126, 252          # forward horizons (trading days)
PARAB_ABS = 0.50            # absolute parabolic threshold on fwd 6m return
SAMPLE_EVERY = 21           # monthly sampling for decile / profile analysis
IC_EVERY = 10               # biweekly sampling for rank-IC series


# ---------------------------------------------------------------- load
def load():
    P = data.build_panel()
    close, high, low, vol = P["close"], P["high"], P["low"], P["volume"]
    member = P["member"]
    spy = data.load_benchmark("SPY")["Close"].reindex(close.index)
    return close, high, low, vol, member, spy


# ---------------------------------------------------------------- features
def days_since_high(close, win=252):
    """Trading days since the most recent 52w-high close (0 = new high today)."""
    rmax = close.rolling(win).max()
    is_hi = (close >= rmax * (1 - 1e-9)) & rmax.notna()
    n = len(close)
    pos = np.arange(n, dtype=float)[:, None]
    last = pd.DataFrame(np.where(is_hi.values, pos, np.nan),
                        index=close.index, columns=close.columns).ffill()
    age = pd.DataFrame(np.broadcast_to(pos, close.shape),
                       index=close.index, columns=close.columns) - last
    return age.where(rmax.notna())


def build_features(close, high, low, vol, spy):
    R = close.pct_change()
    feats = {}
    feats["ret_1m"]  = close / close.shift(21)  - 1
    feats["ret_3m"]  = close / close.shift(63)  - 1
    feats["ret_6m"]  = close / close.shift(126) - 1
    feats["ret_12m"] = close / close.shift(252) - 1
    feats["mom_12_1"] = close.shift(21) / close.shift(252) - 1

    rmax = close.rolling(252).max()
    rmin = close.rolling(252).min()
    feats["dist_52w_high"] = close / rmax - 1          # <= 0
    feats["dist_52w_low"]  = close / rmin - 1          # >= 0
    feats["age_52w_high"]  = days_since_high(close)

    sd20, sd60 = R.rolling(20).std(), R.rolling(60).std()
    feats["vol_20d"] = sd20 * np.sqrt(252)
    feats["vol_ratio_20_60"] = sd20 / sd60             # <1 => compression

    feats["volu_trend"] = vol.rolling(20).mean() / vol.rolling(120).mean()
    feats["updown_21"]  = (R > 0).rolling(21).mean().where(R.notna())
    feats["max_dret_21"] = R.rolling(21).max()

    # beta / corr vs SPY, 120d
    s = spy.pct_change()
    sv = s.values[:, None]
    Rm = R.values
    msk = ~np.isnan(Rm) & ~np.isnan(sv)
    w = 120
    Rm0 = np.where(msk, Rm, 0.0); sm0 = np.where(msk, sv, 0.0)
    k = pd.DataFrame(msk.astype(float)).rolling(w).sum().values
    Sx = pd.DataFrame(Rm0).rolling(w).sum().values
    Sy = pd.DataFrame(sm0).rolling(w).sum().values
    Sxy = pd.DataFrame(Rm0 * sm0).rolling(w).sum().values
    Sxx = pd.DataFrame(Rm0 * Rm0).rolling(w).sum().values
    Syy = pd.DataFrame(sm0 * sm0).rolling(w).sum().values
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy / k - (Sx / k) * (Sy / k)
        vx = Sxx / k - (Sx / k) ** 2
        vy = Syy / k - (Sy / k) ** 2
        beta = np.where(vy > 0, cov / vy, np.nan)
        corr = np.where((vx > 0) & (vy > 0), cov / np.sqrt(vx * vy), np.nan)
    bad = k < 100
    beta[bad] = np.nan; corr[bad] = np.nan
    feats["beta_120"] = pd.DataFrame(beta, index=close.index, columns=close.columns)
    feats["corr_120"] = pd.DataFrame(corr, index=close.index, columns=close.columns)

    # range contraction: 20d high-low range vs 120d
    rng20 = high.rolling(20).max() - low.rolling(20).min()
    rng120 = high.rolling(120).max() - low.rolling(120).min()
    feats["range_contr"] = rng20 / rng120

    # consecutive-ish higher lows: # of rising 10d-low blocks over last 60d (0..6)
    l10 = low.rolling(10).min()
    hl = None
    for j in range(1, 7):
        ind = (l10.shift(10 * (j - 1)) > l10.shift(10 * j)).astype(float)
        ind = ind.where(l10.shift(10 * j).notna())
        hl = ind if hl is None else hl + ind
    feats["higher_lows6"] = hl

    feats["dd_from_ath"] = close / close.cummax() - 1
    return feats


# ---------------------------------------------------------------- outcomes
def build_outcomes(close, member):
    fwd6 = close.shift(-H6) / close - 1
    fwd12 = close.shift(-H12) / close - 1
    valid = member & close.notna()
    return fwd6, fwd12, valid


def regime_series(spy):
    ma200 = spy.rolling(200).mean()
    return (spy > ma200).where(ma200.notna())   # True = risk-on


# ---------------------------------------------------------------- helpers
def xs_rank(row_vals):
    """percentile rank in [0,1] of a 1d array with NaNs."""
    s = pd.Series(row_vals)
    return s.rank(pct=True).values


def md_table(df, fmt="{:.3f}"):
    out = ["| " + " | ".join([df.index.name or ""] + [str(c) for c in df.columns]) + " |",
           "|" + "---|" * (len(df.columns) + 1)]
    for ix, row in df.iterrows():
        cells = [fmt.format(v) if isinstance(v, (int, float, np.floating)) and not pd.isna(v)
                 else ("" if pd.isna(v) else str(v)) for v in row]
        out.append("| " + " | ".join([str(ix)] + cells) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- analyses
def decile_analysis(feats, fwd6, valid, dates, regime):
    """Per-feature decile stats pooled over sampled dates; also split by regime."""
    res = {}
    for name, F in feats.items():
        rows = []
        for d in dates:
            v = valid.loc[d].values & ~np.isnan(F.loc[d].values) & ~np.isnan(fwd6.loc[d].values)
            if v.sum() < 100:
                continue
            f = F.loc[d].values[v]
            y = fwd6.loc[d].values[v]
            pr = pd.Series(f).rank(pct=True).values
            dec = np.minimum((pr * 10).astype(int), 9)
            ymean = y.mean()
            reg = bool(regime.loc[d]) if not pd.isna(regime.loc[d]) else None
            rows.append((dec, y, y - ymean, y > PARAB_ABS, reg))
        # pool
        agg = {k: [[] for _ in range(10)] for k in ("y", "yx", "p")}
        agg_reg = {True: [[[] , []] for _ in range(10)], False: [[[], []] for _ in range(10)]}
        for dec, y, yx, p, reg in rows:
            for q in range(10):
                m = dec == q
                agg["y"][q].append(y[m]); agg["yx"][q].append(yx[m]); agg["p"][q].append(p[m])
                if reg is not None:
                    agg_reg[reg][q][0].append(yx[m]); agg_reg[reg][q][1].append(p[m])
        tab = pd.DataFrame(index=range(1, 11))
        tab.index.name = "decile"
        for q in range(10):
            y = np.concatenate(agg["y"][q]); yx = np.concatenate(agg["yx"][q])
            p = np.concatenate(agg["p"][q])
            tab.loc[q + 1, "mean_fwd6"] = y.mean()
            tab.loc[q + 1, "med_fwd6"] = np.median(y)
            tab.loc[q + 1, "mean_excess6"] = yx.mean()
            tab.loc[q + 1, "P_parab"] = p.mean()
            for reg, lab in ((True, "on"), (False, "off")):
                yy = np.concatenate(agg_reg[reg][q][0]) if agg_reg[reg][q][0] else np.array([np.nan])
                pp = np.concatenate(agg_reg[reg][q][1]) if agg_reg[reg][q][1] else np.array([np.nan])
                tab.loc[q + 1, f"excess6_{lab}"] = np.nanmean(yy) if len(yy) else np.nan
                tab.loc[q + 1, f"P_parab_{lab}"] = np.nanmean(pp) if len(pp) else np.nan
        res[name] = tab
    return res


def rank_ic(feats, fwd6, valid, dates, regime):
    """Spearman IC per date; return DataFrame dates x features."""
    out = pd.DataFrame(index=dates, columns=list(feats), dtype=float)
    for d in dates:
        vrow = valid.loc[d].values
        yraw = fwd6.loc[d].values
        for name, F in feats.items():
            f = F.loc[d].values
            m = vrow & ~np.isnan(f) & ~np.isnan(yraw)
            if m.sum() < 100:
                continue
            fr = pd.Series(f[m]).rank().values
            yr = pd.Series(yraw[m]).rank().values
            out.loc[d, name] = np.corrcoef(fr, yr)[0, 1]
    out["regime_on"] = regime.reindex(dates).values
    return out


def top1pct_profile(feats, fwd6, valid, dates):
    """Median cross-sectional percentile of each feature for per-date top-1%
    fwd-6m winners, plus median raw values winners vs all."""
    pr_win, raw_win, raw_all = {n: [] for n in feats}, {n: [] for n in feats}, {n: [] for n in feats}
    fwd_win = []
    n_win = 0
    for d in dates:
        v = valid.loc[d].values & ~np.isnan(fwd6.loc[d].values)
        if v.sum() < 100:
            continue
        y = fwd6.loc[d].values
        thr = np.nanquantile(y[v], 0.99)
        win = v & (y >= thr)
        n_win += win.sum()
        fwd_win.extend(y[win])
        for name, F in feats.items():
            f = F.loc[d].values
            pr = np.full(len(f), np.nan)
            m = v & ~np.isnan(f)
            pr[m] = pd.Series(f[m]).rank(pct=True).values
            pr_win[name].extend(pr[win])
            raw_win[name].extend(f[win])
            raw_all[name].extend(f[m])
    tab = pd.DataFrame(index=list(feats))
    tab.index.name = "feature"
    for n in feats:
        tab.loc[n, "med_pct_winners"] = np.nanmedian(pr_win[n])
        tab.loc[n, "med_raw_winners"] = np.nanmedian(raw_win[n])
        tab.loc[n, "med_raw_all"] = np.nanmedian(raw_all[n])
    return tab, n_win, np.median(fwd_win)


def biweekly_persistence(close, valid):
    """Spearman IC of F-period formation return rank vs single-period return
    L periods ahead, on the biweekly (10d) grid."""
    idx = close.index
    grid = np.arange(260, len(idx) - 1, 10)        # leave room for formation
    gdates = idx[grid]
    C = close.values
    forms = {1: 10, 2: 20, 3: 30, 6: 60, 13: 130, 26: 260}
    lags = [1, 2, 3, 4, 6, 9, 13]
    res = pd.DataFrame(index=[f"F={f}p({d}d)" for f, d in forms.items()],
                       columns=[f"L={l}" for l in lags], dtype=float)
    res.index.name = "formation"
    for fi, (fp, fd) in enumerate(forms.items()):
        ics = {l: [] for l in lags}
        for gi, g in enumerate(grid):
            if g - fd < 0:
                continue
            form = C[g] / C[g - fd] - 1
            v0 = valid.values[g] & ~np.isnan(form)
            for l in lags:
                a, b = g + 10 * (l - 1), g + 10 * l
                if b >= len(idx):
                    continue
                per = C[b] / C[a] - 1
                m = v0 & ~np.isnan(per)
                if m.sum() < 100:
                    continue
                fr = pd.Series(form[m]).rank().values
                yr = pd.Series(per[m]).rank().values
                ics[l].append(np.corrcoef(fr, yr)[0, 1])
        for l in lags:
            res.iloc[fi, lags.index(l)] = np.mean(ics[l]) if ics[l] else np.nan
    return res


# ---------------------------------------------------------------- main
def main():
    close, high, low, vol, member, spy = load()
    feats = build_features(close, high, low, vol, spy)
    fwd6, fwd12, valid = build_outcomes(close, member)
    regime = regime_series(spy)

    idx = close.index
    last_ok = len(idx) - H6 - 1
    samp = idx[np.arange(260, last_ok, SAMPLE_EVERY)]      # monthly, has 12m history
    samp_ic = idx[np.arange(260, last_ok, IC_EVERY)]       # biweekly

    # ---- base rates
    print("## Base rates\n")
    rates, n_obs = [], 0
    for d in samp:
        v = valid.loc[d].values & ~np.isnan(fwd6.loc[d].values)
        y = fwd6.loc[d].values[v]
        rates.append(((y > PARAB_ABS).mean(), np.median(y), y.mean()))
        n_obs += v.sum()
    rates = np.array(rates)
    print(f"- sampled dates: {len(samp)} (monthly), pooled obs: {n_obs}")
    print(f"- P(fwd6 > +50%) overall: {rates[:,0].mean():.3%}")
    print(f"- median fwd6: {np.median(rates[:,1]):.3%}, mean fwd6: {rates[:,2].mean():.3%}")
    fr12 = []
    for d in samp:
        v = valid.loc[d].values & ~np.isnan(fwd12.loc[d].values)
        if v.sum() > 100:
            fr12.append((fwd12.loc[d].values[v] > 1.0).mean())
    print(f"- P(fwd12 > +100%): {np.mean(fr12):.3%}\n")

    # ---- deciles
    print("## Decile analysis (monthly sampled, per-date deciles)\n")
    dec = decile_analysis(feats, fwd6, valid, samp, regime)
    for name, tab in dec.items():
        print(f"### {name}\n")
        print(md_table(tab, "{:.4f}"))
        print()

    # ---- rank IC
    print("## Rank IC (Spearman, fwd 6m) — biweekly dates\n")
    ic = rank_ic(feats, fwd6, valid, samp_ic, regime)
    icf = ic[list(feats)]
    summ = pd.DataFrame(index=list(feats)); summ.index.name = "feature"
    summ["mean_IC"] = icf.mean()
    summ["IC_IR"] = icf.mean() / icf.std()
    summ["pct_pos"] = (icf > 0).mean()
    on = ic["regime_on"] == True   # noqa: E712
    summ["IC_spy_above200"] = icf[on].mean()
    summ["IC_spy_below200"] = icf[~on].mean()
    print(md_table(summ.sort_values("mean_IC", ascending=False), "{:.4f}"))
    print()
    print("### IC by year\n")
    byyear = icf.groupby(icf.index.year).mean().T
    byyear.index.name = "feature"
    print(md_table(byyear, "{:.3f}"))
    print()
    posyears = (icf.groupby(icf.index.year).mean() > 0).sum().to_frame("years_IC>0")
    posyears["n_years"] = icf.groupby(icf.index.year).mean().notna().sum()
    posyears.index.name = "feature"
    print(md_table(posyears, "{:d}"))
    print()

    # ---- top 1% winners
    print("## Profile of per-date top-1% fwd-6m winners (monthly sampled)\n")
    prof, n_win, med_fwd = top1pct_profile(feats, fwd6, valid, samp)
    print(f"- pooled winner obs: {n_win}, median fwd6 of winners: {med_fwd:.1%}\n")
    print(md_table(prof, "{:.3f}"))
    print()

    # ---- biweekly persistence
    print("## Biweekly relative-strength persistence (mean Spearman IC)\n")
    pers = biweekly_persistence(close, valid)
    print(md_table(pers, "{:.4f}"))
    print()

    ic.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "eda_parabolic_ic.csv"))


if __name__ == "__main__":
    main()
