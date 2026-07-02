"""v3.2 blend selection — DD-throttle fix + candidate sleeves.
Selection on the 2014-01-02..2018-12-31 segment ONLY (never prints post-2018)."""
import sys, glob, os
sys.path.insert(0, "/home/user/bonds/alt")
import numpy as np
import pandas as pd
from sleeve_engine import sharpe, cagr, max_dd, ann_vol
import phoenix_production as prod

SEG_END = "2018-12-31"
CAND = "/tmp/claude-0/-home-user-bonds/86c83463-5faf-5e8c-aa3c-69588fbfea09/scratchpad/candidates/"

base_df = prod.load_sleeve_returns()

def load_cand(name):
    df = pd.read_csv(CAND + name + ".csv", parse_dates=["Date"]).set_index("Date")
    return df.iloc[:, 0].reindex(base_df.index).fillna(0.0)

def overlay_fixed(raw, dd_start=-0.05, dd_floor=-0.15):
    """Tail overlays with the CORRECTED deadband DD throttle + existing vol gate."""
    cum = (1 + raw).cumprod()
    hwm = cum.rolling(252, min_periods=30).max()
    dd = cum / hwm - 1
    dd_mult = ((dd_floor - dd) / (dd_floor - dd_start)).clip(0.0, 1.0)
    sv = raw.rolling(60).std()
    thr = sv.rolling(252, min_periods=60).quantile(0.99)
    gate = pd.Series(np.where(sv <= thr, 1.0, 0.5), index=raw.index)
    total = (dd_mult * gate).shift(2).fillna(1.0)
    tc = total.diff().abs().fillna(0) * 10 / 1e4
    return raw * total - tc, total

def wf_blend(df):
    years = sorted(set(df.loc["2014-01-02":].index.year))
    W = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    for y in years:
        end = pd.Timestamp(f"{y-1}-12-31"); start = end - pd.DateOffset(years=4)
        win = df.loc[start:end]
        active = [c for c in df.columns if win[c].std()*np.sqrt(252) >= 0.05]
        if not active: continue
        srs = {k: sharpe(win[k]) for k in active}
        corr = win[active].corr()
        rho = (corr.sum(axis=1)-1)/max(len(corr)-1,1)
        bud = pd.Series({k: max(srs[k],0.3)*max(1-rho[k],0.05) for k in active})
        w = pd.Series(1/len(active), index=active)
        cov = win[active].cov()
        for _ in range(500):
            rc = w.values*(cov.values@w.values)
            ratio = (bud.values/bud.sum())/np.maximum(rc/max(rc.sum(),1e-18),1e-12)
            w = pd.Series(np.clip(w.values*ratio**0.3,1e-6,None), index=active); w = w/w.sum()
        w = w.clip(upper=0.35); w = w/w.sum()
        W.loc[str(y), active] = w.reindex(active).values
    W = W.loc["2014-01-02":]
    return (df.loc[W.index]*W).sum(axis=1), W

def seg(name, sleeves_extra: dict):
    df = base_df.copy()
    for k, v in sleeves_extra.items():
        df[k] = v
    raw, W = wf_blend(df)
    net, _ = overlay_fixed(raw)
    r = net.loc[:SEG_END]
    w_last_seg = W.loc[:"2018-12-31"].iloc[-1]
    tops = ", ".join(f"{k}:{v:.2f}" for k, v in w_last_seg.sort_values(ascending=False).head(5).items())
    print(f"{name:34s} 14-18: SR {sharpe(r):5.2f}  CAGR {cagr(r)*100:5.1f}%  "
          f"Vol {ann_vol(r)*100:4.1f}%  MDD {max_dd(r)*100:5.1f}%  | 2018w: {tops}")
    return net

c = {n: load_cand(n) for n in
     ["cal_twdn_ph", "crash_recovery", "svxy_postpanic", "vix_postpanic",
      "hyg_lead_sso", "gh52_hyggate_eq", "sector_gh_hedge", "smh_lead_tqqq",
      "credit_breadth_def"]}

print("== v3.2 selection (2014-2018 segment only; corrected DD throttle everywhere) ==")
seg("base7 (throttle fixed)", {})
seg("+CAL(twdn_ph)", {"CAL": c["cal_twdn_ph"]})
seg("+CRC(crash_recovery)", {"CRC": c["crash_recovery"]})
seg("+SVP(svxy_postpanic)", {"SVP": c["svxy_postpanic"]})
seg("+GHR(gh52_hyggate)", {"GHR": c["gh52_hyggate_eq"]})
seg("+HLS(hyg_lead_sso)", {"HLS": c["hyg_lead_sso"]})
seg("+SGH(sector_gh_hedge)", {"SGH": c["sector_gh_hedge"]})
seg("+CAL+CRC", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"]})
seg("+CAL+CRC+SVP", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"], "SVP": c["svxy_postpanic"]})
seg("+CAL+CRC+GHR", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"], "GHR": c["gh52_hyggate_eq"]})
seg("+CAL+CRC+SVP+GHR", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"],
                          "SVP": c["svxy_postpanic"], "GHR": c["gh52_hyggate_eq"]})
seg("+CAL+CRC+SVP+HLS", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"],
                          "SVP": c["svxy_postpanic"], "HLS": c["hyg_lead_sso"]})
seg("+CAL+CRC+SVP+GHR+HLS", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"],
                              "SVP": c["svxy_postpanic"], "GHR": c["gh52_hyggate_eq"],
                              "HLS": c["hyg_lead_sso"]})
seg("+CAL+CRC+SVP+GHR+SGH", {"CAL": c["cal_twdn_ph"], "CRC": c["crash_recovery"],
                              "SVP": c["svxy_postpanic"], "GHR": c["gh52_hyggate_eq"],
                              "SGH": c["sector_gh_hedge"]})
