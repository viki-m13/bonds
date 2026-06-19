"""Exp 33 — settle the premia question with survivorship-bias-free data.
OpenSourceAP (Chen-Zimmermann, peer-reviewed, CRSP-based, release-lags applied):
download 212 long-short premium return series, test if illiquidity/size/etc are
real AND still alive (publication-decay). pip install openassetpricing.
"""
import numpy as np, pandas as pd
from openassetpricing import OpenAP
oap = OpenAP()
df = oap.dl_port('op', 'pandas')      # PredictorPortsFull: monthly LS returns/signal
ls = df[df['port'] == 'LS'].copy(); ls['date'] = pd.to_datetime(ls['date'])
def t_of(sig, lo, hi):
    s = ls[(ls.signalname == sig) & (ls.date >= lo) & (ls.date < hi)]['ret'].dropna()
    return s.mean() / (s.std() + 1e-12) * np.sqrt(len(s)) if len(s) > 24 else np.nan
# % of anomalies still same-sign & |t|>2 in 2016-2024 (publication decay)
alive = tot = flip = 0
for sig in ls.signalname.unique():
    f = ls[(ls.signalname==sig)&(ls.date<'2016-01-01')]['ret']
    r = ls[(ls.signalname==sig)&(ls.date>='2016-01-01')]['ret']
    if len(f) > 24 and len(r) > 24:
        tot += 1
        tr = r.mean()/(r.std()+1e-12)*np.sqrt(len(r))
        if np.sign(f.mean())==np.sign(r.mean()) and abs(tr) > 2: alive += 1
        if np.sign(f.mean())!=np.sign(r.mean()): flip += 1
print(f"Illiquidity 2017-24 t={t_of('Illiquidity','2016-01-01','2025-01-01'):.1f} "
      f"(pre-2004 t={t_of('Illiquidity','1900-01-01','2004-01-01'):.1f})")
print(f"Of {tot} anomalies: {alive} alive (same-sign |t|>2 since 2016), {flip} flipped sign")
