"""Experiment 22 — INSIDER BUYING (SEC Form 4), orthogonal non-price signal.
Objective universe (22 names by pre-2018 liquidity rank, spanning mega->mid cap).
Scrape Form-4 filings, parse open-market PURCHASES (code P, acquired) vs SALES
(code S). Signal = trailing net insider buying. STRICT test: does insider buying
predict forward return relative to QQQ? Split early(2015-2020)/late(2020-2026).
Insider edge is documented strongest in smaller caps; mega-caps rarely buy."""
import sys, warnings, json, re, time, urllib.request, ssl, pickle
warnings.filterwarnings("ignore")
for p in ("/tmp/wave", "/tmp/wave/site", "/tmp/wave/research_lab"):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from research_lab.lab import Lab

ctx = ssl.create_default_context(); ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "research viktormashalov@gmail.com"}
NAMES = ['AAPL', 'DIS', 'SLB', 'REGN', 'BMY', 'WYNN', 'CSX', 'MPC', 'VRTX',
         'CMI', 'APTV', 'LIN', 'SYK', 'FSLR', 'KEY', 'SIRI', 'KSS', 'SWK',
         'PCAR', 'MSI', 'EMN', 'CLF']
t0 = time.time()


def get(url, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                          timeout=30, context=ctx).read()
        except Exception:
            time.sleep(0.5 * (k + 1))
    return b""


cikmap = {v['ticker']: str(v['cik_str']).zfill(10)
          for v in json.loads(get("https://www.sec.gov/files/company_tickers.json")
                              .decode()).values()}
recs = []                                  # (ticker, month, buy$, sell$)
for ti, tk in enumerate(NAMES):
    cik = cikmap.get(tk)
    if not cik:
        print(f"  {tk}: no CIK", flush=True); continue
    sub = get(f"https://data.sec.gov/submissions/CIK{cik}.json")
    if not sub:
        continue
    rec = json.loads(sub.decode())['filings']['recent']
    f4 = [(rec['accessionNumber'][i], rec['filingDate'][i], rec['primaryDocument'][i])
          for i in range(len(rec['form'])) if rec['form'][i] == '4']
    nb = 0
    for acc, dt, doc in f4:
        accnd = acc.replace('-', ''); fn = doc.split('/')[-1]
        raw = get(f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accnd}/{fn}")
        if not raw:
            continue
        x = raw.decode(errors='ignore')
        # iterate non-derivative transactions: code + acq/disp + shares + price
        codes = re.findall(r'<transactionCode>(\w)</transactionCode>', x)
        ad = re.findall(r'<transactionAcquiredDisposedCode>\s*<value>(\w)', x)
        sh = re.findall(r'<transactionShares>\s*<value>([\d.]+)', x)
        pr = re.findall(r'<transactionPricePerShare>\s*<value>([\d.]*)', x)
        b = s = 0.0
        for j, c in enumerate(codes):
            if j >= len(sh):
                break
            shares = float(sh[j] or 0)
            price = float(pr[j]) if j < len(pr) and pr[j] else 0.0
            val = shares * price
            if c == 'P':                    # open-market purchase
                b += val; nb += 1
            elif c == 'S':                  # open-market sale
                s += val
        if b or s:
            recs.append((tk, dt[:7], b, s))
        time.sleep(0.05)
    print(f"  [{ti+1}/{len(NAMES)}] {tk}: {len(f4)} F4s, purchases so far={nb}"
          f"  t={time.time()-t0:.0f}s", flush=True)

df = pd.DataFrame(recs, columns=['tk', 'ym', 'buy', 'sell'])
df.to_pickle("/tmp/wave/_insider.pkl")
g = df.groupby(['tk', 'ym']).sum().reset_index()
print(f"\nscraped {len(df)} filing-records; ticker-months with activity {len(g)};"
      f" total purchase$ {df.buy.sum()/1e6:.0f}M sale$ {df.sell.sum()/1e6:.0f}M",
      flush=True)

# ---- analysis: does net insider buying predict forward rel-to-QQQ return? ----
lab = Lab(); cal = lab.cal; cpx = lab.cpx; q = cpx["QQQ"]
me = cpx.resample("ME").last()
qme = q.resample("ME").last()
g['date'] = pd.to_datetime(g['ym']) + pd.offsets.MonthEnd(0)
g['net'] = g['buy'] - g['sell']
# monthly per-ticker net-buy panel
buy_signal = g.pivot_table(index='date', columns='tk', values='buy', aggfunc='sum').fillna(0)
buy_signal = buy_signal.reindex(me.index).fillna(0)


def fwd_rel(h):
    f = me.shift(-h) / me - 1
    fq = qme.shift(-h) / qme - 1
    return f.sub(fq, axis=0)


for tag, lo, hi in (("EARLY 2015-2020", "2015-01-01", "2020-01-01"),
                    ("LATE 2020-2026", "2020-01-01", "2026-06-01")):
    print(f"\n{tag}:", flush=True)
    for h in (3, 12):
        fr = fwd_rel(h)
        buy_rets, nobuy_rets, ics = [], [], []
        for d in me.index:
            if not (pd.Timestamp(lo) <= d < pd.Timestamp(hi)):
                continue
            row = buy_signal.loc[d, [t for t in NAMES if t in buy_signal.columns]]
            for tk in row.index:
                if tk in fr.columns and np.isfinite(fr.loc[d, tk]):
                    (buy_rets if row[tk] > 0 else nobuy_rets).append(fr.loc[d, tk])
        b = np.array(buy_rets); nb = np.array(nobuy_rets)
        if len(b) >= 10:
            print(f"   fwd{h:>3}m: insider-BUY months n={len(b):4d} mean rel-QQQ "
                  f"{b.mean()*100:+5.1f}%  |  no-buy n={len(nb):5d} mean "
                  f"{nb.mean()*100:+5.1f}%  |  diff {(b.mean()-nb.mean())*100:+5.1f}pp",
                  flush=True)
        else:
            print(f"   fwd{h:>3}m: too few insider-buy months (n={len(b)})", flush=True)
print(f"\nDONE t={time.time()-t0:.0f}s", flush=True)
