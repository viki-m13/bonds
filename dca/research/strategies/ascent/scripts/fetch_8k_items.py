"""LOOP iter 6 data build — 8-K ITEM-level event panel from SEC submissions API.
Ticker->CIK from sec company_tickers.json + Form-345 SUBMISSION.tsv (delisted).
Fetch submissions JSON (recent + archived pages) for the top-2000-$vol universe,
extract every 8-K's (filingDate, items). Output: _8k_items.parquet
(rows: cik, tk, date, items). Polite ~8 req/s.
"""
import glob, io, json, os, time, zipfile, urllib.request, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
UA = {"User-Agent": "research viktormashalov@gmail.com"}
t0 = time.time()
def p(*a): print(*a, flush=True)

def get(u, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read()
        except Exception:
            time.sleep(1.5 * (k + 1))
    return None

# universe: names ever in top-2000 by $vol (monthly panel)
dv = pd.read_pickle(f"{HERE}/_dv_monthly.pkl")
dvr = dv.rank(axis=1, ascending=False)
ever_top = set(dvr.columns[(dvr <= 2000).any(axis=0)])
p(f"universe ever-top2000: {len(ever_top)}")

# ticker -> CIK: current registrants
cur = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
tk2cik = {v["ticker"].upper(): int(v["cik_str"]) for v in cur.values()}
# + delisted via Form-345 SUBMISSION.tsv (ISSUERCIK)
for f in sorted(glob.glob(f"{HERE}/sec_insider/*_form345.zip")):
    try:
        z = zipfile.ZipFile(f)
        sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t", dtype=str,
                          usecols=["ISSUERCIK", "ISSUERTRADINGSYMBOL"])
        for cik, tk in sub.dropna().drop_duplicates().itertuples(index=False):
            tk = str(tk).upper()
            if tk not in tk2cik:
                try: tk2cik[tk] = int(cik)
                except Exception: pass
    except Exception:
        continue
targets = sorted(t for t in ever_top if t in tk2cik)
p(f"mapped tickers: {len(targets)} of {len(ever_top)}  t={time.time()-t0:.0f}s")

rows = []
done_ciks = {}
for i, tk in enumerate(targets):
    cik = tk2cik[tk]
    if cik in done_ciks:
        for d, it in done_ciks[cik]: rows.append((cik, tk, d, it))
        continue
    raw = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", tries=2)
    time.sleep(0.12)
    if not raw:
        continue
    try:
        d = json.loads(raw)
    except Exception:
        continue
    recs = []
    def harvest(block):
        forms = block.get("form", []); dates = block.get("filingDate", []); items = block.get("items", [])
        for f_, dt_, it_ in zip(forms, dates, items):
            if f_ in ("8-K", "8-K/A") and it_:
                recs.append((dt_, it_))
    harvest(d.get("filings", {}).get("recent", {}))
    for extra in d.get("filings", {}).get("files", []):
        raw2 = get("https://data.sec.gov/submissions/" + extra["name"], tries=2)
        time.sleep(0.12)
        if raw2:
            try: harvest(json.loads(raw2))
            except Exception: pass
    done_ciks[cik] = recs
    for dt_, it_ in recs: rows.append((cik, tk, dt_, it_))
    if i % 100 == 0:
        p(f"  {i}/{len(targets)} rows={len(rows)} t={time.time()-t0:.0f}s")

E = pd.DataFrame(rows, columns=["cik", "tk", "date", "items"])
E.to_parquet(f"{HERE}/_8k_items.parquet")
p(f"saved _8k_items.parquet {E.shape}  t={time.time()-t0:.0f}s")
