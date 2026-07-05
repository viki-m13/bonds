"""Lazy Prices (Cohen-Malloy-Nguyen 2020) signal build.
Streaming per-firm cosine similarity between consecutive same-type SEC filings.
Stores ONLY scalar sims keyed (ticker, filing_date, form). Resume-safe.
Output: loop_lazyprices_sims.parquet  cols=[tk,cik,form,filing_date,prev_date,sim,nwords]
"""
import glob, io, json, os, sys, time, zipfile, urllib.request, warnings, re, math
from collections import Counter
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "research viktormashalov@gmail.com"}
OUT = f"{HERE}/loop_lazyprices_sims.parquet"
FORMS = sys.argv[1] if len(sys.argv) > 1 else "10-K"   # "10-K" or "10-Q"
FORM_SET = {"10-K", "10-K405", "10-KSB"} if FORMS == "10-K" else {"10-Q"}
t0 = time.time()
def p(*a): print(*a, flush=True)

def get(u, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60).read()
        except Exception:
            time.sleep(1.0 * (k + 1))
    return None

_tag = re.compile(r"(?s)<[^>]+>")
_sc = re.compile(r"(?is)<script.*?</script>")
_st = re.compile(r"(?is)<style.*?</style>")
_ent = re.compile(r"&#?\w+;")
_non = re.compile(r"[^a-z ]")
def clean_words(raw):
    t = raw.decode("utf-8", "ignore")
    t = _sc.sub(" ", t); t = _st.sub(" ", t); t = _tag.sub(" ", t)
    t = _ent.sub(" ", t); t = _non.sub(" ", t.lower())
    return t.split()

def cos(a, b):
    common = set(a) & set(b)
    if not common: return 0.0
    num = sum(a[w] * b[w] for w in common)
    da = math.sqrt(sum(v * v for v in a.values())); db = math.sqrt(sum(v * v for v in b.values()))
    return num / (da * db) if da and db else float("nan")

# ---- universe: ever top-600 $vol (2005+), most-liquid first ----
D = pd.read_pickle(f"{HERE}/_featmat.pkl")
cols = set(D["cols"]); del D
dv = pd.read_pickle(f"{HERE}/_dv_monthly.pkl").loc["2005-01-01":]
dvr = dv.rank(axis=1, ascending=False)
ever = dvr.columns[(dvr <= 600).any(axis=0)]
ever = [c for c in ever if c in cols]
meanrank = dvr[ever].mean(axis=0).sort_values()   # best (lowest) first
targets = list(meanrank.index)
p(f"universe: {len(targets)} tickers (ever top600 2005+, in featmat)")

# ---- ticker -> CIK ----
cur = json.loads(get("https://www.sec.gov/files/company_tickers.json"))
tk2cik = {v["ticker"].upper(): int(v["cik_str"]) for v in cur.values()}
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
targets = [t for t in targets if t in tk2cik]
p(f"mapped: {len(targets)}  t={time.time()-t0:.0f}s")

# ---- resume ----
rows = []
done_tk = set()
if os.path.exists(OUT):
    prev = pd.read_parquet(OUT)
    prev = prev[prev.form.isin(FORM_SET)]
    rows = prev.to_dict("records")
    done_tk = set(prev.tk.unique())
    p(f"resume: {len(done_tk)} tickers already done ({len(rows)} rows)")

def list_filings(cik):
    """Return list of (form, filingDate, accession, primaryDoc) for FORM_SET, oldest->newest."""
    raw = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", tries=2)
    time.sleep(0.12)
    if not raw: return None
    try: d = json.loads(raw)
    except Exception: return None
    out = []
    def harvest(b):
        for f_, dt_, ac_, pd_ in zip(b.get("form", []), b.get("filingDate", []),
                                     b.get("accessionNumber", []), b.get("primaryDocument", [])):
            if f_ in FORM_SET and pd_ and dt_ >= "2004-01-01":
                out.append((f_, dt_, ac_, pd_))
    harvest(d.get("filings", {}).get("recent", {}))
    for extra in d.get("filings", {}).get("files", []):
        raw2 = get("https://data.sec.gov/submissions/" + extra["name"], tries=2)
        time.sleep(0.12)
        if raw2:
            try: harvest(json.loads(raw2))
            except Exception: pass
    out.sort(key=lambda x: x[1])
    return out

def fetch_doc_words(cik, acc, doc):
    an = acc.replace("-", "")
    u = f"https://www.sec.gov/Archives/edgar/data/{cik}/{an}/{doc}"
    raw = get(u, tries=2)
    time.sleep(0.12)
    if not raw: return None
    return clean_words(raw)

def save():
    pd.DataFrame(rows).to_parquet(OUT)

seen_cik = {}
processed = 0
for i, tk in enumerate(targets):
    if tk in done_tk:
        continue
    cik = tk2cik[tk]
    fl = seen_cik.get(cik)
    if fl is None:
        fl = list_filings(cik)
        seen_cik[cik] = fl if fl is not None else []
    if not fl:
        processed += 1
        continue
    prev_vec = None; prev_date = None
    for (form, dt, acc, doc) in fl:
        w = fetch_doc_words(cik, acc, doc)
        if w is None or len(w) < 200:
            continue
        v = Counter(w)
        if prev_vec is not None:
            s = cos(prev_vec, v)
            rows.append({"tk": tk, "cik": cik, "form": form, "filing_date": dt,
                         "prev_date": prev_date, "sim": s, "nwords": len(w)})
        prev_vec = v; prev_date = dt
    processed += 1
    if processed % 25 == 0:
        save()
        p(f"  {i+1}/{len(targets)} done_tk~{len(done_tk)+processed} rows={len(rows)} t={time.time()-t0:.0f}s")

save()
p(f"DONE {FORMS}: {len(rows)} rows, tickers={pd.DataFrame(rows).tk.nunique() if rows else 0}  t={time.time()-t0:.0f}s")
