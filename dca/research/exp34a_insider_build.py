"""Exp 34a — build a CLEAN monthly insider net-buying panel from SEC's official
Form-345 bulk TSV datasets (no scraping). Point-in-time: keyed on FILING_DATE
(when info became public). Open-market purchases (code P) vs sales (S).
Output: monthly panel of buy$, sell$, n_buy_filings per ticker -> _insider_panel.pkl
"""
import warnings, time, io, zipfile, urllib.request, ssl
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "research viktormashalov@gmail.com"}
t0 = time.time()


def get(u):
    for k in range(3):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA),
                                          timeout=90, context=ctx).read()
        except Exception:
            time.sleep(2 * (k + 1))
    return None


frames = []
for yr in range(2010, 2026):
    for q in (1, 2, 3, 4):
        if (yr, q) > (2025, 2):
            break
        url = ("https://www.sec.gov/files/structureddata/data/"
               f"insider-transactions-data-sets/{yr}q{q}_form345.zip")
        raw = get(url)
        if not raw:
            print(f"  {yr}q{q}: download fail", flush=True); continue
        try:
            z = zipfile.ZipFile(io.BytesIO(raw))
            sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t",
                              dtype=str, usecols=["ACCESSION_NUMBER", "FILING_DATE",
                              "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"])
            nt = pd.read_csv(io.BytesIO(z.read("NONDERIV_TRANS.tsv")), sep="\t",
                             dtype=str, usecols=["ACCESSION_NUMBER", "TRANS_CODE",
                             "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
        except Exception as e:
            print(f"  {yr}q{q}: parse fail {e}", flush=True); continue
        sub = sub[sub.DOCUMENT_TYPE.isin(["4", "4/A"])]
        nt = nt[nt.TRANS_CODE.isin(["P", "S"])].copy()
        nt["sh"] = pd.to_numeric(nt.TRANS_SHARES, errors="coerce")
        nt["px"] = pd.to_numeric(nt.TRANS_PRICEPERSHARE, errors="coerce")
        nt["val"] = nt.sh * nt.px
        nt = nt.dropna(subset=["val"])
        m = nt.merge(sub, on="ACCESSION_NUMBER", how="inner")
        m = m[(m.ISSUERTRADINGSYMBOL.notna()) & (m.ISSUERTRADINGSYMBOL != "NONE")]
        m["fd"] = pd.to_datetime(m.FILING_DATE, errors="coerce")
        m = m.dropna(subset=["fd"])
        m["ym"] = m.fd.values.astype("datetime64[M]")
        m["buy"] = np.where(m.TRANS_CODE == "P", m.val, 0.0)
        m["sell"] = np.where(m.TRANS_CODE == "S", m.val, 0.0)
        m["isbuy"] = (m.TRANS_CODE == "P").astype(int)
        g = m.groupby(["ISSUERTRADINGSYMBOL", "ym"]).agg(
            buy=("buy", "sum"), sell=("sell", "sum"),
            nbuy=("isbuy", "sum")).reset_index()
        frames.append(g)
    print(f"  through {yr}: {sum(len(f) for f in frames)} ticker-months  "
          f"t={time.time()-t0:.0f}s", flush=True)

P = pd.concat(frames).groupby(["ISSUERTRADINGSYMBOL", "ym"]).sum().reset_index()
P.columns = ["tk", "ym", "buy", "sell", "nbuy"]
P.to_pickle("/tmp/wave/_insider_panel.pkl")
print(f"\nDONE: {len(P)} ticker-months, {P.tk.nunique()} tickers, "
      f"{P.ym.min()}->{P.ym.max()}  t={time.time()-t0:.0f}s", flush=True)
