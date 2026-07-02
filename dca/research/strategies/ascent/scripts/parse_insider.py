"""Parse locally-downloaded SEC Form-345 quarterly zips -> _insider_rich.pkl
(schema: tk, ym, buy, sell, nbuyers, off_buy, ceo_buy; keyed on FILING_DATE)."""
import glob, io, os, time, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

import os as _os; HERE = _os.environ.get("ASCENT_WORK", "/tmp/ascent_work"); _os.makedirs(HERE, exist_ok=True)
REPO = _os.environ.get("BONDS_REPO", _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "..")))
t0 = time.time()
frames = []
for f in sorted(glob.glob(f"{HERE}/sec_insider/*_form345.zip")):
    try:
        z = zipfile.ZipFile(f)
        sub = pd.read_csv(io.BytesIO(z.read("SUBMISSION.tsv")), sep="\t", dtype=str,
                          usecols=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL"])
        nt = pd.read_csv(io.BytesIO(z.read("NONDERIV_TRANS.tsv")), sep="\t", dtype=str,
                         usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES",
                                  "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
        try:
            ro = pd.read_csv(io.BytesIO(z.read("REPORTINGOWNER.tsv")), sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
        except Exception:
            ro = pd.DataFrame(columns=["ACCESSION_NUMBER", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
    except Exception as e:
        print(f"  {os.path.basename(f)}: parse fail {e}", flush=True); continue
    nt["sh"] = pd.to_numeric(nt.TRANS_SHARES, errors="coerce")
    nt["px"] = pd.to_numeric(nt.TRANS_PRICEPERSHARE, errors="coerce")
    nt["val"] = (nt.sh * nt.px).fillna(0.0)
    nt["buy"] = np.where((nt.TRANS_CODE == "P") & (nt.TRANS_ACQUIRED_DISP_CD == "A"), nt.val, 0.0)
    nt["sell"] = np.where((nt.TRANS_CODE == "S") & (nt.TRANS_ACQUIRED_DISP_CD == "D"), nt.val, 0.0)
    ab = nt.groupby("ACCESSION_NUMBER")[["buy", "sell"]].sum().reset_index()
    ab = ab.merge(sub, on="ACCESSION_NUMBER").merge(ro.drop_duplicates("ACCESSION_NUMBER"),
                                                    on="ACCESSION_NUMBER", how="left")
    ab = ab[(ab.ISSUERTRADINGSYMBOL.notna()) & (ab.ISSUERTRADINGSYMBOL != "NONE")]
    ab["fd"] = pd.to_datetime(ab.FILING_DATE, errors="coerce", format="mixed")
    ab = ab.dropna(subset=["fd"])
    ab["ym"] = ab.fd.values.astype("datetime64[M]")
    rel = ab.RPTOWNER_RELATIONSHIP.fillna("")
    tit = ab.RPTOWNER_TITLE.fillna("").str.upper()
    ab["is_officer"] = rel.str.contains("Officer")
    ab["is_ceocfo"] = tit.str.contains("CEO|CFO|CHIEF|PRESIDENT", regex=True)
    ab["is_buyer"] = ab.buy > 0
    ab["offbuy_v"] = np.where(ab.is_officer & ab.is_buyer, ab.buy, 0.0)
    ab["ceobuy_v"] = np.where(ab.is_ceocfo & ab.is_buyer, ab.buy, 0.0)
    g = ab.groupby(["ISSUERTRADINGSYMBOL", "ym"]).agg(
        buy=("buy", "sum"), sell=("sell", "sum"), nbuyers=("is_buyer", "sum"),
        off_buy=("offbuy_v", "sum"), ceo_buy=("ceobuy_v", "sum")).reset_index()
    frames.append(g)
    print(f"  {os.path.basename(f)} rows={len(g)} t={time.time()-t0:.0f}s", flush=True)

P = pd.concat(frames).groupby(["ISSUERTRADINGSYMBOL", "ym"]).sum().reset_index()
P.columns = ["tk", "ym", "buy", "sell", "nbuyers", "off_buy", "ceo_buy"]
P.to_pickle(f"{HERE}/_insider_rich.pkl")
print(f"rich panel {len(P)} rows  t={time.time()-t0:.0f}s", flush=True)
