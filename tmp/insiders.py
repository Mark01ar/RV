import io
import json
import time
import zipfile
import pandas as pd
import requests

UA = {"User-Agent": "MaQ research bot research@maq-rv.org"}
d = json.load(open("data/datos.json"))
tick = {x["ticker"] for x in d["tickers"]}
filas = []
for y in range(2006, 2027):
    for q in range(1, 5):
        url = f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
        try:
            r = requests.get(url, headers=UA, timeout=120)
            if r.status_code != 200:
                print("sin", y, q, r.status_code, flush=True)
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            nm = {n.split("/")[-1].upper(): n for n in z.namelist()}
            sub = pd.read_csv(z.open(nm["SUBMISSION.TSV"]), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "FILING_DATE", "ISSUERTRADINGSYMBOL", "DOCUMENT_TYPE"])
            sub["t"] = sub["ISSUERTRADINGSYMBOL"].str.upper().str.strip().str.replace(".", "-", regex=False)
            sub = sub[sub["t"].isin(tick) & (sub["DOCUMENT_TYPE"] == "4")]
            tr = pd.read_csv(z.open(nm["NONDERIV_TRANS.TSV"]), sep="\t", dtype=str,
                             usecols=["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
            tr = tr[tr["TRANS_CODE"].isin(["P", "S"])]
            m = tr.merge(sub, on="ACCESSION_NUMBER")
            m["usd"] = pd.to_numeric(m["TRANS_SHARES"], errors="coerce") * pd.to_numeric(m["TRANS_PRICEPERSHARE"], errors="coerce")
            m["fecha"] = pd.to_datetime(m["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
            filas.append(m[["t", "fecha", "TRANS_CODE", "usd", "ACCESSION_NUMBER"]])
            print("ok", y, q, len(m), flush=True)
        except Exception as e:
            print("err", y, q, e, flush=True)
        time.sleep(0.5)
I = pd.concat(filas)
I.to_csv("tmp/insiders.csv", index=False)
print("total", len(I), I["t"].nunique(), (I["TRANS_CODE"] == "P").sum(), flush=True)
