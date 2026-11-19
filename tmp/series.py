import io
import os
import re
import sys
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
OUT.mkdir(exist_ok=True)
HORIZ = (20, 60, 120)
URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{}q{}_form345.zip"
CEO = re.compile(r"\b(ceo|chief executive|president|cfo|chief financial|principal executive|principal financial)\b", re.I)


def col(df, nombre):
    m = {c.upper(): c for c in df.columns}
    return df[m[nombre.upper()]]


def leer(z, nombre, cols):
    archivo = [n for n in z.namelist() if n.upper().endswith(nombre.upper())][0]
    df = pd.read_csv(z.open(archivo), sep="\t", dtype=str, low_memory=False, on_bad_lines="skip")
    m = {c.upper(): c for c in df.columns}
    faltan = [c for c in cols if c not in m]
    if faltan:
        print(nombre, "faltan", faltan, "hay", list(df.columns))
        return None
    return df[[m[c] for c in cols]].set_axis(cols, axis=1)


def fecha(s):
    a = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    b = pd.to_datetime(s, errors="coerce")
    return a.fillna(b)


def compras(simbolos):
    ua = os.environ.get("SEC_UA", "").strip()
    if not ua:
        print("Falta el secret SEC_UA")
        sys.exit(0)
    s = requests.Session()
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    partes = []
    for anio in range(2018, 2027):
        for q in range(1, 5):
            r = None
            for _ in range(3):
                try:
                    r = s.get(URL.format(anio, q), timeout=120)
                    break
                except Exception:
                    time.sleep(5)
            if r is None or r.status_code != 200:
                print(anio, q, "sin archivo", None if r is None else r.status_code)
                continue
            z = zipfile.ZipFile(io.BytesIO(r.content))
            sub = leer(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERTRADINGSYMBOL"])
            own = leer(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"])
            tr = leer(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS"])
            if sub is None or own is None or tr is None:
                continue
            sub["SYM"] = sub["ISSUERTRADINGSYMBOL"].fillna("").str.upper().str.strip().str.replace(".", "-", regex=False)
            sub = sub[sub["DOCUMENT_TYPE"].isin(["4", "4/A"]) & sub["SYM"].isin(simbolos)]
            tr = tr[(tr["TRANS_CODE"] == "P") & (tr["TRANS_ACQUIRED_DISP_CD"] == "A")]
            own = own.groupby("ACCESSION_NUMBER").agg({"RPTOWNERCIK": "first", "RPTOWNER_RELATIONSHIP": "first", "RPTOWNER_TITLE": "first"}).reset_index()
            d = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
            for c in ("TRANS_SHARES", "TRANS_PRICEPERSHARE", "SHRS_OWND_FOLWNG_TRANS"):
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d[(d["TRANS_SHARES"] > 0) & (d["TRANS_PRICEPERSHARE"] > 0)]
            d["FILING_DATE"] = fecha(d["FILING_DATE"])
            d["TRANS_DATE"] = fecha(d["TRANS_DATE"])
            partes.append(d)
            print(anio, q, len(d))
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def bajar(tks):
    out = []
    for i in range(0, len(tks), 40):
        lote = tks[i:i + 40]
        for _ in range(3):
            try:
                df = yf.download(lote, start="2017-06-01", interval="1d", auto_adjust=True, progress=False, threads=True, group_by="column")["Close"]
                break
            except Exception:
                time.sleep(10)
                df = None
        if df is None:
            continue
        if isinstance(df, pd.Series):
            df = df.to_frame(lote[0])
        out.append(df)
    return pd.concat(out, axis=1)


def main():
    u = json.loads((AQUI / "universo.json").read_text(encoding="utf-8"))
    tks = sorted({x["ticker"] for x in u["tickers"] if x.get("ticker")})
    pe = json.loads((AQUI / "pe.json").read_text(encoding="utf-8"))
    d = compras(set(tks))
    if d.empty:
        print("sin compras")
        return
    rel = d["RPTOWNER_RELATIONSHIP"].fillna("")
    d = d[~(rel.str.contains("TenPercent", case=False) & ~rel.str.contains("Director|Officer", case=False))]
    d["valor"] = d["TRANS_SHARES"] * d["TRANS_PRICEPERSHARE"]
    d["ceo"] = d["RPTOWNER_TITLE"].fillna("").str.contains(CEO)
    d["antes"] = d["SHRS_OWND_FOLWNG_TRANS"] - d["TRANS_SHARES"]
    d["aumento"] = np.where(d["antes"] > 0, d["TRANS_SHARES"] / d["antes"], np.nan)
    d = d.drop_duplicates(["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    d.to_csv(OUT / "compras.csv", index=False)
    print("compras", len(d), "acciones", d["SYM"].nunique())
    px = bajar(tks + ["SPY"])
    spy = px["SPY"].dropna()
    filas, base = [], []
    for t, g in d.groupby("SYM"):
        if t not in px.columns:
            continue
        c = px[t].dropna()
        if len(c) < 300:
            continue
        fw = {n: c.shift(-n) / c - 1 for n in HORIZ}
        sf = {n: spy.shift(-n) / spy - 1 for n in HORIZ}
        ini = pd.Timestamp("2018-01-01")
        for n in HORIZ:
            x = fw[n][c.index >= ini].dropna()
            base.append((t, n, x.mean()))
        mx = c.rolling(252, min_periods=120).max()
        serie_pe = pd.Series(pe.get(t, {}))
        if len(serie_pe):
            serie_pe.index = pd.PeriodIndex(serie_pe.index, freq="M")
        ultimo = None
        for f in sorted(g["FILING_DATE"].dropna().unique()):
            f = pd.Timestamp(f)
            if ultimo is not None and (f - ultimo).days < 30:
                continue
            ventana = g[(g["FILING_DATE"] <= f) & (g["FILING_DATE"] > f - pd.Timedelta(days=30))]
            pos = c.index.searchsorted(f, side="right")
            if pos >= len(c):
                continue
            ultimo = f
            dia = c.index[pos]
            fila = {"ticker": t, "fecha": dia, "insiders": ventana["RPTOWNERCIK"].nunique(), "ceo": bool(ventana["ceo"].any()),
                    "valor": ventana["valor"].sum(), "aumento": ventana["aumento"].max(),
                    "caida": c.iloc[pos] / mx.iloc[pos] - 1 if np.isfinite(mx.iloc[pos]) else np.nan}
            barata = np.nan
            if len(serie_pe):
                m = pd.Period(dia, freq="M") - 1
                hist = serie_pe[(serie_pe.index <= m) & (serie_pe.index > m - 60)]
                hist = hist[hist > 0]
                if len(hist) >= 24 and m in serie_pe.index and serie_pe[m] > 0:
                    barata = float(serie_pe[m] < hist.median())
            fila["barata"] = barata
            for n in HORIZ:
                fila[f"r{n}"] = fw[n].iloc[pos]
                s_ = sf[n].get(dia, np.nan)
                fila[f"m{n}"] = fila[f"r{n}"] - s_
            filas.append(fila)
    ev = pd.DataFrame(filas)
    bs = pd.DataFrame(base, columns=["ticker", "h", "media"])
    for n in HORIZ:
        ev[f"x{n}"] = ev[f"r{n}"] - ev["ticker"].map(bs[bs.h == n].set_index("ticker")["media"])
    ev.to_csv(OUT / "eventos.csv", index=False)
    L = [f"Eventos (primera compra de cada racha, separadas 30 días): {len(ev)} en {ev.ticker.nunique()} acciones, {ev.fecha.min():%Y-%m} a {ev.fecha.max():%Y-%m}"]

    def linea(nombre, x):
        L.append(f"\n## {nombre}  (n={len(x)}, acciones={x.ticker.nunique() if len(x) else 0})")
        for n in HORIZ:
            r_ = x[f"r{n}"].dropna()
            if len(r_) < 10:
                L.append(f"  {n:>3} ruedas | pocos casos")
                continue
            xx = x[["fecha", f"x{n}"]].dropna()
            pf = xx.groupby(pd.to_datetime(xx["fecha"]).dt.to_period(f"{max(1, n // 21)}M"))[f"x{n}"].mean()
            t_ = pf.mean() / pf.std() * np.sqrt(len(pf)) if len(pf) > 2 and pf.std() > 0 else np.nan
            L.append(f"  {n:>3} ruedas | n={len(r_):4d} | media {r_.mean()*100:6.2f}% | mediana {r_.median()*100:6.2f}% | positivos {(r_>0).mean()*100:5.1f}% | vs día cualquiera {xx[f'x{n}'].mean()*100:+6.2f}% (t={t_:4.1f}) | vs SPY {x[f'm{n}'].mean()*100:+6.2f}%")

    linea("Todas las compras", ev)
    linea("Compra de CEO/CFO/presidente", ev[ev.ceo])
    linea("2 o más insiders en 30 días", ev[ev.insiders >= 2])
    linea("3 o más insiders en 30 días", ev[ev.insiders >= 3])
    linea("Monto >= US$ 100.000", ev[ev.valor >= 1e5])
    linea("Monto >= US$ 1 millón", ev[ev.valor >= 1e6])
    linea("Aumenta su tenencia 10% o más", ev[ev.aumento >= 0.10])
    linea("Caída de 30% o más desde máximo 12m", ev[ev.caida <= -0.30])
    linea("Barata: P/E bajo su mediana de 5 años", ev[ev.barata == 1])
    linea("Cara: P/E sobre su mediana de 5 años", ev[ev.barata == 0])
    linea("CEO/CFO + caída 30%", ev[ev.ceo & (ev.caida <= -0.30)])
    linea("CEO/CFO + barata", ev[ev.ceo & (ev.barata == 1)])
    linea("2+ insiders + caída 30%", ev[(ev.insiders >= 2) & (ev.caida <= -0.30)])
    linea("2+ insiders + barata", ev[(ev.insiders >= 2) & (ev.barata == 1)])
    linea("2+ insiders + CEO/CFO", ev[(ev.insiders >= 2) & ev.ceo])
    linea("Monto >= 100k + barata o caída 30%", ev[(ev.valor >= 1e5) & ((ev.barata == 1) | (ev.caida <= -0.30))])
    e2 = ev.copy()
    e2["anio"] = pd.to_datetime(e2.fecha).dt.year
    L.append("\n## Todas por año (60 ruedas)")
    for a, g in e2.groupby("anio"):
        L.append(f"  {a}: n={len(g):4d} | media {g.r60.mean()*100:6.2f}% | vs día cualquiera {g.x60.mean()*100:+6.2f}% | positivos {(g.r60>0).mean()*100:5.1f}%")
    (OUT / "resumen.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
