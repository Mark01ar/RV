import json
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import macro as MA

D = MA.ROOT / "data"
SALIDA = D / "factores.json"
FACTORES = [
    {"id": "tasa", "nombre": "Tasa 10 años EE.UU.", "serie": "DGS10", "fuente": "FRED", "tipo": "nivel", "mov": 1.0, "txt": "la tasa a 10 años sube 1 punto"},
    {"id": "curva", "nombre": "Pendiente de la curva (10a − 2a)", "serie": "T10Y2Y", "fuente": "FRED", "tipo": "nivel", "mov": 0.5, "txt": "la curva se empina 0,5 punto"},
    {"id": "credito", "nombre": "Spread de crédito corporativo", "serie": "BAA10Y", "fuente": "FRED", "tipo": "nivel", "mov": 1.0, "txt": "el spread de crédito sube 1 punto"},
    {"id": "dolar", "nombre": "Dólar global (DXY)", "serie": "DX-Y.NYB", "fuente": "Yahoo", "tipo": "precio", "mov": 0.05, "txt": "el dólar global sube 5%"},
    {"id": "petroleo", "nombre": "Petróleo WTI", "serie": "CL=F", "fuente": "Yahoo", "tipo": "precio", "mov": 0.10, "txt": "el petróleo sube 10%"},
    {"id": "gas", "nombre": "Gas natural", "serie": "NG=F", "fuente": "Yahoo", "tipo": "precio", "mov": 0.10, "txt": "el gas natural sube 10%"},
    {"id": "cobre", "nombre": "Cobre", "serie": "HG=F", "fuente": "Yahoo", "tipo": "precio", "mov": 0.10, "txt": "el cobre sube 10%"},
    {"id": "oro", "nombre": "Oro", "serie": "GC=F", "fuente": "Yahoo", "tipo": "precio", "mov": 0.10, "txt": "el oro sube 10%"},
    {"id": "bitcoin", "nombre": "Bitcoin", "serie": "BTC-USD", "fuente": "Yahoo", "tipo": "precio", "mov": 0.20, "txt": "bitcoin sube 20%"},
    {"id": "yuan", "nombre": "Yuan (dólares por yuan, invertido)", "serie": "CNY=X", "fuente": "Yahoo", "tipo": "precio", "mov": 0.03, "txt": "el yuan se debilita 3%"},
    {"id": "real", "nombre": "Real brasileño (invertido)", "serie": "BRL=X", "fuente": "Yahoo", "tipo": "precio", "mov": 0.05, "txt": "el real se debilita 5%"},
    {"id": "consumo", "nombre": "Confianza del consumidor (U. Michigan)", "serie": "UMCSENT", "fuente": "FRED", "tipo": "nivel", "mov": 5.0, "txt": "la confianza del consumidor sube 5 puntos"},
]
TRAMOS = [("a", "2005-2014", "2005-01", "2014-12"), ("r", "2015-hoy", "2015-01", None)]
MIN_PRUEBA = 36
MIN_RECIENTE = 60
P_CLARA = 0.05
P_RECIENTE = 0.01
ROT = 999
rng = np.random.default_rng(7)


def mensual(s):
    s = s.dropna().resample("ME").last()
    s.index = s.index.to_period("M")
    return s


def bajar(tk):
    yt = tk + ["SPY"] + [f["serie"] for f in FACTORES if f["fuente"] == "Yahoo"]
    px = yf.download(yt, start="2004-11-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.ffill().resample("ME").last()
    px.index = px.index.to_period("M")
    niv = {}
    for f in FACTORES:
        if f["fuente"] == "FRED":
            s, _ = MA.fred(f["serie"])
            niv[f["serie"]] = mensual(s[s.index >= "2004-11-01"])
    return px, niv


def prueba(y, m, f):
    d = pd.concat([y, m, f], axis=1).dropna()
    n = len(d)
    if n < MIN_PRUEBA:
        return None
    yv, mv, fv = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
    q, _ = np.linalg.qr(np.column_stack([np.ones(n), mv]))
    yr = yv - q @ (q.T @ yv)
    k = rng.integers(12, n - 12, ROT)
    F = np.column_stack([fv, fv[(np.arange(n)[:, None] - k[None, :]) % n]])
    Fr = F - q @ (q.T @ F)
    c = (Fr.T @ yr) / (Fr ** 2).sum(axis=0)
    p = (1 + np.sum(np.abs(c[1:]) >= abs(c[0]))) / (ROT + 1)
    return {"c": float(c[0]), "p": round(float(p), 3), "n": int(n), "desde": str(d.index[0])}


def tramo(s, a, b):
    s = s[s.index >= pd.Period(a, "M")]
    return s if b is None else s[s.index <= pd.Period(b, "M")]


def etiqueta(a, r):
    sa = a is not None and a["p"] < P_CLARA
    sr = r is not None and r["p"] < P_CLARA
    if sa and sr:
        return "clara" if np.sign(a["mov"]) == np.sign(r["mov"]) else "vuelta"
    if r is not None and r["p"] < P_RECIENTE and r["n"] >= MIN_RECIENTE:
        return "reciente"
    if sa:
        return "antigua"
    return ""


def movido(c, f):
    x = f["mov"] if f["tipo"] == "nivel" else np.log(1 + f["mov"])
    return round(float(np.exp(c * x) - 1) * 100, 2)


def ultimos(s, f, nmes=3):
    s = s.dropna()
    if f["tipo"] == "nivel":
        ch = s.diff()
    else:
        ch = s.pct_change() * 100
    ch = ch.dropna().iloc[-nmes:]
    return [{"mes": str(i), "cambio": round(float(v), 2), "nivel": round(float(s.loc[i]), 4)} for i, v in ch.items()]


def main():
    em = {x["byma"]: x["ticker"] for x in json.loads((D / "empresas.json").read_text())["empresas"]}
    bl = json.loads((D / "bloques.json").read_text())["bloques"]
    miembros = {b["id"]: [e for e in b["empresas"] if e in em] for b in bl}
    tk = sorted({em[e] for es in miembros.values() for e in es})
    px, niv = bajar(tk)
    hoy = pd.Period(dt.date.today(), "M")
    cerrado = px[px.index < hoy]
    r = np.log(cerrado).diff().iloc[1:]
    fs = {}
    for f in FACTORES:
        if f["fuente"] == "FRED":
            fs[f["id"]] = niv[f["serie"]][niv[f["serie"]].index < hoy].diff()
        else:
            fs[f["id"]] = r[f["serie"]]
    serie = {e: r[em[e]] for es in miembros.values() for e in es if em[e] in r and r[em[e]].notna().sum() >= MIN_PRUEBA}
    for b in bl:
        cols = [serie[e] for e in miembros[b["id"]] if e in serie]
        if cols:
            serie["@" + b["id"]] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
    res = {}
    for f in FACTORES:
        out = {"bloques": {}, "empresas": {}}
        for k, y in serie.items():
            fila = {}
            for cod, _, a, b in TRAMOS:
                z = prueba(tramo(y, a, b), tramo(r["SPY"], a, b), tramo(fs[f["id"]], a, b))
                if z:
                    z["mov"] = movido(z.pop("c"), f)
                fila[cod] = z
            fila["e"] = etiqueta(fila["a"], fila["r"])
            if k.startswith("@"):
                out["bloques"][k[1:]] = fila
            else:
                out["empresas"][k] = fila
        res[f["id"]] = out
        print(f["id"], sum(1 for v in out["empresas"].values() if v["e"] == "clara"), "claras", file=sys.stderr)
    facs = []
    for f in FACTORES:
        if f["fuente"] == "FRED":
            s = niv[f["serie"]]
        else:
            s = px[f["serie"]].dropna()
        facs.append({**f, "ultimos": ultimos(s, f)})
    salida = {
        "actualizado": dt.date.today().isoformat(),
        "tramos": [{"id": c, "nombre": n, "desde": a, "hasta": b} for c, n, a, b in TRAMOS],
        "reglas": {"p_clara": P_CLARA, "p_reciente": P_RECIENTE, "min_meses": MIN_PRUEBA, "min_reciente": MIN_RECIENTE, "rotaciones": ROT},
        "factores": facs,
        "res": res,
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))
    print(f"factores.json {SALIDA.stat().st_size // 1024} KB", file=sys.stderr)


main()
