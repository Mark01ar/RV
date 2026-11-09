import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path(__file__).resolve().parent
D = OUT.parent / "data"
FAC = {
    "tasa": ("DGS10", "nivel", 1.0, "la tasa a 10 años sube 1 punto"),
    "curva": ("T10Y2Y", "nivel", 0.5, "la curva 10a-2a se empina 0,5 punto"),
    "credito": ("BAA10Y", "nivel", 1.0, "el spread de crédito sube 1 punto"),
    "dolar": ("DX-Y.NYB", "precio", 0.05, "el dólar global sube 5%"),
    "petroleo": ("CL=F", "precio", 0.10, "el petróleo sube 10%"),
    "gas": ("NG=F", "precio", 0.10, "el gas natural sube 10%"),
    "cobre": ("HG=F", "precio", 0.10, "el cobre sube 10%"),
    "oro": ("GC=F", "precio", 0.10, "el oro sube 10%"),
    "yuan": ("CNY=X", "precio", 0.03, "el yuan se debilita 3%"),
    "real": ("BRL=X", "precio", 0.05, "el real se debilita 5%"),
    "consumo": ("UMCSENT", "nivel", 5.0, "la confianza del consumidor sube 5 puntos"),
}
MOTORES = {
    "petroleo": ["petroleo", "gas", "real"],
    "bancos": ["tasa", "curva", "credito", "real"],
    "mineria": ["cobre", "oro", "dolar"],
    "electrica": ["gas", "tasa"],
    "defensivo": ["tasa", "dolar"],
    "ciclico": ["consumo", "dolar", "petroleo"],
    "pagos": ["consumo", "dolar", "credito"],
    "aero": ["petroleo", "consumo", "credito"],
    "industriales": ["cobre", "tasa", "dolar"],
    "china": ["yuan", "cobre", "dolar"],
    "ecommerce": ["real", "consumo", "dolar"],
    "medios": ["consumo", "dolar", "tasa"],
    "bigtech": ["tasa", "dolar"],
    "software": ["tasa", "dolar"],
    "hardware": ["dolar", "consumo", "tasa"],
    "fabricacion": ["dolar", "tasa", "yuan"],
}
TRAMOS = {"2005-2014": ("2005-01", "2014-12"), "2015-2026": ("2015-01", None)}
ROT = 999
rng = np.random.default_rng(7)


def bajar(tk):
    yt = tk + ["SPY"] + [v[0] for v in FAC.values() if v[1] == "precio"]
    px = yf.download(yt, start="2004-11-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.ffill().resample("ME").last()
    px.index = px.index.to_period("M")
    r = np.log(px).diff()
    for sid in [v[0] for v in FAC.values() if v[1] == "nivel"]:
        f = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        f = f.dropna().resample("ME").last()
        f.index = f.index.to_period("M")
        r[sid] = f.diff()
    return r.iloc[1:-1]


def coef(y, m, f):
    X = np.column_stack([np.ones(len(y)), m, f])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[2]


def prueba(y, m, f):
    d = pd.concat([y, m, f], axis=1).dropna()
    if len(d) < 36:
        return None
    yv, mv, fv = d.iloc[:, 0].values, d.iloc[:, 1].values, d.iloc[:, 2].values
    c = coef(yv, mv, fv)
    n = len(fv)
    nul = np.array([coef(yv, mv, np.roll(fv, k)) for k in rng.integers(12, n - 12, ROT)])
    p = (1 + np.sum(np.abs(nul) >= abs(c))) / (ROT + 1)
    return {"c": float(c), "p": float(p), "meses": int(n)}


def tramo(s, a, b):
    s = s[s.index >= pd.Period(a, "M")]
    return s if b is None else s[s.index <= pd.Period(b, "M")]


def etiqueta(t1, t2):
    s1 = t1 is not None and t1["p"] < 0.05
    s2 = t2 is not None and t2["p"] < 0.05
    if s1 and s2:
        return "clara" if np.sign(t1["c"]) == np.sign(t2["c"]) else "se dio vuelta"
    if t2 is not None and t2["p"] < 0.01:
        return "solo reciente"
    if s1:
        return "solo antigua"
    return ""


def main():
    em = {x["byma"]: x["ticker"] for x in json.loads((D / "empresas.json").read_text())["empresas"]}
    bl = {b["id"]: b for b in json.loads((D / "bloques.json").read_text())["bloques"]}
    tk = sorted({em[e] for b in MOTORES for e in bl[b]["empresas"] if e in em})
    r = bajar(tk)
    res, lin, cuenta = {}, [], {}
    for bid, mot in MOTORES.items():
        miembros = {e: em[e] for e in bl[bid]["empresas"] if e in em and em[e] in r}
        ret = {e: r[t] for e, t in miembros.items()}
        ret["BLOQUE"] = pd.concat(list(ret.values()), axis=1).mean(axis=1, skipna=True)
        res[bid] = {}
        lin.append(f"\n##### {bl[bid]['nombre']}")
        for nom in mot:
            sid, tipo, mov, txt = FAC[nom]
            f = r[sid]
            res[bid][nom] = {}
            filas = []
            for e, y in ret.items():
                fila = {}
                for tn, (a, b) in TRAMOS.items():
                    z = prueba(tramo(y, a, b), tramo(r["SPY"], a, b), tramo(f, a, b))
                    if z:
                        z["mov"] = round((np.exp(z["c"] * (mov if tipo == "nivel" else np.log(1 + mov))) - 1) * 100, 2)
                    fila[tn] = z
                fila["etiqueta"] = etiqueta(fila["2005-2014"], fila["2015-2026"])
                res[bid][nom][e] = fila
                if e != "BLOQUE":
                    cuenta[fila["etiqueta"] or "nada"] = cuenta.get(fila["etiqueta"] or "nada", 0) + 1

                def fm(z):
                    return f"{z['mov']:+.1f}%" if z else "  -  "
                filas.append(f"  {e:7s} {fm(fila['2005-2014']):>7s} {fm(fila['2015-2026']):>7s}  {fila['etiqueta']}")
            lin.append(f"-- si {txt}")
            lin.extend(filas)
    lin.append(f"\nconteo por empresa: {cuenta}")
    (OUT / "resultado.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    (OUT / "resultado.txt").write_text("\n".join(lin))
    print("\n".join(lin))


main()
