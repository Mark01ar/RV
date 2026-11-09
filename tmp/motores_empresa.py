import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / "scripts"))
import macro as MA

D = R / "data"
OUT = Path(__file__).resolve().parent
FAC = {
    "tasa": ("DGS10", "FRED", "nivel", "Tasa 10a"),
    "curva": ("T10Y2Y", "FRED", "nivel", "Curva 10a-2a"),
    "credito": ("BAA10Y", "FRED", "nivel", "Spread de crédito"),
    "dolar": ("DX-Y.NYB", "Yahoo", "precio", "Dólar global"),
    "petroleo": ("CL=F", "Yahoo", "precio", "Petróleo"),
    "gas": ("NG=F", "Yahoo", "precio", "Gas natural"),
    "cobre": ("HG=F", "Yahoo", "precio", "Cobre"),
    "oro": ("GC=F", "Yahoo", "precio", "Oro"),
    "bitcoin": ("BTC-USD", "Yahoo", "precio", "Bitcoin"),
    "yuan": ("CNY=X", "Yahoo", "precio", "Yuan débil"),
    "real": ("BRL=X", "Yahoo", "precio", "Real débil"),
    "consumo": ("UMCSENT", "FRED", "nivel", "Confianza consumidor"),
    "hierro": ("PIORECRUSDM", "FRED", "precio", "Mineral de hierro"),
    "soja": ("ZS=F", "Yahoo", "precio", "Soja"),
    "uranio": ("PURANUSDM", "FRED", "precio", "Uranio"),
    "acero": ("WPU101", "FRED", "precio", "Acero"),
    "jet": ("DJFUELUSGULF", "FRED", "precio", "Combustible aviones"),
}
TRAMOS = {"a": ("2005-01", "2014-12"), "r": ("2015-01", None)}
ROT = 999
rng = np.random.default_rng(7)


def mensual(s):
    s = s.dropna().resample("ME").last()
    s.index = s.index.to_period("M")
    return s


def bajar(tk):
    yt = tk + ["SPY"] + [v[0] for v in FAC.values() if v[1] == "Yahoo"]
    px = yf.download(yt, start="2004-11-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.ffill().resample("ME").last()
    px.index = px.index.to_period("M")
    hoy = pd.Period(pd.Timestamp.today(), "M")
    px = px[px.index < hoy]
    r = np.log(px).diff().iloc[1:]
    fs = {}
    for k, (sid, fu, tipo, _) in FAC.items():
        try:
            if fu == "FRED":
                s, _ = MA.fred(sid)
                s = mensual(s[s.index >= "2004-11-01"])
                s = s[s.index < hoy]
                fs[k] = s.diff() if tipo == "nivel" else np.log(s).diff()
            else:
                fs[k] = r[sid]
            fs[k] = fs[k].dropna()
            print(f"{k}: {len(fs[k])} meses desde {fs[k].index[0]}", file=sys.stderr)
        except Exception as e:
            print(f"falla {k}: {e}", file=sys.stderr)
    return r, fs


def tramo(s, a, b):
    s = s[s.index >= pd.Period(a, "M")]
    return s if b is None else s[s.index <= pd.Period(b, "M")]


def prueba(y, m, fs, j):
    d = pd.concat([y, m] + fs, axis=1).dropna()
    n = len(d)
    if n < 36:
        return None
    v = d.values
    yv = v[:, 0]
    otros = [v[:, 1]] + [v[:, 2 + i] for i in range(len(fs)) if i != j]
    q, _ = np.linalg.qr(np.column_stack([np.ones(n)] + otros))
    fv = v[:, 2 + j]
    yr = yv - q @ (q.T @ yv)
    k = rng.integers(12, n - 12, ROT)
    F = np.column_stack([fv, fv[(np.arange(n)[:, None] - k[None, :]) % n]])
    Fr = F - q @ (q.T @ F)
    c = (Fr.T @ yr) / (Fr ** 2).sum(axis=0)
    p = (1 + np.sum(np.abs(c[1:]) >= abs(c[0]))) / (ROT + 1)
    return {"c": float(c[0]), "p": float(p), "n": int(n)}


def dos_tramos(y, m, fs, j=0):
    return {t: prueba(tramo(y, a, b), tramo(m, a, b), [tramo(f, a, b) for f in fs], j) for t, (a, b) in TRAMOS.items()}


def etiqueta(z):
    a, r = z["a"], z["r"]
    sa = a is not None and a["p"] < 0.05
    sr = r is not None and r["p"] < 0.05
    if sa and sr:
        return "clara" if np.sign(a["c"]) == np.sign(r["c"]) else "vuelta"
    if r is not None and r["p"] < 0.01 and r["n"] >= 60:
        return "reciente"
    if sa:
        return "antigua"
    return ""


def main():
    em = {x["byma"]: x["ticker"] for x in json.loads((D / "empresas.json").read_text())["empresas"]}
    bl = json.loads((D / "bloques.json").read_text())["bloques"]
    tk = sorted({em[e] for b in bl for e in b["empresas"] if e in em})
    r, fs = bajar(tk)
    spy = r["SPY"]
    sd = {k: float(tramo(f, "2005-01", None).std()) for k, f in fs.items()}
    tip = {k: (sd[k] if FAC[k][2] == "nivel" else sd[k]) for k in fs}
    serie, bloque_de = {}, {}
    for b in bl:
        cols = []
        for e in b["empresas"]:
            t = em.get(e)
            if t in r and r[t].notna().sum() >= 36:
                serie[e] = r[t]
                bloque_de[e] = b["id"]
                cols.append(r[t])
        if cols:
            serie["@" + b["id"]] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)

    def efecto(c, k):
        return round((np.exp(c * tip[k]) - 1) * 100, 2)

    base = {}
    for nom, y in serie.items():
        base[nom] = {}
        for k, f in fs.items():
            z = dos_tramos(y, spy, [f])
            z["e"] = etiqueta(z)
            base[nom][k] = z
    salida, lin = {}, []
    orden = {"clara": 0, "reciente": 1}
    for b in bl:
        lin.append(f"\n##### {b['nombre']}")
        for e in b["empresas"]:
            if e not in serie:
                lin.append(f"  {e:6s} sin historia suficiente")
                continue
            cand = []
            for k, z in base[e].items():
                if z["e"] not in orden:
                    continue
                zb = base.get("@" + bloque_de[e], {}).get(k)
                conf = True
                if z["e"] == "reciente":
                    conf = bool(zb and zb["e"] in ("clara", "reciente") and np.sign(zb["r"]["c"]) == np.sign(z["r"]["c"]))
                ef = efecto(z["r"]["c"], k)
                cand.append((orden[z["e"]] + (0 if conf else 0.5), -abs(ef), k, z["e"] if conf else "reciente*", ef))
            cand.sort()
            kept = []
            for c in cand[:4]:
                k = c[2]
                if not kept:
                    kept.append(c)
                    continue
                fl = [fs[x[2]] for x in kept] + [fs[k]]
                zj = prueba(tramo(serie[e], "2015-01", None), tramo(spy, "2015-01", None), [tramo(f, "2015-01", None) for f in fl], len(fl) - 1)
                if zj and zj["p"] < 0.05 and np.sign(zj["c"]) == np.sign(base[e][k]["r"]["c"]):
                    kept.append(c)
                if len(kept) == 3:
                    break
            sobr = [c[2] for c in cand if c not in kept]
            salida[e] = {"bloque": bloque_de[e], "motores": [{"f": c[2], "e": c[3], "efecto": c[4], "efecto_a": efecto(base[e][c[2]]["a"]["c"], c[2]) if base[e][c[2]]["a"] else None} for c in kept], "absorbidos": sobr}
            if kept:
                txt = "  ·  ".join(f"{FAC[c[2]][3]} {c[4]:+.1f}% ({c[3]})" for c in kept)
                lin.append(f"  {e:6s} {txt}" + (f"   [repetidos: {', '.join(FAC[s][3] for s in sobr)}]" if sobr else ""))
            else:
                lin.append(f"  {e:6s} sin motor macro claro")
    sinm = [e for e, v in salida.items() if not v["motores"]]
    lin.append(f"\nEmpresas con motor: {len(salida) - len(sinm)} de {len(salida)}")
    lin.append("\nMes típico de cada factor (desvío mensual 2005-hoy):")
    for k in fs:
        u = "pts" if FAC[k][2] == "nivel" else "%"
        lin.append(f"  {FAC[k][3]:22s} {tip[k] * (1 if u == 'pts' else 100):.2f} {u}")
    (OUT / "motores.json").write_text(json.dumps({"empresas": salida, "tipico": tip}, ensure_ascii=False))
    (OUT / "motores.txt").write_text("\n".join(lin))
    print("\n".join(lin))


main()
