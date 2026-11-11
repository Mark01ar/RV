import io
import json
import sys
import zipfile
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import macro as MA

D = MA.ROOT / "data"
SALIDA = D / "factores.json"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FACTORES = [
    {"id": "tasa", "nombre": "Tasa a 10 años", "grupo": "Tasas y crédito", "serie": "DGS10", "fuente": "FRED", "tipo": "nivel", "sube": "sube la tasa a 10 años", "baja": "baja la tasa a 10 años"},
    {"id": "curva", "nombre": "Pendiente de la curva", "grupo": "Tasas y crédito", "serie": "T10Y2Y", "fuente": "FRED", "tipo": "nivel", "sube": "se empina la curva (10 años menos 2 años)", "baja": "se aplana la curva (10 años menos 2 años)"},
    {"id": "credito", "nombre": "Spread de crédito", "grupo": "Tasas y crédito", "serie": "BAA10Y", "fuente": "FRED", "tipo": "nivel", "sube": "sube el spread de crédito corporativo", "baja": "baja el spread de crédito corporativo"},
    {"id": "dolar", "nombre": "Dólar global", "grupo": "Monedas", "serie": "DX-Y.NYB", "fuente": "Yahoo", "tipo": "precio", "sube": "se fortalece el dólar global (DXY)", "baja": "se debilita el dólar global (DXY)"},
    {"id": "yuan", "nombre": "Yuan", "grupo": "Monedas", "serie": "CNY=X", "fuente": "Yahoo", "tipo": "precio", "sube": "se debilita el yuan", "baja": "se fortalece el yuan"},
    {"id": "real", "nombre": "Real brasileño", "grupo": "Monedas", "serie": "BRL=X", "fuente": "Yahoo", "tipo": "precio", "sube": "se debilita el real", "baja": "se fortalece el real"},
    {"id": "petroleo", "nombre": "Petróleo", "grupo": "Materias primas", "serie": "CL=F", "fuente": "Yahoo", "tipo": "precio", "sube": "sube el petróleo", "baja": "baja el petróleo"},
    {"id": "gas", "nombre": "Gas natural", "grupo": "Materias primas", "serie": "NG=F", "fuente": "Yahoo", "tipo": "precio", "sube": "sube el gas natural", "baja": "baja el gas natural"},
    {"id": "jet", "nombre": "Combustible de aviones", "grupo": "Materias primas", "serie": "DJFUELUSGULF", "fuente": "FRED", "tipo": "precio", "sube": "sube el combustible de aviones", "baja": "baja el combustible de aviones"},
    {"id": "cobre", "nombre": "Cobre", "grupo": "Materias primas", "serie": "HG=F", "fuente": "Yahoo", "tipo": "precio", "sube": "sube el cobre", "baja": "baja el cobre"},
    {"id": "oro", "nombre": "Oro", "grupo": "Materias primas", "serie": "GC=F", "fuente": "Yahoo", "tipo": "precio", "sube": "sube el oro", "baja": "baja el oro"},
    {"id": "hierro", "nombre": "Mineral de hierro", "grupo": "Materias primas", "serie": "PIORECRUSDM", "fuente": "FRED", "tipo": "precio", "sube": "sube el mineral de hierro", "baja": "baja el mineral de hierro"},
    {"id": "acero", "nombre": "Acero", "grupo": "Materias primas", "serie": "WPU101", "fuente": "FRED", "tipo": "precio", "sube": "sube el acero", "baja": "baja el acero"},
    {"id": "uranio", "nombre": "Uranio", "grupo": "Materias primas", "serie": "PURANUSDM", "fuente": "FRED", "tipo": "precio", "sube": "sube el uranio", "baja": "baja el uranio"},
    {"id": "soja", "nombre": "Soja", "grupo": "Materias primas", "serie": "ZS=F", "fuente": "Yahoo", "tipo": "precio", "sube": "sube la soja", "baja": "baja la soja"},
    {"id": "bitcoin", "nombre": "Bitcoin", "grupo": "Materias primas", "serie": "BTC-USD", "fuente": "Yahoo", "tipo": "precio", "sube": "sube bitcoin", "baja": "baja bitcoin"},
    {"id": "consumo", "nombre": "Confianza del consumidor", "grupo": "Actividad", "serie": "UMCSENT", "fuente": "FRED", "tipo": "nivel", "sube": "sube la confianza del consumidor", "baja": "baja la confianza del consumidor"},
    {"id": "semis", "nombre": "Producción de chips", "grupo": "Actividad", "serie": "IPG3344S", "fuente": "FRED", "tipo": "precio", "sube": "sube la producción de chips en EE.UU.", "baja": "baja la producción de chips en EE.UU."},
    {"id": "pedidos", "nombre": "Pedidos de electrónica", "grupo": "Actividad", "serie": "A34SNO", "fuente": "FRED", "tipo": "precio", "sube": "suben los pedidos de computadoras y electrónica", "baja": "bajan los pedidos de computadoras y electrónica"},
    {"id": "valor", "nombre": "Valor vs crecimiento", "grupo": "Estilo del mercado", "serie": "HML", "fuente": "French", "tipo": "ret", "proxy": ("IVE", "IVW"), "sube": "el mercado prefiere valor a crecimiento", "baja": "el mercado prefiere crecimiento a valor"},
    {"id": "chicas", "nombre": "Chicas vs grandes", "grupo": "Estilo del mercado", "serie": "SMB", "fuente": "French", "tipo": "ret", "proxy": ("IWM", "SPY"), "sube": "las empresas chicas le ganan a las grandes", "baja": "las empresas grandes le ganan a las chicas"},
    {"id": "calidad", "nombre": "Rentables vs no rentables", "grupo": "Estilo del mercado", "serie": "RMW", "fuente": "French", "tipo": "ret", "proxy": ("QUAL", "SPY"), "sube": "el mercado premia a las empresas rentables", "baja": "el mercado premia a las empresas poco rentables"},
    {"id": "inversion", "nombre": "Invierten poco vs mucho", "grupo": "Estilo del mercado", "serie": "CMA", "fuente": "French", "tipo": "ret", "sube": "el mercado castiga a las que invierten mucho", "baja": "el mercado premia a las que invierten mucho"},
    {"id": "momentum", "nombre": "Momentum", "grupo": "Estilo del mercado", "serie": "Mom", "fuente": "French", "tipo": "ret", "proxy": ("MTUM", "SPY"), "sube": "siguen subiendo las que venían subiendo", "baja": "se dan vuelta las que venían subiendo"},
]
F = {f["id"]: f for f in FACTORES}
TRAMOS = {"a": ("2005-01", "2014-12"), "r": ("2015-01", None)}
COD = ["CSI", "REC", "EST", "DES"]
FAMILIA = {"dolar": "Dólar y metales", "cobre": "Dólar y metales", "oro": "Dólar y metales", "hierro": "Dólar y metales", "real": "Dólar y metales", "petroleo": "Energía", "jet": "Energía", "gas": "Energía", "tasa": "Tasas", "curva": "Tasas"}
ENT = {"a": ("2005-01", "2014-12"), "r": ("2015-01", "2022-12")}
TEST = ("2023-01", None)
MIN_OTROS = 3
MIN_MESES = 36
MIN_RECIENTE = 60
ROT = 999
rng = np.random.default_rng(7)


def mensual(s):
    s = s.dropna().resample("ME").last()
    s.index = s.index.to_period("M")
    return s


def french(nombre):
    z = zipfile.ZipFile(io.BytesIO(requests.get(FRENCH + nombre, timeout=120).content))
    txt = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    i = next(n for n, l in enumerate(txt) if l.strip().startswith(","))
    cols = [c.strip() for c in txt[i].split(",")]
    filas = []
    for l in txt[i + 1:]:
        x = [c.strip() for c in l.split(",")]
        if len(x[0]) != 6 or not x[0].isdigit():
            break
        filas.append(x)
    df = pd.DataFrame(filas, columns=["f"] + cols[1:])
    df.index = pd.PeriodIndex(pd.to_datetime(df.pop("f"), format="%Y%m"), freq="M")
    return df.apply(pd.to_numeric, errors="coerce") / 100


def bajar(tk):
    prox = sorted({t for f in FACTORES for t in f.get("proxy", ())})
    yt = sorted(set(tk + ["SPY"] + prox + [f["serie"] for f in FACTORES if f["fuente"] == "Yahoo"]))
    px = yf.download(yt, start="2004-11-01", auto_adjust=True, progress=False, threads=True)["Close"]
    px = px.ffill().resample("ME").last()
    px.index = px.index.to_period("M")
    hoy = pd.Period(dt.date.today(), "M")
    r_todo = np.log(px).diff().iloc[1:]
    r = r_todo[r_todo.index < hoy]
    ff = None
    try:
        ff = pd.concat([french("F-F_Research_Data_5_Factors_2x3_CSV.zip"), french("F-F_Momentum_Factor_CSV.zip")], axis=1)
        ff.columns = [c.strip() for c in ff.columns]
    except Exception as e:
        print(f"french: {e}", file=sys.stderr)
    fs, act = {}, {}
    for f in FACTORES:
        k = f["id"]
        try:
            if f["fuente"] == "French":
                fs[k] = ff[f["serie"]][ff.index >= pd.Period("2004-12", "M")].dropna()
                if f.get("proxy"):
                    a, b = f["proxy"]
                    act[k] = (r_todo[a] - r_todo[b]).dropna()
                else:
                    act[k] = fs[k]
            elif f["fuente"] == "FRED":
                s, _ = MA.fred(f["serie"])
                s = mensual(s[s.index >= "2004-11-01"])
                ch = s.diff() if f["tipo"] == "nivel" else np.log(s).diff()
                act[k] = ch.dropna()
                fs[k] = act[k][act[k].index < hoy]
            else:
                fs[k] = r[f["serie"]].dropna()
                act[k] = r_todo[f["serie"]].dropna()
        except Exception as e:
            print(f"falla {k}: {e}", file=sys.stderr)
    return r, fs, act


def tramo(s, a, b):
    s = s[s.index >= pd.Period(a, "M")]
    return s if b is None else s[s.index <= pd.Period(b, "M")]


def prueba(y, m, fs, j=0):
    d = pd.concat([y, m] + fs, axis=1).dropna()
    n = len(d)
    if n < MIN_MESES:
        return None
    v = d.values
    yv = v[:, 0]
    otros = [v[:, 1]] + [v[:, 2 + i] for i in range(len(fs)) if i != j]
    q, _ = np.linalg.qr(np.column_stack([np.ones(n)] + otros))
    fv = v[:, 2 + j]
    yr = yv - q @ (q.T @ yv)
    k = rng.integers(12, n - 12, ROT)
    Fm = np.column_stack([fv, fv[(np.arange(n)[:, None] - k[None, :]) % n]])
    Fr = Fm - q @ (q.T @ Fm)
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
    if r is not None and r["p"] < 0.01 and r["n"] >= MIN_RECIENTE and not (sa and np.sign(a["c"]) != np.sign(r["c"])):
        return "reciente"
    if sa:
        return "antigua"
    return ""


def compacto(z, tip):
    def uno(t):
        if t is None:
            return None
        return {"ef": round(float(np.exp(t["c"] * tip) - 1) * 100, 2), "p": round(t["p"], 3), "n": t["n"]}
    return {"a": uno(z["a"]), "r": uno(z["r"]), "e": z["e"]}


def motores(nom, base, serie, spy, fs, tip, confirma, rescate):
    orden = {"clara": 0, "reciente": 1}
    cand = []
    for k, z in base[nom].items():
        e = z["e"]
        if e == "reciente" and not (confirma(k, z) or k in rescate):
            e = "indicio"
        if e not in orden:
            continue
        ef = float(np.exp(z["r"]["c"] * tip[k]) - 1) * 100
        cand.append((orden[e], -abs(ef), k, e))
    cand.sort()
    kept = []
    for c in cand:
        if not kept:
            kept.append(c)
            continue
        fl = [fs[x[2]] for x in kept] + [fs[c[2]]]
        zj = prueba(tramo(serie, "2015-01", None), tramo(spy, "2015-01", None), [tramo(f, "2015-01", None) for f in fl], len(fl) - 1)
        if zj and zj["p"] < 0.05 and np.sign(zj["c"]) == np.sign(base[nom][c[2]]["r"]["c"]):
            kept.append(c)
        if len(kept) == 3:
            break
    return [{"f": c[2], "e": c[3]} for c in kept], [c[2] for c in cand if c not in kept]


def validacion(y, spy, f):
    z = {t: prueba(tramo(y, a, b), tramo(spy, a, b), [tramo(f, a, b)]) for t, (a, b) in ENT.items()}
    e = etiqueta(z)
    zt = prueba(tramo(y, *TEST), tramo(spy, *TEST), [tramo(f, *TEST)])
    ref = z["r"] or z["a"]
    if e not in ("clara", "reciente") or zt is None or ref is None:
        return {"v": "", "e_ent": e}
    mismo = np.sign(zt["c"]) == np.sign(ref["c"])
    v = "confirmada" if mismo and zt["p"] < 0.10 else "fallo" if not mismo and zt["p"] < 0.10 else "sin_definir"
    return {"v": v, "e_ent": e, "n_test": zt["n"]}


def regimen(fs, tip):
    m = json.loads((D / "macro.json").read_text())
    reg = pd.Series({x["f"]: x["qc"] for x in m["historial_rt"]})
    reg.index = pd.PeriodIndex(reg.index, freq="M")
    out = {c: [] for c in COD}
    for k, f in fs.items():
        fut = sum(f.shift(-i) for i in range(1, 4))
        sd = tip[k] * np.sqrt(3)
        for c in COD:
            z = {}
            for t, (a, b) in TRAMOS.items():
                d = pd.concat([tramo(fut, a, b), tramo(reg, a, b)], axis=1).dropna()
                if len(d) < 36:
                    z[t] = None
                    continue
                xv, msk = d.iloc[:, 0].values.astype(float), d.iloc[:, 1].values == c
                if msk.sum() < 8 or (~msk).sum() < 8:
                    z[t] = None
                    continue
                dif = xv[msk].mean() - xv[~msk].mean()
                nul = np.array([xv[np.roll(msk, j)].mean() - xv[~np.roll(msk, j)].mean() for j in rng.integers(6, len(d) - 6, ROT)])
                z[t] = {"dif": float(dif), "p": float((1 + np.sum(np.abs(nul) >= abs(dif))) / (ROT + 1)), "meses": int(msk.sum())}
            a, r = z["a"], z["r"]
            sa = a is not None and a["p"] < 0.05
            sr = r is not None and r["p"] < 0.05
            e = ""
            if sa and sr and np.sign(a["dif"]) == np.sign(r["dif"]):
                e = "fuerte"
            elif r is not None and r["p"] < 0.01 and r["meses"] >= 12 and not (sa and np.sign(a["dif"]) != np.sign(r["dif"])):
                e = "reciente"
            if e:
                out[c].append({"f": k, "e": e, "sigma": round(r["dif"] / sd, 2), "meses": r["meses"]})
    return out


def main():
    em = {x["byma"]: x["ticker"] for x in json.loads((D / "empresas.json").read_text())["empresas"]}
    bl = json.loads((D / "bloques.json").read_text())["bloques"]
    tk = sorted({em[e] for b in bl for e in b["empresas"] if e in em})
    r, fs, act = bajar(tk)
    spy = r["SPY"]
    tip = {k: float(tramo(f, "2005-01", None).std()) for k, f in fs.items()}
    serie, bloque_de = {}, {}
    for b in bl:
        cols = []
        for e in b["empresas"]:
            t = em.get(e)
            if t in r and r[t].notna().sum() >= MIN_MESES:
                serie[e] = r[t]
                bloque_de[e] = b["id"]
                cols.append(r[t])
        if cols:
            serie["@" + b["id"]] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
    base = {}
    for nom, y in serie.items():
        base[nom] = {}
        for k, f in fs.items():
            z = dos_tramos(y, spy, [f])
            z["e"] = etiqueta(z)
            base[nom][k] = z
    rescates = {e: set(v) for b in bl for e, v in (b.get("rescates") or {}).items()}
    out_e, out_b = {}, {}
    for b in bl:
        bid = "@" + b["id"]
        if bid in base:
            mot, sol = motores(bid, base, serie[bid], spy, fs, tip, lambda k, z: True, set())
            fb = {k: compacto(z, tip[k]) for k, z in base[bid].items()}
            for k, z in base[bid].items():
                if z["e"] in ("clara", "reciente"):
                    fb[k].update(validacion(serie[bid], spy, fs[k]))
            emps_b = [e for e in b["empresas"] if e in serie]
            comp = {k: sum(1 for e in emps_b if base[e][k]["e"] in ("clara", "reciente") and base[e][k]["r"] and fb[k]["r"] and np.sign(base[e][k]["r"]["c"]) == np.sign(fb[k]["r"]["ef"])) for k in fb}
            mot = [m_ for m_ in mot if comp[m_["f"]] * 2 >= len(emps_b)]
            out_b[b["id"]] = {"f": fb, "motores": mot, "solapados": sol, "n": len(emps_b), "comparten": comp}
        for e in b["empresas"]:
            if e not in serie:
                out_e[e] = {"sin": "historia"}
                continue
            otros = [o for o in b["empresas"] if o in serie and o != e]
            excl = pd.concat([serie[o] for o in otros], axis=1).mean(axis=1, skipna=True) if len(otros) >= MIN_OTROS else None
            cache = {}

            def confirma(k, z, excl=excl, cache=cache):
                if excl is None:
                    return False
                if k not in cache:
                    q = dos_tramos(excl, spy, [fs[k]])
                    q["e"] = etiqueta(q)
                    cache[k] = q
                q = cache[k]
                return bool(q["e"] in ("clara", "reciente") and q["r"] and np.sign(q["r"]["c"]) == np.sign(z["r"]["c"]))
            mot, sol = motores(e, base, serie[e], spy, fs, tip, confirma, rescates.get(e, set()))
            fx = {k: compacto(z, tip[k]) for k, z in base[e].items()}
            for k, z in base[e].items():
                if z["e"] in ("clara", "reciente"):
                    fx[k].update(validacion(serie[e], spy, fs[k]))
            for m_ in mot:
                if m_["f"] in rescates.get(e, set()):
                    m_["rescate"] = True
            out_e[e] = {"f": fx, "motores": mot, "solapados": sol}
    facs = []
    hoy = pd.Period(dt.date.today(), "M")
    for f in FACTORES:
        k = f["id"]
        if k not in fs:
            continue
        a = act[k]
        sd_act = float(a[a.index < hoy].tail(120).std()) if f.get("proxy") else tip[k]
        ult = [{"mes": str(i), "cambio": round(float(v) * (1 if f["tipo"] == "nivel" else 100), 2), "z": round(float(v) / sd_act, 2) if sd_act else None, "en_curso": i == hoy} for i, v in a.tail(3).items()]
        tipico = tip[k] * (1 if f["tipo"] == "nivel" else 100)
        facs.append({**{x: y for x, y in f.items() if x != "proxy"}, "familia": FAMILIA.get(k), "tipico": round(tipico, 2), "unidad": "pts" if f["tipo"] == "nivel" else "%", "desde": str(fs[k].index[0]), "ultimos": ult, "proxy": "/".join(f["proxy"]) if f.get("proxy") else None})
    salida = {
        "actualizado": dt.date.today().isoformat(),
        "tramos": {k: {"desde": a, "hasta": b} for k, (a, b) in TRAMOS.items()},
        "reglas": {"min_meses": MIN_MESES, "min_reciente": MIN_RECIENTE, "rotaciones": ROT},
        "factores": facs,
        "bloques": out_b,
        "empresas": out_e,
        "regimen": regimen(fs, tip),
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))
    con = sum(1 for v in out_e.values() if v.get("motores"))
    print(f"factores.json {SALIDA.stat().st_size // 1024} KB · {con} de {len(out_e)} empresas con motor", file=sys.stderr)


if __name__ == "__main__":
    main()
