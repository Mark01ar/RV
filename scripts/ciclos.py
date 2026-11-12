import re
import sys
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import macro as MA

D = MA.ROOT / "data"
SALIDA = D / "ciclos.json"
UA = {"User-Agent": "MR33 RV research marcosracimo@gmail.com"}
MESES = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
HYPER = {"MSFT": 789019, "GOOGL": 1652044, "AMZN": 1018724, "META": 1326801}
FABS = {"INTC": 50863, "MU": 723125, "TXN": 97476}
CAPEX = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
COSTO = ["CostOfGoodsAndServicesSold", "CostOfRevenue"]
INVENT = ["InventoryNet"]
EMPRESAS = {
    "SNDK": {
        "indicadores": ["hyper"],
        "pronos_de": "memorias",
        "solo": "nand",
        "per": "3y",
        "nota": "Datos del ciclo de la memoria NAND, que es lo que vende Sandisk. No son motores medidos: la acción suele moverse antes que estos datos. Sandisk cotiza sola desde febrero de 2025, así que hay poca historia propia.",
    }
}
BLOQUES = {
    "memorias": {
        "indicadores": ["nanya", "mu_inv", "hyper", "mu_capex"],
        "empresas": ["MU", "SKHY"],
        "ciclos": [["2018", "2017-01", "2019-12"], ["2021", "2020-06", "2022-12"]],
        "base": "nanya",
        "pronos": "trendforce.csv",
        "nota": "Datos del ciclo de la memoria DRAM. No son motores medidos: la acción suele moverse antes que estos datos, que confirman en qué parte del ciclo estamos.",
        "dir": {"mu_capex": -1},
    },
    "chips": {
        "indicadores": ["tsmc", "chips_inv", "hyper"],
        "empresas": ["NVDA", "AMD", "AVGO", "MRVL", "ALAB", "QCOM"],
        "ciclos": [["2018", "2018-01", "2019-06"], ["2022", "2021-06", "2023-12"]],
        "base": "tsmc",
        "nota": "Datos del ciclo de los chips. No son motores medidos: la acción suele moverse antes que estos datos, que confirman en qué parte del ciclo estamos.",
    },
    "fabricacion": {
        "indicadores": ["tsmc", "fab_capex", "hyper"],
        "ciclos": [["2018", "2018-01", "2019-06"], ["2022", "2021-06", "2023-12"]],
        "base": "tsmc",
        "nota": "Datos del ciclo de la fabricación de chips. TSMC vende chips; ASML, Applied Materials y Lam Research venden las máquinas para fabricarlos, así que dependen de cuánto invierten los fabricantes. No son motores medidos: la acción suele moverse antes que estos datos.",
    },
}


MESES3 = [m[:3] for m in MESES]


def leer_web(url, y):
    h = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    txt = re.sub(r"<[^>]+>", " ", h)
    txt = re.sub(r"\s+", " ", txt).lower()
    out = {}
    for i in range(12):
        g = re.search(rf"\b(?:{MESES[i]}|{MESES3[i]})\.?\s+([\d,]{{5,}})\s", txt)
        if g:
            out[pd.Period(f"{y}-{i + 1:02d}", "M")] = int(g.group(1).replace(",", ""))
    return out


TWSE = {}


def twse(codigo, escala):
    if not TWSE:
        try:
            for x in requests.get("https://openapi.twse.com.tw/v1/opendata/t187ap05_L", headers={"User-Agent": "Mozilla/5.0"}, timeout=60).json():
                TWSE[str(x.get("公司代號"))] = x
        except Exception as e:
            print(f"twse: {e}", file=sys.stderr)
            TWSE["-"] = None
    x = TWSE.get(codigo)
    if not x:
        return {}
    ym = str(x["資料年月"])
    per = pd.Period(f"{int(ym[:-2]) + 1911}-{ym[-2:]}", "M")
    v = lambda k: float(x[k]) / escala
    return {per: v("營業收入-當月營收"), per - 1: v("營業收入-上月營收"), per - 12: v("營業收入-去年當月營收")}


def mensual(archivo, url, nombre, corto, fuente, que, codigo=None, escala=1):
    f = D / archivo
    s = pd.read_csv(f, index_col=0)["ventas"]
    s.index = pd.PeriodIndex(s.index, freq="M")
    hoy = dt.date.today()
    nuevos = twse(codigo, escala) if codigo else {}
    if not nuevos and url:
        for y in sorted({hoy.year - 1, hoy.year}):
            try:
                nuevos.update(leer_web(url.format(y=y), y))
            except Exception as e:
                print(f"{archivo} {y}: {e}", file=sys.stderr)
    print(f"{archivo} web: {len(nuevos)} meses, último {max(nuevos) if nuevos else None}")
    if nuevos:
        n = pd.Series(nuevos)
        ok = all(abs(n[k] / s[k] - 1) < 0.01 for k in n.index if k in s.index)
        if ok:
            s = pd.concat([s, n[~n.index.isin(s.index)]]).sort_index()
            pd.DataFrame({"mes": s.index.astype(str), "ventas": s.values.round(0).astype(int)}).to_csv(f, index=False)
        else:
            print(f"{archivo}: la web no coincide con la historia, no se actualiza", file=sys.stderr)
    t3 = s.rolling(3).sum().dropna()
    a = s.iloc[-3:].sum() / s.iloc[-15:-12].sum() - 1 if len(s) >= 15 else None
    p = s.iloc[-3:].sum() / s.iloc[-6:-3].sum() - 1 if len(s) >= 6 else None
    yy = (t3 / t3.shift(12) - 1).dropna()
    prev = yy[yy.index < pd.Period("2023-01", "M")]
    return {
        "max_previo": [float(prev.max()), str(prev.idxmax())],
        "base3": [[str(k), float(v)] for k, v in t3.items()],
        "nombre": nombre, "corto": corto, "frec": "mensual", "fuente": fuente, "que": que,
        "ultimo": str(s.index[-1]),
        "anual": a, "trim": p,
        "serie": [[str(k), float(v)] for k, v in t3.items() if k >= pd.Period("2015-01", "M")],
    }


def nanya():
    return mensual("nanya.csv", "https://www.nanya.com/en/IR/36/Monthly%20Revenue?Year={y}",
                   "Ventas de memoria DRAM (Nanya)", "Ventas DRAM",
                   "Nanya Technology (Taiwán), fabricante de DRAM pura. Publica sus ventas cerca del día 10 del mes siguiente.",
                   "Suma de los últimos 3 meses, para sacar el ruido de cada mes. Es lo más cercano gratis al precio de la memoria: si suben, la DRAM se vende más o más cara.",
                   codigo="2408", escala=1)


def tsmc():
    return mensual("tsmc.csv", None,
                   "Ventas de TSMC", "Ventas TSMC",
                   "TSMC, que fabrica los chips de casi todas las empresas de diseño (NVDA, AMD, AVGO, Apple, QCOM). Publica sus ventas cerca del día 10 del mes siguiente.",
                   "Suma de los últimos 3 meses. Es el mejor termómetro mensual de la demanda de chips: casi todo lo que diseñan NVDA, AMD o Broadcom pasa por sus fábricas.",
                   codigo="2330", escala=1000)


HECHOS = {}


def hechos(cik):
    if cik not in HECHOS:
        HECHOS[cik] = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=UA, timeout=60).json().get("facts", {}).get("us-gaap", {})
    return HECHOS[cik]


def instantes(cik, conceptos):
    g = hechos(cik)
    d = {}
    for pri, n in enumerate(conceptos):
        for e in g.get(n, {}).get("units", {}).get("USD", []):
            if "start" in e or e.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                continue
            per = pd.Period(pd.Timestamp(e["end"]) - pd.Timedelta(days=10), "Q")
            k = (per, pri)
            if k not in d or e["filed"] > d[k][1]:
                d[k] = (float(e["val"]), e["filed"])
    out = {}
    for (per, pri), (v, _) in sorted(d.items(), key=lambda z: -z[0][1]):
        out[per] = v
    return pd.Series(out).sort_index()


def trimestres(cik, conceptos=CAPEX):
    g = hechos(cik)
    d = {}
    for pri, n in enumerate(conceptos):
        for e in g.get(n, {}).get("units", {}).get("USD", []):
            if "start" not in e or e.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                continue
            k = (e["start"], e["end"])
            if k not in d or (pri, e["filed"]) < (d[k][2], "") or (pri == d[k][2] and e["filed"] > d[k][1]):
                d[k] = (float(e["val"]), e["filed"], pri)
    q = {}
    por = {}
    for (a, b), (v, _, _) in d.items():
        por.setdefault(a, []).append((b, v))
    for a, xs in por.items():
        xs.sort()
        prev_end, prev_v = a, 0.0
        for b, v in xs:
            dias = (pd.Timestamp(b) - pd.Timestamp(prev_end)).days
            if 80 <= dias <= 100:
                per = pd.Period(pd.Timestamp(b) - pd.Timedelta(days=10), "Q")
                q.setdefault(per, v - prev_v)
            prev_end, prev_v = b, v
    return pd.Series(q).sort_index()


def capex(ciks):
    out = {}
    for t, c in ciks.items():
        try:
            out[t] = trimestres(c)
        except Exception as e:
            print(f"capex {t}: {e}", file=sys.stderr)
    df = pd.DataFrame(out).sort_index()
    df = df[df.index >= pd.Period("2013Q1", "Q")]
    return df


def capex_ind(df, nombre, corto, que, fuente):
    df = df.dropna()
    tot = df.sum(axis=1)
    ttm = tot.rolling(4).sum().dropna()
    a = ttm.iloc[-1] / ttm.iloc[-5] - 1 if len(ttm) >= 5 else None
    p = tot.iloc[-1] / tot.iloc[-5] - 1 if len(tot) >= 5 else None
    serie = [[str(k.asfreq("M", "end")), float(v) / 1e9] for k, v in ttm.items() if k >= pd.Period("2015Q1", "Q")]
    yy = (ttm / ttm.shift(4) - 1).dropna()
    prev = yy[yy.index < pd.Period("2023Q1", "Q")]
    ult = df.iloc[-1]
    return {
        "nombre": nombre, "corto": corto, "frec": "trimestral", "fuente": fuente, "que": que,
        "ultimo": str(df.index[-1]), "anual": a, "trim": p, "unidad": "miles de millones de USD (últimos 12 meses)",
        "detalle": {k: round(float(v) / 1e9, 1) for k, v in ult.items()},
        "valor": round(float(ttm.iloc[-1]) / 1e9, 1),
        "max_previo": [float(prev.max()), str(prev.idxmax())] if len(prev) else None,
        "serie": serie,
    }


def inventario(ciks, nombre, corto, fuente, que):
    inv = sum(instantes(c, INVENT) for c in ciks.values())
    cg = sum(trimestres(c, COSTO) for c in ciks.values())
    dias = (inv / cg * 91).dropna()
    dias = dias[dias.index >= pd.Period("2014Q1", "Q")]
    h = dias[dias.index >= pd.Period("2015Q1", "Q")]
    ult = float(dias.iloc[-1])
    return {
        "nombre": nombre, "corto": corto, "frec": "trimestral", "fuente": fuente, "que": que,
        "ultimo": str(dias.index[-1]),
        "anual": float(dias.iloc[-1] / dias.iloc[-5] - 1) if len(dias) >= 5 else None,
        "trim": float(dias.iloc[-1] / dias.iloc[-2] - 1) if len(dias) >= 2 else None,
        "lectura": f"Hoy: <b>{ult:.0f} días</b> de stock. Promedio desde 2015: {h.mean():.0f} días. Lo más alto fue {h.max():.0f} días ({str(h.idxmax())[-1]}.º trim. {str(h.idxmax())[:4]}); lo más bajo, {h.min():.0f} días ({str(h.idxmin())[-1]}.º trim. {str(h.idxmin())[:4]}). En el último trimestre pasó de {float(dias.iloc[-2]):.0f} a {ult:.0f} días.",
        "serie": [[str(k.asfreq("M", "end")), round(float(v), 1)] for k, v in h.items()],
    }


def precios(tks):
    px = yf.download(sorted(set(tks)), start="2014-10-01", auto_adjust=True, progress=False, threads=False)["Close"]
    px = px.ffill().resample("ME").last()
    px.index = px.index.to_period("M")
    return {t: [[str(k), round(float(v), 4)] for k, v in px[t].dropna().items() if k >= pd.Period("2015-01", "M")] for t in px.columns}


def main():
    ind = {}
    try:
        ind["nanya"] = nanya()
    except Exception as e:
        print(f"nanya: {e}", file=sys.stderr)
    try:
        h = capex(HYPER)
        print(h.tail(6).div(1e9).round(1).to_string(), file=sys.stderr)
        ind["hyper"] = capex_ind(h, "Inversión de Microsoft, Google, Amazon y Meta", "Inversión big tech",
                                 "Suma de lo que invierten las cuatro en centros de datos, servidores y chips, en los últimos 12 meses. Es la demanda: de acá sale la compra de memoria HBM y de chips para IA.",
                                 "Balances trimestrales presentados a la SEC (compras de propiedad y equipo).")
    except Exception as e:
        print(f"hyper: {e}", file=sys.stderr)
    try:
        m = capex({"MU": 723125})
        print(m.tail(6).div(1e9).round(2).to_string(), file=sys.stderr)
        ind["mu_capex"] = capex_ind(m, "Inversión de Micron", "Inversión Micron",
                                    "Lo que invierte Micron en fábricas, en los últimos 12 meses. Es la oferta futura: cuando los fabricantes invierten fuerte a la vez, uno o dos años después suele sobrar memoria y los precios caen. Es una aproximación: Samsung y SK Hynix no presentan balances a la SEC, así que no están.",
                                    "Balances trimestrales de Micron presentados a la SEC.")
    except Exception as e:
        print(f"mu_capex: {e}", file=sys.stderr)
    try:
        ind["mu_inv"] = inventario({"MU": 723125}, "Días de inventario de Micron", "Inventario Micron",
                                   "Balances trimestrales de Micron presentados a la SEC (inventario sobre costo de ventas).",
                                   "Cuántos días de ventas tiene Micron guardados en stock. Es la señal clásica del ciclo: cuando la memoria se vende sola el stock baja; cuando empieza a sobrar, se acumula y los precios caen después.")
        print("mu_inv", ind["mu_inv"]["serie"][-8:], file=sys.stderr)
    except Exception as e:
        print(f"mu_inv: {e}", file=sys.stderr)
    try:
        ind["tsmc"] = tsmc()
    except Exception as e:
        print(f"tsmc: {e}", file=sys.stderr)
    try:
        ind["chips_inv"] = inventario({"NVDA": 1045810, "AMD": 2488}, "Días de inventario de NVDA y AMD", "Inventario NVDA+AMD",
                                      "Balances trimestrales de Nvidia y AMD presentados a la SEC (inventario sobre costo de ventas, las dos juntas).",
                                      "Cuántos días de ventas tienen Nvidia y AMD guardados en stock. Si sube rápido, los chips se están vendiendo menos de lo que fabrican; en 2022 fue la primera señal del bajón de los chips.")
        print("chips_inv", ind["chips_inv"]["serie"][-6:], file=sys.stderr)
    except Exception as e:
        print(f"chips_inv: {e}", file=sys.stderr)
    try:
        fb = capex(FABS)
        print(fb.tail(6).div(1e9).round(2).to_string(), file=sys.stderr)
        ind["fab_capex"] = capex_ind(fb, "Inversión de fabricantes de chips (Intel, Micron y Texas Instruments)", "Inversión fábricas",
                                     "Lo que invierten en fábricas Intel, Micron y Texas Instruments, en los últimos 12 meses. Es lo que compran a ASML, Applied Materials y Lam Research: cuando sube, venden más equipos. TSMC y Samsung no presentan balances trimestrales a la SEC, así que no están.",
                                     "Balances trimestrales presentados a la SEC (compras de propiedad y equipo).")
    except Exception as e:
        print(f"fab_capex: {e}", file=sys.stderr)
    em = {x["byma"]: x["ticker"] for x in json.loads((D / "empresas.json").read_text())["empresas"]}
    bl = {b["id"]: b for b in json.loads((D / "bloques.json").read_text())["bloques"]}
    tks = {b: [em[e] for e in bl[b]["empresas"] if e in em] for b in BLOQUES if b in bl}
    try:
        px = precios([t for v in tks.values() for t in v] + ["SPY"])
    except Exception as e:
        print(f"precios: {e}", file=sys.stderr)
        px = {}
    for b, v in BLOQUES.items():
        if v.get("base") in ind:
            t3 = pd.Series({pd.Period(k, "M"): x for k, x in ind[v["base"]]["base3"]})
            v["picos"] = [[n, str(t3[a:z].idxmax())] for n, a, z in v["ciclos"]]
    for k in ind:
        ind[k].pop("base3", None)
    pronos = {}
    for b, v in BLOQUES.items():
        f = D / v.get("pronos", "-")
        if f.exists():
            t = pd.read_csv(f)
            pronos[b] = {
                "fuente": "Pronósticos de TrendForce publicados al empezar cada trimestre (cuánto esperan que cambie el precio de contrato contra el trimestre anterior). Se cargan a mano cuatro veces por año.",
                "serie": [[r.trimestre, r.publicado, *(None if pd.isna(x) else float(x) for x in (r.dram_min, r.dram_max, r.nand_min, r.nand_max)), "" if pd.isna(r.nota) else r.nota] for r in t.itertuples()],
            }
    out = {
        "actualizado": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "indicadores": ind,
        "bloques": {b: {**v, "indicadores": [i for i in v["indicadores"] if i in ind]} for b, v in BLOQUES.items()},
        "precios": {t: px[t] for t in px},
        "pronos": pronos,
        "empresas": {e: {**v, "indicadores": [i for i in v["indicadores"] if i in ind]} for e, v in EMPRESAS.items()},
    }
    SALIDA.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    for k, v in ind.items():
        print(k, v["ultimo"], v.get("anual"), v.get("trim"), v.get("valor"), v.get("detalle"))


if __name__ == "__main__":
    main()
