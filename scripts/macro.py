import io
import json
import math
import os
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
SALIDA = ROOT / "data" / "macro.json"
KEY = os.environ.get("FRED_API_KEY", "").strip()
INICIO = "1995-01-01"
PESO_DATO = 0.6
MIN_HIST = 36
UMBRAL_CAMBIO = 0.45
SECTORES = {"XLK": "Tecnología", "XLC": "Comunicaciones", "XLY": "Consumo discrecional", "XLF": "Financieras",
            "XLI": "Industriales", "XLB": "Materiales", "XLE": "Energía", "XLV": "Salud",
            "XLP": "Consumo básico", "XLU": "Utilities", "XLRE": "Real estate"}

CUADRANTES = {
    "CSI": {"nombre": "Crecimiento sin inflación", "g": 1, "i": -1,
            "guia": "Suelen andar mejor growth y tecnología, beta alto y duración larga."},
    "REC": {"nombre": "Recalentamiento", "g": 1, "i": 1,
            "guia": "Suelen andar mejor value y cíclicas (energía, materiales, industriales). Cuidado con múltiplos altos."},
    "EST": {"nombre": "Estanflación", "g": -1, "i": 1,
            "guia": "Suelen andar mejor defensivas y activos reales. Conviene evitar beta alto."},
    "DES": {"nombre": "Desinflación / recesión", "g": -1, "i": -1,
            "guia": "Suelen andar mejor calidad y defensivas (utilities, consumo básico, salud)."},
}


def fred(sid):
    if KEY:
        try:
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": sid, "api_key": KEY, "file_type": "json",
                                     "observation_start": INICIO}, timeout=60)
            if r.ok:
                obs = r.json().get("observations", [])
                s = pd.Series({o["date"]: o["value"] for o in obs})
                s = pd.to_numeric(s, errors="coerce").dropna()
                s.index = pd.to_datetime(s.index)
                if len(s):
                    return s, "api"
        except Exception as e:
            print(f"fred api {sid}: {e}", file=sys.stderr)
    r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={INICIO}", timeout=60)
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["fecha", "v"]
    s = pd.Series(pd.to_numeric(df["v"], errors="coerce").values, index=pd.to_datetime(df["fecha"])).dropna()
    return s, "csv"


def yahoo(tickers):
    import yfinance as yf
    d = yf.download(tickers, start="1998-01-01", interval="1d", auto_adjust=True, progress=False)
    c = d["Close"] if isinstance(d.columns, pd.MultiIndex) else d[["Close"]]
    c.index = pd.to_datetime(c.index).tz_localize(None)
    return c


def mes(s):
    return s.resample("ME").last()


def infl_impulso(s):
    m = mes(s)
    ann3 = (m / m.shift(3)) ** 4 - 1
    yoy = m / m.shift(12) - 1
    return (ann3 - yoy) * 100, yoy * 100, f"{yoy.dropna().iloc[-1] * 100:.1f}% anual · {ann3.dropna().iloc[-1] * 100:.1f}% anualizado 3m"


def indicadores(f, y):
    out = []

    def add(id_, grupo, tipo, nombre, expl, imp, valor, fecha, signo=1):
        out.append({"id": id_, "grupo": grupo, "tipo": tipo, "nombre": nombre, "expl": expl,
                    "imp": imp.dropna() * signo, "valor": valor, "fecha": fecha})

    s = f["CFNAIMA3"]
    m = mes(s)
    add("CFNAIMA3", "crecimiento", "dato", "Actividad (CFNAI, Fed de Chicago)",
        "Resume 85 indicadores de actividad. Positivo = la economía crece por encima de su tendencia. Mide si mejoró o empeoró en 3 meses.",
        m.diff(3), f"{m.iloc[-1]:+.2f}", s.index[-1])

    s = f["ICSA"]
    m = mes(s.rolling(4).mean())
    add("ICSA", "crecimiento", "dato", "Pedidos de subsidio por desempleo",
        "Promedio de 4 semanas. Si suben, el mercado laboral se enfría (por eso cuenta al revés).",
        np.log(m).diff(3) * 100, f"{m.iloc[-1] / 1000:,.0f} mil", s.index[-1], -1)

    s = f["INDPRO"]
    m = mes(s)
    yoy = (m / m.shift(12) - 1) * 100
    add("INDPRO", "crecimiento", "dato", "Producción industrial",
        "Variación anual. Mide si se aceleró o frenó en los últimos 3 meses.",
        yoy.diff(3), f"{yoy.iloc[-1]:+.1f}% anual", s.index[-1])

    s = f["PAYEMS"]
    m = mes(s).diff()
    add("PAYEMS", "crecimiento", "dato", "Creación de empleo (nóminas)",
        "Empleos creados por mes: promedio de 3 meses contra promedio de 12 meses.",
        m.rolling(3).mean() - m.rolling(12).mean(), f"{m.rolling(3).mean().iloc[-1]:,.0f} mil/mes", s.index[-1])

    for sid, nombre in [("CPIAUCSL", "Inflación general (CPI)"), ("CPILFESL", "Inflación núcleo (CPI core)"),
                        ("PCEPILFE", "Inflación núcleo (PCE core)")]:
        s = f[sid]
        imp, _, val = infl_impulso(s)
        add(sid, "inflacion", "dato", nombre,
            "Inflación anualizada de los últimos 3 meses contra la de 12 meses. Positivo = se acelera.",
            imp, val, s.index[-1])

    s = f["BAA10Y"]
    m = mes(s)
    add("BAA10Y", "crecimiento", "mercado", "Spread de bonos corporativos",
        "Cuánto más pagan las empresas (BAA) que el Tesoro a 10 años. Si se agranda, el mercado ve menos crecimiento (cuenta al revés).",
        m.diff(3), f"{m.iloc[-1]:.2f}%", s.index[-1], -1)

    fy = y.dropna(how="all").index[-1]
    if {"XLY", "XLP"} <= set(y.columns):
        r = mes(y["XLY"] / y["XLP"]).dropna()
        add("XLYXLP", "crecimiento", "mercado", "Cíclicas vs defensivas",
            "Consumo discrecional contra consumo básico. Si las cíclicas ganan, el mercado descuenta más crecimiento.",
            np.log(r).diff(3) * 100, f"{(r.iloc[-1] / r.iloc[-4] - 1) * 100:+.1f}% en 3m", fy)
    if {"HG=F", "GC=F"} <= set(y.columns):
        r = mes(y["HG=F"] / y["GC=F"]).dropna()
        add("COBREORO", "crecimiento", "mercado", "Cobre vs oro",
            "El cobre depende de la actividad y el oro es refugio. Si el cobre le gana al oro, el mercado ve más crecimiento.",
            np.log(r).diff(3) * 100, f"{(r.iloc[-1] / r.iloc[-4] - 1) * 100:+.1f}% en 3m", fy)

    s = f["T5YIE"]
    m = mes(s)
    add("T5YIE", "inflacion", "mercado", "Inflación esperada a 5 años (breakeven)",
        "La inflación que descuentan los bonos del Tesoro. Mide cuánto cambió en 3 meses.",
        m.diff(3), f"{m.iloc[-1]:.2f}%", s.index[-1])
    if "CL=F" in y.columns:
        r = mes(y["CL=F"]).dropna()
        r = r[r > 0]
        add("PETROLEO", "inflacion", "mercado", "Petróleo (WTI)",
            "Suba del petróleo en 3 meses. Empuja la inflación de corto plazo.",
            np.log(r).diff(3) * 100, f"USD {r.iloc[-1]:.1f}", fy)
    if {"TIP", "IEF"} <= set(y.columns):
        r = mes(y["TIP"] / y["IEF"]).dropna()
        add("TIPIEF", "inflacion", "mercado", "Bonos indexados vs nominales",
            "Si los bonos ajustados por inflación le ganan a los nominales, el mercado espera más inflación.",
            np.log(r).diff(3) * 100, f"{(r.iloc[-1] / r.iloc[-4] - 1) * 100:+.1f}% en 3m", fy)
    return out


def zexp(s):
    m = s.expanding(MIN_HIST).mean()
    sd = s.expanding(MIN_HIST).std()
    return ((s - m) / sd).clip(-3, 3)


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cuadrante(g, i):
    for k, v in CUADRANTES.items():
        if (g >= 0) == (v["g"] > 0) and (i >= 0) == (v["i"] > 0):
            return k


def probs_de(g, i):
    pg, pi = phi(g), phi(i)
    return {"CSI": pg * (1 - pi), "REC": pg * pi, "EST": (1 - pg) * pi, "DES": (1 - pg) * (1 - pi)}


def confirmar(hist):
    conf, prev = [], None
    for k, (_, r) in enumerate(hist.iterrows()):
        cand = r.q
        if prev is None:
            prev = cand
        elif cand != prev:
            dos_meses = k > 0 and hist["q"].iloc[k - 1] == cand
            fuerte = probs_de(r.g, r.i)[cand] > UMBRAL_CAMBIO
            if dos_meses or fuerte:
                prev = cand
        conf.append(prev)
    return conf


def sectores_por_regimen(y, hist, rezago=1):
    cols = [t for t in list(SECTORES) + ["SPY"] if t in y.columns]
    m = y[cols].resample("ME").last().pct_change()
    reg = hist["qc"].copy()
    reg.index = reg.index.to_period("M")
    m.index = m.index.to_period("M")
    sig = reg.shift(rezago).reindex(m.index)
    out = {}
    for q in CUADRANTES:
        meses = sig[sig == q].index
        filas = []
        for t, nombre in SECTORES.items():
            if t not in m.columns:
                continue
            ex = (m.loc[meses, t] - m.loc[meses, "SPY"]).dropna()
            if len(ex) < 6:
                continue
            filas.append({"etf": t, "nombre": nombre, "exceso": float(ex.mean() * 12),
                          "gana": float((ex > 0).mean()), "n": int(len(ex))})
        out[q] = sorted(filas, key=lambda x: -x["exceso"])
    return out


def main():
    ids = ["CFNAIMA3", "ICSA", "INDPRO", "PAYEMS", "CPIAUCSL", "CPILFESL", "PCEPILFE", "BAA10Y", "T5YIE"]
    f, fuentes = {}, set()
    for sid in ids:
        s, src = fred(sid)
        f[sid] = s
        fuentes.add(src)
        print(f"{sid}: {len(s)} obs hasta {s.index[-1].date()} ({src})")
    y = yahoo(sorted(set(["XLY", "XLP", "HG=F", "GC=F", "CL=F", "TIP", "IEF", "SPY"]) | set(SECTORES)))
    inds = indicadores(f, y)

    hoy = pd.Timestamp.today().to_period("M").to_timestamp("M")
    idx = pd.date_range("1999-01-31", hoy, freq="ME")
    Z = {}
    for ind in inds:
        z = zexp(ind["imp"]).reindex(idx).ffill(limit=4)
        Z[ind["id"]] = z
        ind["z"] = None if pd.isna(z.iloc[-1]) else float(z.iloc[-1])
        ind["impulso"] = float(ind["imp"].iloc[-1])
    Z = pd.DataFrame(Z)

    def grupo(g, t):
        cols = [x["id"] for x in inds if x["grupo"] == g and x["tipo"] == t]
        return Z[cols].mean(axis=1)

    comp = {}
    for g in ["crecimiento", "inflacion"]:
        d, m = grupo(g, "dato"), grupo(g, "mercado")
        c = (PESO_DATO * d + (1 - PESO_DATO) * m).fillna(d).fillna(m)
        comp[g] = {"dato": zexp(d), "mercado": zexp(m), "total": zexp(c)}

    G, I = comp["crecimiento"]["total"], comp["inflacion"]["total"]
    hist = pd.DataFrame({"g": G, "i": I}).dropna()
    hist = hist[hist.index >= "2000-01-01"]
    hist["q"] = [cuadrante(a, b) for a, b in zip(hist["g"], hist["i"])]
    hist["qc"] = confirmar(hist)

    g0, i0 = float(hist["g"].iloc[-1]), float(hist["i"].iloc[-1])
    probs = probs_de(g0, i0)
    crudo = max(probs, key=probs.get)
    actual = hist["qc"].iloc[-1]
    q = hist["qc"].tolist()
    dur = 1
    while dur < len(q) and q[-1 - dur] == q[-1]:
        dur += 1

    def ult(s):
        s = s.dropna()
        return None if s.empty else float(s.iloc[-1])

    lectura = {}
    for g in ["crecimiento", "inflacion"]:
        d, m = ult(comp[g]["dato"]), ult(comp[g]["mercado"])
        lectura[g] = {"dato": d, "mercado": m, "total": ult(comp[g]["total"]),
                      "diverge": d is not None and m is not None and (d >= 0) != (m >= 0)}

    salida = {
        "actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "fuente_fred": "/".join(sorted(fuentes)),
        "peso_dato": PESO_DATO,
        "g": g0, "i": i0, "probs": probs, "actual": actual, "crudo": crudo, "duracion": dur,
        "umbral": UMBRAL_CAMBIO, "sectores": sectores_por_regimen(y, hist, 1),
        "sectores_mismo": sectores_por_regimen(y, hist, 0),
        "cambios": int((hist["qc"] != hist["qc"].shift()).sum() - 1),
        "cambios_sin_regla": int((hist["q"] != hist["q"].shift()).sum() - 1),
        "cuadrantes": CUADRANTES, "lectura": lectura,
        "indicadores": [{k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else v)
                         for k, v in x.items() if k != "imp"} for x in inds],
        "historial": [{"f": d.strftime("%Y-%m"), "g": round(float(r.g), 3), "i": round(float(r.i), 3), "q": r.q, "qc": r.qc}
                      for d, r in hist.iterrows()],
    }
    SALIDA.write_text(json.dumps(salida, ensure_ascii=False, separators=(",", ":")))
    print(f"cambios de regimen: {salida['cambios']} con regla, {salida['cambios_sin_regla']} sin regla")
    print(f"regimen: {CUADRANTES[actual]['nombre']} (crudo {crudo}) g={g0:.2f} i={i0:.2f} probs={ {k: round(v, 2) for k, v in probs.items()} }")


if __name__ == "__main__":
    main()
