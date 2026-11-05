import json
import math
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
UNIVERSO = DATA / "universo_rv.csv"
SECTORES = DATA / "sectores.json"
SEMANAL = DATA / "precios_semanales.csv"
SALIDA = DATA / "datos.json"
D912 = "https://data912.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (monitor-rv)"}
BENCH = "SPY"
LIQ_ALTA = 1_000_000
LIQ_MEDIA = 200_000
LIQ_BAJA = 30_000


def ratio_num(r):
    a, b = str(r).strip().split(":")
    return float(a) / float(b)


def leer_universo():
    u = pd.read_csv(UNIVERSO, dtype=str).fillna("")
    u = u[u["estado"] == "INCLUIDO"].copy()
    u["cedears_por_accion"] = u["ratio"].map(ratio_num)
    return u.reset_index(drop=True)


def descargar_precios(tickers):
    import yfinance as yf
    frames = []
    for i in range(0, len(tickers), 40):
        lote = tickers[i:i + 40]
        for intento in range(3):
            try:
                df = yf.download(lote, period="10y", interval="1d", auto_adjust=True,
                                 progress=False, threads=True, group_by="column")
                break
            except Exception as e:
                print(f"reintento lote {i}: {e}", file=sys.stderr)
                time.sleep(10)
        else:
            continue
        if df is None or df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            cierre = df["Close"]
        else:
            cierre = df[["Close"]].rename(columns={"Close": lote[0]})
        frames.append(cierre)
        time.sleep(2)
    if not frames:
        return pd.DataFrame()
    p = pd.concat(frames, axis=1, sort=False)
    p = p.loc[:, ~p.columns.duplicated()]
    p.index = pd.to_datetime(p.index).tz_localize(None)
    return p.sort_index()


def actualizar_sectores(tickers):
    import yfinance as yf
    cache = json.loads(SECTORES.read_text()) if SECTORES.exists() else {}
    hoy = dt.date.today()
    for t in tickers:
        c = cache.get(t, {})
        fecha = c.get("fecha")
        vieja = fecha is None or (hoy - dt.date.fromisoformat(fecha)).days > 30
        if c.get("sector") and not vieja:
            continue
        try:
            info = yf.Ticker(t).info or {}
            cache[t] = {
                "sector": info.get("sector") or c.get("sector"),
                "industria": info.get("industry") or c.get("industria"),
                "pais": info.get("country") or c.get("pais"),
                "moneda": info.get("currency") or c.get("moneda"),
                "mcap": info.get("marketCap") or c.get("mcap"),
                "fecha": hoy.isoformat(),
            }
        except Exception as e:
            print(f"sin info {t}: {e}", file=sys.stderr)
            cache.setdefault(t, {"sector": None})
        time.sleep(0.4)
    SECTORES.write_text(json.dumps(cache, ensure_ascii=False, indent=1))
    return cache


def pedir_json(url):
    for intento in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception as e:
            print(f"error {url}: {e}", file=sys.stderr)
        time.sleep(3)
    return None


def columna(df, opciones):
    for o in opciones:
        if o in df.columns:
            return o
    return None


def byma_vivo():
    data = pedir_json(f"{D912}/live/arg_cedears") or []
    return {x.get("symbol"): x for x in data if isinstance(x, dict)}


MUESTRA = {}


def usa_vivo():
    data = pedir_json(f"{D912}/live/usa_stocks") or []
    return {x.get("symbol"): x.get("c") for x in data if isinstance(x, dict) and x.get("c")}


def ultimo_habil():
    d = pd.Timestamp.now(tz="America/Argentina/Buenos_Aires").tz_localize(None).normalize()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def completar_hoy(precios, usa):
    hoy = ultimo_habil()
    precios = precios.loc[:hoy].copy()
    n = 0
    for t, px in usa.items():
        if t in precios.columns:
            serie = precios[t].dropna()
            if len(serie) and serie.index[-1] < hoy:
                precios.loc[hoy, t] = float(px)
                n += 1
    print(f"data912 usa: {len(usa)} especies, cierre {hoy.date()} completado en {n}")
    return precios.sort_index()


def byma_historico(sym):
    data = pedir_json(f"{D912}/historical/cedears/{sym}")
    if "hist" not in MUESTRA:
        MUESTRA["hist"] = str(data)[:400]
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return None
    try:
        df = pd.DataFrame(data)
    except Exception:
        return None
    cf, cc, cv = columna(df, ["date", "fecha", "d"]), columna(df, ["c", "close", "cierre"]), columna(df, ["v", "volume", "volumen"])
    if not (cf and cc and cv):
        return None
    df = df[[cf, cc, cv]].rename(columns={cf: "fecha", cc: "c", cv: "v"})
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df = df.dropna().sort_values("fecha")
    df["monto"] = pd.to_numeric(df["c"], errors="coerce") * pd.to_numeric(df["v"], errors="coerce")
    return df


def datos_byma(u):
    vivo = byma_vivo()
    print(f"byma vivo: {len(vivo)} especies; ejemplo: {next(iter(vivo.values()), None)}")
    filas = {}
    for _, r in u.iterrows():
        sym = r["byma"]
        v = vivo.get(sym, {})
        px = v.get("c") or v.get("px_ask") or v.get("px_bid")
        try:
            hist = byma_historico(sym)
        except Exception as e:
            print(f"historico {sym}: {e}", file=sys.stderr)
            hist = None
        monto20 = None
        fecha = ultimo_habil().strftime("%Y-%m-%d") if px else None
        if hist is not None and len(hist):
            monto20 = float(hist["monto"].tail(20).median())
            if not px:
                ult = pd.to_numeric(hist["c"], errors="coerce").dropna()
                if len(ult):
                    px = float(ult.iloc[-1])
                    fecha = hist["fecha"].iloc[-1].strftime("%Y-%m-%d")
        elif v.get("v") and px:
            monto20 = float(v["v"]) * float(px)
        filas[sym] = {"px_ars": float(px) if px else None, "monto_ars": monto20, "px_fecha": fecha}
        time.sleep(0.15)
    print(f"historico ejemplo: {MUESTRA.get('hist')}")
    print(f"con monto: {sum(1 for f in filas.values() if f['monto_ars'])}")
    return filas


def serie_usd(precios, t, fx):
    if t not in precios.columns:
        return None
    s = precios[t].dropna()
    if t.endswith(".KS") and fx is not None:
        s = (s / fx.reindex(s.index).ffill()).dropna()
    return s if len(s) else None


def retorno(s, n):
    if s is None or len(s) <= n:
        return None
    return float(s.iloc[-1] / s.iloc[-1 - n] - 1)


def metricas(s, bench):
    if s is None or len(s) < 30:
        return {}
    m = {
        "precio": float(s.iloc[-1]),
        "r1m": retorno(s, 21),
        "r3m": retorno(s, 63),
        "r6m": retorno(s, 126),
        "r12m": retorno(s, 252),
        "anios": round((s.index[-1] - s.index[0]).days / 365.25, 1),
    }
    if len(s) > 252:
        m["mom12_1"] = float(s.iloc[-22] / s.iloc[-253] - 1)
    if len(s) >= 200:
        m["vs_ma200"] = float(s.iloc[-1] / s.tail(200).mean() - 1)
    d = np.log(s).diff().dropna().tail(252)
    if len(d) > 60:
        m["vol"] = float(d.std() * math.sqrt(252))
        u = s.tail(252)
        m["dd12m"] = float((u / u.cummax() - 1).min())
    if bench is not None:
        sw = s.resample("W-FRI").last().pct_change()
        bw = bench.resample("W-FRI").last().pct_change()
        j = pd.concat([sw, bw], axis=1, sort=False).dropna().tail(104)
        if len(j) > 40:
            m["beta"] = float(j.iloc[:, 0].cov(j.iloc[:, 1]) / j.iloc[:, 1].var())
    return m


def clase_liquidez(usd):
    if usd is None:
        return "Sin dato"
    if usd >= LIQ_ALTA:
        return "Alta"
    if usd >= LIQ_MEDIA:
        return "Media"
    if usd >= LIQ_BAJA:
        return "Baja"
    return "Muy baja"


def limpiar(x):
    if isinstance(x, dict):
        return {k: limpiar(v) for k, v in x.items()}
    if isinstance(x, list):
        return [limpiar(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, float):
        return round(x, 6)
    return x


def construir(u, precios, sectores, byma):
    fx = precios["KRW=X"].dropna() if "KRW=X" in precios.columns else None
    bench = precios[BENCH].dropna() if BENCH in precios.columns else None
    filas = []
    for _, r in u.iterrows():
        t = r["ticker_us"]
        s = serie_usd(precios, t, fx)
        m = metricas(s, bench)
        b = byma.get(r["byma"], {})
        implicito = None
        if b.get("px_ars") and s is not None and m and not t.endswith(".KS"):
            ref = s.loc[:b["px_fecha"]] if b.get("px_fecha") else s
            if len(ref) and (not b.get("px_fecha") or ref.index[-1] == pd.Timestamp(b["px_fecha"])):
                implicito = b["px_ars"] * r["cedears_por_accion"] / float(ref.iloc[-1])
        info = sectores.get(t, {})
        filas.append({
            "byma": r["byma"], "ticker": t, "nombre": r["nombre"], "ratio": r["ratio"],
            "nota": r["motivo"], "sector": info.get("sector"), "industria": info.get("industria"),
            "pais": info.get("pais"), "mcap": info.get("mcap"),
            "px_ars": b.get("px_ars"), "px_fecha": b.get("px_fecha"), "monto_ars": b.get("monto_ars"), "ccl_impl": implicito,
            "sin_datos": not m, **m,
        })
    df = pd.DataFrame(filas)
    monto_usd_aprox = df["monto_ars"] / df["ccl_impl"].median()
    liquidas = df[(monto_usd_aprox >= LIQ_ALTA) & df["ccl_impl"].notna()]
    base = liquidas if len(liquidas) >= 10 else df[df["ccl_impl"].notna()]
    ccl = float(base["ccl_impl"].median()) if len(base) else None
    for f in filas:
        f["monto_usd"] = f["monto_ars"] / ccl if (f["monto_ars"] and ccl) else None
        f["liquidez"] = clase_liquidez(f["monto_usd"])
        f["premio"] = f["ccl_impl"] / ccl - 1 if (f["ccl_impl"] and ccl) else None
        if f["premio"] is not None and abs(f["premio"]) > 0.3:
            f["premio"] = None
            f["nota"] = "; ".join(x for x in [f["nota"], "Verificar ratio en BYMA"] if x)
    spy = metricas(bench, None) if bench is not None else {}
    return {
        "actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "ccl": ccl, "spy": spy, "n": len(filas),
        "sin_datos": [f["ticker"] for f in filas if f["sin_datos"]],
        "tickers": filas,
    }


def guardar_semanal(precios):
    w = precios.resample("W-FRI").last().dropna(how="all")
    w.index = w.index.strftime("%Y-%m-%d")
    w.round(4).to_csv(SEMANAL)


def main():
    u = leer_universo()
    tickers = sorted(set(u["ticker_us"]) | {BENCH, "KRW=X"})
    print(f"universo: {len(u)}")
    precios = descargar_precios(tickers)
    print(f"precios: {precios.shape}")
    if precios.empty:
        sys.exit("no se pudieron bajar precios")
    precios = completar_hoy(precios, usa_vivo())
    sectores = actualizar_sectores(sorted(set(u["ticker_us"])))
    byma = datos_byma(u)
    print(f"byma con precio: {sum(1 for v in byma.values() if v['px_ars'])}")
    salida = construir(u, precios, sectores, byma)
    guardar_semanal(precios)
    SALIDA.write_text(json.dumps(limpiar(salida), ensure_ascii=False, separators=(",", ":")))
    print(f"ccl implicito: {salida['ccl']}  sin datos: {salida['sin_datos']}")


if __name__ == "__main__":
    main()

