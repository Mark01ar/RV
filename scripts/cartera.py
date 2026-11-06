import json
import math
import datetime as dt
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CARTERAS = DATA / "carteras"
SALIDA = DATA / "cartera.json"
N = 5
LIQ_OK = {"Alta", "Media", "Baja"}
COSTO = 0.005
LETRAS_MES = 0.0035


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def armar(datos):
    elig = [x for x in datos["tickers"] if x.get("liquidez") in LIQ_OK and num(x.get("mom12_1")) is not None
            and num(x.get("vs_ma200")) is not None and num(x.get("precio"))]
    df = pd.DataFrame(elig).set_index("ticker")
    df["score"] = (df["mom12_1"].rank(pct=True) + df["vs_ma200"].rank(pct=True)) / 2 * 100
    df = df.sort_values("score", ascending=False)
    ranking = []
    for k, (t, r) in enumerate(df.head(2 * N).iterrows(), start=1):
        ranking.append({"puesto": k, "ticker": t, "byma": r["byma"], "nombre": r["nombre"], "sector": r.get("sector"),
                        "score": float(r["score"]), "mom12_1": float(r["mom12_1"]), "vs_ma200": float(r["vs_ma200"]),
                        "precio": float(r["precio"]), "px_ars": num(r.get("px_ars")), "liquidez": r["liquidez"],
                        "en_cartera": bool(k <= N and r["vs_ma200"] > 0)})
    return ranking, len(df)


def ret_cartera(pos, precios_fin, n):
    rets = []
    for p in pos:
        if p["en_cartera"]:
            fin = precios_fin.get(p["ticker"])
            rets.append(fin / p["precio"] - 1 if fin else 0.0)
        else:
            rets.append(LETRAS_MES)
    return sum(rets) / n if rets else 0.0


def main():
    datos = json.loads((DATA / "datos.json").read_text())
    fund = {}
    try:
        fund = {x["ticker"]: x for x in json.loads((DATA / "fundamentales.json").read_text())["tickers"]}
    except Exception:
        pass
    hoy = dt.date.today()
    mes = hoy.strftime("%Y-%m")
    CARTERAS.mkdir(exist_ok=True)
    archivo = CARTERAS / f"{mes}.json"
    if not archivo.exists():
        ranking, n_elig = armar(datos)
        archivo.write_text(json.dumps({
            "mes": mes, "fecha": hoy.isoformat(), "n": N, "spy": datos["spy"]["precio"], "ccl": datos.get("ccl"),
            "n_elegibles": n_elig, "posiciones": [r for r in ranking if r["puesto"] <= N],
            "suplentes": [r for r in ranking if r["puesto"] > N]}, ensure_ascii=False, indent=1))
    actual = json.loads(archivo.read_text())
    precios = {x["ticker"]: num(x.get("precio")) for x in datos["tickers"]}
    info = {x["ticker"]: x for x in datos["tickers"]}

    meses = sorted(CARTERAS.glob("*.json"))
    hist = [json.loads(p.read_text()) for p in meses]
    track, acum_c, acum_s = [], 1.0, 1.0
    for i, c in enumerate(hist):
        if i + 1 < len(hist):
            sig = hist[i + 1]
            pf = {p["ticker"]: p["precio"] for p in sig["posiciones"] + sig.get("suplentes", [])}
            pf = {**precios, **pf}
            spy_fin, parcial = sig["spy"], False
        else:
            pf, spy_fin, parcial = precios, datos["spy"]["precio"], True
        nc = c.get("n", len(c["posiciones"]))
        r = ret_cartera(c["posiciones"], pf, nc)
        if i > 0:
            prev = {p["ticker"] for p in hist[i - 1]["posiciones"] if p["en_cartera"]}
            nuevo = {p["ticker"] for p in c["posiciones"] if p["en_cartera"]}
            np_ = hist[i - 1].get("n", len(hist[i - 1]["posiciones"]))
            r -= COSTO * (len(nuevo - prev) / nc + len(prev - nuevo) / np_)
        else:
            r -= COSTO * sum(1 for p in c["posiciones"] if p["en_cartera"]) / nc
        rs = spy_fin / c["spy"] - 1
        acum_c *= 1 + r
        acum_s *= 1 + rs
        track.append({"mes": c["mes"], "desde": c["fecha"], "cartera": r, "spy": rs, "dif": r - rs,
                      "acum_cartera": acum_c - 1, "acum_spy": acum_s - 1, "parcial": parcial})

    cambios = None
    if len(hist) >= 2:
        prev = {p["ticker"]: p for p in hist[-2]["posiciones"] if p["en_cartera"]}
        nuevo = {p["ticker"]: p for p in actual["posiciones"] if p["en_cartera"]}
        cambios = {"vender": [prev[t]["byma"] for t in prev if t not in nuevo],
                   "comprar": [nuevo[t]["byma"] for t in nuevo if t not in prev],
                   "mantener": [nuevo[t]["byma"] for t in nuevo if t in prev]}

    na = actual.get("n", len(actual["posiciones"]))
    pos = []
    for p in actual["posiciones"] + actual.get("suplentes", []):
        t = p["ticker"]
        f = fund.get(t, {})
        x = info.get(t, {})
        pa = precios.get(t)
        pos.append({**p, "precio_hoy": pa, "desde_armado": (pa / p["precio"] - 1) if pa else None,
                    "px_ars_hoy": num(x.get("px_ars")), "premio": num(x.get("premio")), "liquidez_hoy": x.get("liquidez"),
                    "vs_ma200_hoy": num(x.get("vs_ma200")), "total": (f.get("scores") or {}).get("total"),
                    "proximo_balance": (f.get("raw") or {}).get("proximo_balance")})
    SALIDA.write_text(json.dumps({
        "actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "mes": actual["mes"], "fecha": actual["fecha"], "n_elegibles": actual.get("n_elegibles"),
        "parametros": {"n": na, "liquidez": sorted(LIQ_OK), "costo": COSTO},
        "spy_desde": datos["spy"]["precio"] / actual["spy"] - 1,
        "posiciones": [p for p in pos if p["puesto"] <= na], "suplentes": [p for p in pos if p["puesto"] > na],
        "cambios": cambios, "track": track}, ensure_ascii=False, separators=(",", ":")))
    print(f"cartera {actual['mes']}: {[p['byma'] for p in actual['posiciones'] if p['en_cartera']]}")


if __name__ == "__main__":
    main()
