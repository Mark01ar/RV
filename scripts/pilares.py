import json
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import macro as MA

SALIDA = MA.ROOT / "data" / "pilares.json"


def semanal(s, anios=3):
    s = s.dropna()
    s = s[s.index >= s.index[-1] - pd.Timedelta(days=365 * anios)]
    w = s.resample("W-FRI").last().dropna()
    return {"f": [d.strftime("%Y-%m-%d") for d in w.index], "v": [round(float(x), 4) for x in w.values]}


def atras(s, dias):
    s = s.dropna()
    obj = s.index[-1] - pd.Timedelta(days=dias)
    a = s[s.index <= obj]
    return float(a.iloc[-1]) if len(a) else None


def pctl(s, anios=10):
    s = s.dropna()
    s = s[s.index >= s.index[-1] - pd.Timedelta(days=365 * anios)]
    return float((s < s.iloc[-1]).mean()) if len(s) > 50 else None


def ficha(id_, pilar, nombre, unidad, s, expl, lectura, dec=2):
    s = s.dropna()
    if not len(s):
        return None
    v = float(s.iloc[-1])
    return {"id": id_, "pilar": pilar, "nombre": nombre, "unidad": unidad, "dec": dec, "valor": round(v, 4),
            "fecha": s.index[-1].strftime("%Y-%m-%d"), "m1": atras(s, 30), "m3": atras(s, 91), "a1": atras(s, 365),
            "pctl10": pctl(s), "min10": float(s[s.index >= s.index[-1] - pd.Timedelta(days=3650)].min()),
            "max10": float(s[s.index >= s.index[-1] - pd.Timedelta(days=3650)].max()),
            "serie": semanal(s), "expl": expl, "lectura": lectura(s)}


def main():
    f = {}
    for sid in ["DFF", "DGS2", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "VIXCLS", "WALCL", "WTREGEN", "RRPONTSYD", "DFII10"]:
        try:
            f[sid], _ = MA.fred(sid)
        except Exception as e:
            print(f"fred {sid}: {e}", file=sys.stderr)
    y = MA.yahoo(["SPY", "RSP", "DX-Y.NYB", "^VIX"])
    out = []

    def add(x):
        if x:
            out.append(x)

    if "DFF" in f:
        add(ficha("fed", "tasas", "Tasa de la Fed", "%", f["DFF"],
                  "Tasa de referencia de la Reserva Federal (efectiva).",
                  lambda s: "Subió en el último año." if s.iloc[-1] > (atras(s, 365) or 0) + 0.2 else "Bajó en el último año." if s.iloc[-1] < (atras(s, 365) or 0) - 0.2 else "Sin cambios importantes en el último año."))
    if "DGS2" in f and "DFF" in f:
        d = (f["DGS2"] - f["DFF"].reindex(f["DGS2"].index, method="ffill")).dropna()
        def lec(s):
            v = s.iloc[-1]
            n = abs(v) / 0.25
            if v < -0.15:
                return f"El mercado espera bajas de tasa: la tasa a 2 años está {abs(v):.2f} puntos debajo de la de la Fed, como {n:.0f} recortes de 0,25."
            if v > 0.15:
                return f"El mercado espera subas de tasa: la tasa a 2 años está {v:.2f} puntos arriba de la de la Fed."
            return "El mercado no espera grandes cambios de tasa."
        add(ficha("tasa2fed", "tasas", "Tasa a 2 años menos tasa de la Fed", "pts", d,
                  "Lo que el mercado espera de la Fed para los próximos dos años. Negativo: espera bajas.", lec))
    if "DGS10" in f:
        add(ficha("t10", "tasas", "Tasa a 10 años", "%", f["DGS10"],
                  "Rendimiento del bono del Tesoro a 10 años. Es la tasa contra la que compiten las acciones.",
                  lambda s: f"{'Subió' if s.iloc[-1] > (atras(s, 91) or 0) else 'Bajó'} {abs(s.iloc[-1] - (atras(s, 91) or s.iloc[-1])):.2f} puntos en 3 meses."))
    if "DFII10" in f:
        add(ficha("real10", "tasas", "Tasa real a 10 años", "%", f["DFII10"],
                  "Tasa a 10 años descontada la inflación esperada (bonos indexados). Cuando sube, pesa sobre las acciones de crecimiento y sobre el oro.",
                  lambda s: "Alta para los últimos 10 años." if pctl(s) and pctl(s) > 0.8 else "Baja para los últimos 10 años." if pctl(s) is not None and pctl(s) < 0.2 else "En niveles normales de los últimos 10 años."))
    if "T10Y2Y" in f:
        def lec(s):
            v = s.iloc[-1]
            mn = s[s.index >= s.index[-1] - pd.Timedelta(days=730)].min()
            if v < 0:
                return "Curva invertida: la tasa corta paga más que la larga. Históricamente anticipó recesiones, con mucho retraso."
            if mn < 0:
                return "La curva se desinvirtió en los últimos dos años. Históricamente, las recesiones llegaron después de desinvertirse, no durante la inversión."
            return "Curva con pendiente positiva normal."
        add(ficha("curva", "tasas", "Pendiente de la curva (10 años − 2 años)", "pts", f["T10Y2Y"],
                  "Diferencia entre la tasa a 10 y a 2 años.", lec))
    if "BAMLH0A0HYM2" in f:
        def lec(s):
            v = s.iloc[-1]
            if v < 3.5:
                return "Muy bajo: el mercado de crédito está tranquilo y paga poco por el riesgo. Poco colchón si algo sale mal."
            if v > 5.5:
                return "Alto: el mercado de crédito muestra estrés. Suele acompañar caídas de las acciones."
            return "En niveles normales."
        add(ficha("hy", "credito", "Spread de bonos high yield", "%", f["BAMLH0A0HYM2"],
                  "Cuánto más que el Tesoro pagan las empresas de peor calificación. Sube rápido cuando el mercado se asusta.", lec))
    vix = f.get("VIXCLS")
    if vix is None or not len(vix) and "^VIX" in y:
        vix = y["^VIX"]
    if vix is not None:
        add(ficha("vix", "credito", "VIX (volatilidad esperada)", "", vix,
                  "Volatilidad que descuentan las opciones del S&P 500 para el próximo mes. Es el termómetro del miedo.",
                  lambda s: "Calma: menos de 15." if s.iloc[-1] < 15 else "Miedo: más de 25." if s.iloc[-1] > 25 else "Normal: entre 15 y 25.", dec=1))
    if "SPY" in y and "RSP" in y:
        r = (y["RSP"] / y["SPY"]).dropna()
        def lec(s):
            c = s.iloc[-1] / (atras(s, 365) or s.iloc[-1]) - 1
            if c < -0.05:
                return f"El S&P con igual peso rinde {abs(c)*100:.0f}% menos que el normal en un año: la suba la hacen pocas empresas grandes."
            if c > 0.05:
                return f"El S&P con igual peso rinde {c*100:.0f}% más que el normal en un año: la suba es amplia."
            return "La suba está repartida de forma pareja."
        add(ficha("amplitud", "amplitud", "S&P con igual peso / S&P normal", "x", r,
                  "Si sube, la mayoría de las empresas acompaña; si baja, el índice depende de pocas grandes.", lec, dec=3))
    if "DX-Y.NYB" in y:
        add(ficha("dxy", "dolar", "Dólar global (DXY)", "", y["DX-Y.NYB"],
                  "El dólar contra las principales monedas. Un dólar fuerte pesa sobre emergentes, materias primas y las ventas afuera de las empresas de EE.UU.",
                  lambda s: f"{'Se fortaleció' if s.iloc[-1] > (atras(s, 91) or 0) else 'Se debilitó'} {abs(s.iloc[-1] / (atras(s, 91) or s.iloc[-1]) - 1)*100:.1f}% en 3 meses.", dec=1))
    if all(k in f for k in ["WALCL", "WTREGEN", "RRPONTSYD"]):
        w = f["WALCL"] / 1e3
        t = f["WTREGEN"].reindex(w.index, method="ffill") / 1e3
        rr = f["RRPONTSYD"].reindex(w.index, method="ffill")
        liq = (w - t - rr).dropna()
        add(ficha("liquidez", "dolar", "Liquidez neta de la Fed", "B", liq,
                  "Balance de la Fed menos la cuenta del Tesoro y los pases pasivos, en miles de millones de dólares. Cuando crece, suele acompañar subas de activos de riesgo.",
                  lambda s: f"{'Creció' if s.iloc[-1] > (atras(s, 91) or 0) else 'Cayó'} {abs(s.iloc[-1] - (atras(s, 91) or s.iloc[-1])):.0f} mil millones en 3 meses.", dec=0))
    val = {}
    try:
        import yfinance as yf
        i = yf.Ticker("SPY").info
        pe, pef = i.get("trailingPE"), i.get("forwardPE")
        t10 = float(f["DGS10"].iloc[-1]) if "DGS10" in f else None
        val = {"pe": pe, "pe_fwd": pef, "t10": t10, "rend_ganancias": (1 / pef * 100) if pef else None,
               "prima": (1 / pef * 100 - t10) if pef and t10 is not None else None}
    except Exception as e:
        print(f"valuacion: {e}", file=sys.stderr)
    SALIDA.write_text(json.dumps({"actualizado": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                                  "indicadores": out, "valuacion": val}, ensure_ascii=False, separators=(",", ":")))
    print("pilares", len(out), val)


if __name__ == "__main__":
    main()
