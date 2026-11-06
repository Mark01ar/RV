import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

COSTO_IDA_VUELTA = 0.012
d = json.load(open("data/datos.json"))
tickers = [x["ticker"] for x in d["tickers"] if not x.get("sin_datos") and not x["ticker"].endswith(".KS")]
LIQ = {x["ticker"] for x in d["tickers"] if x.get("liquidez") in ("Alta", "Media", "Baja")}

partes = []
for i in range(0, len(tickers), 40):
    lote = tickers[i:i + 40]
    for _ in range(3):
        try:
            df = yf.download(lote, start="2005-01-01", auto_adjust=True, progress=False, group_by="column", threads=True)["Close"]
            break
        except Exception as e:
            print("reintento", e, flush=True)
            time.sleep(10)
    partes.append(df)
    time.sleep(2)
close = pd.concat(partes, axis=1)
close = close.loc[:, ~close.columns.duplicated()]
close.index = pd.to_datetime(close.index).tz_localize(None)
spy = yf.download("SPY", start="2005-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
cal = spy.dropna().index
close = close.reindex(cal).ffill(limit=3)
S = spy.reindex(cal).values
C = close.values
cols = list(close.columns)
print("precios", close.shape, flush=True)

ev = []
for k, t in enumerate(cols):
    try:
        e = yf.Ticker(t).get_earnings_dates(limit=100)
    except Exception:
        e = None
    if e is None or not len(e):
        continue
    e = e.reset_index()
    fc = e.columns[0]
    e[fc] = pd.to_datetime(e[fc], utc=True).dt.tz_convert("America/New_York")
    e = e[e["Reported EPS"].notna()].sort_values(fc)
    for _, r in e.iterrows():
        x = r[fc]
        dia = x.tz_localize(None).normalize()
        pos = cal.searchsorted(dia)
        if x.hour >= 16 and pos < len(cal) and cal[pos] == dia:
            pos += 1
        if pos < 7 or pos + 62 >= len(cal):
            continue
        sp = r.get("Surprise(%)")
        ev.append({"t": t, "c": k, "d0": pos, "fecha": cal[pos], "sorpresa": float(sp) if pd.notna(sp) else np.nan})
    time.sleep(0.25)
E = pd.DataFrame(ev).sort_values("d0").reset_index(drop=True)
print("eventos", len(E), E["fecha"].min(), E["fecha"].max(), flush=True)


def ret(c, a, b):
    pa, pb = C[a, c], C[b, c]
    return pb / pa - 1 if (pa > 0 and pb > 0 and np.isfinite(pa) and np.isfinite(pb)) else np.nan


def sret(a, b):
    return S[b] / S[a] - 1


E["ear"] = [ret(r.c, r.d0 - 1, r.d0 + 2) - sret(r.d0 - 1, r.d0 + 2) for r in E.itertuples()]
E = E[E["ear"].notna()].reset_index(drop=True)

umbral = []
hist = []
j = 0
for r in E.itertuples():
    while j < len(E) and E.at[j, "d0"] + 2 < r.d0:
        hist.append(E.at[j, "ear"])
        j += 1
    umbral.append(np.percentile(hist, 80) if len(hist) >= 300 else np.nan)
E["umbral"] = umbral

beats = []
for t, g in E.groupby("t"):
    s = (g["sorpresa"] > 0).astype(float).where(g["sorpresa"].notna())
    prev = s.shift(1).rolling(8, min_periods=8).sum()
    beats.append(prev)
E["beats8"] = pd.concat(beats)


def evento(c, a, b):
    x = ret(c, a, b)
    return x - sret(a, b) if np.isfinite(x) else np.nan


def calendario(trades):
    n = len(cal)
    sr = np.zeros(n)
    cnt = np.zeros(n)
    for c, a, b in trades:
        rr = C[a + 1:b + 1, c] / C[a:b, c] - 1
        ok = np.isfinite(rr)
        sr[a + 1:b + 1][ok] += rr[ok]
        cnt[a + 1:b + 1][ok] += 1
    sretd = np.r_[0, S[1:] / S[:-1] - 1]
    inv = cnt > 0
    ex = np.where(inv, sr / np.where(inv, cnt, 1) - sretd, 0.0)
    return pd.Series(ex, index=cal), inv, cnt


def resumen(nombre, sel, a_off, b_off, solo=None):
    out = {}
    for per, (pa, pb) in {"2007-2018": ("2007", "2018"), "2019-2026": ("2019", "2026"), "total": ("2007", "2026")}.items():
        g = sel[(sel["fecha"] >= pd.Timestamp(pa + "-01-01")) & (sel["fecha"] <= pd.Timestamp(pb + "-12-31"))]
        if solo is not None:
            g = g[g["t"].isin(solo)]
        if len(g) < 30:
            continue
        trades = [(r.c, r.d0 + a_off, r.d0 + b_off) for r in g.itertuples()]
        exs = np.array([evento(c, a, b) for c, a, b in trades])
        exs = exs[np.isfinite(exs)]
        neto = exs - COSTO_IDA_VUELTA
        serie, inv, cnt = calendario(trades)
        serie = serie[(serie.index >= pd.Timestamp(pa + "-01-01")) & (serie.index <= pd.Timestamp(pb + "-12-31"))]
        anios = len(serie) / 252
        bruto_anual = float(serie.sum() / anios)
        n_prom = float(cnt[inv].mean()) if inv.any() else 1
        drag = len(trades) / anios * COSTO_IDA_VUELTA / max(n_prom, 1)
        mens = serie.resample("ME").sum()
        out[per] = {"eventos": int(len(exs)), "exceso_medio": round(float(exs.mean()) * 100, 2),
                    "neto_medio": round(float(neto.mean()) * 100, 2), "mediana": round(float(np.median(exs)) * 100, 2),
                    "acierto": round(float((exs > 0).mean()) * 100), "acierto_neto": round(float((neto > 0).mean()) * 100),
                    "cartera_bruto_anual": round(bruto_anual * 100, 1), "cartera_neto_anual": round((bruto_anual - drag) * 100, 1),
                    "posiciones_prom": round(n_prom, 1), "dias_invertido": round(float(inv[(cal >= pd.Timestamp(pa + "-01-01")) & (cal <= pd.Timestamp(pb + "-12-31"))].mean()) * 100),
                    "t_mensual": round(float(mens.mean() / mens.std() * np.sqrt(len(mens))), 2) if mens.std() > 0 else None}
    print(f"R|{nombre}|" + json.dumps(out), flush=True)


top = E[E["umbral"].notna() & (E["ear"] > E["umbral"])]
for H in [20, 40, 60]:
    resumen(f"A_{H}", top, 2, 2 + H)
resumen("A_40_liquidas", top, 2, 42, solo=LIQ)
resumen("A2_40", top[top["sorpresa"] > 0], 2, 42)
E["umbral_bajo"] = [np.nan] * len(E)
abajo = E[E["umbral"].notna() & (E["ear"] < -E["umbral"])]
resumen("A_peores_40", abajo, 2, 42)
resumen("B", E[E["beats8"] >= 6], -6, 1)
resumen("C", E, -6, 1)
