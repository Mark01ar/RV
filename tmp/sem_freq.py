import json
import numpy as np
import pandas as pd
import yfinance as yf

m = json.load(open("data/macro.json"))
rt = pd.DataFrame(m["historial_rt"])
rt["fin"] = pd.to_datetime(rt["f"]) + pd.offsets.MonthEnd(0)
frena = rt.set_index("fin")["qc"].isin(["EST", "DES"])
spy = yf.download("SPY", start="2003-01-01", auto_adjust=True, progress=False)["Close"].squeeze()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
ma200 = spy.rolling(200).mean()
pm = spy.resample("ME").last()
ma10 = pm.rolling(10).mean()


def frena_en(d):
    s = frena[frena.index <= d]
    return bool(s.iloc[-1]) if len(s) else None


def fwd(d, n):
    i = spy.index.get_indexer([d], method="pad")[0]
    return float(spy.iloc[i + n] / spy.iloc[i] - 1) if i + n < len(spy) else np.nan


def evaluar(fechas, bajo_fn):
    rows = []
    for d in fechas:
        f = frena_en(d)
        b = bajo_fn(d)
        if f is None or b is None:
            continue
        rows.append({"d": d, "rojo": f and b, "amar": f != b, "r1m": fwd(d, 21), "r3m": fwd(d, 63)})
    return pd.DataFrame(rows).set_index("d")


dias = spy.loc["2005-01-01":].index
semanas = dias.to_series().groupby(dias.to_period("W-FRI")).max().values
quinc = semanas[::2]
meses = dias.to_series().groupby(dias.to_period("M")).max().values
bajo_m10 = lambda d: (lambda i: None if pd.isna(ma10.loc[:d].iloc[-1]) else bool(pm.loc[:d].iloc[-1] < ma10.loc[:d].iloc[-1]))(0)
bajo_200 = lambda d: None if pd.isna(ma200.loc[d]) else bool(spy.loc[d] < ma200.loc[d])
V = {"Mensual (actual)": evaluar(meses, bajo_m10), "Mensual MA200d": evaluar(meses, bajo_200),
     "Quincenal": evaluar(quinc, bajo_200), "Semanal": evaluar(semanas, bajo_200)}

pico = spy.cummax()
dd = spy / pico - 1
eps = [("2007-10-09", "2009-03-09"), ("2011-04-29", "2011-10-03"), ("2015-05-21", "2016-02-11"), ("2018-09-20", "2018-12-24"),
       ("2020-02-19", "2020-03-23"), ("2022-01-03", "2022-10-12"), ("2025-02-19", "2025-04-08")]
picos = spy.loc["2025-06-01":"2026-06-30"]
p_ = picos.loc[:"2026-04-30"].idxmax() if len(picos) else None
if p_ is not None:
    t_ = spy.loc[p_:"2026-06-30"].idxmin()
    eps.append((str(p_)[:10], str(t_)[:10]))
for k, R in V.items():
    R = R.dropna(subset=["r1m"])
    rj, ot = R[R["rojo"]], R[~R["rojo"]]
    vr, nv = R[~R["rojo"] & ~R["amar"]], R[R["amar"]]
    o = {"obs": len(R), "pct_rojo": round(len(rj) / len(R) * 100),
         "r1m_rojo": round(rj["r1m"].mean() * 100, 2), "r1m_amar": round(nv["r1m"].mean() * 100, 2), "r1m_verde": round(vr["r1m"].mean() * 100, 2),
         "r3m_rojo": round(rj["r3m"].mean() * 100, 2), "r3m_amar": round(nv["r3m"].mean() * 100, 2), "r3m_verde": round(vr["r3m"].mean() * 100, 2),
         "neg1m_rojo": round(float((rj["r1m"] < 0).mean()) * 100), "neg1m_verde": round(float((vr["r1m"] < 0).mean()) * 100),
         "neg3m_rojo": round(float((rj["r3m"] < 0).mean()) * 100), "neg3m_verde": round(float((vr["r3m"] < 0).mean()) * 100),
         "vol1m_rojo": round(rj["r1m"].std() * 100, 1), "vol1m_verde": round(vr["r1m"].std() * 100, 1)}
    s = R["rojo"].astype(int)
    o["encendidas"] = int(((s.diff() == 1)).sum())
    epi = {}
    for a, b in eps:
        w = R.loc[a:b]
        primero = w.index[w["rojo"]].min() if w["rojo"].any() else None
        caida = float(spy.loc[b] / spy.loc[a] - 1)
        if primero is not None:
            hecho = float(spy.loc[:primero].iloc[-1] / spy.loc[a] - 1)
            epi[a[:7]] = f"caida {caida*100:.0f}% | rojo {str(primero)[:10]} con {hecho*100:.0f}% ya caido"
        else:
            epi[a[:7]] = f"caida {caida*100:.0f}% | nunca rojo"
    o["episodios"] = epi
    print("R|" + k + "|" + json.dumps(o, ensure_ascii=False), flush=True)
