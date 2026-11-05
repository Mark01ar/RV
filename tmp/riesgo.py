import json
import numpy as np
import pandas as pd
import yfinance as yf

COSTO = 0.005
m = json.load(open("data/macro.json"))
reg = pd.Series({pd.Period(h["f"], "M"): h["qc"] for h in m["historial"]})
p = yf.download(["SPY", "XLV", "XLP", "XLU", "^IRX"], start="1998-12-01", auto_adjust=True, progress=False)["Close"]
pm = p.resample("ME").last()
pm.index = pm.index.to_period("M")
r = pm[["SPY", "XLV", "XLP", "XLU"]].pct_change()
caja = (pm["^IRX"] / 100 / 12).shift(1)
ma10 = pm["SPY"].rolling(10).mean()
ult = pd.Period(pd.Timestamp.today(), "M") - 1
meses = pd.period_range("2005-01", ult, freq="M")
DEF = ["XLV", "XLP", "XLU"]


def peso(regla, t):
    q = reg.get(t - 1)
    frena = q in ("EST", "DES")
    bajo = pm.loc[t - 1, "SPY"] < ma10.loc[t - 1]
    if regla == "SPY":
        return {"SPY": 1.0}
    if regla == "R1":
        return {s: 1 / 3 for s in DEF} if frena else {"SPY": 1.0}
    if regla == "R2":
        return {"SPY": 0.5, "CAJA": 0.5} if q == "EST" else {"SPY": 1.0}
    if regla == "R3":
        return {"CAJA": 1.0} if bajo else {"SPY": 1.0}
    if regla == "R4":
        return {"CAJA": 1.0} if (frena and bajo) else {"SPY": 1.0}


def correr(regla):
    filas, prev = [], None
    for t in meses:
        w = peso(regla, t)
        ret = sum(v * (caja.loc[t] if k == "CAJA" else r.loc[t, k]) for k, v in w.items())
        if prev is not None:
            keys = set(w) | set(prev)
            ret -= COSTO * sum(abs(w.get(k, 0) - prev.get(k, 0)) for k in keys)
        prev = w
        filas.append({"mes": t, "ret": ret, "caja": caja.loc[t], "invertido": 1 - w.get("CAJA", 0),
                      "cambio": 0})
    return pd.DataFrame(filas).set_index("mes")


def stats(d):
    s = d["ret"]
    c = (1 + s).cumprod()
    anual = c.iloc[-1] ** (12 / len(s)) - 1
    vol = s.std() * np.sqrt(12)
    sharpe = (s - d["caja"]).mean() / s.std() * np.sqrt(12)
    dd = (c / c.cummax() - 1).min()
    peor12 = (1 + s).rolling(12).apply(np.prod, raw=True).min() - 1
    out = {"anual": round(anual * 100, 1), "vol": round(vol * 100, 1), "sharpe": round(sharpe, 2),
           "caida_max": round(dd * 100), "peor_12m": round(peor12 * 100), "invertido": round(d["invertido"].mean() * 100)}
    for a, b in [(2005, 2018), (2019, 2026)]:
        x = s[(s.index.year >= a) & (s.index.year <= b)]
        cx = (1 + x).cumprod()
        out[f"anual_{a}_{b}"] = round((cx.iloc[-1] ** (12 / len(x)) - 1) * 100, 1)
        out[f"caida_{a}_{b}"] = round((cx / cx.cummax() - 1).min() * 100)
    eps = {"2008": ("2007-10", "2009-03"), "2020": ("2020-02", "2020-03"), "2022": ("2022-01", "2022-09")}
    out["episodios"] = {k: round(((1 + s.loc[a:b]).prod() - 1) * 100) for k, (a, b) in eps.items()}
    return out


print("RESULTADO " + json.dumps({k: stats(correr(k)) for k in ["SPY", "R1", "R2", "R3", "R4"]}))
