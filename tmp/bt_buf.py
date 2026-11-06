import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

COSTO = 0.006
N = 20
d = json.load(open("data/datos.json"))
tickers = [x["ticker"] for x in d["tickers"] if not x.get("sin_datos")]
LIQ = {x["ticker"] for x in d["tickers"] if x.get("liquidez") in ("Alta", "Media", "Baja")}

frames = {c: [] for c in ["Close", "High", "Low", "Volume"]}
for i in range(0, len(tickers), 40):
    lote = tickers[i:i + 40]
    for _ in range(3):
        try:
            df = yf.download(lote, start="2003-06-01", auto_adjust=True, progress=False, group_by="column", threads=True)
            break
        except Exception as e:
            print("reintento", e, flush=True)
            time.sleep(10)
    for c in frames:
        frames[c].append(df[c])
    time.sleep(2)
P = {c: pd.concat(v, axis=1).loc[:, lambda x: ~x.columns.duplicated()] for c, v in frames.items()}
for c in P:
    P[c].index = pd.to_datetime(P[c].index).tz_localize(None)
close = P["Close"]
spy = yf.download("SPY", start="2003-06-01", auto_adjust=True, progress=False)["Close"].squeeze()
irx = yf.download("^IRX", start="2003-06-01", auto_adjust=True, progress=False)["Close"].squeeze()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
irx.index = pd.to_datetime(irx.index).tz_localize(None)
cal = spy.dropna().index
for c in P:
    P[c] = P[c].reindex(cal)
P["Close"] = P["Close"].ffill(limit=3)
close = P["Close"]
print("precios", close.shape, float(close.iloc[-1].notna().mean()), flush=True)

dias = close.index
fechas = dias.to_series().groupby(dias.to_period("M")).max()
fechas = fechas[(fechas >= "2005-12-01")]
fechas = fechas.iloc[:-1] if fechas.index[-1] == pd.Period(pd.Timestamp.today(), "M") else fechas
cm = close.reindex(dias).ffill(limit=5)
ma200 = close.rolling(200, min_periods=150).mean()


FE = list(fechas.values)
pm = cm.loc[FE]
mom = pm.shift(1) / pm.shift(12) - 1
vsma = pm / ma200.loc[FE] - 1
fwd = pm.shift(-1) / pm - 1
spym = spy.reindex(dias).ffill().loc[FE]
spyf = spym.shift(-1) / spym - 1
cajaf = (irx.reindex(dias).ffill().loc[FE] / 100 / 12)
def rk(s):
    return s.rank(pct=True)
liq = [t for t in close.columns if t in LIQ]
VAR = [(n, b) for n in (3, 5, 10) for b in (False, True)]
nom = lambda n, b: f"N{n}" + ("_perm" if b else "")
EST = [nom(n, b) for n, b in VAR]
hold = {k: set() for k in EST}
filas = []
for f in FE[12:-1]:
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & mom.columns.isin(liq)
    if ok.sum() < 60:
        continue
    m, v, r = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok]
    orden = list(((rk(m) + rk(v)) / 2).sort_values(ascending=False).index)
    pos = {t: i + 1 for i, t in enumerate(orden)}
    fl = {"f": f, "caja": cajaf.loc[f], "spy": spyf.loc[f], "Univ": float(r.mean())}
    for n, b in VAR:
        k = nom(n, b)
        if b:
            keep = {t for t in hold[k] if t in pos and pos[t] <= 20 and v[t] > 0}
            for t in orden[:10]:
                if len(keep) >= n:
                    break
                if t not in keep and v[t] > 0:
                    keep.add(t)
            nuevo = keep
        else:
            nuevo = {t for t in orden[:n] if v[t] > 0}
        cambios = len(nuevo ^ hold[k])
        bruto = (sum(r[t] for t in nuevo) + (n - len(nuevo)) * fl["caja"]) / n
        fl[k] = bruto
        fl[k + "_giro"] = cambios / n
        hold[k] = nuevo
    filas.append(fl)
R = pd.DataFrame(filas).set_index("f")
R.index = pd.to_datetime(R.index)
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
out = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7]}
for a_, b_ in [("2007", "2018"), ("2019", "2026"), ("2007", "2026")]:
    sub = R.loc[a_:b_]
    ys = (1 + sub["spy"]).groupby(sub.index.year).prod() - 1
    o = {"SPY": round(an(sub["spy"]) * 100, 1), "Univ": round(an(sub["Univ"]) * 100, 1)}
    for k in EST:
        g = sub[k + "_giro"]
        res = {"rot_anual": round(float(g.mean() * 12 / 2) * 100)}
        for c in (0.005, 0.008):
            x = sub[k] - c * g
            mix = 0.75 * sub["spy"] + 0.25 * x
            cc, cm_ = (1 + x).cumprod(), (1 + mix).cumprod()
            y = (1 + x).groupby(x.index.year).prod() - 1
            ym = (1 + mix).groupby(mix.index.year).prod() - 1
            res[f"c{c*1000:.0f}"] = {"anual": round(an(x) * 100, 1), "vs_spy": round((an(x) - an(sub["spy"])) * 100, 1),
                                    "vs_univ": round((an(x) - an(sub["Univ"])) * 100, 1), "dd": round(float((cc / cc.cummax() - 1).min()) * 100),
                                    "anios_gana_spy": f"{int((y > ys).sum())}/{len(y)}", "peor_anio": round(float(y.min()) * 100),
                                    "mix75_25": round(an(mix) * 100, 1), "mix_vs_spy": round((an(mix) - an(sub["spy"])) * 100, 1),
                                    "mix_dd": round(float((cm_ / cm_.cummax() - 1).min()) * 100), "mix_anios_gana": f"{int((ym > ys).sum())}/{len(ym)}"}
        o[k] = res
    out[f"{a_}-{b_}"] = o
for kk, vv_ in out.items():
    print("R|" + kk + "|" + json.dumps(vv_), flush=True)
