import json
import numpy as np
import pandas as pd
import yfinance as yf

E4 = ["SPY", "QQQ", "RSP", "IVE"]
px = yf.download(E4 + ["^IRX"], start="2003-01-01", auto_adjust=True, progress=False)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None)
pm = px[E4].resample("ME").last()
irx = px["^IRX"].resample("ME").last()
ret = pm.pct_change()
fwd = ret.shift(-1)
m = json.load(open("data/macro.json"))
rt = pd.DataFrame(m["historial_rt"])
rt.index = pd.to_datetime(rt["f"]) + pd.offsets.MonthEnd(0)
reg = rt["qc"].reindex(pm.index)
NOM = {"REC": "Recalentamiento", "EST": "Estanflacion", "DES": "Desinflacion", "CSI": "Crec sin inflacion"}

print("primer dato", {e: str(pm[e].first_valid_index())[:7] for e in E4}, flush=True)
D = pd.DataFrame({"reg": reg}).join(ret.add_suffix("_m")).join(fwd.add_suffix("_f")).loc["2005-01":"2026-10"].dropna()
for modo, suf in [("mismo_mes", "_m"), ("mes_siguiente", "_f")]:
    for (a_, b_) in [("2005", "2018"), ("2019", "2026"), ("2005", "2026")]:
        sub = D.loc[a_:b_]
        o = {}
        for r_, g in sub.groupby("reg"):
            row = {"meses": len(g)}
            for e in ["QQQ", "RSP", "IVE"]:
                ex = g[e + suf] - g["SPY" + suf]
                row[e] = f"{ex.mean()*1200:+.1f}%/a gana {int((ex>0).mean()*100)}%"
            o[NOM.get(r_, r_)] = row
        print("R|" + modo + "|" + a_ + "-" + b_ + "|" + json.dumps(o, ensure_ascii=False), flush=True)

COSTO = 0.005
filas = []
prev = {}
fechas = [f for f in pm.index if pd.Timestamp("2005-12-31") <= f <= pd.Timestamp("2026-09-30")]
for f in fechas:
    r_next = fwd.loc[f]
    if r_next[E4].isna().any():
        continue
    hist = D.loc[:f].iloc[:-1]
    rg = reg.loc[f]
    sel = {}
    h = hist[hist["reg"] == rg]
    if len(h) >= 12:
        sel["Regimen"] = max(E4, key=lambda e: h[e + "_f"].mean())
    else:
        sel["Regimen"] = "SPY"
    sel["Regimen_perfecto"] = None
    m6 = pm.loc[:f].iloc[-1] / pm.loc[:f].iloc[-7] - 1
    m12 = pm.loc[:f].iloc[-2] / pm.loc[:f].iloc[-13] - 1
    sel["Mom6"] = m6[E4].idxmax()
    sel["Mom12_1"] = m12[E4].idxmax()
    fila = {"f": f, "SPY": r_next["SPY"], "Igual4": r_next[E4].mean(), "reg": rg}
    for k, e in sel.items():
        if k == "Regimen_perfecto":
            g = D[D["reg"] == rg]
            e = max(E4, key=lambda x: g[x + "_f"].mean())
        full = r_next[e]
        tilt = 0.5 * r_next["SPY"] + 0.5 * r_next[e]
        c = COSTO * (1 if prev.get(k) not in (None, e) else 0)
        fila[k + "_100"] = full - c * 2
        fila[k + "_tilt"] = tilt - c
        fila[k + "_sel"] = e
        prev[k] = e
    filas.append(fila)
R = pd.DataFrame(filas).set_index("f")
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
for (a_, b_) in [("2006", "2018"), ("2019", "2026"), ("2006", "2026")]:
    sub = R.loc[a_:b_]
    o = {"SPY": round(an(sub["SPY"]) * 100, 1), "Igual4": round(an(sub["Igual4"]) * 100, 1)}
    ys = (1 + sub["SPY"]).groupby(sub.index.year).prod() - 1
    for k in ["Regimen", "Regimen_perfecto", "Mom6", "Mom12_1"]:
        for v in ["_100", "_tilt"]:
            x = sub[k + v]
            c = (1 + x).cumprod()
            y = (1 + x).groupby(x.index.year).prod() - 1
            o[k + v] = {"anual": round(an(x) * 100, 1), "vs_spy": round((an(x) - an(sub["SPY"])) * 100, 1),
                        "dd": round(float((c / c.cummax() - 1).min()) * 100), "anios": f"{int((y > ys).sum())}/{len(y)}"}
        o[k + "_cambios"] = int((sub[k + "_sel"] != sub[k + "_sel"].shift()).sum())
        o[k + "_mix"] = sub[k + "_sel"].value_counts(normalize=True).round(2).to_dict()
    print("S|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)
ult = R.iloc[-1]
print("HOY|", str(R.index[-1])[:7], ult["reg"], {k: ult[k + "_sel"] for k in ["Regimen", "Mom6", "Mom12_1"]}, flush=True)
