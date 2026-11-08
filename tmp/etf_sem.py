import json
import numpy as np
import pandas as pd
import yfinance as yf

E4 = ["SPY", "QQQ", "RSP", "IVE"]
px = yf.download(E4, start="2003-01-01", auto_adjust=True, progress=False)["Close"]
px.index = pd.to_datetime(px.index).tz_localize(None)
px = px.dropna()
spy = px["SPY"]
ma200 = spy.rolling(200).mean()
m = json.load(open("data/macro.json"))
rt = pd.DataFrame(m["historial_rt"])
qc = pd.Series(rt["qc"].values, index=pd.to_datetime(rt["f"]) + pd.offsets.MonthEnd(0))
viernes = pd.date_range("2005-01-07", px.index[-1], freq="14D")
cortes = sorted({px.index[px.index <= v][-1] for v in viernes if (px.index <= v).any()})
filas = []
for d in cortes:
    q = qc[qc.index <= d]
    if not len(q) or pd.isna(ma200.loc[d]):
        continue
    fr = q.iloc[-1] in ("EST", "DES")
    ba = spy.loc[d] < ma200.loc[d]
    filas.append({"d": d, "e": "rojo" if fr and ba else ("amarillo" if fr or ba else "verde")})
S = pd.DataFrame(filas).set_index("d")
P = px.loc[S.index]
r = P.pct_change().shift(-1)
S = S.join(r).dropna()
for h, nom in [(1, "2sem"), (6, "3m")]:
    fw = (P.shift(-h) / P - 1).reindex(S.index)
    for a_, b_ in [("2005", "2018"), ("2019", "2026"), ("2005", "2026")]:
        o = {}
        sub = fw.loc[a_:b_].join(S["e"]).dropna()
        for e_, g in sub.groupby("e"):
            o[e_] = {"n": len(g), "SPY": round(g["SPY"].mean() * 100, 2)}
            for t in ["QQQ", "RSP", "IVE"]:
                ex = g[t] - g["SPY"]
                o[e_][t] = f"{ex.mean()*100:+.2f} gana {int((ex>0).mean()*100)}%"
        print("C|" + nom + "|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)

REG = {"Sem_QQQ_SPY_IVE": {"verde": "QQQ", "amarillo": "SPY", "rojo": "IVE"},
       "Sem_QQQ_o_SPY": {"verde": "QQQ", "amarillo": "SPY", "rojo": "SPY"},
       "Sem_QQQ_SPY_RSP": {"verde": "QQQ", "amarillo": "SPY", "rojo": "RSP"},
       "Sem_QQQ_IVE": {"verde": "QQQ", "amarillo": "IVE", "rojo": "IVE"}}
COSTO = 0.005
out = {}
R = pd.DataFrame(index=S.index)
R["SPY"], R["QQQ"], R["Igual4"] = S["SPY"], S["QQQ"], S[E4].mean(axis=1)
for k, mp in REG.items():
    sel = S["e"].map(mp)
    x = pd.Series([S.loc[d, sel.loc[d]] for d in S.index], index=S.index)
    cambio = (sel != sel.shift()).astype(int)
    cambio.iloc[0] = 0
    R[k] = x - cambio * COSTO * 2
    out[k + "_cambios"] = int(cambio.sum())
an = lambda s: (1 + s).prod() ** (365.25 / ((s.index[-1] - s.index[0]).days + 14)) - 1
for a_, b_ in [("2005", "2018"), ("2019", "2026"), ("2005", "2026")]:
    sub = R.loc[a_:b_]
    ys = (1 + sub["SPY"]).groupby(sub.index.year).prod() - 1
    o = {}
    for k in sub.columns:
        x = sub[k]
        c = (1 + x).cumprod()
        y = (1 + x).groupby(x.index.year).prod() - 1
        o[k] = {"anual": round(an(x) * 100, 1), "vs_spy": round((an(x) - an(sub["SPY"])) * 100, 1),
                "dd": round(float((c / c.cummax() - 1).min()) * 100), "anios_vs_spy": f"{int((y > ys).sum())}/{len(y)}",
                "vol": round(x.std() * np.sqrt(26) * 100, 1)}
    print("S|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)
print("X|" + json.dumps(out) + "|hoy " + S["e"].iloc[-1], flush=True)
