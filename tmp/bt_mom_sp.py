import io
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

url = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
H = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text))
H["date"] = pd.to_datetime(H["date"])
H = H.sort_values("date")
H["set"] = H["tickers"].str.split(",").apply(lambda l: {x.strip().replace(".", "-") for x in l})
TODOS = sorted(set().union(*H[H["date"] >= "1999-06-01"]["set"]))
print("tickers historicos", len(TODOS), flush=True)
fr = []
for i in range(0, len(TODOS), 80):
    lote = TODOS[i:i + 80]
    for _ in range(3):
        try:
            df = yf.download(lote, start="1999-01-01", auto_adjust=True, progress=False, group_by="column", threads=True)
            break
        except Exception:
            time.sleep(10)
    fr.append(df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": lote[0]}))
    time.sleep(2)
close = pd.concat(fr, axis=1)
close = close.loc[:, ~close.columns.duplicated()]
ref = yf.download(["SPY", "RSP", "^IRX"], start="1999-01-01", auto_adjust=True, progress=False)["Close"]
for x in (close, ref):
    x.index = pd.to_datetime(x.index).tz_localize(None)
cal = ref["SPY"].dropna().index
close = close.reindex(cal).ffill(limit=5)
fechas = cal.to_series().groupby(cal.to_period("M")).max()
FE = pd.DatetimeIndex(fechas[(fechas >= "2000-01-01")].values[:-1])
pm = close.loc[FE]
ma200 = close.rolling(200, min_periods=150).mean().loc[FE]
mom = pm.shift(1) / pm.shift(12) - 1
vsma = pm / ma200 - 1
fwd = pm.shift(-1) / pm - 1
caja = ref["^IRX"].reindex(FE).ffill() / 100 / 12
spyf = ref["SPY"].reindex(FE).pct_change().shift(-1)
cob = []


def miembros(f):
    r = H[H["date"] <= f]
    return r["set"].iloc[-1] if len(r) else set()


def rk(s):
    return s.rank(pct=True)


VAR = []
for n in (5, 10, 20, 50):
    for ma in (True, False):
        for perm in (False, True):
            VAR.append((n, ma, perm, "comb"))
VAR += [(10, True, False, "mom"), (20, True, False, "mom")]
nom = lambda v: f"N{v[0]}{'_MA' if v[1] else ''}{'_perm' if v[2] else ''}{'_solo12-1' if v[3]=='mom' else ''}"
COSTOS = (0.005, 0.01)
hold = {nom(v): set() for v in VAR}
filas = []
for f in FE[13:-1]:
    mem = miembros(f)
    con = [t for t in mem if t in close.columns]
    ok = [t for t in con if pd.notna(mom.loc[f, t]) and pd.notna(vsma.loc[f, t]) and pd.notna(fwd.loc[f, t])]
    cob.append((len(mem), len(ok)))
    if len(ok) < 200:
        continue
    m_, v_, r_ = mom.loc[f, ok], vsma.loc[f, ok], fwd.loc[f, ok]
    sc = {"comb": (rk(m_) + rk(v_)) / 2, "mom": m_}
    fl = {"f": f, "caja": float(caja.loc[f]), "spy": float(spyf.loc[f]), "Univ": float(r_.mean()), "n": len(ok)}
    for v in VAR:
        n, ma, perm, tipo = v
        k = nom(v)
        orden = list(sc[tipo].sort_values(ascending=False).index)
        pos = {t: i + 1 for i, t in enumerate(orden)}
        if perm:
            keep = {t for t in hold[k] if t in pos and pos[t] <= 2 * n and (not ma or v_[t] > 0)}
            for t in orden[:n]:
                if len(keep) >= n:
                    break
                if t not in keep and (not ma or v_[t] > 0):
                    keep.add(t)
            nuevo = keep
        else:
            nuevo = {t for t in orden[:n] if (not ma or v_[t] > 0)}
        bruto = (sum(r_[t] for t in nuevo) + (n - len(nuevo)) * fl["caja"]) / n
        fl[k] = bruto
        fl[k + "_g"] = len(nuevo ^ hold[k]) / n
        hold[k] = nuevo
    filas.append(fl)
cb = np.array(cob)
print("R|cobertura|" + json.dumps({"pct_total": round(float(cb[:, 1].sum() / cb[:, 0].sum()) * 100),
                                   "pct_2001": round(float(cb[:24, 1].mean() / cb[:24, 0].mean()) * 100),
                                   "pct_2010": round(float(cb[108:120, 1].mean() / cb[108:120, 0].mean()) * 100)}), flush=True)
R = pd.DataFrame(filas).set_index("f")
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
print("R|meses|" + json.dumps({"n": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7], "empresas_med": int(R["n"].median())}), flush=True)
for a_, b_ in [("2001", "2009"), ("2010", "2018"), ("2019", "2026"), ("2001", "2026")]:
    sub = R.loc[a_:b_]
    ys = (1 + sub["spy"]).groupby(sub.index.year).prod() - 1
    o = {"SPY": round(an(sub["spy"]) * 100, 1), "Univ": round(an(sub["Univ"]) * 100, 1)}
    for v in VAR:
        k = nom(v)
        res = {"rot": round(float(sub[k + "_g"].mean() * 6) * 100)}
        for c in COSTOS:
            x = sub[k] - c * sub[k + "_g"]
            cc = (1 + x).cumprod()
            y = (1 + x).groupby(x.index.year).prod() - 1
            mix = 0.75 * sub["spy"] + 0.25 * x
            cm = (1 + mix).cumprod()
            res[f"c{int(c*1000)}"] = [round(an(x) * 100, 1), round((an(x) - an(sub["spy"])) * 100, 1), round(float((cc / cc.cummax() - 1).min()) * 100),
                                      f"{int((y > ys).sum())}/{len(y)}", round((an(mix) - an(sub["spy"])) * 100, 1), round(float((cm / cm.cummax() - 1).min()) * 100)]
        o[k] = res
    print("R|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)
for k in ["N5_MA", "N10_MA", "N20_MA", "N10_MA_perm"]:
    x = R[k] - 0.005 * R[k + "_g"]
    y = (1 + x).groupby(x.index.year).prod() - 1
    ys = (1 + R["spy"]).groupby(R.index.year).prod() - 1
    print("A|" + k + "|" + json.dumps({str(a): round(float((y[a] - ys[a]) * 100)) for a in y.index}), flush=True)
