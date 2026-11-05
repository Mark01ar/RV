import json
import os
import sys
import time
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, "scripts")
import macro as M

KEY = os.environ["FRED_API_KEY"]
REV = ["CFNAIMA3", "ICSA", "INDPRO", "PAYEMS", "CPIAUCSL", "CPILFESL", "PCEPILFE"]
NOREV = ["BAA10Y", "T5YIE"]


def vint(sid):
    rows, off = [], 0
    while True:
        for k in range(4):
            r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                             params={"series_id": sid, "api_key": KEY, "file_type": "json",
                                     "observation_start": "1995-01-01", "realtime_start": "1776-07-04",
                                     "realtime_end": "9999-12-31", "limit": 100000, "offset": off}, timeout=180)
            if r.ok:
                break
            time.sleep(5)
        r.raise_for_status()
        o = r.json()["observations"]
        rows += o
        if len(o) < 100000:
            break
        off += 100000
    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["rs"] = pd.to_datetime(df["realtime_start"])
    df["re"] = pd.to_datetime(df["realtime_end"].replace("9999-12-31", "2262-01-01"))
    df["d"] = pd.to_datetime(df["date"])
    return df[["d", "rs", "re", "value"]]


V = {sid: vint(sid) for sid in REV}
FIRST = {sid: v.sort_values("rs").groupby("d").first() for sid, v in V.items()}
cobertura = {sid: v["rs"].min().strftime("%Y-%m-%d") for sid, v in V.items()}
print("cobertura", cobertura, flush=True)


def asof(sid, T):
    v = V[sid]
    s = v[(v["rs"] <= T) & (v["re"] >= T)].groupby("d")["value"].last()
    lag = pd.Timedelta(days=12 if sid == "ICSA" else 62)
    fi = FIRST[sid]
    extra = fi[(fi["rs"] > T) & (fi.index + lag <= T)]["value"]
    extra = extra[~extra.index.isin(s.index)]
    return pd.concat([s, extra]).sort_index()


NR = {sid: M.fred(sid)[0] for sid in NOREV}
y = M.yahoo(["XLY", "XLP", "HG=F", "GC=F", "CL=F", "TIP", "IEF", "SPY", "^IRX"])


def g_i(T):
    f = {sid: asof(sid, T) for sid in REV}
    f.update({sid: s.loc[:T] for sid, s in NR.items()})
    yt = y.loc[:T]
    inds = M.indicadores(f, yt)
    idx = pd.date_range("1999-01-31", T, freq="ME")
    Z = pd.DataFrame({x["id"]: M.zexp(x["imp"]).reindex(idx).ffill(limit=4) for x in inds})

    def grupo(g, t):
        return Z[[x["id"] for x in inds if x["grupo"] == g and x["tipo"] == t]].mean(axis=1)

    out = []
    for g in ["crecimiento", "inflacion"]:
        d, m = grupo(g, "dato"), grupo(g, "mercado")
        c = (M.PESO_DATO * d + (1 - M.PESO_DATO) * m).fillna(d).fillna(m)
        out.append(float(M.zexp(c).iloc[-1]))
    return out


fin = pd.Timestamp.today().to_period("M").to_timestamp("M") - pd.offsets.MonthEnd(1)
filas = []
for T in pd.date_range("2004-12-31", fin, freq="ME"):
    g, i = g_i(T)
    filas.append({"f": T, "g": g, "i": i})
rt = pd.DataFrame(filas).set_index("f")
rt["q"] = [M.cuadrante(a, b) for a, b in zip(rt["g"], rt["i"])]
rt["qc"] = M.confirmar(rt)

mj = json.load(open("data/macro.json"))
rev = pd.DataFrame(mj["historial"])
rev.index = pd.to_datetime(rev["f"]) + pd.offsets.MonthEnd(0)
rev = rev.loc["2004-12-31":fin]

comun = rt.index.intersection(rev.index)
frena_rt = rt.loc[comun, "qc"].isin(["EST", "DES"])
frena_rev = rev.loc[comun, "qc"].isin(["EST", "DES"])
acuerdo = {"cuadrante_crudo": float((rt.loc[comun, "q"] == rev.loc[comun, "q"]).mean()),
           "cuadrante_confirmado": float((rt.loc[comun, "qc"] == rev.loc[comun, "qc"]).mean()),
           "crecimiento_frena": float((frena_rt == frena_rev).mean()),
           "corr_g": float(rt.loc[comun, "g"].corr(rev.loc[comun, "g"])),
           "corr_i": float(rt.loc[comun, "i"].corr(rev.loc[comun, "i"]))}

s_rt = M.semaforo(y, rt[["g", "i", "q", "qc"]])
s_rev = M.semaforo(y, rev[["g", "i", "q", "qc"]])


def corto(s):
    r = s["resultados"]
    return {"regla": {k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in r["regla"].items()},
            "spy": {k: (round(v, 3) if isinstance(v, float) else {a: round(b, 3) for a, b in v.items()}) for k, v in r["spy"].items()},
            "tiempo_fuera": round(r["tiempo_fuera"], 3),
            "tramos": {k: {a: round(b, 3) for a, b in v.items()} for k, v in r["tramos"].items()},
            "alarmas": [(a["desde"], a["meses"], None if a["spy_durante"] is None else round(a["spy_durante"], 3)) for a in s["alarmas"]],
            "hoy": s["estado"]}


print("RESULTADO " + json.dumps({"cobertura": cobertura, "acuerdo": acuerdo, "tiempo_real": corto(s_rt), "revisado": corto(s_rev),
                                  "ultimos_rt": [(d.strftime("%Y-%m"), q, qc) for d, q, qc in zip(rt.index[-6:], rt["q"][-6:], rt["qc"][-6:])]}))
