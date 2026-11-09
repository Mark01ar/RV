import json
import numpy as np
import pandas as pd
import yfinance as yf

m = json.load(open("data/macro.json"))
rt = pd.Series({x["f"]: x["qc"] for x in m["historial_rt"]})
rt.index = pd.PeriodIndex(rt.index, freq="M")
em = {x["byma"]: x for x in json.load(open("data/empresas.json"))["empresas"]}
B = [b for b in json.load(open("data/bloques.json"))["bloques"] if b["id"] != "aparte"]
tk = sorted({em[t]["ticker"] for b in B for t in b["empresas"]})
px = yf.download(tk + ["SPY"], period="12y", interval="1mo", auto_adjust=True, progress=False)["Close"]
px.index = px.index.to_period("M")
px = px[~px.index.duplicated()]
r = px.pct_change(fill_method=None)
ex = r[tk].sub(r["SPY"], axis=0).iloc[:-1]
ex = ex[ex.index >= pd.Period("2015-01", "M")]
Q = ["CSI", "REC", "EST", "DES"]


def stats(reg):
    out = {}
    for b in B:
        ts = [em[t]["ticker"] for t in b["empresas"] if em[t]["ticker"] in ex.columns]
        for q in Q:
            difs = []
            for t in ts:
                x = ex[t].dropna()
                rr = reg.reindex(x.index)
                a = x[rr == q]
                o = x[(rr != q) & rr.notna()]
                if len(a) < 12 or len(o) < 12:
                    continue
                difs.append((a.mean() - o.mean()) * 12)
            if len(difs) < 3:
                continue
            md = float(np.median(difs))
            same = sum(np.sign(d) == np.sign(md) for d in difs) / len(difs)
            out[(b["id"], q)] = (md, same, len(difs))
    return out


reg_real = rt.shift(1).reindex(ex.index)
real = stats(reg_real)
vals = reg_real.values
n = len(vals)
sims = []
for k in range(6, n - 6, 2):
    sims.append(stats(pd.Series(np.roll(vals, k), index=ex.index)))
res = []
for key, (md, same, nn) in real.items():
    dist = [s[key][0] for s in sims if key in s]
    p = float(np.mean([abs(d) >= abs(md) for d in dist])) if dist else None
    res.append({"bloque": key[0], "fase": key[1], "md": round(md, 4), "same": round(same, 2), "n": nn, "p": p, "nsim": len(dist)})
cons = lambda S: sum(1 for (md, same, nn) in S.values() if abs(md) >= 0.05 and same >= 0.75)
print("RESUMEN", json.dumps({"meses": n, "consistentes_real": cons(real), "consistentes_azar": [cons(s) for s in sims]}))
print("DETALLE", json.dumps(res))
