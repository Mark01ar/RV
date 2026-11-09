import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

D = Path(__file__).resolve().parent.parent / "data"
FASES = ["Crecimiento sin inflación", "Recalentamiento", "Estanflación", "Desinflación"]
COD = {"CSI": FASES[0], "REC": FASES[1], "EST": FASES[2], "DES": FASES[3]}


def main():
    m = json.loads((D / "macro.json").read_text())
    rt = pd.Series({x["f"]: COD[x["qc"]] for x in m["historial_rt"]})
    rt.index = pd.PeriodIndex(rt.index, freq="M")
    em = json.loads((D / "empresas.json").read_text())["empresas"]
    bl = [b for b in json.loads((D / "bloques.json").read_text())["bloques"] if b["id"] != "aparte"]
    tk = sorted({x["ticker"] for x in em})
    px = yf.download(tk + ["SPY"], period="12y", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = px.index.to_period("M")
    px = px[~px.index.duplicated()]
    r = px.pct_change(fill_method=None)
    ex = r[[t for t in tk if t in r.columns]].sub(r["SPY"], axis=0).iloc[:-1]
    ex = ex[ex.index >= pd.Period("2015-01", "M")]
    reg = rt.shift(1).reindex(ex.index)

    def dif(x, rr, fz):
        a = x[rr == fz]
        o = x[(rr != fz) & rr.notna()]
        if len(a) < 12 or len(o) < 12:
            return None
        return a, o

    out = []
    for x in em:
        t = x["ticker"]
        if t not in ex.columns:
            continue
        s = ex[t].dropna()
        if len(s) < 36:
            continue
        sd = float(s.std())
        rr = reg.reindex(s.index)
        res = {}
        for fz in FASES:
            z = dif(s, rr, fz)
            if not z:
                continue
            a, o = z
            d = (a.mean() - o.mean()) * 12
            se = sd * 12 * math.sqrt(1 / len(a) + 1 / len(o))
            res[fz] = {"dif": round(float(d), 4), "t": round(float(d / se), 2), "meses": int(len(a)), "vs_spy": round(float(a.mean() * 12), 4)}
        if res:
            out.append({"b": x["byma"], "ticker": t, "nombre": x["nombre"], "r": res})

    def stats(rg):
        o = {}
        for b in bl:
            ts = [x["ticker"] for x in em if x["byma"] in b["empresas"] and x["ticker"] in ex.columns]
            for fz in FASES:
                ds = []
                for t in ts:
                    s = ex[t].dropna()
                    z = dif(s, rg.reindex(s.index), fz)
                    if z:
                        ds.append((z[0].mean() - z[1].mean()) * 12)
                if len(ds) >= 3:
                    md = float(np.median(ds))
                    o[(b["id"], fz)] = (md, sum(np.sign(v) == np.sign(md) for v in ds), len(ds))
        return o

    real = stats(reg)
    v = reg.values
    n = len(v)
    sims = [stats(pd.Series(np.roll(v, k), index=ex.index)) for k in range(6, n - 6, 2)]
    bloques = {}
    for (bid, fz), (md, mismo, nn) in real.items():
        dist = [s[(bid, fz)][0] for s in sims if (bid, fz) in s]
        p = float(np.mean([abs(d) >= abs(md) for d in dist])) if dist else None
        bloques.setdefault(bid, {})[fz] = {"md": round(md, 4), "mismo": int(mismo), "n": int(nn), "p": None if p is None else round(p, 3)}
    (D / "escenarios.json").write_text(json.dumps({"fases": FASES, "meses": n, "desde": str(ex.index[0]), "pruebas": len(sims),
                                                   "empresas": out, "bloques": bloques}, ensure_ascii=False, separators=(",", ":")))
    print("escenarios", len(out), "bloques", len(bloques), "meses", n)


if __name__ == "__main__":
    main()
