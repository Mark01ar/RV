import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import macro as M
D = Path(__file__).resolve().parent.parent / "data"
FASES = ["CSI", "REC", "EST", "DES"]
ETF = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "IBB", "IGV", "SMH", "KBE", "XME", "ITA"]
MIN = 12


def main():
    f = {sid: M.fred(sid)[0] for sid in ["CFNAIMA3", "ICSA", "INDPRO", "PAYEMS", "CPIAUCSL", "CPILFESL", "PCEPILFE", "BAA10Y", "T5YIE"]}
    y = M.yahoo(["XLY", "XLP", "HG=F", "GC=F", "CL=F", "TIP", "IEF", "SPY"])
    inds = [x for x in M.indicadores(f, y) if x["tipo"] == "mercado"]
    idx = pd.date_range("1999-01-31", pd.Timestamp.today().to_period("M").to_timestamp("M"), freq="ME")
    Z = pd.DataFrame({x["id"]: M.zexp(x["imp"]).reindex(idx).ffill(limit=1) for x in inds})
    G = M.zexp(Z[[x["id"] for x in inds if x["grupo"] == "crecimiento"]].mean(axis=1))
    I = M.zexp(Z[[x["id"] for x in inds if x["grupo"] == "inflacion"]].mean(axis=1))
    hist = pd.DataFrame({"g": G, "i": I}).dropna()
    hist["q"] = [M.cuadrante(a, b) for a, b in zip(hist["g"], hist["i"])]
    hist["qc"] = M.confirmar(hist)
    print("regimen mercado desde", hist.index[0].date(), [x["id"] for x in inds])
    old = json.loads((D / "macro.json").read_text())
    ov = pd.Series({x["f"]: x["qc"] for x in old["historial_rt"]})
    rt = pd.Series(hist["qc"].values, index=pd.PeriodIndex(hist.index, freq="M"))
    ov.index = pd.PeriodIndex(ov.index, freq="M")
    cm = rt.reindex(ov.index).dropna()
    print("coincide con el regimen actual:", round(float((cm == ov.reindex(cm.index)).mean()), 2), "cambios/mes mercado:", round(float((rt != rt.shift()).mean()), 2), "actual:", round(float((ov != ov.shift()).mean()), 2))
    em = json.loads((D / "empresas.json").read_text())["empresas"]
    bl = [b for b in json.loads((D / "bloques.json").read_text())["bloques"] if b["id"] != "aparte"]
    tk = sorted({x["ticker"] for x in em} | set(ETF))
    px = yf.download(tk + ["SPY"], start="2004-06-01", interval="1mo", auto_adjust=True, progress=False)["Close"]
    px.index = px.index.to_period("M"); px = px[~px.index.duplicated()]
    r = px.pct_change(fill_method=None)
    ex = r[[t for t in tk if t in r.columns]].sub(r["SPY"], axis=0).iloc[:-1]
    reg = rt.shift(1).reindex(ex.index)
    ex = ex[reg.notna()]; reg = reg[reg.notna()]
    cols = list(ex.columns)
    unidades = {}
    for b in bl:
        ts = [x["ticker"] for x in em if x["byma"] in b["empresas"] and x["ticker"] in cols]
        unidades["B:" + b["id"]] = ([cols.index(t) for t in ts], 3)
    for e in ETF:
        if e in cols:
            unidades["E:" + e] = ([cols.index(e)], 1)

    def stats(X, v):
        V = ~np.isnan(X); Xz = np.where(V, X, 0.0)
        o = {}
        for fz in FASES:
            mk = (v == fz)[:, None]
            na = (V & mk).sum(0); no = (V & ~mk).sum(0)
            sa = (Xz * mk).sum(0); so = (Xz * ~mk).sum(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                d = (sa / na - so / no) * 12
            ok = (na >= MIN) & (no >= MIN)
            for u, (ix, mn) in unidades.items():
                ds = [d[i] for i in ix if ok[i]]
                if len(ds) >= mn:
                    o[(u, fz)] = float(np.median(ds))
        return o

    def tramo(a, b):
        s = (reg.index >= pd.Period(a, "M")) & (reg.index <= pd.Period(b, "M"))
        X = ex.values[s]; v = reg.values[s]; n = len(v)
        real = stats(X, v)
        sims = [stats(X, np.roll(v, k)) for k in range(6, n - 6, 2)]
        p = {k: float(np.mean([abs(s_[k]) >= abs(md) for s_ in sims if k in s_])) for k, md in real.items()}
        ep = {fz: int(((v == fz) & (np.r_[True, v[1:] != v[:-1]])).sum()) for fz in FASES}
        me = {fz: int((v == fz).sum()) for fz in FASES}
        return real, p, sims, X, v, n, ep, me

    A = tramo("2005-01", "2014-12"); B = tramo("2015-01", "2030-12"); T = tramo("2005-01", "2030-12")
    for nm, t in [("A 2005-2014", A), ("B 2015-2026", B), ("TOTAL", T)]:
        real, p = t[0], t[1]
        for g in ["B:", "E:"]:
            ks = [k for k in real if k[0].startswith(g)]
            print(f"{nm} {g} pruebas {len(ks)} pasan p<0.05: {sum(p[k] < 0.05 for k in ks)} esperado {0.05 * len(ks):.1f} | meses {t[5]} | episodios {t[6]} | meses x fase {t[7]}")
    for g in ["B:", "E:"]:
        ks = [k for k in A[0] if k in B[0] and k[0].startswith(g)]
        a = np.array([A[0][k] for k in ks]); b = np.array([B[0][k] for k in ks])
        c = float(np.corrcoef(a, b)[0, 1]); sg = float(np.mean(np.sign(a) == np.sign(b)))
        cs = []
        for s_ in B[2]:
            bb = np.array([s_.get(k, np.nan) for k in ks]); ok = ~np.isnan(bb)
            cs.append(float(np.corrcoef(a[ok], bb[ok])[0, 1]))
        print(f"REPLICA {g} pares {len(ks)} corr A-B {c:.2f} p {np.mean([x >= c for x in cs]):.3f} | mismo signo {sg:.2f}")
        pas = [k for k in ks if A[1][k] < 0.05]
        print(f"  pasan en A: {len(pas)}; mismo signo en B: {sum(np.sign(A[0][k]) == np.sign(B[0][k]) for k in pas)}; ademas p<0.10 en B: {sum(np.sign(A[0][k]) == np.sign(B[0][k]) and B[1][k] < 0.10 for k in pas)}")
    print("DETALLE unidad fase | A md p | B md p | TOTAL md p")
    for k in sorted(T[0]):
        f = lambda t: f"{t[0][k] * 100:+5.0f} {t[1][k]:.2f}" if k in t[0] else "   -    - "
        mk = " *" if (k in A[1] and A[1][k] < 0.05) or (k in B[1] and B[1][k] < 0.05) or T[1][k] < 0.05 else ""
        print(f"{k[0]:16s} {k[1]} | {f(A)} | {f(B)} | {f(T)}{mk}")


main()
