import json
import numpy as np
import pandas as pd
import yfinance as yf

SECT = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE"]
K = 3
COSTO = 0.01
m = json.load(open("data/macro.json"))
reg = pd.Series({pd.Period(h["f"], "M"): h["qc"] for h in m["historial"]})
p = yf.download(SECT + ["SPY"], start="1999-12-01", auto_adjust=True, progress=False)["Close"]
r = p.resample("ME").last().pct_change()
r.index = r.index.to_period("M")
ex = r[SECT].sub(r["SPY"], axis=0)
ult = pd.Period(pd.Timestamp.today(), "M") - 1
meses = pd.period_range("2005-01", ult, freq="M")


def correr(rezago):
    filas, prev_r, prev_e = [], set(), set()
    for t in meses:
        q = reg.get(t - 1)
        if q is None or t not in ex.index:
            continue
        hist = ex.index[ex.index < t]
        sig = reg.shift(rezago).reindex(hist)
        tab = ex.loc[sig[sig == q].index].mean().dropna()
        tab = tab[[s for s in tab.index if pd.notna(ex.loc[t, s])]]
        if len(tab) < 2 * K:
            continue
        tab = tab.sort_values(ascending=False)
        rec, evi = set(tab.index[:K]), set(tab.index[-K:])
        c_r = COSTO * len(rec - prev_r) / K if prev_r else COSTO
        c_e = COSTO * len(evi - prev_e) / K if prev_e else COSTO
        prev_r, prev_e = rec, evi
        filas.append({"mes": t, "q": q, "spy": r.loc[t, "SPY"],
                      "rec": ex.loc[t, list(rec)].mean() + r.loc[t, "SPY"] - c_r,
                      "evi": ex.loc[t, list(evi)].mean() + r.loc[t, "SPY"] - c_e,
                      "eq": r.loc[t, [s for s in SECT if pd.notna(r.loc[t, s])]].mean()})
    return pd.DataFrame(filas).set_index("mes")


def resumen(d, col):
    exm = d[col] - d["spy"]
    anios = len(d) / 12
    tot = (1 + d[col]).prod() ** (1 / anios) - 1
    spy = (1 + d["spy"]).prod() ** (1 / anios) - 1
    top2 = exm.sort_values(ascending=False).iloc[:max(1, len(d) // 20)].sum()
    porano = exm.groupby(d.index.year).sum()
    return {"anual": round(tot * 100, 1), "spy_anual": round(spy * 100, 1), "exceso_anual": round((tot - spy) * 100, 1),
            "gana_meses": round((exm > 0).mean() * 100), "t": round(exm.mean() / exm.std() * np.sqrt(len(exm)), 2),
            "anios_gana": f"{(porano > 0).sum()}/{len(porano)}",
            "exceso_sin_mejor_5pct": round((exm.sum() - top2) / anios * 100, 1),
            "por_periodo": {f"{a}-{b}": round(exm[(d.index.year >= a) & (d.index.year <= b)].mean() * 1200, 1)
                            for a, b in [(2005, 2012), (2013, 2019), (2020, 2026)]},
            "por_regimen": {q: round(exm[d["q"] == q].mean() * 1200, 1) for q in ["CSI", "REC", "EST", "DES"]}}


out = {}
for nombre, rez in [("teoria", 0), ("historial", 1)]:
    d = correr(rez)
    out[nombre] = {"n": len(d), "recomendados": resumen(d, "rec"), "evitados": resumen(d, "evi")}
out["n_cambios"] = int((reg.loc["2005-01":] != reg.loc["2005-01":].shift()).sum())
print("RESULTADO " + json.dumps(out, ensure_ascii=False))
