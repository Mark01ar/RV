import json
import pandas as pd
import yfinance as yf
from pathlib import Path

SECT = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE"]
N = 18
K = 3
m = json.load(open("data/macro.json"))
reg = pd.Series({pd.Period(h["f"], "M"): h["qc"] for h in m["historial"]})
p = yf.download(SECT + ["SPY"], start="1999-12-01", auto_adjust=True, progress=False)["Close"]
r = p.resample("ME").last().pct_change()
r.index = r.index.to_period("M")
ex = r[SECT].sub(r["SPY"], axis=0)
ult = pd.Period(pd.Timestamp.today(), "M") - 1
prueba = pd.period_range(ult - N + 1, ult, freq="M")
antes = ex.index[ex.index < prueba[0]]


def tabla(rezago):
    sig = reg.shift(rezago).reindex(antes)
    return {q: ex.loc[sig[sig == q].index].mean().dropna().sort_values(ascending=False) for q in ["CSI", "REC", "EST", "DES"]}


out = {"meses": [], "tablas": {}}
for nombre, rez in [("teoria", 0), ("historial", 1)]:
    t = tabla(rez)
    out["tablas"][nombre] = {q: [(s, round(float(v) * 12, 4)) for s, v in t[q].items()] for q in t}
    for mes in prueba:
        q = reg.get(mes - 1)
        if q is None or mes not in ex.index:
            continue
        rec, evi = list(t[q].index[:K]), list(t[q].index[-K:])
        out["meses"].append({"modelo": nombre, "mes": str(mes), "regimen": q, "rec": rec, "evi": evi,
                             "ex_rec": float(ex.loc[mes, rec].mean()), "ex_evi": float(ex.loc[mes, evi].mean()),
                             "spy": float(r.loc[mes, "SPY"]),
                             "detalle": {s: round(float(ex.loc[mes, s]), 4) for s in SECT if pd.notna(ex.loc[mes, s])}})
Path("data/analisis_18m.json").write_text(json.dumps(out, indent=1))
