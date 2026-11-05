import json
import numpy as np
import pandas as pd
import yfinance as yf

SECT = ["XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE", "XLV", "XLP", "XLU", "XLRE"]
K = 3
COSTO = 0.01
m = json.load(open("data/macro.json"))
reg = pd.Series({pd.Period(h["f"], "M"): h["qc"] for h in m["historial"]})
p = yf.download(SECT + ["SPY"], start="1998-12-01", auto_adjust=True, progress=False)["Close"]
pm = p.resample("ME").last()
pm.index = pm.index.to_period("M")
r = pm.pct_change()
ex = r[SECT].sub(r["SPY"], axis=0)
mom = pm[SECT].shift(1) / pm[SECT].shift(12) - 1
ult = pd.Period(pd.Timestamp.today(), "M") - 1
meses = pd.period_range("2005-01", ult, freq="M")


def evitar(t):
    q = reg.get(t - 1)
    if q is None:
        return set()
    hist = ex.index[ex.index < t]
    sig = reg.shift(1).reindex(hist)
    tab = ex.loc[sig[sig == q].index].mean().dropna().sort_values()
    return set(tab.index[:K])


def correr(con_regimen, cada):
    filas, cart = [], set()
    for k, t in enumerate(meses):
        disp = [s for s in SECT if pd.notna(r.loc[t, s]) and pd.notna(mom.loc[t - 1, s])] if (t - 1) in mom.index else []
        if len(disp) < 2 * K:
            continue
        nueva = cart
        if k % cada == 0 or not cart:
            m1 = mom.loc[t, disp]
            if con_regimen:
                ev = evitar(t)
                m1 = m1[[s for s in m1.index if s not in ev]]
            nueva = set(m1.sort_values(ascending=False).index[:K])
        costo = COSTO * len(nueva - cart) / K if cart else COSTO
        cart = nueva
        eq = r.loc[t, disp].mean()
        filas.append({"mes": t, "est": r.loc[t, list(cart)].mean() - costo, "eq": eq, "spy": r.loc[t, "SPY"]})
    return pd.DataFrame(filas).set_index("mes")


def stats(d):
    exm = d["est"] - d["eq"]
    def anual(s):
        return (1 + s).prod() ** (12 / len(s)) - 1
    def dd(s):
        c = (1 + s).cumprod()
        return (c / c.cummax() - 1).min()
    out = {"anual": round(anual(d["est"]) * 100, 1), "igualpeso": round(anual(d["eq"]) * 100, 1), "spy": round(anual(d["spy"]) * 100, 1),
           "exceso": round((anual(d["est"]) - anual(d["eq"])) * 100, 1), "t": round(exm.mean() / exm.std() * np.sqrt(len(exm)), 2),
           "gana_meses": round((exm > 0).mean() * 100), "anios": f"{(exm.groupby(d.index.year).sum() > 0).sum()}/{d.index.year.nunique()}",
           "caida_max": round(dd(d["est"]) * 100), "caida_max_igualpeso": round(dd(d["eq"]) * 100)}
    for a, b in [(2005, 2018), (2019, 2026)]:
        s = d[(d.index.year >= a) & (d.index.year <= b)]
        out[f"exceso_{a}_{b}"] = round((anual(s["est"]) - anual(s["eq"])) * 100, 1)
    return out


res = {}
for nombre, reg_, cada in [("A_momentum_mensual", False, 1), ("B_momentum_regimen_mensual", True, 1),
                           ("C_momentum_trimestral", False, 3), ("D_momentum_regimen_trimestral", True, 3)]:
    res[nombre] = stats(correr(reg_, cada))
print("RESULTADO " + json.dumps(res))
