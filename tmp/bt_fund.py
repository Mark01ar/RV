import json
import time
import numpy as np
import pandas as pd
import yfinance as yf

COSTO = 0.006
N = 20
d = json.load(open("data/datos.json"))
tickers = [x["ticker"] for x in d["tickers"] if not x.get("sin_datos")]
LIQ = {x["ticker"] for x in d["tickers"] if x.get("liquidez") in ("Alta", "Media", "Baja")}

frames = {c: [] for c in ["Close", "High", "Low", "Volume"]}
for i in range(0, len(tickers), 40):
    lote = tickers[i:i + 40]
    for _ in range(3):
        try:
            df = yf.download(lote, start="2003-06-01", auto_adjust=True, progress=False, group_by="column", threads=True)
            break
        except Exception as e:
            print("reintento", e, flush=True)
            time.sleep(10)
    for c in frames:
        frames[c].append(df[c])
    time.sleep(2)
P = {c: pd.concat(v, axis=1).loc[:, lambda x: ~x.columns.duplicated()] for c, v in frames.items()}
for c in P:
    P[c].index = pd.to_datetime(P[c].index).tz_localize(None)
close = P["Close"]
spy = yf.download("SPY", start="2003-06-01", auto_adjust=True, progress=False)["Close"].squeeze()
irx = yf.download("^IRX", start="2003-06-01", auto_adjust=True, progress=False)["Close"].squeeze()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
irx.index = pd.to_datetime(irx.index).tz_localize(None)
cal = spy.dropna().index
for c in P:
    P[c] = P[c].reindex(cal)
P["Close"] = P["Close"].ffill(limit=3)
close = P["Close"]
print("precios", close.shape, float(close.iloc[-1].notna().mean()), flush=True)

dias = close.index
fechas = dias.to_series().groupby(dias.to_period("M")).max()
fechas = fechas[(fechas >= "2005-12-01")]
fechas = fechas.iloc[:-1] if fechas.index[-1] == pd.Period(pd.Timestamp.today(), "M") else fechas
cm = close.reindex(dias).ffill(limit=5)
ma200 = close.rolling(200, min_periods=150).mean()


FE = list(fechas.values)
pm = cm.loc[FE]
mom = pm.shift(1) / pm.shift(12) - 1
vsma = pm / ma200.loc[FE] - 1
fwd = pm.shift(-1) / pm - 1
spym = spy.reindex(dias).ffill().loc[FE]
spyf = spym.shift(-1) / spym - 1
cajaf = (irx.reindex(dias).ffill().loc[FE] / 100 / 12)
def rk(s):
    return s.rank(pct=True)
sec = json.load(open("data/sectores.json"))
liq = [t for t in close.columns if t in LIQ]


def fila(df, *nombres):
    for n in nombres:
        if n in df.index:
            return df.loc[n]
    return None


reg = []
for i, t in enumerate(close.columns):
    for _ in range(2):
        try:
            tk = yf.Ticker(t)
            inc, bal, cf = tk.income_stmt, tk.balance_sheet, tk.cashflow
            try:
                cur = tk.info.get("financialCurrency")
            except Exception:
                cur = None
            break
        except Exception:
            inc = bal = cf = None
            time.sleep(3)
    if inc is None or inc.empty or bal is None or bal.empty:
        continue
    for fy in inc.columns:
        g = lambda df, *n: (lambda s: float(s[fy]) if s is not None and fy in s.index and pd.notna(s[fy]) else np.nan)(fila(df, *n)) if df is not None and not df.empty else np.nan
        rev, gp, op, ni = g(inc, "Total Revenue"), g(inc, "Gross Profit"), g(inc, "Operating Income", "EBIT"), g(inc, "Net Income", "Net Income Common Stockholders")
        ebitda = g(inc, "EBITDA", "Normalized EBITDA")
        ta, cl, debt, sh = g(bal, "Total Assets"), g(bal, "Current Liabilities"), g(bal, "Total Debt"), g(bal, "Ordinary Shares Number", "Share Issued")
        cfo, fcf = g(cf, "Operating Cash Flow"), g(cf, "Free Cash Flow")
        reg.append({"t": t, "fy": pd.Timestamp(fy), "rev": rev, "gp": gp, "op": op, "ni": ni, "ebitda": ebitda, "ta": ta, "cl": cl,
                    "debt": debt, "sh": sh, "cfo": cfo, "fcf": fcf, "usd": cur == "USD"})
    if i % 25 == 0:
        print("bal", i, len(reg), flush=True)
    time.sleep(0.3)
B = pd.DataFrame(reg)
B["disp"] = B["fy"] + pd.Timedelta(days=90)
B["roic"] = B["op"] / (B["ta"] - B["cl"])
B["mop"] = B["op"] / B["rev"]
B["mb"] = B["gp"] / B["rev"]
B["deb"] = -(B["debt"] / B["ebitda"]).where(B["ebitda"] > 0, 99)
B["acc"] = -(B["ni"] - B["cfo"]) / B["ta"]
B = B.sort_values(["t", "fy"])
B["dil"] = -B.groupby("t")["sh"].pct_change()
print("balances", len(B), B["t"].nunique(), str(B["fy"].min())[:10], flush=True)

EST = ["Base", "Fund", "Calidad", "Valor", "BaseSinPeor", "Combo", "Univ"]
filas, prev, spread = [], {k: set() for k in EST}, []
for f in FE[12:-1]:
    pf = pd.Timestamp(f)
    if pf < pd.Timestamp("2022-01-01"):
        continue
    ok = mom.loc[f].notna() & vsma.loc[f].notna() & fwd.loc[f].notna() & mom.columns.isin(liq)
    m, v, r = mom.loc[f][ok], vsma.loc[f][ok], fwd.loc[f][ok]
    ult = B[(B["disp"] <= pf) & B["t"].isin(m.index)].groupby("t").tail(1).set_index("t")
    ult = ult[ult["disp"] > pf - pd.Timedelta(days=500)]
    if len(ult) < 60:
        continue
    px = pm.loc[f, ult.index]
    mcap = px * ult["sh"]
    ult["ey"] = (ult["ni"] / mcap).where(ult["usd"])
    ult["fy_"] = (ult["fcf"] / mcap).where(ult["usd"])
    ult["sector"] = [(sec.get(t) or {}).get("sector") for t in ult.index]
    ult["ey_s"] = ult.groupby("sector")["ey"].rank(pct=True)
    cal = ult[["roic", "mop", "mb", "deb", "acc", "dil"]].rank(pct=True).mean(axis=1)
    val = ult[["ey_s", "fy_"]].rank(pct=True).mean(axis=1)
    idx = ult.index
    mm, vv, rr = m[idx], v[idx], r[idx]
    mo = (rk(mm) + rk(vv)) / 2
    fund = (25 * rk(mo) + 25 * cal.rank(pct=True) + 20 * val.rank(pct=True).fillna(0.5)) / 70
    qv = (cal.rank(pct=True) + val.rank(pct=True).fillna(0.5)) / 2
    base_all = (rk(m) + rk(v)) / 2
    peor = set(qv[qv <= qv.quantile(1 / 3)].index)
    sc = {"Base": base_all, "Fund": fund, "Calidad": cal, "Valor": val.dropna(),
          "BaseSinPeor": base_all.drop(list(peor), errors="ignore"),
          "Combo": (rk(mm) + rk(vv) + cal.rank(pct=True) + val.rank(pct=True).fillna(0.5)) / 4}
    fl = {"f": f, "caja": cajaf.loc[f], "spy": spyf.loc[f], "Univ": float(r.mean()), "n": len(idx)}
    for s_, s in sc.items():
        top = list(s.sort_values(ascending=False).index[:10])
        hold = {t for t in top if v[t] > 0}
        fl[s_] = (sum(r[t] for t in hold) + (10 - len(hold)) * fl["caja"]) / 10 - COSTO * len(hold ^ prev[s_]) / 10
        prev[s_] = hold
    for nm, s in {"fund": fund, "calidad": cal, "valor": val.dropna(), "qv": qv}.items():
        a, b = s[s >= s.quantile(2 / 3)].index, s[s <= s.quantile(1 / 3)].index
        fl["sp_" + nm] = float(r[a].mean() - r[b].mean())
    filas.append(fl)
R = pd.DataFrame(filas).set_index("f")
R.index = pd.to_datetime(R.index)
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
o = {"meses": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7], "n_med": int(R["n"].median()), "SPY": round(an(R["spy"]) * 100, 1)}
yb = (1 + R["Base"]).groupby(R.index.year).prod() - 1
for s_ in EST:
    x = R[s_]
    c = (1 + x).cumprod()
    y = (1 + x).groupby(x.index.year).prod() - 1
    o[s_] = {"anual": round(an(x) * 100, 1), "vs_base": round((an(x) - an(R["Base"])) * 100, 1), "vs_univ": round((an(x) - an(R["Univ"])) * 100, 1),
             "dd": round(float((c / c.cummax() - 1).min()) * 100), "anios": {str(k): round(float(w) * 100) for k, w in y.items()}}
for nm in ["fund", "calidad", "valor", "qv"]:
    s = R["sp_" + nm]
    o["tercios_" + nm] = {"mensual": round(float(s.mean()) * 100, 2), "t": round(float(s.mean() / s.std() * np.sqrt(len(s))), 2), "meses_pos": round(float((s > 0).mean()) * 100)}
for kk, vv_ in o.items():
    print("R|" + kk + "|" + json.dumps(vv_), flush=True)
