import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

UA = {"User-Agent": "MaQ RV research marcosracimo@gmail.com"}
COSTO = 0.005
d = json.load(open("data/datos.json"))
US = [x for x in d["tickers"] if x.get("pais") == "United States" and not x.get("sin_datos") and not x["ticker"].endswith(".KS")]
SECT = {x["ticker"]: x.get("sector") for x in US}
LIQ = {x["ticker"] for x in US if x.get("liquidez") in ("Alta", "Media", "Baja")}
tickers = [x["ticker"] for x in US]

mapa = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
cik = {v["ticker"].upper(): int(v["cik_str"]) for v in mapa.values()}

INGRESOS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]
CONC = {"rev": INGRESOS, "gp": ["GrossProfit"], "op": ["OperatingIncomeLoss"], "ni": ["NetIncomeLoss", "ProfitLoss"],
        "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"], "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
        "ta": ["Assets"], "cl": ["LiabilitiesCurrent"], "ca": ["AssetsCurrent"], "ltd": ["LongTermDebt", "LongTermDebtNoncurrent"],
        "std": ["DebtCurrent", "LongTermDebtCurrent"], "eq": ["StockholdersEquity"]}
INST = {"ta", "cl", "ca", "ltd", "std", "eq"}


def serie(f, nombres, inst):
    rows = []
    g = f.get("facts", {}).get("us-gaap", {})
    for pri, n in enumerate(nombres):
        if n not in g:
            continue
        for u, lst in g[n].get("units", {}).items():
            if u != "USD":
                continue
            for e in lst:
                if e.get("form") not in ("10-K", "10-K/A") or e.get("fp") != "FY":
                    continue
                end = pd.Timestamp(e["end"])
                if not inst:
                    if "start" not in e:
                        continue
                    dur = (end - pd.Timestamp(e["start"])).days
                    if dur < 330 or dur > 400:
                        continue
                rows.append({"end": end, "filed": pd.Timestamp(e["filed"]), "val": float(e["val"]), "pri": pri})
    if not rows:
        return pd.DataFrame(columns=["end", "filed", "val"])
    r = pd.DataFrame(rows).sort_values(["end", "pri", "filed"])
    r = r.groupby("end").first().reset_index()
    return r[["end", "filed", "val"]]


def acciones(f):
    rows = []
    for ns, n in [("dei", "EntityCommonStockSharesOutstanding")]:
        for u, lst in f.get("facts", {}).get(ns, {}).get(n, {}).get("units", {}).items():
            for e in lst:
                rows.append({"filed": pd.Timestamp(e["filed"]), "val": float(e["val"])})
    if not rows:
        return pd.Series(dtype=float)
    s = pd.DataFrame(rows).groupby("filed")["val"].sum().sort_index()
    return s


SEC = {}
for i, t in enumerate(tickers):
    c = cik.get(t.upper()) or cik.get(t.upper().replace("-", ".")) or cik.get(t.upper().replace("-", ""))
    if not c:
        continue
    try:
        f = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json", headers=UA, timeout=60).json()
    except Exception as e:
        print("err", t, e, flush=True)
        continue
    SEC[t] = {"series": {k: serie(f, v, k in INST) for k, v in CONC.items()}, "sh": acciones(f)}
    time.sleep(0.15)
    if i % 40 == 0:
        print("sec", i, len(SEC), flush=True)
print("sec_ok", len(SEC), "de", len(tickers), flush=True)

raw = yf.download(tickers + ["SPY", "^IRX"], start="2008-01-01", auto_adjust=False, progress=False, group_by="column", threads=True)
adj = raw["Adj Close"]
cls = raw["Close"]
adj.index = pd.to_datetime(adj.index).tz_localize(None)
cls.index = adj.index
cal = adj["SPY"].dropna().index
adj, cls = adj.reindex(cal).ffill(limit=3), cls.reindex(cal).ffill(limit=3)
spl = {}
for t in SEC:
    try:
        s = yf.Ticker(t).splits
        s.index = pd.to_datetime(s.index).tz_localize(None)
        spl[t] = s[s > 0]
    except Exception:
        spl[t] = pd.Series(dtype=float)
print("precios", adj.shape, flush=True)

fechas = cal.to_series().groupby(cal.to_period("M")).max()
fechas = fechas[(fechas >= "2009-01-01")]
fechas = fechas.iloc[:-1]
FE = list(fechas.values)
am = adj.loc[FE]
ma200 = adj.rolling(200, min_periods=150).mean().loc[FE]
mom = am.shift(1) / am.shift(12) - 1
vsma = am / ma200 - 1
fwd = am.shift(-1) / am - 1
caja = adj["^IRX"].loc[FE] / 100 / 12


def pit(df, f, anios_atras=0):
    x = df[df["filed"] <= f]
    if x.empty:
        return np.nan
    e = x["end"].max()
    if anios_atras:
        e2 = x[(x["end"] <= e - pd.Timedelta(days=330)) & (x["end"] >= e - pd.Timedelta(days=400))]
        return float(e2.sort_values("end")["val"].iloc[-1]) if len(e2) else np.nan
    if e < f - pd.Timedelta(days=500):
        return np.nan
    return float(x.loc[x["end"] == e, "val"].iloc[0])


def precio_sin_ajuste(t, f):
    p = cls.loc[f, t]
    s = spl.get(t, pd.Series(dtype=float))
    fac = float(np.prod(s[s.index > f].values)) if len(s) else 1.0
    return p * fac


def rk(s):
    return s.rank(pct=True)


def metricas(t, f):
    S = SEC[t]["series"]
    g = lambda k, a=0: pit(S[k], f, a)
    rev, gp, op, ni, cfo, capex, da, ta, cl, ca, ltd, std = (g(k) for k in ["rev", "gp", "op", "ni", "cfo", "capex", "da", "ta", "cl", "ca", "ltd", "std"])
    if np.isnan(ta) or np.isnan(ni):
        return None
    rev1, gp1, ni1, ta1, ltd1, ca1, cl1 = (g(k, 1) for k in ["rev", "gp", "ni", "ta", "ltd", "ca", "cl"])
    sh = SEC[t]["sh"]
    sh = sh[sh.index <= f]
    nsh = float(sh.iloc[-1]) if len(sh) and sh.index[-1] >= f - pd.Timedelta(days=200) else np.nan
    sh1 = sh[sh.index <= f - pd.Timedelta(days=330)]
    nsh1 = float(sh1.iloc[-1]) if len(sh1) and sh1.iloc[-1] > 0 else np.nan
    nsh = nsh if nsh and nsh > 0 else np.nan
    mcap = precio_sin_ajuste(t, f) * nsh if not np.isnan(nsh) else np.nan
    deuda = np.nansum([ltd, std]) if not (np.isnan(ltd) and np.isnan(std)) else np.nan
    ebitda = op + (0 if np.isnan(da) else da) if not np.isnan(op) else np.nan
    roa, roa1 = ni / ta, (ni1 / ta1 if not np.isnan(ni1) and not np.isnan(ta1) else np.nan)
    fs = [roa > 0, (cfo > 0) if not np.isnan(cfo) else np.nan, (roa > roa1) if not np.isnan(roa1) else np.nan,
          (cfo > ni) if not np.isnan(cfo) else np.nan,
          (ltd / ta < ltd1 / ta1) if not (np.isnan(ltd) or np.isnan(ltd1) or np.isnan(ta1)) else np.nan,
          (ca / cl > ca1 / cl1) if not any(np.isnan(v) for v in [ca, cl, ca1, cl1]) else np.nan,
          (nsh <= nsh1 * 1.01) if not (np.isnan(nsh) or np.isnan(nsh1)) else np.nan,
          (gp / rev > gp1 / rev1) if not any(np.isnan(v) for v in [gp, rev, gp1, rev1]) else np.nan,
          (rev / ta > rev1 / ta1) if not any(np.isnan(v) for v in [rev, rev1, ta1]) else np.nan]
    fsv = [v for v in fs if v is not np.nan and not (isinstance(v, float) and np.isnan(v))]
    return {"roic": op / (ta - cl) if not (np.isnan(op) or np.isnan(cl)) and ta - cl > 0 else np.nan,
               "mop": op / rev if rev and not np.isnan(op) else np.nan,
               "mb": gp / rev if rev and not np.isnan(gp) else np.nan,
               "deb": -(deuda / ebitda) if ebitda and ebitda > 0 and not np.isnan(deuda) else (-99 if ebitda and ebitda <= 0 else np.nan),
               "acc": -(ni - cfo) / ta if not np.isnan(cfo) else np.nan,
               "dil": -(nsh / nsh1 - 1) if not (np.isnan(nsh) or np.isnan(nsh1)) else np.nan,
               "fs": (sum(fsv) / len(fsv)) if len(fsv) >= 6 else np.nan,
               "ey": ni / mcap if mcap and mcap > 0 else np.nan,
               "fcfy": (cfo - (0 if np.isnan(capex) else capex)) / mcap if mcap and mcap > 0 and not np.isnan(cfo) else np.nan,
               "sy": rev / mcap if mcap and mcap > 0 and rev else np.nan}


EST = ["Base", "Fund", "Calidad", "Valor", "Combo", "BaseSinPeor", "Univ"]
filas, prev = [], {k: set() for k in EST}
for f in FE[12:-1]:
    f = pd.Timestamp(f)
    ok = [t for t in SEC if pd.notna(mom.loc[f, t]) and pd.notna(vsma.loc[f, t]) and pd.notna(fwd.loc[f, t])]
    rows = {}
    for t in ok:
        try:
            x = metricas(t, f)
        except (ZeroDivisionError, ValueError, KeyError, IndexError, TypeError):
            x = None
        if x:
            rows[t] = x
    if len(rows) < 60:
        continue
    X = pd.DataFrame(rows).T
    X["sector"] = [SECT.get(t) for t in X.index]
    for c in ["ey", "sy"]:
        X[c + "_s"] = X.groupby("sector")[c].rank(pct=True)
    cal_ = X[["roic", "mop", "mb", "deb", "acc", "dil", "fs"]].rank(pct=True).mean(axis=1)
    val_ = X[["ey_s", "fcfy", "sy_s"]].rank(pct=True).mean(axis=1)
    idx = X.index
    m_, v_, r_ = mom.loc[f, idx], vsma.loc[f, idx], fwd.loc[f, idx]
    mo = (rk(m_) + rk(v_)) / 2
    fund = (25 * rk(mo) + 25 * rk(cal_) + 20 * rk(val_).fillna(0.5)) / 70
    qv = (rk(cal_) + rk(val_).fillna(0.5)) / 2
    peor = set(qv[qv <= qv.quantile(1 / 3)].index)
    sc = {"Base": mo, "Fund": fund, "Calidad": cal_, "Valor": val_.dropna(), "BaseSinPeor": mo.drop(list(peor)),
          "Combo": (rk(m_) + rk(v_) + rk(cal_) + rk(val_).fillna(0.5)) / 4}
    fl = {"f": f, "n": len(idx), "caja": float(caja.loc[f]), "spy": float(fwd.loc[f, "SPY"]), "Univ": float(r_.mean())}
    for s_, s in sc.items():
        top = list(s.sort_values(ascending=False).index[:10])
        hold = {t for t in top if v_[t] > 0}
        fl[s_] = (sum(r_[t] for t in hold) + (10 - len(hold)) * fl["caja"]) / 10 - COSTO * len(hold ^ prev[s_]) / 10
        prev[s_] = hold
    for nm, s in {"fund": fund, "calidad": cal_, "valor": val_.dropna(), "qv": qv, "mom": mo}.items():
        a, b = s[s >= s.quantile(2 / 3)].index, s[s <= s.quantile(1 / 3)].index
        fl["sp_" + nm] = float(r_[a].mean() - r_[b].mean())
    filas.append(fl)
R = pd.DataFrame(filas).set_index("f")
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
print("R|meses|" + json.dumps({"n": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7], "empresas_med": int(R["n"].median())}), flush=True)
for a_, b_ in [("2010", "2018"), ("2019", "2026"), ("2010", "2026")]:
    sub = R.loc[a_:b_]
    o = {"SPY": round(an(sub["spy"]) * 100, 1)}
    for s_ in EST:
        x = sub[s_]
        c = (1 + x).cumprod()
        o[s_] = {"anual": round(an(x) * 100, 1), "vs_univ": round((an(x) - an(sub["Univ"])) * 100, 1), "dd": round(float((c / c.cummax() - 1).min()) * 100)}
    for nm in ["fund", "calidad", "valor", "qv", "mom"]:
        s = sub["sp_" + nm]
        o["T_" + nm] = {"mensual": round(float(s.mean()) * 100, 2), "t": round(float(s.mean() / s.std() * np.sqrt(len(s))), 2), "pos": round(float((s > 0).mean()) * 100)}
    print("R|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)
ya = {}
for nm in ["fund", "calidad", "valor", "mom"]:
    ya[nm] = {str(k): round(float(v) * 100, 1) for k, v in ((1 + R["sp_" + nm]).groupby(R.index.year).prod() - 1).items()}
print("R|anual_tercios|" + json.dumps(ya), flush=True)
