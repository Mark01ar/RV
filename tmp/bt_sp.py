import io
import json
import time
import numpy as np
import pandas as pd
import requests
import yfinance as yf

UA = {"User-Agent": "MaQ RV research marcosracimo@gmail.com"}
COSTO = 0.005
url = "https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
H = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text))
H["date"] = pd.to_datetime(H["date"])
H = H.sort_values("date")
H["set"] = H["tickers"].str.split(",").apply(lambda l: {x.strip().replace(".", "-") for x in l})
TODOS = sorted(set().union(*H[H["date"] >= "2008-01-01"]["set"]))
print("tickers historicos", len(TODOS), flush=True)


def miembros(f):
    r = H[H["date"] <= f]
    return r["set"].iloc[-1] if len(r) else set()


raw = {}
for i in range(0, len(TODOS), 80):
    lote = TODOS[i:i + 80]
    for _ in range(3):
        try:
            df = yf.download(lote, start="2008-01-01", auto_adjust=False, progress=False, group_by="column", threads=True)
            break
        except Exception as e:
            time.sleep(10)
    for c in ["Adj Close", "Close"]:
        raw.setdefault(c, []).append(df[c] if isinstance(df.columns, pd.MultiIndex) else df[[c]].rename(columns={c: lote[0]}))
    time.sleep(2)
adj = pd.concat(raw["Adj Close"], axis=1)
cls = pd.concat(raw["Close"], axis=1)
adj = adj.loc[:, ~adj.columns.duplicated()]
cls = cls.loc[:, ~cls.columns.duplicated()]
spy = yf.download(["SPY", "RSP", "^IRX"], start="2008-01-01", auto_adjust=True, progress=False)["Close"]
for x in (adj, cls, spy):
    x.index = pd.to_datetime(x.index).tz_localize(None)
cal = spy["SPY"].dropna().index
adj, cls = adj.reindex(cal).ffill(limit=5), cls.reindex(cal).ffill(limit=5)
con_precio = [t for t in adj.columns if adj[t].notna().sum() > 250]
print("con precio", len(con_precio), "de", len(TODOS), flush=True)

mapa = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
cik = {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in mapa.values()}
INGRESOS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueGoodsNet"]
CONC = {"rev": INGRESOS, "gp": ["GrossProfit"], "op": ["OperatingIncomeLoss"], "ni": ["NetIncomeLoss", "ProfitLoss"],
        "cfo": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
        "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"], "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
        "ta": ["Assets"], "cl": ["LiabilitiesCurrent"], "ca": ["AssetsCurrent"], "ltd": ["LongTermDebt", "LongTermDebtNoncurrent"],
        "std": ["DebtCurrent", "LongTermDebtCurrent"]}
INST = {"ta", "cl", "ca", "ltd", "std"}
fechas = cal.to_series().groupby(cal.to_period("M")).max()
FE = pd.DatetimeIndex(fechas[(fechas >= "2009-06-01")].values[:-1])


def serie(f, nombres, inst):
    rows = []
    g = f.get("facts", {}).get("us-gaap", {})
    for pri, n in enumerate(nombres):
        for u, lst in g.get(n, {}).get("units", {}).items():
            if u != "USD":
                continue
            for e in lst:
                if e.get("form") not in ("10-K", "10-K/A") or e.get("fp") != "FY":
                    continue
                end = pd.Timestamp(e["end"])
                if not inst:
                    if "start" not in e:
                        continue
                    du = (end - pd.Timestamp(e["start"])).days
                    if du < 330 or du > 400:
                        continue
                rows.append((end, pd.Timestamp(e["filed"]), float(e["val"]), pri))
    if not rows:
        return None
    r = pd.DataFrame(rows, columns=["end", "filed", "val", "pri"]).sort_values(["end", "pri", "filed"]).groupby("end").first().reset_index()
    r = r[r["filed"] <= r["end"] + pd.Timedelta(days=400)]
    return r


def panel(r):
    out = pd.DataFrame(index=FE, columns=["v", "v1", "end"], dtype=float)
    if r is None or r.empty:
        return out
    r = r.sort_values("filed")
    vals = r.set_index("end")["val"]
    best_end, rec = None, []
    for _, x in r.iterrows():
        if best_end is None or x["end"] > best_end:
            best_end = x["end"]
        rec.append((x["filed"], best_end))
    s = pd.DataFrame(rec, columns=["filed", "end"]).groupby("filed")["end"].max()
    e_at = s.reindex(s.index.union(FE)).ffill().reindex(FE)
    v, v1 = [], []
    for e in e_at:
        if pd.isna(e):
            v.append(np.nan); v1.append(np.nan); continue
        v.append(vals.get(e, np.nan))
        prev = vals[(vals.index >= e - pd.Timedelta(days=400)) & (vals.index <= e - pd.Timedelta(days=330))]
        v1.append(prev.iloc[-1] if len(prev) else np.nan)
    out["v"], out["v1"], out["end"] = v, v1, [(e.value if pd.notna(e) else np.nan) for e in e_at]
    return out


DATA = {}
n_ok = 0
for i, t in enumerate(con_precio):
    c = cik.get(t.upper())
    if not c:
        continue
    try:
        f = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json", headers=UA, timeout=60).json()
    except Exception:
        continue
    P = {}
    for k, v in CONC.items():
        p = panel(serie(f, v, k in INST))
        P[k], P[k + "1"] = p["v"].astype(float), p["v1"].astype(float)
        if k == "ta":
            P["end"] = p["end"]
    rows = []
    for u, lst in f.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).items():
        for e in lst:
            rows.append((pd.Timestamp(e["filed"]), float(e["val"])))
    if rows:
        sh = pd.DataFrame(rows, columns=["filed", "val"]).groupby("filed")["val"].sum().sort_index()
        sh = sh[sh > 0]
        P["sh"] = sh.reindex(sh.index.union(FE)).ffill(limit=None).reindex(FE)
        last = pd.Series(sh.index, index=sh.index).reindex(sh.index.union(FE)).ffill().reindex(FE)
        P["sh"] = P["sh"].where((FE - last) <= pd.Timedelta(days=200))
        P["sh1"] = P["sh"].shift(12)
    DATA[t] = pd.DataFrame(P)
    n_ok += 1
    time.sleep(0.12)
    if i % 100 == 0:
        print("sec", i, n_ok, flush=True)
print("sec_ok", len(DATA), flush=True)

spl = {}
for t in DATA:
    try:
        s = yf.Ticker(t).splits
        s.index = pd.to_datetime(s.index).tz_localize(None)
        spl[t] = s[s > 0]
    except Exception:
        spl[t] = pd.Series(dtype=float)
fac = pd.DataFrame(1.0, index=FE, columns=list(DATA))
for t, s in spl.items():
    for d_, r in s.items():
        fac.loc[fac.index < d_, t] *= r
print("splits ok", flush=True)

T = list(DATA)
am = adj.reindex(columns=T).loc[FE]
ma200 = adj.reindex(columns=T).rolling(200, min_periods=150).mean().loc[FE]
mom = am.shift(1) / am.shift(12) - 1
vsma = am / ma200 - 1
fwd = am.shift(-1) / am - 1
pxr = cls.reindex(columns=T).loc[FE] * fac
caja = spy["^IRX"].reindex(FE) / 100 / 12
spyf = (spy["SPY"].reindex(FE).shift(-1) / spy["SPY"].reindex(FE) - 1)
rspf = (spy["RSP"].reindex(FE).shift(-1) / spy["RSP"].reindex(FE) - 1)
G = {k: pd.DataFrame({t: DATA[t][k] if k in DATA[t] else np.nan for t in T}, index=FE) for k in
     ["rev", "rev1", "gp", "gp1", "op", "ni", "ni1", "cfo", "capex", "da", "ta", "ta1", "cl", "cl1", "ca", "ca1", "ltd", "ltd1", "std", "sh", "sh1", "end"]}
endt = G["end"].apply(lambda c: pd.to_datetime(c, errors="coerce"))
fresco = endt.ge(pd.DataFrame({t: FE - pd.Timedelta(days=500) for t in T}, index=FE))
for k in G:
    if k != "end":
        G[k] = G[k].where(fresco)
with np.errstate(all="ignore"):
    sdiv = lambda a, b: (a / b).where(b.abs() > 0)
    mcap = pxr * G["sh"]
    deuda = G["ltd"].fillna(0) + G["std"].fillna(0)
    deuda = deuda.where(G["ltd"].notna() | G["std"].notna())
    ebitda = G["op"] + G["da"].fillna(0)
    M = {"roic": sdiv(G["op"], G["ta"] - G["cl"]).where((G["ta"] - G["cl"]) > 0),
         "mop": sdiv(G["op"], G["rev"]), "mb": sdiv(G["gp"], G["rev"]),
         "deb": (-sdiv(deuda, ebitda)).where(ebitda > 0, -99).where(ebitda.notna() & deuda.notna()),
         "acc": -sdiv(G["ni"] - G["cfo"], G["ta"]),
         "dil": -(sdiv(G["sh"], G["sh1"]) - 1),
         "ey": sdiv(G["ni"], mcap).where(mcap > 0), "fcfy": sdiv(G["cfo"] - G["capex"].fillna(0), mcap).where(mcap > 0),
         "sy": sdiv(G["rev"], mcap).where(mcap > 0)}
    roa, roa1 = sdiv(G["ni"], G["ta"]), sdiv(G["ni1"], G["ta1"])
    tests = [roa > 0, G["cfo"] > 0, roa > roa1, G["cfo"] > G["ni"], sdiv(G["ltd"], G["ta"]) < sdiv(G["ltd1"], G["ta1"]),
             sdiv(G["ca"], G["cl"]) > sdiv(G["ca1"], G["cl1"]), G["sh"] <= G["sh1"] * 1.01,
             sdiv(G["gp"], G["rev"]) > sdiv(G["gp1"], G["rev1"]), sdiv(G["rev"], G["ta"]) > sdiv(G["rev1"], G["ta1"])]
    valid = [roa.notna(), G["cfo"].notna(), roa1.notna(), G["cfo"].notna() & G["ni"].notna(), G["ltd1"].notna() & G["ltd"].notna(),
             G["ca1"].notna() & G["cl1"].notna(), G["sh1"].notna(), G["gp1"].notna() & G["gp"].notna(), G["rev1"].notna()]
    num_ = sum((a & b).astype(float) for a, b in zip(tests, valid))
    den_ = sum(b.astype(float) for b in valid)
    M["fs"] = (num_ / den_).where(den_ >= 6)
sec_ = {}
try:
    info = requests.get("https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv", timeout=60).text
    cc = pd.read_csv(io.StringIO(info))
    sec_ = dict(zip(cc["Symbol"].str.replace(".", "-"), cc["GICS Sector"]))
except Exception:
    pass
print("sectores", len(sec_), flush=True)


def rk(s):
    return s.rank(pct=True)


EST = ["Base", "Fund", "Calidad", "Valor", "Combo", "BaseSinPeor"]
filas, prev = [], {(k, n): set() for k in EST for n in (10, 50)}
cobertura = []
for f in FE[12:-1]:
    mem = {t for t in miembros(f)}
    cobertura.append((len(mem), len(mem & set(T)), int(sum(1 for t in mem & set(T) if pd.notna(fwd.loc[f, t]) and pd.notna(M["roic"].loc[f, t]) or pd.notna(M["mop"].loc[f, t])))))
    ok = [t for t in mem if t in T and pd.notna(mom.loc[f, t]) and pd.notna(vsma.loc[f, t]) and pd.notna(fwd.loc[f, t]) and pd.notna(G["ta"].loc[f, t])]
    if len(ok) < 150:
        continue
    X = pd.DataFrame({k: M[k].loc[f, ok] for k in M})
    X["sector"] = [sec_.get(t, "?") for t in ok]
    for c in ["ey", "sy"]:
        X[c + "_s"] = X.groupby("sector")[c].rank(pct=True)
    cal_ = X[["roic", "mop", "mb", "deb", "acc", "dil", "fs"]].rank(pct=True).mean(axis=1)
    val_ = X[["ey_s", "fcfy", "sy_s"]].rank(pct=True).mean(axis=1)
    m_, v_, r_ = mom.loc[f, ok], vsma.loc[f, ok], fwd.loc[f, ok]
    mo = (rk(m_) + rk(v_)) / 2
    fund = (25 * rk(mo) + 25 * rk(cal_) + 20 * rk(val_).fillna(0.5)) / 70
    qv = (rk(cal_) + rk(val_).fillna(0.5)) / 2
    peor = list(qv[qv <= qv.quantile(1 / 3)].index)
    sc = {"Base": mo, "Fund": fund, "Calidad": cal_, "Valor": val_.dropna(), "BaseSinPeor": mo.drop(peor),
          "Combo": (rk(m_) + rk(v_) + rk(cal_) + rk(val_).fillna(0.5)) / 4}
    fl = {"f": f, "n": len(ok), "caja": float(caja.loc[f]), "spy": float(spyf.loc[f]), "rsp": float(rspf.loc[f]), "Univ": float(r_.mean())}
    for s_, s in sc.items():
        for n in (10, 50):
            top = list(s.sort_values(ascending=False).index[:n])
            hold = {t for t in top if v_[t] > 0}
            fl[f"{s_}{n}"] = (sum(r_[t] for t in hold) + (n - len(hold)) * fl["caja"]) / n - COSTO * len(hold ^ prev[(s_, n)]) / n
            prev[(s_, n)] = hold
    for nm, s in {"fund": fund, "calidad": cal_, "valor": val_.dropna(), "qv": qv, "mom": mo,
                  "roic": X["roic"].dropna(), "mb": X["mb"].dropna(), "fs": X["fs"].dropna(), "acc": X["acc"].dropna(), "ey": X["ey_s"].dropna(), "fcfy": X["fcfy"].dropna()}.items():
        a, b = s[s >= s.quantile(0.8)].index, s[s <= s.quantile(0.2)].index
        fl["sp_" + nm] = float(r_[a].mean() - r_[b].mean())
    filas.append(fl)
cb = np.array(cobertura)
print("R|cobertura|" + json.dumps({"miembros_prom": round(float(cb[:, 0].mean())), "con_precio_y_sec": round(float(cb[:, 1].mean())), "pct": round(float(cb[:, 1].mean() / cb[:, 0].mean()) * 100),
                                   "pct_2010": round(float(cb[:12, 1].mean() / cb[:12, 0].mean()) * 100)}), flush=True)
R = pd.DataFrame(filas).set_index("f")
an = lambda s: (1 + s).prod() ** (12 / len(s)) - 1
print("R|meses|" + json.dumps({"n": len(R), "desde": str(R.index[0])[:7], "hasta": str(R.index[-1])[:7], "empresas_med": int(R["n"].median())}), flush=True)
for a_, b_ in [("2010", "2018"), ("2019", "2026"), ("2010", "2026")]:
    sub = R.loc[a_:b_]
    o = {"SPY": round(an(sub["spy"]) * 100, 1), "RSP": round(an(sub["rsp"]) * 100, 1), "Univ": round(an(sub["Univ"]) * 100, 1)}
    for s_ in EST:
        for n in (10, 50):
            x = sub[f"{s_}{n}"]
            c = (1 + x).cumprod()
            o[f"{s_}{n}"] = {"anual": round(an(x) * 100, 1), "vs_univ": round((an(x) - an(sub["Univ"])) * 100, 1), "dd": round(float((c / c.cummax() - 1).min()) * 100)}
    for nm in ["fund", "calidad", "valor", "qv", "mom", "roic", "mb", "fs", "acc", "ey", "fcfy"]:
        s = sub["sp_" + nm].dropna()
        o["Q_" + nm] = {"mensual": round(float(s.mean()) * 100, 2), "t": round(float(s.mean() / s.std() * np.sqrt(len(s))), 2), "pos": round(float((s > 0).mean()) * 100)}
    print("R|" + a_ + "-" + b_ + "|" + json.dumps(o), flush=True)
ya = {nm: {str(k): round(float(v) * 100, 1) for k, v in ((1 + R["sp_" + nm]).groupby(R.index.year).prod() - 1).items()} for nm in ["calidad", "valor", "mom", "fund"]}
print("R|anual_quintiles|" + json.dumps(ya), flush=True)
