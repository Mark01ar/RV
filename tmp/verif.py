import json
import time
import requests
import yfinance as yf

UA = {"User-Agent": "MaQ RV research marcosracimo@gmail.com"}
d = json.load(open("data/empresas.json"))["empresas"]
mal = []
for x in d:
    t = x["ticker"]
    f = json.load(open(f"data/fichas/{t}.json"))
    try:
        info = yf.Ticker(t).info
    except Exception:
        continue
    usd = info.get("financialCurrency") in (None, "USD")
    tq = f.get("trimestral", [])[-4:]
    an = f.get("anual") or [{}]
    u = an[-1]
    vt = sum(q["ventas"] for q in tq) if len(tq) == 4 and all(q.get("ventas") for q in tq) else None
    nt = sum(q["neto"] for q in tq) if len(tq) == 4 and all(q.get("neto") is not None for q in tq) else None
    comp = [("ventas_ttm", vt, info.get("totalRevenue"), 0.15), ("neto_ttm", nt, info.get("netIncomeToCommon"), 0.30),
            ("caja", u.get("caja"), info.get("totalCash"), 0.5), ("deuda", u.get("deuda"), info.get("totalDebt"), 0.6)]
    for nom, mio, yh, tol in comp:
        if not usd or mio is None or not yh:
            continue
        if abs(mio / yh - 1) > tol and abs(mio - yh) > 1e9:
            mal.append((x["byma"], nom, round(mio / 1e9, 1), round(yh / 1e9, 1)))
    if not info.get("trailingPE") and nt and nt > 0:
        mal.append((x["byma"], "pe_falta_con_ganancia", round(nt / 1e9, 2), None))
    time.sleep(0.3)
for m in mal:
    print("M|" + json.dumps(m), flush=True)
mapa = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
cik = {v["ticker"]: int(v["cik_str"]) for v in mapa.values()}
for t in ["NVDA", "AAPL", "GOOGL"]:
    g = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik[t]:010d}.json", headers=UA, timeout=60).json()["facts"]["us-gaap"]
    for k, v in g.items():
        if any(w in k for w in ["Cash", "Marketable", "ShortTermInvest"]) and "USD" in v.get("units", {}):
            vals = [e for e in v["units"]["USD"] if e.get("form") == "10-K"]
            if vals and vals[-1]["end"] >= "2024":
                print("T|" + t + "|" + k + "|" + vals[-1]["end"] + "|" + str(round(vals[-1]["val"] / 1e9, 1)), flush=True)
