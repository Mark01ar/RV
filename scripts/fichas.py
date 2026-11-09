import json
import math
import os
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "fichas"
UA = {"User-Agent": "MaQ RV research marcosracimo@gmail.com"}
ETF_SECTOR = {"Technology": "XLK", "Communication Services": "XLC", "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP",
              "Financial Services": "XLF", "Healthcare": "XLV", "Industrials": "XLI", "Energy": "XLE", "Basic Materials": "XLB",
              "Real Estate": "XLRE", "Utilities": "XLU"}
SECTOR_ES = {"Technology": "Tecnología", "Communication Services": "Comunicaciones", "Consumer Cyclical": "Consumo discrecional",
             "Consumer Defensive": "Consumo básico", "Financial Services": "Financiero", "Healthcare": "Salud", "Industrials": "Industria",
             "Energy": "Energía", "Basic Materials": "Materiales", "Real Estate": "Inmobiliario", "Utilities": "Servicios públicos"}
CRIPTO = {"MSTR", "HUT", "IREN", "RIOT", "COIN"}


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def lin(x, pts):
    if x is None:
        return None
    xs, ys = zip(*pts)
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for (a, fa), (b, fb) in zip(pts, pts[1:]):
        if a <= x <= b:
            return fa + (fb - fa) * (x - a) / (b - a)


def media(v):
    v = [x for x in v if x is not None]
    return round(sum(v) / len(v), 1) if v else None


def liquidez(m):
    if m is None:
        return "Sin dato"
    return "Alta" if m >= 1e6 else "Media" if m >= 2e5 else "Baja" if m >= 3e4 else "Muy baja"


def fila(df, *nombres):
    if df is None or getattr(df, "empty", True):
        return None
    for n in nombres:
        if n in df.index:
            return df.loc[n]
    return None


def fx_anual(cur):
    if not cur or cur == "USD":
        return lambda y: 1.0
    try:
        s = yf.download(f"{cur}=X", start="2010-01-01", progress=False, auto_adjust=True)["Close"].squeeze().dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        prom = s.groupby(s.index.year).mean()
        ult = float(s.iloc[-1])
        return lambda y: float(prom.get(y, ult))
    except Exception:
        return lambda y: None


SEC_C = {"rev": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
         "gp": ["GrossProfit"], "op": ["OperatingIncomeLoss"], "ni": ["NetIncomeLoss", "ProfitLoss"],
         "cfo": ["NetCashProvidedByUsedInOperatingActivities"], "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
         "da": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
         "int": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt"],
         "ta": ["Assets"], "ca": ["AssetsCurrent"], "cl": ["LiabilitiesCurrent"], "eq": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
         "cashsti": ["CashCashEquivalentsAndShortTermInvestments"],
         "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
         "sti": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
         "ltd": ["LongTermDebtNoncurrent", "LongTermDebt"], "std": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"]}
INST = {"ta", "ca", "cl", "eq", "cash", "cashsti", "sti", "ltd", "std"}


def sec_anual(ticker, cik):
    c = cik.get(ticker.upper().replace("-", "."), cik.get(ticker.upper()))
    if not c:
        return None
    try:
        f = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{c:010d}.json", headers=UA, timeout=60).json()
    except Exception:
        return None
    g = f.get("facts", {}).get("us-gaap", {})
    out = {}
    for k, nombres in SEC_C.items():
        rows = []
        for pri, n in enumerate(nombres):
            for u, lst in g.get(n, {}).get("units", {}).items():
                if u != "USD":
                    continue
                for e in lst:
                    if e.get("form") not in ("10-K", "10-K/A") or e.get("fp") != "FY":
                        continue
                    end = pd.Timestamp(e["end"])
                    if k not in INST:
                        if "start" not in e:
                            continue
                        d = (end - pd.Timestamp(e["start"])).days
                        if d < 330 or d > 400:
                            continue
                    rows.append((end, pd.Timestamp(e["filed"]), float(e["val"]), pri))
        if rows:
            r = pd.DataFrame(rows, columns=["end", "filed", "val", "pri"]).sort_values(["end", "pri", "filed"], ascending=[True, True, False])
            r = r.groupby("end").first()
            out[k] = r["val"]
    if "rev" not in out or "ni" not in out:
        return None
    sh = []
    for u, lst in f.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).items():
        sh += [(pd.Timestamp(e["end"]), float(e["val"])) for e in lst]
    df = pd.DataFrame(out)
    df = df[df.index >= df.index.max() - pd.Timedelta(days=365 * 10 + 30)]
    if sh:
        s = pd.DataFrame(sh, columns=["end", "v"]).groupby("end")["v"].sum().sort_index()
        df["sh"] = [s[s.index <= e + pd.Timedelta(days=120)].iloc[-1] if (s.index <= e + pd.Timedelta(days=120)).any() else np.nan for e in df.index]
    df["fuente"] = "SEC"
    return df


def moneda_real(k, info, conv):
    try:
        inc = k.income_stmt
        raw = num(fila(inc, "Total Revenue").dropna().iloc[0])
        ttm = num(info.get("totalRevenue"))
        f = conv(dt.date.today().year)
        if raw and ttm and f and f != 1:
            if 0.4 < raw / ttm < 2.5:
                return conv
            if 0.4 < raw * f / ttm < 2.5:
                return lambda y: 1.0
    except Exception:
        pass
    return conv


def yahoo_anual(k, conv):
    inc, bal, cf = k.income_stmt, k.balance_sheet, k.cashflow
    if inc is None or inc.empty:
        return None
    rows = {}
    for fy in inc.columns:
        g = lambda df, *n: (lambda s: num(s[fy]) if s is not None and fy in s.index else None)(fila(df, *n))
        rows[pd.Timestamp(fy)] = {"rev": g(inc, "Total Revenue"), "gp": g(inc, "Gross Profit"), "op": g(inc, "Operating Income", "EBIT"),
                                  "ni": g(inc, "Net Income", "Net Income Common Stockholders"), "da": g(cf, "Depreciation And Amortization", "Depreciation Amortization Depletion"),
                                  "int": g(inc, "Interest Expense"), "cfo": g(cf, "Operating Cash Flow"),
                                  "capex": (lambda v: -v if v is not None else None)(g(cf, "Capital Expenditure")),
                                  "ta": g(bal, "Total Assets"), "ca": g(bal, "Current Assets"), "cl": g(bal, "Current Liabilities"),
                                  "eq": g(bal, "Stockholders Equity", "Common Stock Equity"), "cashsti": g(bal, "Cash Cash Equivalents And Short Term Investments"), "cash": g(bal, "Cash And Cash Equivalents"),
                                  "ltd": g(bal, "Long Term Debt"), "std": g(bal, "Current Debt"), "sh": g(bal, "Ordinary Shares Number", "Share Issued")}
    df = pd.DataFrame(rows).T.sort_index().astype(float)
    df = df[df["rev"].notna() | df["ni"].notna()]
    for y in df.index:
        f = conv(y.year)
        if f is None:
            return None
        cols = [c for c in df.columns if c != "sh"]
        df.loc[y, cols] = df.loc[y, cols] / f
    df["fuente"] = "Yahoo"
    return df


def anual_tabla(df):
    t = []
    for e, r in df.iterrows():
        g = lambda k: num(r.get(k))
        rev, op, ni, cfo, capex = g("rev"), g("op"), g("ni"), g("cfo"), g("capex")
        deuda = (g("ltd") or 0) + (g("std") or 0) if (g("ltd") is not None or g("std") is not None) else None
        ebitda = op + (g("da") or 0) if op is not None else None
        fcf = cfo - (capex or 0) if cfo is not None else None
        caja = g("cashsti") if g("cashsti") is not None else ((g("cash") or 0) + (g("sti") or 0) if g("cash") is not None else None)
        t.append({"anio": int(e.year), "cierre": e.strftime("%Y-%m"), "ventas": rev, "bruto": g("gp"), "ebit": op, "ebitda": ebitda, "neto": ni,
                  "fco": cfo, "capex": capex, "fcf": fcf, "deuda": deuda, "caja": caja, "patrimonio": g("eq"), "activo": g("ta"),
                  "intereses": abs(g("int")) if g("int") is not None else None, "acciones": g("sh"), "act_corr": g("ca"), "pas_corr": g("cl"),
                  "m_bruto": g("gp") / rev if rev and g("gp") is not None else None, "m_ebit": op / rev if rev and op is not None else None,
                  "m_neto": ni / rev if rev and ni is not None else None, "m_fcf": fcf / rev if rev and fcf is not None else None, "fuente": r.get("fuente")})
    return t


def trimestral(k, conv):
    q = k.quarterly_income_stmt
    if q is None or q.empty:
        return []
    out = []
    for c in sorted(q.columns)[-8:]:
        g = lambda *n: (lambda s: num(s[c]) if s is not None and c in s.index else None)(fila(q, *n))
        f = conv(pd.Timestamp(c).year) or 1
        rev, op, ni = g("Total Revenue"), g("Operating Income", "EBIT"), g("Net Income", "Net Income Common Stockholders")
        out.append({"trimestre": pd.Timestamp(c).strftime("%Y-%m"), "ventas": rev / f if rev is not None else None,
                    "ebit": op / f if op is not None else None, "neto": ni / f if ni is not None else None,
                    "m_ebit": op / rev if rev and op is not None else None})
    return out


def analistas(k, info, precio, px):
    a = {"recomendacion": info.get("recommendationKey"), "n_analistas": num(info.get("numberOfAnalystOpinions")),
         "objetivo": num(info.get("targetMeanPrice")), "objetivo_alto": num(info.get("targetHighPrice")), "objetivo_bajo": num(info.get("targetLowPrice"))}
    a["potencial"] = a["objetivo"] / precio - 1 if a["objetivo"] and precio else None
    try:
        rc = k.recommendations
        if rc is not None and len(rc):
            r0 = rc.iloc[0]
            a["votos"] = {c: int(r0[c]) for c in ["strongBuy", "buy", "hold", "sell", "strongSell"] if c in rc.columns}
    except Exception:
        pass
    try:
        et = k.eps_trend
        for p in ["0y", "+1y"]:
            if p in et.index:
                r = et.loc[p]
                cur = num(r.get("current"))
                a[f"eps_{p}"] = cur
                for d in ["7daysAgo", "30daysAgo", "90daysAgo"]:
                    v = num(r.get(d))
                    a[f"rev_{p}_{d[:-7]}"] = cur / v - 1 if cur is not None and v else None
    except Exception:
        pass
    try:
        rv = k.eps_revisions
        if "0y" in rv.index:
            r = rv.loc["0y"]
            a["suben_30d"] = num(r.get("upLast30days"))
            a["bajan_30d"] = num(r.get("downLast30days"))
    except Exception:
        pass
    try:
        ge = k.revenue_estimate
        if "+1y" in ge.index:
            a["crec_ventas_1y"] = num(ge.loc["+1y"].get("growth"))
        ee = k.earnings_estimate
        if "+1y" in ee.index:
            a["crec_eps_1y"] = num(ee.loc["+1y"].get("growth"))
    except Exception:
        pass
    sorp = []
    try:
        ed = k.get_earnings_dates(limit=12)
        if ed is not None and len(ed):
            ed = ed.reset_index()
            col = ed.columns[0]
            ahora = pd.Timestamp.now(tz=ed[col].dt.tz)
            fut = ed[ed[col] > ahora]
            if len(fut):
                a["proximo_balance"] = fut[col].min().tz_convert("America/New_York").strftime("%Y-%m-%d")
            ed = ed[ed[col] <= ahora].head(4)
            for _, r in ed.iterrows():
                f = r[col].tz_convert("America/New_York")
                d0 = f.tz_localize(None).normalize()
                reac = reac5 = None
                if px is not None and len(px):
                    antes = px[px.index < d0] if f.hour < 12 else px[px.index <= d0]
                    despues = px[px.index > antes.index[-1]] if len(antes) else px.iloc[0:0]
                    if len(antes) and len(despues):
                        reac = float(despues.iloc[0] / antes.iloc[-1] - 1)
                        reac5 = float(despues.iloc[min(4, len(despues) - 1)] / antes.iloc[-1] - 1)
                sorp.append({"fecha": f.strftime("%Y-%m-%d"), "eps": num(r.get("Reported EPS")), "estimado": num(r.get("EPS Estimate")),
                             "sorpresa": (num(r.get("Surprise(%)")) or 0) / 100 if num(r.get("Surprise(%)")) is not None else None, "reaccion": reac, "reaccion_5d": reac5})
    except Exception as e:
        print(f"balances {k.ticker}: {e}", file=sys.stderr)
    a["sorpresas"] = sorp
    return a


def precio_bloque(px, spy, etf):
    s = px.dropna()
    if len(s) < 60:
        return {}
    u = float(s.iloc[-1])
    r = lambda d: float(u / s.iloc[-1 - d] - 1) if len(s) > d else None
    ma200 = float(s.tail(200).mean()) if len(s) >= 200 else None
    w = s.resample("W-FRI").last().dropna()
    sw = spy.reindex(w.index, method="pad")
    rw, rs = w.pct_change().dropna().tail(104), sw.pct_change().dropna().tail(104)
    rs = rs.reindex(rw.index)
    beta = float(np.cov(rw, rs)[0, 1] / np.var(rs)) if len(rw) > 30 else None
    s5 = s[s.index >= s.index[-1] - pd.Timedelta(days=365 * 5)]
    ys = s[s.index.year == s.index[-1].year]
    w5 = w[w.index >= w.index[-1] - pd.Timedelta(days=365 * 5)]
    serie = {"f": [d.strftime("%Y-%m-%d") for d in w5.index], "p": [round(float(x), 4) for x in w5],
             "spy": [round(float(x), 4) for x in spy.reindex(w5.index, method="pad")],
             "etf": [round(float(x), 4) for x in etf.reindex(w5.index, method="pad")] if etf is not None else None}
    r12 = r(252)
    return {"precio": u, "r1m": r(21), "r3m": r(63), "r6m": r(126), "r12m": r12, "ytd": float(u / ys.iloc[0] - 1) if len(ys) else None,
            "r12m_spy": float(spy.iloc[-1] / spy.iloc[-253] - 1) if len(spy) > 253 else None,
            "mom12_1": float(s.iloc[-22] / s.iloc[-253] - 1) if len(s) > 253 else None,
            "vs_ma200": u / ma200 - 1 if ma200 else None, "max52": float(s.tail(252).max()), "vs_max52": u / float(s.tail(252).max()) - 1,
            "vol1y": float(s.pct_change().tail(252).std() * math.sqrt(252)), "beta": beta,
            "caida_max_5y": float((s5 / s5.cummax() - 1).min()), "serie": serie}


def posicion(k, info):
    p = {"short_float": num(info.get("shortPercentOfFloat")), "short_ratio": num(info.get("shortRatio")),
         "short_cambio": (num(info.get("sharesShort")) / num(info.get("sharesShortPriorMonth")) - 1) if num(info.get("sharesShort")) and num(info.get("sharesShortPriorMonth")) else None,
         "institucionales": num(info.get("heldPercentInstitutions")), "insiders": num(info.get("heldPercentInsiders"))}
    try:
        ip = k.insider_purchases
        if ip is not None and len(ip):
            ip = ip.set_index(ip.columns[0])
            g = lambda fila_, col: num(ip.loc[fila_, col]) if fila_ in ip.index and col in ip.columns else None
            p["dir_compras"] = g("Purchases", "Trans")
            p["dir_ventas"] = g("Sales", "Trans")
            p["dir_neto_acciones"] = g("Net Shares Purchased (Sold)", "Shares")
            p["dir_neto_pct"] = g("% Net Shares Purchased (Sold)", "Shares")
    except Exception:
        pass
    return p


def ultimo_balance(a, tr):
    s = (a.get("sorpresas") or [None])[0]
    if not s:
        return None
    q = tr[-1] if tr else {}
    q4 = tr[-5] if len(tr) >= 5 else None
    yoy = q["ventas"] / q4["ventas"] - 1 if q4 and q4.get("ventas") and q.get("ventas") else None
    sp = s.get("sorpresa")
    veredicto = None if sp is None else "Superó lo esperado" if sp > 0.02 else "No alcanzó lo esperado" if sp < -0.02 else "En línea con lo esperado"
    f = lambda x, d=2: f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    pc = lambda x, d=1: ("+" if x > 0 else "") + f(x * 100, d) + "%"
    partes = []
    if s.get("eps") is not None and s.get("estimado") is not None:
        partes.append(f"Ganó {f(s['eps'])} por acción contra {f(s['estimado'])} esperado" + (f": {veredicto.lower()} por {f(abs(sp) * 100, 1)}%." if sp is not None and veredicto != "En línea con lo esperado" else ", en línea con lo esperado."))
    if yoy is not None:
        t = f"Ventas del trimestre {pc(yoy, 0)} contra el mismo trimestre del año anterior"
        if q.get("m_ebit") is not None and q4 and q4.get("m_ebit") is not None:
            t += f"; margen EBIT {f(q['m_ebit'] * 100, 0)}% ({f(q4['m_ebit'] * 100, 0)}% un año antes)"
        partes.append(t + ".")
    if s.get("reaccion") is not None:
        r = s["reaccion"]
        partes.append(f"Al día siguiente la acción {'subió' if r > 0 else 'cayó'} {f(abs(r) * 100, 1)}%" + (", a pesar del resultado." if (sp or 0) > 0.02 and r < -0.02 or (sp or 0) < -0.02 and r > 0.02 else "."))
    return {"fecha": s["fecha"], "eps": s.get("eps"), "estimado": s.get("estimado"), "sorpresa": sp, "veredicto": veredicto,
            "reaccion": s.get("reaccion"), "reaccion_5d": s.get("reaccion_5d"), "trimestre": q.get("trimestre"), "ventas": q.get("ventas"), "ventas_yoy": yoy,
            "m_ebit": q.get("m_ebit"), "m_ebit_prev": q4.get("m_ebit") if q4 else None, "comentario": " ".join(partes)}


def logo(t, b, info):
    d = DATA / "logos"
    d.mkdir(exist_ok=True)
    arch = d / f"{b}.png"
    if arch.exists() and arch.stat().st_size > 200:
        return
    fuentes = [f"https://financialmodelingprep.com/image-stock/{t.split('.')[0]}.png"]
    web = (info.get("website") or "").replace("https://", "").replace("http://", "").strip("/").split("/")[0]
    if web:
        fuentes.append(f"https://www.google.com/s2/favicons?domain={web}&sz=128")
    for url in fuentes:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if r.ok and r.headers.get("content-type", "").startswith("image") and len(r.content) > 500:
                arch.write_bytes(r.content)
                return
        except Exception:
            pass


def es_banco(info):
    ind = (info.get("industry") or "").lower()
    return info.get("sector") == "Financial Services" and any(x in ind for x in ["bank", "capital markets", "insurance", "credit services"]) and "credit services" not in ind


def notas_absolutas(ficha):
    an = [x for x in ficha.get("anual", []) if x.get("ventas")]
    if not an:
        return {}
    u = an[-1]
    banco = ficha["tipo"] == "banco"
    det = {}
    if banco:
        roe = u["neto"] / u["patrimonio"] if u.get("neto") is not None and u.get("patrimonio") else None
        roa = u["neto"] / u["activo"] if u.get("neto") is not None and u.get("activo") else None
        eqa = u["patrimonio"] / u["activo"] if u.get("patrimonio") and u.get("activo") else None
        det["calidad"] = {"ROE": roe, "ROA": roa}
        cal = media([lin(roe, [(0, 1), (0.05, 3), (0.10, 6), (0.15, 8), (0.20, 10)]), lin(roa, [(0, 1), (0.005, 4), (0.01, 6), (0.015, 8), (0.02, 10)])])
        det["solidez"] = {"Patrimonio/activo": eqa}
        sol = media([lin(eqa, [(0.04, 2), (0.07, 5), (0.10, 7), (0.13, 9), (0.16, 10)])])
    else:
        capital = (u.get("patrimonio") or 0) + (u.get("deuda") or 0) - (u.get("caja") or 0)
        roic = u["ebit"] * 0.79 / capital if u.get("ebit") is not None and capital > 0 else None
        ms = [x["m_ebit"] for x in an[-5:] if x.get("m_ebit") is not None]
        estab = float(np.std(ms)) if len(ms) >= 3 else None
        det["calidad"] = {"ROIC": roic, "Margen EBIT": u.get("m_ebit"), "Margen flujo libre": u.get("m_fcf"), "Variación del margen (5 años)": estab}
        cal = media([lin(roic, [(0, 1), (0.05, 4), (0.10, 6), (0.15, 7.5), (0.25, 9), (0.40, 10)]),
                     lin(u.get("m_ebit"), [(-0.05, 1), (0, 2), (0.10, 5), (0.20, 7), (0.30, 9), (0.40, 10)]),
                     lin(u.get("m_fcf"), [(-0.05, 1), (0, 3), (0.05, 5), (0.15, 7.5), (0.25, 9.5)]),
                     lin(estab, [(0.01, 10), (0.03, 8), (0.06, 5), (0.10, 2)]) if estab is not None else None])
        dn = (u.get("deuda") or 0) - (u.get("caja") or 0)
        lev = dn / u["ebitda"] if u.get("ebitda") and u["ebitda"] > 0 else (99 if dn > 0 else -1)
        cob = u["ebit"] / u["intereses"] if u.get("ebit") is not None and u.get("intereses") else None
        cr = u["act_corr"] / u["pas_corr"] if u.get("act_corr") and u.get("pas_corr") else None
        det["solidez"] = {"Deuda neta / EBITDA": lev if lev != 99 else None, "Cobertura de intereses": cob, "Liquidez corriente": cr}
        sol = media([lin(lev, [(-0.01, 10), (0.5, 9), (1, 8), (2, 6), (3, 4), (4, 2.5), (6, 1)]),
                     lin(cob, [(1, 1), (2, 3), (5, 6), (10, 8), (20, 10)]) if cob is not None else (9 if dn <= 0 else None),
                     lin(cr, [(0.7, 2), (1, 5), (1.5, 7), (2, 9)])])
    cagr = None
    if len(an) >= 4 and an[-4]["ventas"] and an[-4]["ventas"] > 0 and u["ventas"] > 0:
        cagr = (u["ventas"] / an[-4]["ventas"]) ** (1 / 3) - 1
    a = ficha.get("analistas", {})
    det["crecimiento"] = {"Ventas, 3 años (anual)": cagr, "Ventas esperadas próximo año": a.get("crec_ventas_1y"), "Ganancia esperada próximo año": a.get("crec_eps_1y")}
    pts = [(-0.10, 1), (0, 3), (0.05, 5), (0.10, 7), (0.20, 9), (0.30, 10)]
    cre = media([lin(cagr, pts), lin(a.get("crec_ventas_1y"), pts), lin(a.get("crec_eps_1y"), [(-0.10, 1), (0, 3), (0.08, 5), (0.15, 7), (0.25, 9), (0.40, 10)])])
    return {"calidad": cal, "solidez": sol, "crecimiento": cre, "detalle": det}


def main():
    solo = [x.strip() for x in os.environ.get("FICHAS_SOLO", "").split(",") if x.strip()]
    u = pd.read_csv(DATA / "universo_fichas.csv")
    datos = json.loads((DATA / "datos.json").read_text())
    ccl = datos.get("ccl")
    by_t = {x["ticker"]: x for x in datos["tickers"]}
    try:
        vivo = {x["symbol"]: x for x in requests.get("https://data912.com/live/arg_cedears", timeout=30).json()}
    except Exception:
        vivo = {}
    tick = list(u["ticker_us"])
    px = yf.download(tick + ["SPY"] + list(ETF_SECTOR.values()), period="6y", auto_adjust=True, progress=False, threads=True)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    spy = px["SPY"].dropna()
    try:
        mapa = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
        cik = {v["ticker"].upper(): int(v["cik_str"]) for v in mapa.values()}
    except Exception:
        cik = {}
    OUT.mkdir(exist_ok=True)
    previo = {}
    try:
        previo = {x["ticker"]: x for x in json.loads((DATA / "empresas.json").read_text())["empresas"]}
    except Exception:
        pass
    lista, fichas = [], {}
    for _, r in u.iterrows():
        t, b = r["ticker_us"], r["byma"]
        k = yf.Ticker(t)
        try:
            info = k.info or {}
        except Exception:
            info = {}
        try:
            logo(t, b, info)
        except Exception:
            pass
        s = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
        etf_t = ETF_SECTOR.get(info.get("sector"))
        pb = precio_bloque(s, spy, px[etf_t].dropna() if etf_t in px.columns else None)
        x = by_t.get(t, {})
        ratio = [float(v) for v in str(r["ratio"]).split(":")] if ":" in str(r["ratio"]) else None
        px_ars = num(x.get("px_ars")) or num((vivo.get(b) or {}).get("c"))
        precio = pb.get("precio")
        premio = num(x.get("premio"))
        if premio is None and px_ars and precio and ratio and ccl and not t.endswith(".KS"):
            p = px_ars * ratio[0] / ratio[1] / precio / ccl - 1
            premio = p if abs(p) <= 0.3 else None
        mcap = num(info.get("marketCap"))
        fxc = None
        if info.get("currency") and info.get("currency") != "USD":
            fxc = fx_anual(info["currency"])(dt.date.today().year)
            if fxc:
                mcap = mcap / fxc if mcap else None
                for c_ in ["precio", "max52"]:
                    if pb.get(c_):
                        pb[c_] = pb[c_] / fxc
        usd = info.get("financialCurrency") in (None, "USD")
        tipo = "cripto" if b in CRIPTO else "banco" if es_banco(info) or b in {"AXP"} else "inmobiliaria" if info.get("sector") == "Real Estate" else "general"
        val = {"pe": num(info.get("trailingPE")), "pe_fwd": num(info.get("forwardPE")), "ev_ebitda": num(info.get("enterpriseToEbitda")) if usd else None,
               "ps": num(info.get("priceToSalesTrailing12Months")) if usd else None, "pb": num(info.get("priceToBook")) if usd else None,
               "fcf_yield": num(info.get("freeCashflow")) / mcap if num(info.get("freeCashflow")) and mcap and usd else None,
               "div_yield": (num(info.get("dividendYield")) or 0) / 100 if num(info.get("dividendYield")) is not None else None}
        e = {"puesto": int(r["puesto"]), "byma": b, "ticker": t, "nombre": r["nombre"], "ratio": r["ratio"],
             "sector": SECTOR_ES.get(info.get("sector"), info.get("sector")), "sector_en": info.get("sector"), "industria": info.get("industry"),
             "pais": info.get("country"), "moneda_reporte": info.get("financialCurrency"), "mcap": mcap, "tipo": tipo,
             "precio": pb.get("precio"), "px_ars": px_ars, "premio": premio, "monto_usd": num(r["monto_usd_dia"]), "liquidez": liquidez(num(r["monto_usd_dia"])),
             "r1m": pb.get("r1m"), "r3m": pb.get("r3m"), "r12m": pb.get("r12m"), "vs_ma200": pb.get("vs_ma200"), "mom12_1": pb.get("mom12_1"),
             "rel6m": (pb["r6m"] - float(spy.iloc[-1] / spy.iloc[-127] - 1)) if pb.get("r6m") is not None else None, "val": val}
        if not solo or t in solo or b in solo:
            try:
                conv = moneda_real(k, info, fx_anual(info.get("financialCurrency")))
                df = sec_anual(t, cik) if info.get("financialCurrency") in (None, "USD") and info.get("country") == "United States" else None
                if df is None:
                    df = yahoo_anual(k, conv)
                an = anual_tabla(df) if df is not None else []
                if df is not None and df["fuente"].iloc[0] == "SEC" and an and an[-1]["anio"] < dt.date.today().year - 1:
                    y = yahoo_anual(k, conv)
                    if y is not None:
                        an = [x for x in an if x["anio"] < min(z["anio"] for z in anual_tabla(y))] + anual_tabla(y)
                ana = analistas(k, info, precio, s)
                if fxc:
                    for c_ in ["objetivo", "objetivo_alto", "objetivo_bajo"]:
                        if ana.get(c_):
                            ana[c_] = ana[c_] / fxc
                hist_pe = []
                for x_ in an:
                    if not (x_.get("neto") and x_["neto"] > 0):
                        continue
                    d = pd.Timestamp(x_["cierre"]) + pd.offsets.MonthEnd(0)
                    p_ = s[s.index <= d]
                    if not len(p_) or (d - p_.index[-1]).days >= 10:
                        continue
                    if mcap and precio:
                        hist_pe.append({"anio": x_["anio"], "pe": float(p_.iloc[-1]) / precio * mcap / x_["neto"]})
                ficha = {**{k_: v for k_, v in e.items()}, "descripcion": info.get("longBusinessSummary"), "web": info.get("website"),
                         "empleados": num(info.get("fullTimeEmployees")), "precio_bloque": pb, "anual": an, "trimestral": trimestral(k, conv),
                         "analistas": ana, "pe_historia": hist_pe, "posicion": posicion(k, info)}
                ficha["ultimo_balance"] = ultimo_balance(ana, ficha["trimestral"])
                tq = ficha["trimestral"][-4:]
                u_ = an[-1] if an else {}
                ventas_ttm = sum(q["ventas"] for q in tq) if len(tq) == 4 and all(q.get("ventas") for q in tq) else u_.get("ventas")
                neto_ttm = sum(q["neto"] for q in tq) if len(tq) == 4 and all(q.get("neto") is not None for q in tq) else u_.get("neto")
                if mcap:
                    if ventas_ttm:
                        val["ps"] = mcap / ventas_ttm
                    if u_.get("fcf") is not None:
                        val["fcf_yield"] = u_["fcf"] / mcap
                    if u_.get("ebitda") and u_["ebitda"] > 0 and tipo != "banco":
                        val["ev_ebitda"] = (mcap + (u_.get("deuda") or 0) - (u_.get("caja") or 0)) / u_["ebitda"]
                    if not usd and neto_ttm and neto_ttm > 0:
                        val["pe"] = mcap / neto_ttm
                if tipo == "banco":
                    val["ev_ebitda"] = None
                    val["fcf_yield"] = None
                e["val"] = val
                ficha["val"] = val
                ficha["notas_abs"] = notas_absolutas(ficha)
                e["proximo_balance"] = ana.get("proximo_balance")
                e["rev90"] = ana.get("rev_0y_90")
                e["sorpresa_media"] = media([z["sorpresa"] for z in ana.get("sorpresas", [])])
                e["arriba_abajo"] = ((ana.get("suben_30d") or 0) - (ana.get("bajan_30d") or 0)) / max((ana.get("suben_30d") or 0) + (ana.get("bajan_30d") or 0), 1)
                e["notas_abs"] = {k_: v for k_, v in ficha["notas_abs"].items() if k_ != "detalle"}
                e["ficha"] = True
                fichas[t] = ficha
            except Exception as ex:
                print(f"ficha {t}: {type(ex).__name__}: {ex}", file=sys.stderr)
        elif (OUT / f"{t}.json").exists():
            try:
                fv = json.loads((OUT / f"{t}.json").read_text())
                e["notas_abs"] = {c: (fv.get("notas") or {}).get(c) for c in ["calidad", "solidez", "crecimiento"]}
                for c in ["proximo_balance", "rev90", "sorpresa_media", "arriba_abajo"]:
                    e[c] = (previo.get(t) or {}).get(c)
                e["proximo_balance"] = e["proximo_balance"] or (fv.get("analistas") or {}).get("proximo_balance")
                e["ficha"] = True
            except Exception:
                pass
        lista.append(e)
        time.sleep(0.4)
    salida(lista, fichas, ccl)


def salida(lista, fichas, ccl):
    L = pd.DataFrame(lista).set_index("ticker")
    for c in ["rev90", "arriba_abajo", "sorpresa_media", "ficha", "notas_abs", "proximo_balance"]:
        if c not in L.columns:
            L[c] = None
    rk = lambda s_: s_.rank(pct=True)
    tend = (rk(L["mom12_1"]) + rk(L["vs_ma200"]) + rk(L["rel6m"])) / 3
    V = pd.DataFrame({t: x for t, x in L["val"].items()}).T.apply(pd.to_numeric, errors="coerce")
    V["ey"] = 1 / V["pe"].where(V["pe"] > 0)
    V["ee"] = 1 / V["ev_ebitda"].where(V["ev_ebitda"] > 0)
    V["sector"] = L["sector"]
    V["tipo"] = L["tipo"]
    tam = V.groupby("sector")["ey"].transform("count")
    vr = []
    for c in ["ey", "ee", "fcf_yield"]:
        por_sec = V.groupby("sector")[c].rank(pct=True)
        vr.append(por_sec.where(tam >= 5, V[c].rank(pct=True)))
    valn = pd.concat(vr, axis=1).mean(axis=1)
    an_n = pd.concat([rk(pd.to_numeric(L["rev90"], errors="coerce")), rk(pd.to_numeric(L["arriba_abajo"], errors="coerce")),
                      rk(pd.to_numeric(L["sorpresa_media"], errors="coerce"))], axis=1).mean(axis=1)
    sal = []
    for t, x in L.iterrows():
        d = x.to_dict()
        d["ticker"] = t
        na = d.get("notas_abs") if isinstance(d.get("notas_abs"), dict) else {}
        nt = {"calidad": na.get("calidad"), "solidez": na.get("solidez"), "crecimiento": na.get("crecimiento"),
              "valuacion": None if d["tipo"] == "cripto" or pd.isna(valn.get(t)) else round(1 + 9 * float(valn[t]), 1),
              "tendencia": None if pd.isna(tend.get(t)) else round(1 + 9 * float(tend[t]), 1),
              "analistas": None if not d.get("ficha") or pd.isna(an_n.get(t)) else round(1 + 9 * float(an_n[t]), 1)}
        nt["negocio"] = media([nt["calidad"], nt["solidez"]])
        pes = {"calidad": 25, "crecimiento": 20, "solidez": 15, "valuacion": 15, "analistas": 15, "tendencia": 10}
        disp = {k_: nt[k_] for k_ in pes if nt.get(k_) is not None}
        nt["general"] = round(sum(nt[k_] * pes[k_] for k_ in disp) / sum(pes[k_] for k_ in disp), 1) if len(disp) >= 3 else None
        d["notas"] = nt
        d.pop("notas_abs", None)
        d = {k_: (None if isinstance(v, float) and not math.isfinite(v) else v) for k_, v in d.items()}
        sal.append(d)
        if t in fichas:
            f = fichas[t]
            f["notas"] = nt
            f["notas_detalle"] = f["notas_abs"].get("detalle", {})
            f["notas_detalle"]["valuacion"] = {"P/E": d["val"]["pe"], "EV/EBITDA": d["val"]["ev_ebitda"], "Flujo libre / precio": d["val"]["fcf_yield"],
                                              "Comparado contra": f"su sector ({int(tam.get(t))} empresas)" if tam.get(t, 0) >= 5 else "todas las 125"}
            f["notas_detalle"]["tendencia"] = {"Momentum 12-1": d["mom12_1"], "Distancia a la MA200": d["vs_ma200"], "Contra el SPY, 6 meses": d["rel6m"]}
            f["notas_detalle"]["analistas"] = {"Revisión de ganancias, 90 días": d.get("rev90"), "Revisiones netas, 30 días": d.get("arriba_abajo"), "Sorpresa promedio, 4 balances": d.get("sorpresa_media")}
            f.pop("notas_abs", None)
            secv = V[V["sector"] == f["sector"]]
            f["pares"] = {c: num(secv[c].median()) for c in ["pe", "pe_fwd", "ev_ebitda", "ps", "fcf_yield"]}
            f["pares"]["n"] = int(len(secv))
            f["actualizado"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes")
            (OUT / f"{t}.json").write_text(json.dumps(f, ensure_ascii=False, default=lambda o: None, separators=(",", ":")).replace("NaN", "null"))
    (DATA / "empresas.json").write_text(json.dumps({"actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"), "ccl": ccl,
                                                    "empresas": sal}, ensure_ascii=False, default=lambda o: None, separators=(",", ":")).replace("NaN", "null"))
    print(f"empresas: {len(sal)}; fichas: {len(fichas)}")


if __name__ == "__main__":
    main()
