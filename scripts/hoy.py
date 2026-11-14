import re
import sys
import json
import time
import datetime as dt
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ciclos import UA

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
SALIDA = D / "hoy.json"
AHORA = dt.datetime.now(dt.timezone.utc)
HOY = AHORA.date()
WEB = {"User-Agent": "Mozilla/5.0"}
MERCADO = {"SPY": "S&P 500", "QQQ": "Nasdaq 100", "^VIX": "VIX", "^TNX": "Tasa 10 años", "CL=F": "Petróleo", "BTC-USD": "Bitcoin", "DX-Y.NYB": "Dólar global", "GC=F": "Oro"}
FOMC = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]
MACRO = [
    ("Consumer Price Index", "Inflación (CPI)", "10:30", 3),
    ("Employment Situation", "Empleo (nóminas no agrícolas)", "10:30", 3),
    ("Personal Income and Outlays", "Inflación PCE e ingresos", "10:30", 2),
    ("Gross Domestic Product", "PBI", "10:30", 2),
    ("Producer Price Index", "Inflación mayorista (PPI)", "10:30", 2),
    ("Advance Monthly Sales for Retail", "Ventas minoristas", "10:30", 2),
    ("Job Openings and Labor Turnover", "Vacantes laborales (JOLTS)", "12:00", 1),
    ("Unemployment Insurance Weekly Claims", "Pedidos de desempleo", "10:30", 1),
    ("Surveys of Consumers", "Confianza del consumidor (Michigan)", "12:00", 1),
    ("Industrial Production", "Producción industrial", "11:15", 1),
    ("New Residential Construction", "Construcción de viviendas", "10:30", 1),
]
ITEMS = {"1.01": "acuerdo importante", "1.02": "fin de un acuerdo importante", "1.03": "quiebra", "1.05": "incidente de ciberseguridad",
         "2.01": "compra o venta de activos", "2.02": "resultados", "2.03": "nueva deuda", "2.05": "reestructuración", "2.06": "desvalorización de activos",
         "3.01": "aviso de la bolsa", "3.02": "venta de acciones", "4.02": "corrección de balances", "5.01": "cambio de control",
         "5.02": "cambio de directivos", "8.01": "otro hecho relevante"}
BASURA = re.compile(r"(stocks? to (buy|own|watch|hold)|should you (buy|sell)|better buy|buy now|\b\d+ (best |top |undervalued |growth |ai )?(stocks|reasons|things)\b|millionaire|prediction|forever|zacks rank|top picks|is it time to|here's why|what to know|could soar|no-brainer|dividend stock|trending stock|undervalued|fully valued|valuation look|investment narrative|earnings call transcript|price target|should investors|is .* a buy)", re.I)
FUENTES_FUERA = {"Simply Wall St.", "Motley Fool", "Zacks", "MarketBeat", "Insider Monkey", "24/7 Wall St.", "Benzinga Insights", "TipRanks Auto-Generated Newsdesk", "ACCESS Newswire", "GlobeNewswire", "PR Newswire", "Business Wire"}
MERCADO_KW = re.compile(r"(stock|market|fed\b|rate|yield|treasury|inflation|job|payroll|tariff|econom|earnings|dow|s&p|nasdaq|oil|bitcoin|dollar|wall street|recession|gdp|bond|ai\b|chip)", re.I)
ALIAS = {"GOOGL": ["google", "alphabet", "youtube", "waymo", "gemini"], "META": ["meta", "facebook", "instagram", "zuckerberg"], "TSM": ["tsmc", "taiwan semiconductor"],
         "BRK-B": ["berkshire", "buffett"], "MSTR": ["microstrategy", "strategy", "saylor"], "DIS": ["disney"], "MSFT": ["microsoft", "azure", "openai"],
         "AMZN": ["amazon", "aws"], "AAPL": ["apple", "iphone"], "NVDA": ["nvidia", "huang", "kress"], "TSLA": ["tesla", "musk"], "AVGO": ["broadcom"],
         "ORCL": ["oracle"], "CRWV": ["coreweave"], "NBIS": ["nebius"], "IREN": ["iren"], "MU": ["micron"], "INTC": ["intel"], "AMD": ["amd"],
         "XOM": ["exxon"], "CVX": ["chevron"], "JPM": ["jpmorgan", "jp morgan", "dimon"], "GS": ["goldman"], "MELI": ["mercadolibre", "mercado libre"],
         "BABA": ["alibaba"], "PBR": ["petrobras"], "VIST": ["vista"], "SPOT": ["spotify"], "NFLX": ["netflix"], "LLY": ["lilly"], "NVO": ["novo nordisk", "wegovy", "ozempic"],
         "UNH": ["unitedhealth"], "BA": ["boeing"], "GE": ["ge aerospace"], "UAL": ["united airlines"], "DAL": ["delta"], "AAL": ["american airlines"], "MA": ["mastercard"], "V": ["visa"],
         "PYPL": ["paypal", "venmo"], "HOOD": ["robinhood"], "COIN": ["coinbase"], "ANET": ["arista"], "ADBE": ["adobe"], "CRM": ["salesforce"], "PLTR": ["palantir"], "SNDK": ["sandisk"], "WDC": ["western digital"]}
GRUPOS = {"chips": "ia", "fabricacion": "ia", "memorias": "ia", "infraia": "ia", "bigtech": "ia", "software": "ia", "hardware": "ia",
          "electrica": "energia", "petroleo": "energia", "mineria": "energia",
          "bancos": "finanzas", "pagos": "finanzas", "cripto": "finanzas"}


def log(*a):
    print(*a, file=sys.stderr)


def leer(nombre, defecto=None):
    try:
        return json.loads((D / nombre).read_text())
    except Exception:
        return defecto


def num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except Exception:
        return None


TRAD = {"ok": 0, "mal": 0}
GLOSA = [(r"\bOPI [Aa]ntr[óo]pica\b", "salida a bolsa de Anthropic"), (r"\b[Aa]ntr[óo]pic[oa]s?\b", "Anthropic"), (r"\bOPI\b", "salida a bolsa"), (r"\bAlfabeto\b", "Alphabet"), (r"\bOr[áa]culo\b", "Oracle"), (r"\bManzana\b", "Apple"), (r"\bAmazonas\b", "Amazon")]


def traducir(textos):
    textos = [t.replace("\n", " ").strip() for t in textos]
    if not textos:
        return []
    out = list(textos)
    lotes, lote, largo = [], [], 0
    for k, t in enumerate(textos):
        if largo + len(t) > 1500 and lote:
            lotes.append(lote)
            lote, largo = [], 0
        lote.append(k)
        largo += len(t) + 1
    if lote:
        lotes.append(lote)
    fallas = 0
    for lote in lotes:
        if fallas >= 3:
            break
        q = "\n".join(textos[k] for k in lote)
        try:
            r = requests.post("https://translate.googleapis.com/translate_a/single", params={"client": "gtx", "sl": "en", "tl": "es", "dt": "t"},
                              data={"q": q}, headers=WEB, timeout=30)
            r.raise_for_status()
            js = r.json()
            tr = "".join(seg[0] for seg in js[0] if seg and seg[0])
            partes = tr.split("\n")
            if len(partes) == len(lote):
                for k, v in zip(lote, partes):
                    v = v.strip()
                    for pat, rep in GLOSA:
                        v = re.sub(pat, rep, v)
                    out[k] = v or textos[k]
                TRAD["ok"] += len(lote)
            else:
                TRAD["mal"] += len(lote)
                log("traduccion partes", len(partes), len(lote))
        except Exception as e:
            fallas += 1
            TRAD["mal"] += len(lote)
            log("traduccion", str(e)[:120])
        time.sleep(0.6)
    log("traduccion", TRAD)
    return out


def universo():
    emp = leer("empresas.json", {}).get("empresas", [])
    bl = leer("bloques.json", {}).get("bloques", [])
    de = {}
    for b in bl:
        for e in b["empresas"]:
            de[e] = b
    ideas = pd.read_csv(D / "ideas.csv", dtype=str).fillna("")
    por_tk = {}
    for x in emp:
        b = de.get(x["byma"])
        por_tk[x["ticker"]] = {"ticker": x["ticker"], "byma": x["byma"], "nombre": x.get("nombre"), "bloque": b["nombre"] if b else None,
                               "bloque_id": b["id"] if b else None, "pais": x.get("pais"), "mcap": x.get("mcap")}
    for t in ideas["ticker"]:
        por_tk.setdefault(t, {"ticker": t, "byma": t, "nombre": None, "bloque": None, "bloque_id": None, "pais": None, "mcap": None})
    return por_tk, ideas


def precios(tks):
    px = yf.download(sorted(set(tks)), period="14mo", auto_adjust=True, progress=False, threads=True)["Close"]
    px.index = pd.to_datetime(px.index).tz_localize(None)
    return px


def mercado(px):
    out = {}
    for t, n in MERCADO.items():
        if t not in px.columns:
            continue
        s = px[t].dropna()
        if len(s) < 7:
            continue
        if t == "^TNX":
            out[t] = {"nombre": n, "valor": float(s.iloc[-1]), "dia": float(s.iloc[-1] - s.iloc[-2]), "semana": float(s.iloc[-1] - s.iloc[-6]), "pts": True}
        else:
            out[t] = {"nombre": n, "valor": float(s.iloc[-1]), "dia": float(s.iloc[-1] / s.iloc[-2] - 1), "semana": float(s.iloc[-1] / s.iloc[-6] - 1)}
    spy = px["SPY"].dropna()
    return {"fecha": str(spy.index[-1].date()), "datos": out}


def proximos_balances(fichas):
    out = {}
    for b, f in fichas.items():
        pb = (f.get("analistas") or {}).get("proximo_balance")
        if pb:
            out[f["ticker"]] = pb[:10]
    return out


def hora_balance(t):
    try:
        ed = yf.Ticker(t).get_earnings_dates(limit=8)
        if ed is None or not len(ed):
            return None, None
        idx = pd.DatetimeIndex(ed.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York")
        fut = [d for d in idx if d.date() >= HOY - dt.timedelta(days=1)]
        if not fut:
            return None, None
        d = min(fut)
        h = d.hour + d.minute / 60
        cuando = "antes de la apertura" if h < 9.5 else "después del cierre" if h >= 16 else "durante la rueda" if h > 0 else None
        return str(d.date()), cuando
    except Exception as e:
        log("hora", t, e)
        return None, None


def ideas_calc(ideas, px, uni, fichas, prox):
    spy = px["SPY"].dropna()
    out = []
    for r in ideas.itertuples():
        t = r.ticker
        x = {"ticker": t, "tipo": r.tipo, "desde": r.desde, "comentario": r.comentario, "byma": uni.get(t, {}).get("byma", t),
             "nombre": uni.get(t, {}).get("nombre"), "bloque": uni.get(t, {}).get("bloque"), "alertas": []}
        if t not in px.columns or px[t].dropna().empty:
            x["sin_precio"] = True
            out.append(x)
            continue
        s = px[t].dropna()
        d0 = pd.Timestamp(r.desde)
        base = s[s.index <= d0]
        b0 = base.index[-1] if len(base) else s.index[0]
        p0, p = float(s[b0]), float(s.iloc[-1])
        sb = spy[spy.index <= b0]
        q0 = float(sb.iloc[-1]) if len(sb) else float(spy.iloc[0])
        q = float(spy.iloc[-1])
        x["nueva"] = b0 == s.index[-1]
        x["precio"], x["precio_desde"], x["fecha_base"] = p, p0, str(b0.date())
        x["ret"] = p / p0 - 1
        x["ret_spy"] = q / q0 - 1
        x["rel"] = (1 + x["ret"]) / (1 + x["ret_spy"]) - 1
        x["dia"] = float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) > 1 else None
        rr = (s[s.index >= b0] / s[b0]) / (spy.reindex(s.index).ffill()[s.index >= b0] / q0)
        x["serie"] = [round(float(v) - 1, 4) for v in rr.values][-130:]
        ma = s.rolling(200).mean()
        if not np.isnan(ma.iloc[-1]):
            arriba = (s > ma).iloc[-11:]
            x["sobre_ma200"] = bool(arriba.iloc[-1])
            x["dist_ma200"] = float(s.iloc[-1] / ma.iloc[-1] - 1)
            if arriba.iloc[:-1].nunique() > 1 or arriba.iloc[-1] != arriba.iloc[0]:
                x["alertas"].append(["cruce", ("▲ Cruzó hacia arriba" if arriba.iloc[-1] else "▼ Cruzó hacia abajo") + " su promedio de 200 días"])
            elif r.tipo == "OW" and not arriba.iloc[-1]:
                x["alertas"].append(["ma", f"▼ Debajo de su promedio de 200 días ({x['dist_ma200'] * 100:.0f}%)".replace(".", ",")])
            elif r.tipo == "UW" and arriba.iloc[-1]:
                x["alertas"].append(["ma", f"▲ Arriba de su promedio de 200 días (+{x['dist_ma200'] * 100:.0f}%)".replace(".", ",")])
        y = s.iloc[-252:]
        if p >= y.max() * 0.98:
            x["alertas"].append(["max", "Cerca de su máximo de 52 semanas"])
        elif p <= y.min() * 1.02:
            x["alertas"].append(["min", "Cerca de su mínimo de 52 semanas"])
        pb = prox.get(t)
        if pb:
            dd = (pd.Timestamp(pb).date() - HOY).days
            if 0 <= dd <= 21:
                x["alertas"].append(["balance", f"● Balance el {pb[8:10]}/{pb[5:7]}" + (" (hoy)" if dd == 0 else f" (en {dd} días)")])
        out.append(x)
    return out


def balances(fichas, prox, uni, ideas_tk, px):
    dias = []
    d = HOY
    while len(dias) < 5:
        if d.weekday() < 5:
            dias.append(d)
        d += dt.timedelta(days=1)
    fin = dias[-1]
    cand = {t: f for t, f in prox.items() if HOY <= pd.Timestamp(f).date() <= fin}
    for t in ideas_tk:
        if t not in prox and t not in cand:
            try:
                cal = yf.Ticker(t).calendar or {}
                ed = cal.get("Earnings Date")
                ed = ed[0] if isinstance(ed, list) and ed else ed
                if ed and HOY <= pd.Timestamp(ed).date() <= fin:
                    cand[t] = str(pd.Timestamp(ed).date())
            except Exception:
                pass
    prox_out = []
    for t, f in cand.items():
        f2, cuando = hora_balance(t)
        if f2 and HOY <= pd.Timestamp(f2).date() <= fin:
            f = f2
        u = uni.get(t, {})
        prox_out.append({"ticker": t, "byma": u.get("byma", t), "nombre": u.get("nombre"), "fecha": f, "cuando": cuando,
                         "bloque": u.get("bloque"), "idea": t in ideas_tk, "grande": (u.get("mcap") or 0) > 3e11})
        time.sleep(0.3)
    ya = {}
    for b, fi in fichas.items():
        ub = fi.get("ultimo_balance") or {}
        if ub.get("fecha") and 0 <= (HOY - pd.Timestamp(ub["fecha"]).date()).days <= 7:
            ya[fi["ticker"]] = {"ticker": fi["ticker"], "byma": b, "fecha": ub["fecha"], "veredicto": ub.get("veredicto"), "reaccion": ub.get("reaccion"), "sorpresa": ub.get("sorpresa")}
    recientes = [t for t, f in prox.items() if 1 <= (HOY - pd.Timestamp(f).date()).days <= 8 and t not in ya]
    for t in recientes + [t for t in ideas_tk if t not in prox and t not in ya]:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=8)
            ed = ed.dropna(subset=["Reported EPS"]) if ed is not None else None
            if ed is None or not len(ed):
                continue
            idx = pd.DatetimeIndex(ed.index)
            idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx
            i = int(np.argmax(idx))
            d = idx[i]
            if not (0 <= (HOY - d.date()).days <= 8):
                continue
            row = ed.iloc[i]
            est, rep_ = num(row.get("EPS Estimate")), num(row.get("Reported EPS"))
            sor = (rep_ - est) / abs(est) if est and rep_ is not None else None
            s = px[t].dropna() if t in px.columns else pd.Series(dtype=float)
            reac = None
            if len(s):
                dd = pd.Timestamp(d.date())
                if d.hour >= 16:
                    a_, b_ = s[s.index <= dd], s[s.index > dd]
                    reac = float(b_.iloc[0] / a_.iloc[-1] - 1) if len(a_) and len(b_) else None
                else:
                    a_, b_ = s[s.index < dd], s[s.index >= dd]
                    reac = float(b_.iloc[0] / a_.iloc[-1] - 1) if len(a_) and len(b_) else None
            ver = None if sor is None else "Superó lo esperado" if sor > 0.02 else "No alcanzó lo esperado" if sor < -0.02 else "En línea con lo esperado"
            u = uni.get(t, {})
            ya[t] = {"ticker": t, "byma": u.get("byma", t), "fecha": str(d.date()), "veredicto": ver, "reaccion": reac, "sorpresa": sor}
            time.sleep(0.3)
        except Exception as e:
            log("ya", t, e)
    ya = list(ya.values())
    return {"dias": [str(x) for x in dias], "proximos": sorted(prox_out, key=lambda z: z["fecha"]), "ya": sorted(ya, key=lambda z: z["fecha"])}


def calendario(dias):
    import os
    key = os.environ.get("FRED_API_KEY", "").strip()
    out = []
    ini, fin = dias[0], dias[-1]
    if key:
        try:
            r = requests.get("https://api.stlouisfed.org/fred/releases/dates", params={"api_key": key, "file_type": "json", "realtime_start": ini, "realtime_end": fin,
                                                                                        "include_release_dates_with_no_data": "true", "limit": 1000, "sort_order": "asc"}, timeout=30).json()
            vistos = set()
            for x in r.get("release_dates", []):
                for en, es, hora, peso in MACRO:
                    if x["release_name"].lower().startswith(en.lower()) and (x["date"], es) not in vistos:
                        vistos.add((x["date"], es))
                        out.append({"fecha": x["date"], "nombre": es, "hora": hora, "peso": peso})
        except Exception as e:
            log("fred calendario", e)
    for f in FOMC:
        d = pd.Timestamp(f).date()
        if ini <= str(d) <= fin:
            out.append({"fecha": str(d), "nombre": "Decisión de tasa de la Fed", "hora": "16:00", "peso": 3})
        m = d + dt.timedelta(days=21)
        if ini <= str(m) <= fin:
            out.append({"fecha": str(m), "nombre": "Minutas de la Fed", "hora": "16:00", "peso": 2})
    return sorted(out, key=lambda z: (z["fecha"], z["hora"]))


def fecha_iso(v):
    try:
        if isinstance(v, (int, float)):
            return dt.datetime.fromtimestamp(v, dt.timezone.utc)
        return pd.Timestamp(v).tz_convert("UTC").to_pydatetime() if pd.Timestamp(v).tzinfo else pd.Timestamp(v).tz_localize("UTC").to_pydatetime()
    except Exception:
        return None


def noticias_yahoo(tks):
    out = []
    for t in tks:
        try:
            ns = yf.Ticker(t).news or []
        except Exception as e:
            log("news", t, e)
            ns = []
        for n in ns:
            c = n.get("content") if isinstance(n.get("content"), dict) else n
            if c.get("contentType") and c.get("contentType") not in ("STORY", "VIDEO"):
                continue
            titulo = c.get("title")
            url = ((c.get("canonicalUrl") or {}).get("url") or (c.get("clickThroughUrl") or {}).get("url") or c.get("link"))
            fuente = (c.get("provider") or {}).get("displayName") or c.get("publisher")
            f = fecha_iso(c.get("pubDate") or c.get("displayTime") or c.get("providerPublishTime"))
            if not titulo or not f:
                RAW["sin_fecha"] = RAW.get("sin_fecha", 0) + 1
                continue
            if fuente in FUENTES_FUERA:
                continue
            out.append({"titulo": titulo, "url": url, "fuente": fuente or "Yahoo", "fecha": f, "tickers": [t], "tipo": "empresa"})
        time.sleep(0.15)
    return out


RAW = {}


def rss(url, tipo, fuente, maximo=20):
    try:
        x = ET.fromstring(requests.get(url, headers=WEB, timeout=30).content)
    except Exception as e:
        log("rss", url, e)
        return []
    out = []
    its = list(x.iter("item"))
    log("rss", url, len(its), [(i.findtext("pubDate"), (i.findtext("title") or "")[:40]) for i in its[:2]])
    for it in its:
        titulo = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        f = fecha_iso(it.findtext("pubDate") or it.findtext("{http://purl.org/dc/elements/1.1/}date"))
        if titulo and f:
            out.append({"titulo": titulo, "url": link, "fuente": fuente, "fecha": f, "tickers": [], "tipo": tipo})
    return out[:maximo]


def sec_8k(uni):
    try:
        m = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=60).json()
    except Exception as e:
        log("sec tickers", e)
        return []
    ck = {v["ticker"].upper(): int(v["cik_str"]) for v in m.values()}
    out = []
    n_ok = n_8k = 0
    for t, u in uni.items():
        c = ck.get(t.upper().replace(".", "-")) or ck.get(t.upper())
        if not c:
            continue
        n_ok += 1
        try:
            r = requests.get(f"https://data.sec.gov/submissions/CIK{c:010d}.json", headers=UA, timeout=30).json()["filings"]["recent"]
        except Exception as e:
            log("sec", t, e)
            continue
        for i, fm in enumerate(r["form"]):
            if fm not in ("8-K", "8-K/A"):
                continue
            fd = pd.Timestamp(r["filingDate"][i]).date()
            if (HOY - fd).days > 4:
                break
            n_8k += 1
            its = [z.strip() for z in str(r.get("items", [""] * len(r["form"]))[i]).split(",") if z.strip()]
            que = [ITEMS[z] for z in its if z in ITEMS]
            if not que:
                continue
            acc = r["accessionNumber"][i]
            url = f"https://www.sec.gov/Archives/edgar/data/{c}/{acc.replace('-', '')}/{r['primaryDocument'][i]}"
            f = fecha_iso(r.get("acceptanceDateTime", [None] * len(r["form"]))[i]) or fecha_iso(str(fd))
            out.append({"titulo": f"{u.get('nombre') or t} informó a la SEC: {', '.join(dict.fromkeys(que))}", "url": url, "fuente": "SEC", "fecha": f,
                        "tickers": [t], "tipo": "sec", "items": its, "es": True})
        time.sleep(0.12)
    log("sec empresas", n_ok, "8-K recientes", n_8k, "con items", len(out))
    return out


def alias(u):
    out = {}
    for t, x in u.items():
        a = [t.lower(), (x.get("byma") or "").lower()] + ALIAS.get(t, [])
        n = re.sub(r"[^a-z0-9 &.-]", " ", (x.get("nombre") or "").lower()).split()
        n = [w for w in n if w not in ("inc", "corp", "co", "the", "group", "holdings", "company", "ltd", "plc", "sa", "nv", "class", "a", "b", "de")]
        if n and len(n[0]) > 3:
            a.append(n[0])
        out[t] = [w for w in dict.fromkeys(a) if w and len(w) >= 2]
    return out


def menciona(titulo, palabras):
    tl = " " + re.sub(r"[^a-z0-9&.-]", " ", titulo.lower()) + " "
    return any(re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", tl) for p in palabras)


def norm(s):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower()).split() if len(w) > 2]


def juntar(items):
    items = sorted(items, key=lambda z: z["fecha"], reverse=True)
    out = []
    for it in items:
        w = set(norm(it["titulo"]))
        hecho = False
        for o in out:
            if o["tipo"] != it["tipo"]:
                continue
            ow = o["_w"]
            if w and ow and len(w & ow) / len(w | ow) >= 0.6:
                for t in it["tickers"]:
                    if t not in o["tickers"]:
                        o["tickers"].append(t)
                if it["fuente"] not in o["fuentes"]:
                    o["fuentes"].append(it["fuente"])
                o["n"] += 1
                hecho = True
                break
        if not hecho:
            out.append({**it, "_w": w, "fuentes": [it["fuente"]], "n": 1})
    for o in out:
        o.pop("_w", None)
    return out


def polymarket():
    try:
        conf = pd.read_csv(D / "predicciones.csv", dtype=str).fillna("")
    except Exception as e:
        log("predicciones.csv", e)
        return []
    out = []
    for r in conf.itertuples():
        evs = []
        for u in [f"https://gamma-api.polymarket.com/public-search?q={requests.utils.quote(r.buscar)}&limit_per_type=20&events_status=active",
                  f"https://gamma-api.polymarket.com/events?closed=false&active=true&limit=50&title={requests.utils.quote(r.buscar)}"]:
            try:
                js = requests.get(u, headers=WEB, timeout=30).json()
                evs += js.get("events", []) if isinstance(js, dict) else js
            except Exception as e:
                log("polymarket", u, e)
        cand = [e for e in evs if re.search(r.patron, e.get("title") or "", re.I) and not e.get("closed")]
        log("polymarket", r.buscar, len(evs), [e.get("title") for e in evs[:6]], "->", [e.get("title") for e in cand[:3]])
        if not cand:
            continue
        ev = max(cand, key=lambda e: num(e.get("volume")) or 0)
        mks = ev.get("markets") or []
        if not mks or not all("outcomePrices" in m for m in mks):
            try:
                ev = requests.get(f"https://gamma-api.polymarket.com/events/slug/{ev.get('slug')}", headers=WEB, timeout=30).json()
                mks = ev.get("markets") or []
            except Exception as e:
                log("polymarket evento", e)
        ops = []
        for m in mks:
            if m.get("closed") or m.get("active") is False:
                continue
            if re.match(r"^(candidate|person|player|other|team) ?[a-z0-9]?$", (m.get("groupItemTitle") or "").strip(), re.I) or (num(m.get("volume")) or 0) <= 0:
                continue
            try:
                pr = json.loads(m.get("outcomePrices") or "[]")
                oc = [o.lower() for o in json.loads(m.get("outcomes") or "[]")]
                p = float(pr[0])
            except Exception:
                continue
            if len(mks) == 1:
                ops.append({"label": "Sí" if oc[:1] == ["yes"] else (json.loads(m.get("outcomes"))[0]), "p": p, "semana": num(m.get("oneWeekPriceChange"))})
            else:
                ops.append({"label": m.get("groupItemTitle") or m.get("question"), "p": p, "semana": num(m.get("oneWeekPriceChange"))})
        fechas = []
        for o in ops:
            try:
                fechas.append(pd.Timestamp(o["label"]) if re.search(r"(january|february|march|april|may|june|july|august|september|october|november|december)", o["label"] or "", re.I) else None)
            except Exception:
                fechas.append(None)
        if ops and all(f is not None for f in fechas):
            hoy_ = pd.Timestamp(HOY)
            orden = sorted(zip(fechas, ops), key=lambda z: z[0])
            ops = [o for f, o in orden if f >= hoy_ and o["p"] >= 0.02][:3] or [o for f, o in orden][:3]
        else:
            ops = sorted(ops, key=lambda z: -z["p"])[:3]
        if ops:
            out.append({"titulo": r.titulo, "es": True, "url": f"https://polymarket.com/event/{ev.get('slug')}", "vol": num(ev.get("volume")) or 0,
                        "titulo_en": ev.get("title"), "opciones": ops})
    return out

def argentina(datos):
    out = {}
    try:
        ds = {x["casa"]: x for x in requests.get("https://dolarapi.com/v1/dolares", headers=WEB, timeout=30).json()}
        mep, ccl = num(ds.get("bolsa", {}).get("venta")), num(ds.get("contadoconliqui", {}).get("venta"))
        out.update({"mep": mep, "ccl": ccl, "canje": ccl / mep - 1 if mep and ccl else None, "fecha_dolar": ds.get("contadoconliqui", {}).get("fechaActualizacion")})
    except Exception as e:
        log("dolarapi", e)
    if not out.get("ccl") and datos:
        out["ccl"] = datos.get("ccl")
        out["ccl_implicito"] = True
    try:
        h = requests.get("https://api.argentinadatos.com/v1/finanzas/indices/riesgo-pais", headers=WEB, timeout=30).json()
        h = sorted(h, key=lambda z: z["fecha"])
        out["riesgo"] = h[-1]["valor"]
        out["riesgo_fecha"] = h[-1]["fecha"]
        sem = [z for z in h if z["fecha"] <= str(pd.Timestamp(h[-1]["fecha"]).date() - dt.timedelta(days=7))]
        out["riesgo_semana"] = h[-1]["valor"] - sem[-1]["valor"] if sem else None
    except Exception as e:
        log("riesgo pais", e)
    return out


def main():
    uni, ideas = universo()
    fichas = {}
    for p in (D / "fichas").glob("*.json"):
        try:
            f = json.loads(p.read_text())
            fichas[f["byma"]] = f
        except Exception:
            pass
    ideas_tk = list(ideas["ticker"])
    out = {"actualizado": AHORA.strftime("%Y-%m-%dT%H:%M:%SZ")}
    px = precios(list(uni) + list(MERCADO))
    out["mercado"] = mercado(px)
    prox = proximos_balances(fichas)
    out["ideas"] = ideas_calc(ideas, px, uni, fichas, prox)
    try:
        prev = json.loads(SALIDA.read_text())
        if prev["mercado"]["fecha"] > out["mercado"]["fecha"] and set(i["ticker"] for i in prev.get("ideas", [])) == set(ideas_tk):
            log("precios atrasados", out["mercado"]["fecha"], "se mantienen los de", prev["mercado"]["fecha"])
            out["mercado"], out["ideas"] = prev["mercado"], prev["ideas"]
    except Exception as e:
        log("previo", e)
    try:
        out["balances"] = balances(fichas, prox, uni, ideas_tk, px)
    except Exception as e:
        log("balances", e)
    dias = out.get("balances", {}).get("dias") or []
    if dias:
        out["calendario"] = calendario(dias)
    items = []
    try:
        items += noticias_yahoo(ideas_tk + [t for t in uni if t not in ideas_tk])
    except Exception as e:
        log("yahoo", e)
    try:
        items += sec_8k({t: u for t, u in uni.items() if u.get("pais") in ("United States", None)})
    except Exception as e:
        log("sec", e)
    items += rss("https://www.federalreserve.gov/feeds/press_all.xml", "fed", "Reserva Federal", 10)
    items += rss("https://www.federalreserve.gov/feeds/speeches.xml", "fed", "Reserva Federal", 10)
    gen = rss("https://www.cnbc.com/id/20910258/device/rss/rss.html", "general", "CNBC", 25) + rss("https://www.cnbc.com/id/100003114/device/rss/rss.html", "general", "CNBC", 25) + rss("https://feeds.content.dowjones.io/public/rss/mw_topstories", "general", "MarketWatch", 25)
    corte = AHORA - dt.timedelta(hours=40)
    al = alias(uni)
    log("items crudos", len(items), {k: sum(1 for i in items if i["tipo"] == k) for k in ("empresa", "sec", "fed")})
    lim = []
    for i in items:
        if i["fecha"] < corte or BASURA.search(i["titulo"]):
            continue
        if i["tipo"] == "empresa":
            i["tickers"] = [t for t in i["tickers"] if menciona(i["titulo"], al.get(t, [t.lower()]))]
            if not i["tickers"]:
                continue
        lim.append(i)
    items = lim
    log("items filtrados", len(items))
    gen = [i for i in gen if i["fecha"] >= AHORA - dt.timedelta(hours=20) and not BASURA.search(i["titulo"]) and MERCADO_KW.search(i["titulo"])][:12]
    noticias = juntar(items + gen)
    noticias = sorted(noticias, key=lambda z: z["fecha"], reverse=True)[:120]
    trad = traducir([n["titulo"] for n in noticias if not n.get("es")])
    j = 0
    for n in noticias:
        if not n.get("es"):
            n["titulo_en"] = n["titulo"]
            n["titulo"] = trad[j]
            j += 1
        n["fecha"] = n["fecha"].strftime("%Y-%m-%dT%H:%M:%SZ")
        n["bloques"] = list(dict.fromkeys(uni[t]["bloque"] for t in n["tickers"] if uni.get(t, {}).get("bloque")))
        gs = [GRUPOS.get(uni[t]["bloque_id"], "otros") for t in n["tickers"] if uni.get(t, {}).get("bloque_id")]
        n["grupo"] = "macro" if n["tipo"] in ("fed", "general") else (gs[0] if gs else "otros")
        n["mov"] = {}
        for t in n["tickers"][:3]:
            if t in px.columns and len(px[t].dropna()) > 1 and "SPY" in px.columns:
                s, q = px[t].dropna(), px["SPY"].dropna()
                n["mov"][t] = float((s.iloc[-1] / s.iloc[-2]) / (q.iloc[-1] / q.iloc[-2]) - 1)
    out["noticias"] = noticias
    ult = {}
    for n in noticias:
        for t in n["tickers"]:
            ult.setdefault(t, n)
    for x in out["ideas"]:
        n = ult.get(x["ticker"])
        if n:
            x["noticia"] = {"titulo": n["titulo"], "url": n["url"], "fuente": n["fuentes"][0], "fecha": n["fecha"]}
    try:
        pm = polymarket()
        labs = [o for m in pm for o in m["opciones"] if re.search(r"(bps|increase|decrease|change|hike|cut|january|february|march|april|may|june|july|august|september|october|november|december|before|by |yes|no$)", o["label"] or "", re.I)]
        tr = traducir([o["label"] for o in labs])
        for o, t in zip(labs, tr):
            o["label_en"], o["label"] = o["label"], t
        out["predicciones"] = pm
    except Exception as e:
        log("predicciones", e)
    try:
        out["argentina"] = argentina(leer("datos.json", {}))
    except Exception as e:
        log("argentina", e)
    SALIDA.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":"), default=str))
    print("ideas", len(out["ideas"]), "balances", len(out.get("balances", {}).get("proximos", [])), "ya", len(out.get("balances", {}).get("ya", [])),
          "calendario", len(out.get("calendario", [])), "noticias", len(noticias), "sec", sum(1 for n in noticias if n["tipo"] == "sec"),
          "fed", sum(1 for n in noticias if n["tipo"] == "fed"), "general", sum(1 for n in noticias if n["tipo"] == "general"),
          "predicciones", len(out.get("predicciones", [])), "argentina", out.get("argentina"))


if __name__ == "__main__":
    main()
