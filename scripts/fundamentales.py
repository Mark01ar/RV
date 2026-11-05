import json
import math
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SALIDA = DATA / "fundamentales.json"
SNAP = DATA / "snapshots"
PESOS = {"revisiones": 0.30, "momentum": 0.25, "calidad": 0.25, "valuacion": 0.20}
FINANCIERAS = {"Financial Services"}
CAMPOS = ["forwardPE", "trailingPE", "returnOnAssets", "returnOnEquity", "operatingMargins", "grossMargins",
          "freeCashflow", "marketCap", "totalDebt", "totalCash", "ebitda", "revenueGrowth", "earningsGrowth",
          "recommendationMean", "numberOfAnalystOpinions", "targetMeanPrice", "currentPrice", "financialCurrency",
          "currency", "mostRecentQuarter"]


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def filas(df):
    if df is None or len(df) == 0:
        return {}
    return {str(i): r.to_dict() for i, r in df.iterrows()}


def cambio(cur, old):
    cur, old = num(cur), num(old)
    if cur is None or old in (None, 0):
        return None
    return (cur - old) / abs(old)


def bajar(t):
    k = yf.Ticker(t)
    info = k.info or {}
    if len(info) < 20:
        raise ValueError("info vacía")
    d = {c: info.get(c) for c in CAMPOS}
    try:
        tr = filas(k.eps_trend)
    except Exception:
        tr = {}
    try:
        rv = filas(k.eps_revisions)
    except Exception:
        rv = {}
    d["rev90_0y"] = cambio(tr.get("0y", {}).get("current"), tr.get("0y", {}).get("90daysAgo"))
    d["rev90_1y"] = cambio(tr.get("+1y", {}).get("current"), tr.get("+1y", {}).get("90daysAgo"))
    up = sum(num(rv.get(p, {}).get("upLast30days")) or 0 for p in ["0y", "+1y"])
    dn = sum(num(rv.get(p, {}).get("downLast30days")) or 0 for p in ["0y", "+1y"])
    d["amplitud"] = (up - dn) / (up + dn) if up + dn > 0 else None
    d["rev_suben"], d["rev_bajan"] = up, dn
    try:
        e = k.get_earnings_dates(limit=8)
    except Exception:
        e = None
    if e is not None and len(e):
        e = e.reset_index()
        col = e.columns[0]
        e[col] = pd.to_datetime(e[col], utc=True)
        ahora = pd.Timestamp.now(tz="UTC")
        pas = e[(e[col] <= ahora) & e["Reported EPS"].notna()].sort_values(col)
        fut = e[e[col] > ahora].sort_values(col)
        if len(pas):
            u = pas.iloc[-1]
            d["ultimo_balance"] = u[col].tz_convert("America/New_York").isoformat()
            s = num(u.get("Surprise(%)"))
            d["sorpresa"] = s / 100 if s is not None else None
        if len(fut):
            d["proximo_balance"] = fut.iloc[0][col].tz_convert("America/New_York").isoformat()
    return d


def tipos_cambio(monedas):
    fx = {"USD": 1.0}
    for m in monedas:
        if m in fx or not m:
            continue
        base, div = (m.upper(), 100.0) if m in ("GBp", "ILA", "ZAc") else (m, 1.0)
        try:
            h = yf.Ticker(f"{base}USD=X").history(period="10d")["Close"].dropna()
            if len(h):
                fx[m] = float(h.iloc[-1]) / div
        except Exception as e:
            print(f"fx {m}: {e}", file=sys.stderr)
    return fx


def ohlcv(tickers):
    frames = {}
    for i in range(0, len(tickers), 40):
        lote = tickers[i:i + 40]
        for _ in range(3):
            try:
                df = yf.download(lote, period="1y", interval="1d", auto_adjust=False, progress=False,
                                 group_by="column", threads=True)
                break
            except Exception as e:
                print(f"ohlcv lote {i}: {e}", file=sys.stderr)
                time.sleep(10)
        else:
            continue
        for c in ["High", "Low", "Close", "Volume"]:
            if isinstance(df.columns, pd.MultiIndex) and c in df.columns.get_level_values(0):
                frames.setdefault(c, []).append(df[c])
        time.sleep(2)
    out = {c: pd.concat(v, axis=1) for c, v in frames.items()}
    for c in out:
        out[c] = out[c].loc[:, ~out[c].columns.duplicated()]
        out[c].index = pd.to_datetime(out[c].index).tz_localize(None)
    return out


def avwap(px, t, ultimo):
    if not ultimo or "Close" not in px or t not in px["Close"].columns:
        return None, None
    f = pd.Timestamp(ultimo)
    ancla = f.tz_localize(None).normalize()
    if f.hour >= 16:
        ancla += pd.Timedelta(days=1)
    h, l, c, v = (px[k][t] for k in ["High", "Low", "Close", "Volume"])
    df = pd.DataFrame({"h": h, "l": l, "c": c, "v": v}).dropna()
    df = df[df.index >= ancla]
    if len(df) < 2 or df["v"].sum() <= 0:
        return None, None
    tp = (df["h"] + df["l"] + df["c"]) / 3
    vw = float((tp * df["v"]).sum() / df["v"].sum())
    return float(df["c"].iloc[-1] / vw - 1), df.index[0].strftime("%Y-%m-%d")


def pct(s, mayor_mejor=True):
    s = pd.to_numeric(s, errors="coerce")
    r = (s if mayor_mejor else -s).rank(pct=True) * 100
    return r.where(s.notna())


def pct_sector(s, sector, minimo=8):
    s = pd.to_numeric(s, errors="coerce")
    glob = pct(s)
    out = glob.copy()
    for sec, idx in s.groupby(sector).groups.items():
        sub = s.loc[idx]
        if sub.notna().sum() >= minimo:
            out.loc[idx] = (sub.rank(pct=True) * 100).where(sub.notna())
    return out


def componente(df, cols, minimo):
    sub = df[cols]
    n = sub.notna().sum(axis=1)
    return sub.mean(axis=1).where(n >= minimo)


def limpiar(x):
    if isinstance(x, dict):
        return {k: limpiar(v) for k, v in x.items()}
    if isinstance(x, list):
        return [limpiar(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return round(v, 6) if math.isfinite(v) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def main():
    datos = json.loads((DATA / "datos.json").read_text())
    univ = [x for x in datos["tickers"] if not x.get("sin_datos")]
    previo = {}
    if SALIDA.exists():
        try:
            previo = {x["ticker"]: x.get("raw", {}) for x in json.loads(SALIDA.read_text())["tickers"]}
        except Exception:
            previo = {}

    raw, fallidos = {}, []
    for k, x in enumerate(univ):
        t = x["ticker"]
        for intento in range(3):
            try:
                raw[t] = bajar(t)
                break
            except Exception as e:
                if intento == 2:
                    print(f"fallo {t}: {e}", file=sys.stderr)
                    fallidos.append(t)
                    if t in previo:
                        raw[t] = {**previo[t], "dato_previo": True}
                time.sleep(4 * (intento + 1))
        time.sleep(0.5)
        if k % 50 == 0:
            print(f"{k}/{len(univ)}", flush=True)

    fx = tipos_cambio({r.get("financialCurrency") for r in raw.values() if r.get("financialCurrency")})
    px = ohlcv([x["ticker"] for x in univ])

    filas_df = []
    for x in univ:
        t = x["ticker"]
        r = raw.get(t, {})
        fin = x.get("sector") in FINANCIERAS
        tasa = fx.get(r.get("financialCurrency") or "USD")
        fcf = num(r.get("freeCashflow"))
        mcap = num(r.get("marketCap"))
        ebitda = num(r.get("ebitda"))
        deuda = num(r.get("totalDebt"))
        caja = num(r.get("totalCash"))
        fpe = num(r.get("forwardPE"))
        gm = num(r.get("grossMargins"))
        dist_avwap, ancla = avwap(px, t, r.get("ultimo_balance"))
        precio = num(r.get("currentPrice"))
        objetivo = num(r.get("targetMeanPrice"))
        filas_df.append({
            "ticker": t, "byma": x["byma"], "nombre": x["nombre"], "sector": x.get("sector"),
            "liquidez": x.get("liquidez"), "monto_usd": x.get("monto_usd"), "nota": x.get("nota"),
            "rev90_0y": num(r.get("rev90_0y")), "rev90_1y": num(r.get("rev90_1y")),
            "amplitud": num(r.get("amplitud")), "sorpresa": num(r.get("sorpresa")),
            "mom12_1": x.get("mom12_1"), "vs_ma200": x.get("vs_ma200"),
            "roa": num(r.get("returnOnAssets")), "margen_op": None if fin else num(r.get("operatingMargins")),
            "margen_bruto": None if (fin or gm in (None, 0)) else gm,
            "deuda_ebitda": None if (fin or not ebitda or ebitda <= 0 or deuda is None) else (deuda - (caja or 0)) / ebitda,
            "ey": (1 / fpe) if fpe not in (None, 0) else None,
            "fcf_yield": None if (fin or fcf is None or not mcap or tasa is None) else fcf * tasa / mcap,
            "forward_pe": fpe, "crec_ventas": num(r.get("revenueGrowth")),
            "upside": (objetivo / precio - 1) if (objetivo and precio) else None,
            "recom": num(r.get("recommendationMean")), "n_analistas": num(r.get("numberOfAnalystOpinions")),
            "rev_suben": r.get("rev_suben"), "rev_bajan": r.get("rev_bajan"),
            "ultimo_balance": r.get("ultimo_balance"), "proximo_balance": r.get("proximo_balance"),
            "avwap_dist": dist_avwap, "avwap_ancla": ancla, "dato_previo": bool(r.get("dato_previo")),
            "sin_fund": t not in raw,
        })
    df = pd.DataFrame(filas_df).set_index("ticker")

    p = pd.DataFrame(index=df.index)
    p["rev90_0y"] = pct(df["rev90_0y"])
    p["rev90_1y"] = pct(df["rev90_1y"])
    p["amplitud"] = pct(df["amplitud"])
    p["sorpresa"] = pct(df["sorpresa"])
    p["mom12_1"] = pct(df["mom12_1"])
    p["vs_ma200"] = pct(df["vs_ma200"])
    p["roa"] = pct(df["roa"])
    p["margen_op"] = pct(df["margen_op"])
    p["margen_bruto"] = pct(df["margen_bruto"])
    p["deuda_ebitda"] = pct(df["deuda_ebitda"], mayor_mejor=False)
    p["ey"] = pct_sector(df["ey"], df["sector"])
    p["fcf_yield"] = pct(df["fcf_yield"])

    comp = pd.DataFrame(index=df.index)
    comp["revisiones"] = componente(p, ["rev90_0y", "rev90_1y", "amplitud", "sorpresa"], 2)
    comp["momentum"] = componente(p, ["mom12_1", "vs_ma200"], 1)
    comp["calidad"] = componente(p, ["roa", "margen_op", "margen_bruto", "deuda_ebitda"], 1)
    comp["valuacion"] = componente(p, ["ey", "fcf_yield"], 1)
    w = pd.Series(PESOS)
    disp = comp.notna().mul(w, axis=1)
    total = (comp.fillna(0).mul(w, axis=1).sum(axis=1) / disp.sum(axis=1)).where(comp.notna().sum(axis=1) >= 3)
    comp["total"] = total
    comp["puesto"] = total.rank(ascending=False, method="min")

    salida = []
    for t, r in df.iterrows():
        c = comp.loc[t]
        salida.append({
            "ticker": t, "byma": r["byma"], "nombre": r["nombre"], "sector": r["sector"],
            "liquidez": r["liquidez"], "monto_usd": r["monto_usd"], "nota": r["nota"],
            "scores": {k: (None if pd.isna(c[k]) else float(c[k])) for k in ["total", "revisiones", "momentum", "calidad", "valuacion"]},
            "puesto": None if pd.isna(c["puesto"]) else int(c["puesto"]),
            "raw": {k: (None if (isinstance(r[k], float) and pd.isna(r[k])) else r[k]) for k in
                    ["rev90_0y", "rev90_1y", "amplitud", "sorpresa", "mom12_1", "vs_ma200", "roa", "margen_op",
                     "margen_bruto", "deuda_ebitda", "ey", "fcf_yield", "forward_pe", "crec_ventas", "upside", "recom",
                     "n_analistas", "rev_suben", "rev_bajan", "ultimo_balance", "proximo_balance", "avwap_dist",
                     "avwap_ancla", "dato_previo"]},
            "sin_fund": bool(r["sin_fund"]),
        })
    hoy = dt.date.today().isoformat()
    SALIDA.write_text(json.dumps(limpiar({
        "actualizado": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "pesos": PESOS, "n": len(salida), "con_score": int(comp["total"].notna().sum()),
        "fallidos": fallidos, "tickers": salida}), ensure_ascii=False, separators=(",", ":")))
    SNAP.mkdir(exist_ok=True)
    snap = comp[["total", "revisiones", "momentum", "calidad", "valuacion"]].round(2)
    snap["vs_ma200"] = df["vs_ma200"].round(4)
    snap["avwap_dist"] = df["avwap_dist"].round(4)
    snap.to_csv(SNAP / f"fund_{hoy}.csv")
    print(f"fundamentals: {len(salida)} tickers, {int(comp['total'].notna().sum())} con score, {len(fallidos)} fallidos")
    print(comp.sort_values("total", ascending=False).head(10).round(1).to_string())


if __name__ == "__main__":
    main()
