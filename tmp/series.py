import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
OUT.mkdir(exist_ok=True)
HORIZ = (20, 60, 120)
VIGENCIA = 504
INICIO = pd.Timestamp("2021-11-01")


def bajar(tks):
    partes = {k: [] for k in ("Close", "High", "Low", "Volume")}
    for i in range(0, len(tks), 40):
        lote = tks[i:i + 40]
        for _ in range(3):
            try:
                df = yf.download(lote, period="7y", interval="1d", auto_adjust=True, progress=False, threads=True, group_by="column")
                break
            except Exception:
                time.sleep(10)
                df = None
        if df is None or df.empty:
            continue
        for k in partes:
            x = df[k]
            if isinstance(x, pd.Series):
                x = x.to_frame(lote[0])
            partes[k].append(x)
    return {k: pd.concat(v, axis=1) for k, v in partes.items()}


def eventos_ticker(c, h, l, v, umbral):
    c = c.dropna()
    if len(c) < 300:
        return []
    h = h.reindex(c.index).fillna(c).values
    l = l.reindex(c.index).fillna(c).values
    vv = v.reindex(c.index).values
    cc = c.values
    idx = c.index
    base = pd.Series(vv).rolling(20).mean().shift(1).values
    rel = vv / base
    vrel_dia = rel
    spikes = [k for k in range(21, len(cc)) if np.isfinite(rel[k]) and rel[k] >= umbral]
    sel = []
    for k in spikes:
        if sel and k - sel[-1] < 5:
            if rel[k] > rel[sel[-1]]:
                sel[-1] = k
            continue
        sel.append(k)
    out = []
    for s in sel:
        alto, bajo = h[s], l[s]
        lado = 0
        ultimo = -99
        for k in range(s + 1, min(len(cc), s + VIGENCIA)):
            p = cc[k]
            nuevo = 1 if p > alto else -1 if p < bajo else lado
            tipo = None
            if lado == -1 and nuevo == 1:
                tipo = "ruptura"
            elif lado == 1 and nuevo == -1:
                tipo = "perforacion"
            elif lado == 1 and nuevo == 1 and l[k] <= alto and k - ultimo > 10:
                tipo = "rebote"
            elif lado == -1 and nuevo == -1 and h[k] >= bajo and k - ultimo > 10:
                tipo = "rechazo"
            if tipo:
                ultimo = k
            if tipo and idx[k] >= INICIO and k - s >= 5:
                out.append((idx[k], tipo, k, float(vrel_dia[k]) if np.isfinite(vrel_dia[k]) else np.nan))
            lado = nuevo
    return out


def main():
    u = json.loads((AQUI / "universo.json").read_text(encoding="utf-8"))
    tks = sorted({x["ticker"] for x in u["tickers"] if x.get("ticker")})
    px = bajar(tks + ["SPY"])
    C, H, L, V = px["Close"], px["High"], px["Low"], px["Volume"]
    spy = C["SPY"].dropna()
    spyf = {n: spy.shift(-n) / spy - 1 for n in HORIZ}
    filas = []
    base_rows = []
    for t in tks:
        if t not in C.columns:
            continue
        c = C[t].dropna()
        if len(c) < 300:
            continue
        fw = {n: (c.shift(-n) / c - 1) for n in HORIZ}
        ventana = c.index >= INICIO
        for n in HORIZ:
            x = fw[n][ventana].dropna()
            base_rows.append((t, n, x.mean(), (x > 0).mean(), len(x)))
        for umbral in (2.0, 3.0):
            vistos = set()
            for f, tipo, k, vr in eventos_ticker(c, H[t], L[t], V[t], umbral):
                if (f, tipo) in vistos:
                    continue
                vistos.add((f, tipo))
                fila = {"ticker": t, "fecha": f, "tipo": tipo, "umbral": umbral, "vol_dia": vr}
                for n in HORIZ:
                    r_ = fw[n].iloc[k]
                    fila[f"r{n}"] = r_
                    s_ = spyf[n].get(f, np.nan)
                    fila[f"m{n}"] = r_ - s_ if np.isfinite(r_) and np.isfinite(s_) else np.nan
                filas.append(fila)
    ev = pd.DataFrame(filas)
    bs = pd.DataFrame(base_rows, columns=["ticker", "h", "media", "pos", "n"])
    for n in HORIZ:
        m = bs[bs.h == n].set_index("ticker")["media"]
        ev[f"x{n}"] = ev[f"r{n}"] - ev["ticker"].map(m)
    ev.to_csv(OUT / "eventos.csv", index=False)
    L_ = []
    def linea(nombre, d):
        L_.append(f"\n## {nombre}  (eventos: {len(d)}, acciones: {d.ticker.nunique()})")
        for n in HORIZ:
            r_ = d[f"r{n}"].dropna()
            x = d[["fecha", f"x{n}"]].dropna()
            mk = d[f"m{n}"].dropna()
            pf = x.groupby(pd.to_datetime(x["fecha"]).dt.to_period(f"{max(1, n // 21)}M"))[f"x{n}"].mean()
            t_ = pf.mean() / pf.std() * np.sqrt(len(pf)) if len(pf) > 2 and pf.std() > 0 else np.nan
            L_.append(f"  {n:>3} ruedas | n={len(r_):5d} | media {r_.mean()*100:6.2f}% | mediana {r_.median()*100:6.2f}% | positivos {(r_>0).mean()*100:5.1f}% | vs día cualquiera {x[f'x{n}'].mean()*100:+6.2f}% (t={t_:4.1f}) | vs SPY {mk.mean()*100:+6.2f}%")
    L_.append("BASE: día cualquiera, misma ventana")
    for n in HORIZ:
        b = bs[bs.h == n]
        w = b.n / b.n.sum()
        L_.append(f"  {n:>3} ruedas | media ponderada {(b.media*w).sum()*100:6.2f}% | positivos {(b.pos*w).sum()*100:5.1f}%")
    for umbral in (3.0, 2.0):
        for tipo in ("ruptura", "perforacion", "rebote", "rechazo"):
            d = ev[(ev.umbral == umbral) & (ev.tipo == tipo)]
            linea(f"umbral {umbral}x · {tipo}", d)
        d = ev[(ev.umbral == umbral) & (ev.tipo == "ruptura") & (ev.vol_dia >= 1.5)]
        linea(f"umbral {umbral}x · ruptura con volumen >= 1,5x", d)
    d = ev[(ev.umbral == 3.0) & (ev.tipo == "ruptura")].copy()
    d["anio"] = pd.to_datetime(d.fecha).dt.year
    L_.append("\n## Rupturas 3x por año (60 ruedas, vs día cualquiera)")
    for a, g in d.groupby("anio"):
        L_.append(f"  {a}: n={len(g):4d} | media {g.r60.mean()*100:6.2f}% | vs día cualquiera {g.x60.mean()*100:+6.2f}% | positivos {(g.r60>0).mean()*100:5.1f}%")
    (OUT / "resumen.txt").write_text("\n".join(L_), encoding="utf-8")
    print("\n".join(L_))


if __name__ == "__main__":
    main()
