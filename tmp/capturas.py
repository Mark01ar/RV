import sys
import threading
import functools
import http.server
from pathlib import Path

from playwright.sync_api import sync_playwright

R = Path(__file__).resolve().parent.parent
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


H = functools.partial(Q, directory=str(R))
http.server.ThreadingHTTPServer.allow_reuse_address = True
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
errores = []


def entrar(pg):
    pg.goto("http://127.0.0.1:8765/index.html")
    pg.fill("#gp", "3140")
    pg.press("#gp", "Enter")
    pg.wait_for_timeout(1500)


def tab(pg, t):
    pg.click(f'nav button[data-t="{t}"]')
    pg.wait_for_timeout(1500)


def foto(pg, nombre, full=True):
    pg.screenshot(path=str(OUT / f"{nombre}.png"), full_page=full)


with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    for disp, vp in (("cel", {"width": 390, "height": 844}), ("pc", {"width": 1366, "height": 900})):
        pg = b.new_page(viewport=vp, device_scale_factor=2 if disp == "cel" else 1)
        pg.on("console", lambda m: errores.append(f"{disp} console {m.type}: {m.text}") if m.type == "error" else None)
        pg.on("pageerror", lambda e: errores.append(f"{disp} pageerror: {e}"))
        entrar(pg)
        tab(pg, "macro")
        foto(pg, f"{disp}_1_macro")
        tab(pg, "escenarios")
        foto(pg, f"{disp}_2_escenarios_petroleo")
        pg.click('[data-ef="valor"]')
        pg.wait_for_timeout(500)
        pg.click('[data-ev="empresas"]')
        pg.wait_for_timeout(800)
        foto(pg, f"{disp}_3_escenarios_valor_empresas")
        tab(pg, "empresas")
        pg.click('[data-b="petroleo"]')
        pg.wait_for_timeout(3000)
        foto(pg, f"{disp}_4_bloque_petroleo")
        tab(pg, "empresas")
        pg.click('[data-b="aparte"]')
        pg.wait_for_timeout(3000)
        foto(pg, f"{disp}_5_bloque_aparte")
        tab(pg, "empresas")
        pg.click('[data-b="farma"]')
        pg.wait_for_timeout(3000)
        pg.click('tr[data-e="LLY"]')
        pg.wait_for_timeout(1500)
        pg.click('[data-ft="sensib"]')
        pg.wait_for_timeout(800)
        foto(pg, f"{disp}_6_ficha_lly")
        tab(pg, "empresas")
        pg.click('[data-b="mineria"]')
        pg.wait_for_timeout(3000)
        pg.click('tr[data-e="FCX"]')
        pg.wait_for_timeout(1500)
        foto(pg, f"{disp}_7_ficha_fcx_resumen", full=False)
        pg.click('.ftabs [data-ft="sensib"]')
        pg.wait_for_timeout(800)
        foto(pg, f"{disp}_8_ficha_fcx")
        pg.close()
    b.close()
srv.shutdown()
print("\n".join(errores) if errores else "sin errores de JavaScript")
