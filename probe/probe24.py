import time
from pathlib import Path
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 800, "height": 500})
    t = time.time(); pg.goto(Path("site/test.html").resolve().as_uri()); print("load", round(time.time() - t, 2))
    for label, css in [("display filter on", ""), ("display filter off", ".out{filter:none!important}")]:
        if css: pg.add_style_tag(content=css)
        t = time.time()
        try:
            pg.screenshot(timeout=60000); print(label, "screenshot", round(time.time() - t, 2), "s", flush=True)
        except Exception as e:
            print(label, "timeout", flush=True)
    b.close()
