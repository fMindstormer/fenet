import io, random
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
random.seed(7)
pat = np.array([[random.random() < 0.35 for _ in range(14)] for _ in range(14)])
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1.25, 1.5, 2):
        for extra in ("", ".out{position:fixed;left:20px;top:20px}"):
            pg = b.new_page(viewport={"width": 800, "height": 500}, device_scale_factor=dpr)
            pg.goto(Path("site/test.html").resolve().as_uri())
            # show the mirror itself: no network filter, no display filter
            pg.add_style_tag(content=extra + "body{background:#404040 !important} .mirror rect.m { transition: none !important; } .out { filter: none !important; }")
            pg.evaluate("""() => document.querySelector('.mirror').removeAttribute('filter')""")
            pg.evaluate("""cells => { const r = document.querySelectorAll('.mirror rect.m'); cells.forEach((v, i) => r[i].style.fill = v ? '#fff' : '#000'); }""", pat.ravel().tolist())
            box = pg.locator(".out").bounding_box()
            full = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("L")).astype(int)
            ox, oy = box["x"] * dpr, box["y"] * dpr
            best = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    patch = full[int(oy) + dy:int(oy) + dy + 14, int(ox) + dx:int(ox) + dx + 14]
                    err = np.abs(patch / 255 - pat).max()
                    grey = int(((patch > 20) & (patch < 235)).sum())
                    if best is None or err < best[0]:
                        best = (err, dx, dy, grey)
            print(f"dpr {dpr} {'fixed' if extra else 'flow'} origin ({ox:.2f},{oy:.2f}): max error {best[0]:.3f} at offset {best[1:3]}, grey pixels {best[3]}")
            pg.close()
    b.close()
