"""Run test digits through the page in Chromium: set the mirror cells, read the net's one-hot pixels."""
import io
import json
import sys
import time
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
import train as T
from compile import N

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
dprs = [float(d) for d in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1"])]
X, y, Xt, yt = T.load()
xt = T.batch_grid(Xt, range(n), None, aug=False)
model = dict(np.load("model.npz"))
net = T.Net(); net.p = {k: model[k] for k in ("K1", "b1", "K2", "b2", "K3", "b3", "W", "c")}
pred = net.predict(xt)
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in dprs:
        pg = b.new_page(viewport={"width": 800, "height": 500}, device_scale_factor=dpr)
        pg.goto(Path("site/test.html").resolve().as_uri())
        pg.add_style_tag(content=".mirror rect.m { transition: none !important; } .out { filter: none !important; }")
        box = pg.locator(".out").bounding_box()
        ox, oy = round(box["x"] * dpr), round(box["y"] * dpr)
        agree = correct = bad = 0
        t0 = time.time()
        for i in range(n):
            cells = xt[i, 0].ravel().tolist()
            pg.evaluate("""cells => { const r = document.querySelectorAll('.mirror rect.m');
                cells.forEach((v, i) => r[i].style.fill = v ? '#fff' : '#000'); }""", cells)
            img = Image.open(io.BytesIO(pg.screenshot(clip={"x": box["x"], "y": box["y"], "width": 50 / dpr + 2, "height": 4 / dpr + 2}))).convert("L")
            on = [k for k in range(10) if img.getpixel((4 * k + 1, 1)) > 128]
            ink = img.getpixel((41, 1)) > 128
            agree += on == [pred[i]]
            correct += on == [yt[i]]
            bad += len(on) != 1
        print(f"dpr {dpr}: {n} digits, agrees with the model on {agree}, correct {correct}, not exactly one winner {bad} "
              f"({(time.time() - t0) / n * 1000:.0f} ms each)", flush=True)
        pg.close()
    b.close()
