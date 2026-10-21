import io
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright
import train as T
from train_mlp import MLP, blur
from compile_mlp import SLOTS
X, y, Xt, yt = T.load()
grids = T.batch_grid(Xt, range(40), None, aug=False)[:, 0]
net = MLP(); net.p = dict(np.load("mlp.npz")); pred = net.predict(blur(grids))
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 800, "height": 500}, device_scale_factor=1.25)
    pg.goto(Path("site/test.html").resolve().as_uri())
    pg.add_style_tag(content="body{background:#404040 !important} .mirror rect.m { transition: none !important; } .out { filter: none !important; }")
    box = pg.locator(".out").bounding_box()
    print("box", box, "device origin", box["x"] * 1.25, box["y"] * 1.25)
    full = None
    for i in range(40):
        pg.evaluate("""cells => { const r = document.querySelectorAll('.mirror rect.m'); cells.forEach((v, i) => r[i].style.fill = v ? '#fff' : '#000'); }""", grids[i].ravel().tolist())
        img = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("L"))
        ox, oy = int(box["x"] * 1.25), int(box["y"] * 1.25)
        patch = img[oy - 2:oy + 18, ox - 2:ox + 18]
        lit = np.argwhere(patch > 128)
        if i < 6 or (len(lit) and pred[i] != None):
            pass
        # where is the lit 3x3 block (centre)?
        if len(lit):
            cy, cx = lit.mean(0) - 2
            k = pred[i]; sx, sy = SLOTS[k]
            if abs(cx - sx) > 0.6 or abs(cy - sy) > 0.6 or i < 4:
                print(f"digit {i}: model says {k} (slot {sx},{sy}); lit block centre ({cx:.1f},{cy:.1f}), {len(lit)} lit pixels")
    b.close()
