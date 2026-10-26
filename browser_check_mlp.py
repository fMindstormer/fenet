"""Test digits through the real page in Chromium: paint the mirror cells, read the printed answer.

    python browser_check_mlp.py [n] [ratios]     e.g. 200 1,1.25,2
"""
import os, sys, time
import numpy as np
from pathlib import Path
from playwright.sync_api import sync_playwright
import train as T
from train_mlp import MLP, blur
from readout import launch, read

n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
dprs = [float(d) for d in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1"])]
_, _, (Xt, yt) = T.load()
grids = T.batch_grid(Xt, range(n), None, aug=False)[:, 0]
net = MLP(); net.p = dict(np.load("mlp.npz"))
pred = net.predict(blur(grids))
with sync_playwright() as p:
    b = launch(p)
    for dpr in dprs:
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=dpr)
        pg.goto(Path("site/test.html").resolve().as_uri())
        pg.add_style_tag(content=".mirror rect.m { transition: none !important; } .ex { display: none !important; }")
        pg.wait_for_timeout(1000)
        box = pg.locator(".out").bounding_box()
        agree = correct = bad = 0
        times = []
        for i in range(n):
            pg.evaluate("""cells => { const r = document.querySelectorAll('.mirror rect.m');
                cells.forEach((v, i) => r[i].style.fill = v ? '#fff' : '#000'); }""", grids[i].ravel().tolist())
            t0 = time.time()
            got = read(pg, box)
            times.append(time.time() - t0)
            agree += got == str(pred[i]); correct += got == str(yt[i]); bad += len(got) != 1
        print(f"dpr {dpr}: {n} digits, the page prints the model's answer for {agree}, the true digit for {correct}, "
              f"something unreadable for {bad}; render+read median {1000 * np.median(times):.0f} ms", flush=True)
        pg.close()
    b.close()
