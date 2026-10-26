"""Draw with real mouse events in Chromium and read the dot-matrix answer off the screen, as a first visitor
would: the first digit straight over the opening example, then Clear before each of the others."""
import os, sys, time
import numpy as np
from pathlib import Path
from playwright.sync_api import sync_playwright
from readout import STROKES, launch, read

def stroke(pg, gbox, path, steps=6):
    cell = gbox["width"] / 14
    pts = [(gbox["x"] + (c + 0.5) * cell, gbox["y"] + (r + 0.5) * cell) for c, r in path]
    pg.mouse.move(*pts[0]); pg.mouse.down()
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = int(max(abs(x1 - x0), abs(y1 - y0)) / cell * 2) + 1
        for t in range(1, n + 1):
            pg.mouse.move(x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n)
    pg.mouse.up()

ONLY_OPEN = bool(os.environ.get("ONLY_OPEN"))
dprs = [float(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1.0]
with sync_playwright() as p:
    b = launch(p)
    for dpr in dprs:
        pg = b.new_page(viewport={"width": 1100, "height": 900}, device_scale_factor=dpr)
        pg.goto(Path("site/test.html").resolve().as_uri())
        if os.environ.get("EXTRA_CSS"):
            pg.add_style_tag(content=os.environ["EXTRA_CSS"])
        pg.wait_for_timeout(1000)
        mark = np.array([0xd1, 0x36, 0x2b]); dim = np.array([0x6f, 0x9b, 0xc8])
        box = pg.locator(".out").bounding_box(); gbox = pg.locator(".grid").bounding_box()
        res = [f"opening example: {read(pg, box, mark, dim)}"]
        res[0] += " " + str(pg.evaluate("[devicePixelRatio, getComputedStyle(document.querySelector('.out')).visibility, getComputedStyle(document.querySelector('.out')).filter]"))
        if ONLY_OPEN:
            print(f"dpr {dpr}: {res[0]}", flush=True); pg.close(); continue
        clear = pg.locator(".clear")
        for k, (want, paths) in enumerate(STROKES.items()):
            if k:                                     # a first visitor draws straight over the example; Clear after that
                clear.click(); pg.wait_for_timeout(50)
            for path in paths:
                stroke(pg, gbox, path)
            t0 = time.time(); got = read(pg, box, mark, dim); dt = time.time() - t0
            res.append(f"drew {want}{' over the example' if k == 0 else ''}: read {got} ({1000 * dt:.0f} ms to screenshot)")
            if k == 0:
                shown = pg.evaluate("[...document.querySelectorAll('.ex')].map(e => getComputedStyle(e).display)")
                res.append("example after the first stroke: " + ("gone" if set(shown) == {"none"} else f"still shown {shown}"))
        clear.click(); pg.wait_for_timeout(100)
        res.append(f"after Clear: {read(pg, box, mark, dim)}")
        print(f"dpr {dpr}: " + "; ".join(res), flush=True)
        pg.close()
    b.close()
