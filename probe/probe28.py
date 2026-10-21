"""GPU raster and a deep filter: layer 1 and a merge of m layer-2 neurons in ONE filter, as the page had it
before the layers were split. For each m, is the result transparent? Software raster reuses layer 1 for every
neuron; GPU raster computes it again for each one, and past a limit gives up.

    PYTHONPATH=. python probe/probe28.py [m,m,...]          RASTER=sw for software raster
"""
import io, json, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
from compile_mlp import N, build, to_svg
from readout import launch

m = dict(np.load("mlp.npz")); alpha = json.load(open("alpha.json"))["alpha"]
cells = json.load(open("sample.json"))["cells"]
rects = "".join(f'<rect x="{i % N}" y="{i // N}" width="1" height="1" fill="#fff"/>' for i, v in enumerate(cells) if v)


def page(k):
    g = build(m, alpha)
    l1 = g.cuts[0] - 1                                   # the layer-1 merge
    l2 = next(i for i, (_, kind, _) in enumerate(g.ops) if kind == "merge" and i > l1)
    name, kind, p = g.ops[l2]
    g.ops = g.ops[:l2] + [(name, kind, dict(p, srcs=p["srcs"][:1 + k]))]   # flood + k neurons, then stop
    g.cuts = []                                          # one filter
    return (f'<!doctype html><body style="margin:0;background:#808080"><svg width="40" height="40">'
            f'<defs>{to_svg(g, "f")}</defs><g filter="url(#f1)" shape-rendering="crispEdges">'
            f'<rect width="{N}" height="{N}" fill="#000"/>{rects}</g></svg></body>')


ks = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [8, 16, 32, 40, 43, 44, 48, 96]
tmp = Path(tempfile.gettempdir()) / "p28.html"
with sync_playwright() as p:
    b = launch(p)
    for k in ks:
        tmp.write_text(page(k))
        pg = b.new_page(viewport={"width": 200, "height": 200})
        pg.goto(tmp.as_uri()); pg.wait_for_timeout(300)
        im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)
        print(f"layer 1 + {k:3d} layer-2 neurons in one filter: {'transparent' if (im[:N, :N] == 128).all() else 'drawn'}", flush=True)
        pg.close()
    b.close()
