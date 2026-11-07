"""Which device pixel does a CSS-unit crop read, inside a will-change layer, at fractional ratios?
A 14x14 pattern of distinct greys sits in a group scaled by 1/ratio (one cell = one device pixel).
An extract filter copies each pixel's crop into its own 4x4-CSS-px block; we read the blocks back."""
import io, sys, tempfile
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright

N = 14
vals = np.arange(N * N).reshape(N, N) + 40          # distinct greys 40..235

def page(d, mode):
    px = 1 / max(d, 1)
    rects = "".join(f'<rect x="{x}" y="{y}" width="1" height="1" fill="rgb({v},{v},{v})"/>' for (y, x), v in np.ndenumerate(vals))
    prims = []
    for y in range(N):
        for x in range(N):
            if mode == "A": cx, cy, w = x * px, y * px, px
            elif mode == "B": cx, cy, w = (x - .25) * px, (y - .25) * px, 1.5 * px
            else: cx, cy, w = (x + .25) * px, (y + .25) * px, .5 * px
            i = y * N + x
            prims.append(f'<feOffset in="SourceGraphic" x="{cx:.6g}" y="{cy:.6g}" width="{w:.6g}" height="{w:.6g}" result="a{i}"/>'
                         f'<feTile in="a{i}" x="{4 * x}" y="{20 + 4 * y}" width="4" height="4" result="b{i}"/>')
    merge = "<feMerge>" + "".join(f'<feMergeNode in="b{i}"/>' for i in range(N * N)) + "</feMerge>"
    filt = (f'<filter id="ex" x="0" y="0" width="60" height="80" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" '
            f'color-interpolation-filters="sRGB">{"".join(prims)}{merge}</filter>')
    return (f'<!doctype html><html><head><style>body{{margin:0;background:#808080}} .wrap{{padding:13.37px 0 0 21.1px}}'
            f'.out{{display:block;will-change:transform;filter:url(#ex)}} .net{{transform-origin:0 0;transform:scale({px:.7f})}}</style></head>'
            f'<body><div class="wrap"><svg class="out" width="60" height="80" viewBox="0 0 60 80"><defs>{filt}</defs>'
            f'<g class="net" shape-rendering="crispEdges">{rects}</g></svg></div></body></html>')

tmp = str(Path(tempfile.gettempdir()) / "p27.html")
with sync_playwright() as p:
    b = p.chromium.launch()
    for d in [float(v) for v in sys.argv[1].split(",")]:
        for mode in "ABC":
            open(tmp, "w").write(page(d, mode))
            pg = b.new_page(viewport={"width": 300, "height": 300}, device_scale_factor=d)
            pg.goto(Path(tmp).as_uri()); pg.wait_for_timeout(300)
            box = pg.locator(".out").bounding_box()
            im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("L")).astype(int)
            ok = empty = 0; wrong = []
            for y in range(N):
                for x in range(N):
                    cxp, cyp = (box["x"] + 4 * x + 2) * d, (box["y"] + 20 + 4 * y + 2) * d
                    got = im[int(cyp), int(cxp)]
                    if got == vals[y, x]: ok += 1
                    elif got == 128: empty += 1
                    else:
                        z = np.argwhere(vals == got); wrong.append((x, y, tuple(z[0][::-1] - (y, x)[::-1]) if len(z) else got))
            print(f"ratio {d} crop {mode}: exact {ok}/196, empty {empty}, other {len(wrong)} e.g. {wrong[:4]}", flush=True)
            pg.close()
    b.close()
