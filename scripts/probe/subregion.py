"""feConvolveMatrix with a one-pixel subregion: does it read input pixels outside that pixel?

    python -m scripts.probe.subregion [engines]     default chromium,firefox,webkit

A 7x7 source of distinct reds; each case is one convolution whose kernel picks a single source pixel,
read back at the result pixel. With edgeMode="none", Chromium reads the source beyond the subregion
and Firefox and WebKit read transparent black there; with edgeMode="duplicate" all three repeat the
subregion's own edge. (WebKit also returns 13 for a dark 10 through a 7x7 kernel, a precision loss
of its own.) The page depends on neither: each neuron convolves the whole image and an
feOffset moves the one result it needs (fenet/compile_mlp.py). upstream/ has the tests and the issue.
"""
import io
import sys

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

N = 7
CASES = [  # name, kernel order, target, result pixel, source pixel it should read
    ("identity", 3, (1, 1), (3, 3), (3, 3)),
    ("right neighbour", 3, (1, 1), (3, 3), (4, 3)),
    ("7x7, far corner", 7, (3, 3), (3, 3), (0, 0)),
]


def red(x, y):
    return 10 + 5 * (x + N * y)


def kernel(order, target, at, read):
    """Spec: result(X, Y) = sum source(X - targetX + j, Y - targetY + i) * kernel[order-1-i][order-1-j]."""
    j, i = read[0] - at[0] + target[0], read[1] - at[1] + target[1]
    k = np.zeros((order, order)); k[order - 1 - i, order - 1 - j] = 1
    return " ".join(str(int(v)) for v in k.ravel())


def page(cases):
    src = "".join(f'<rect x="{x}" y="{y}" width="1" height="1" fill="rgb({red(x, y)},0,0)"/>' for y in range(N) for x in range(N))
    out = []
    for k, (_, order, target, at, read, sub, edge) in enumerate(cases):
        region = f'x="{at[0]}" y="{at[1]}" width="1" height="1"' if sub else ""
        out.append(f'<svg width="{N}" height="{N}" style="position:absolute;left:{10 + 10 * k}px;top:10px" shape-rendering="crispEdges">'
                   f'<filter id="f{k}" x="0" y="0" width="{N}" height="{N}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" '
                   f'color-interpolation-filters="sRGB"><feConvolveMatrix order="{order}" targetX="{target[0]}" targetY="{target[1]}" '
                   f'kernelMatrix="{kernel(order, target, at, read)}" divisor="1" edgeMode="{edge}" preserveAlpha="true" {region}/></filter>'
                   f'<g filter="url(#f{k})">{src}</g></svg>')
    return f'<!doctype html><body style="margin:0;background:#fff">{"".join(out)}</body>'


if __name__ == "__main__":
    cases = [(*c, sub, edge) for edge in ("none", "duplicate") for sub in (False, True) for c in CASES]
    with sync_playwright() as p:
        for engine in (sys.argv[1] if len(sys.argv) > 1 else "chromium,firefox,webkit").split(","):
            b = getattr(p, engine).launch()
            pg = b.new_page(viewport={"width": 10 + 10 * len(cases), "height": 30}); pg.set_content(page(cases)); pg.wait_for_timeout(300)
            im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)
            print(f"== {engine} {b.version}")
            for k, (name, order, target, at, read, sub, edge) in enumerate(cases):
                got = im[10 + at[1], 10 + 10 * k + at[0], 0]
                print(f"  edgeMode={edge:9s} {'1-pixel' if sub else 'whole':7s} subregion, {name:16s}: "
                      f"red {got:3d}, the source pixel is {red(*read):3d}{'' if got == red(*read) else '  <- differs'}")
            b.close()
