"""Read the dot-matrix answer off a screenshot of the page."""
import io
import os
import numpy as np
from PIL import Image
from build_page import FONT, GX, GY, PITCH

MARK = np.array([0xd1, 0x36, 0x2b])     # --mark, light theme
DIM = np.array([0x6f, 0x9b, 0xc8])      # --dropout, light theme

STROKES = {  # cell paths (col, row), drawn as press-drag-release
    "7": [[(3, 3), (10, 3), (6, 12)]],
    "1": [[(7, 2), (7, 12)]],
    "0": [[(7, 2), (4, 4), (3, 8), (5, 11), (8, 11), (10, 8), (10, 4), (7, 2)]],
    "4": [[(8, 2), (3, 8), (11, 8)], [(9, 4), (9, 12)]],
    "2": [[(4, 4), (6, 2), (9, 3), (9, 6), (4, 11), (11, 11)]],
}


def launch(p):
    """Full Chromium, which rasterizes on the GPU where there is one, as desktop browsers do (the headless
    shell is software only). RASTER=sw forces software raster."""
    flag = "--disable-gpu-rasterization" if os.environ.get("RASTER") == "sw" else "--enable-gpu-rasterization"
    return p.chromium.launch(channel="chromium", args=[flag])


def read(pg, box, mark=MARK, dim=DIM):
    im = np.asarray(Image.open(io.BytesIO(pg.screenshot())).convert("RGB")).astype(int)
    d = pg.evaluate("devicePixelRatio")
    lit = ""
    for r in range(7):
        row = ""
        for c in range(5):
            s = box["width"] / 280                     # the readout is scaled up below ratio 1
            x, y = box["x"] + s * (GX + c * PITCH + PITCH / 2), box["y"] + s * (GY + r * PITCH + PITCH / 2)
            px = im[int(y * d), int(x * d)]
            row += "M" if np.abs(px - mark).sum() < 60 else "D" if np.abs(px - dim).sum() < 60 else "0"
        lit += row + " "
    for ch, pat in FONT.items():
        if ch == "all":
            continue
        if lit.replace("D", "1").split() == pat.split() and "M" not in lit:
            return f"empty ({ch})" if ch == "?" else f"dim {ch}"
        if lit.replace("M", "1").replace("0", "0").replace("D", "0").split() == pat.split():
            return ch
    return "unreadable " + lit

