"""Compute group inside a fixed-size SVG, scaled by 1/DPR with CSS: is one local unit one device pixel for a filter?"""
import io, random
from PIL import Image
from playwright.sync_api import sync_playwright
N = 14
random.seed(5)
pat = [[random.random() < 0.3 for _ in range(N)] for _ in range(N)]
flt = ('<filter id="f" x="0" y="0" width="40" height="20" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">'
       '<feConvolveMatrix order="3 1" kernelMatrix="1 0 0" divisor="1" preserveAlpha="true" edgeMode="none"/></filter>')
rects = '<rect width="40" height="20" fill="#000"/>' + "".join(f'<rect x="{c}" y="{r}" width="1" height="1" fill="#fff"/>' for r in range(N) for c in range(N) if pat[r][c])
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 1.25, 1.5, 2, 2.5, 3):
        pg = b.new_page(viewport={"width": 200, "height": 200}, device_scale_factor=dpr)
        res = []
        for left in (16, 17.3):
            html = (f'<!doctype html><html><body style="margin:0;background:#888">'
                    f'<svg style="position:absolute;left:{left}px;top:16px" width="160" height="160" viewBox="0 0 160 160" shape-rendering="crispEdges">'
                    f'<defs>{flt}</defs><g style="transform:scale({1/dpr})"><g filter="url(#f)">{rects}</g></g></svg></body></html>')
            pg.set_content(html)
            img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
            ox, oy = round(left * dpr), round(16 * dpr)
            best = min((sum((img.getpixel((ox + dx + c, oy + dy + r)) > 128) != pat[r][c + 1]
                            for r in range(N) for c in range(N - 1)), dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
            res.append(f"left {left}: {best[0]} wrong (offset {best[1]},{best[2]})")
        print(f"dpr {dpr}:", "; ".join(res))
        pg.close()
    b.close()
