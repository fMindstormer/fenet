"""Gather the 14x14 drawing (cells of 8 CSS px) into 14x14 device pixels with a displacement map, at several DPRs."""
import io, random
from PIL import Image
from playwright.sync_api import sync_playwright

N, CELL = 14, 8
def gather_filter(fid, dpr):
    # output device pixel i (user x = i/dpr) must read cell centre (i + 0.5)*CELL: displacement = x*(CELL*dpr - 1) + CELL/2
    W = N / dpr                      # user extent of the tiny grid
    S = 2 * (W * (CELL * dpr - 1) + CELL)   # scale with headroom
    c0 = 0.5 + (CELL / 2) / S
    c1 = 0.5 + (W * (CELL * dpr - 1) + CELL / 2) / S
    def ramp(axis, ch):
        x2, y2 = (W, 0) if axis == "x" else (0, W)
        col = lambda c: f"rgb({c*100:.4f}%,0%,0%)" if ch == "R" else f"rgb(0%,{c*100:.4f}%,0%)"
        svg = (f"<svg xmlns='http://www.w3.org/2000/svg' width='{W}' height='{W}' viewBox='0 0 {W} {W}'>"
               f"<linearGradient id='g' gradientUnits='userSpaceOnUse' x1='0' y1='0' x2='{x2}' y2='{y2}'>"
               f"<stop offset='0' stop-color='{col(c0)}'/><stop offset='1' stop-color='{col(c1)}'/></linearGradient>"
               f"<rect width='{W}' height='{W}' fill='url(%23g)'/></svg>")
        return "data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E").replace("'", "%27").replace(" ", "%20").replace("#", "%23")
    return (f'<filter id="{fid}" x="0" y="0" width="{N*CELL}" height="{N*CELL}" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">'
            f'<feImage href="{ramp("x", "R")}" x="0" y="0" width="{W}" height="{W}" preserveAspectRatio="none" result="rx"/>'
            f'<feImage href="{ramp("y", "G")}" x="0" y="0" width="{W}" height="{W}" preserveAspectRatio="none" result="ry"/>'
            f'<feComposite in="rx" in2="ry" operator="arithmetic" k2="1" k3="1" x="0" y="0" width="{W}" height="{W}" result="map"/>'
            f'<feDisplacementMap in="SourceGraphic" in2="map" scale="{S}" xChannelSelector="R" yChannelSelector="G" x="0" y="0" width="{W}" height="{W}"/>'
            f'</filter>')

random.seed(3)
pattern = [[random.random() < 0.3 for _ in range(N)] for _ in range(N)]
cells = "".join(f'<i style="background:{"#000" if pattern[r][c] else "#fff"}"></i>' for r in range(N) for c in range(N))
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 1.25, 1.5, 2, 3):
        pg = b.new_page(viewport={"width": 200, "height": 200}, device_scale_factor=dpr)
        html = (f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{gather_filter("g", dpr)}</defs></svg>'
                f'<style>.m{{position:absolute;left:16px;top:16px;display:grid;grid-template-columns:repeat({N},{CELL}px);filter:url(#g)}} .m i{{width:{CELL}px;height:{CELL}px}}</style>'
                f'<div class="m">{cells}</div></body></html>')
        pg.set_content(html)
        img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
        ox = oy = round(16 * dpr)
        got = [[img.getpixel((ox + c, oy + r)) < 128 for c in range(N)] for r in range(N)]
        bad = sum(got[r][c] != pattern[r][c] for r in range(N) for c in range(N))
        print(f"dpr {dpr}: {bad} of {N*N} cells wrong")
        pg.close()
    b.close()
