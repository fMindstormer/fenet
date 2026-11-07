import io
from PIL import Image
from playwright.sync_api import sync_playwright
# source: 1-px wide vertical stripes whose red value encodes x (x = 0..239)
stripes = "linear-gradient(90deg," + ",".join(f"rgb({x},0,0) {x}px,rgb({x},0,0) {x+1}px" for x in range(240)) + ")"
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 2):
        pg = b.new_page(viewport={"width": 300, "height": 60}, device_scale_factor=dpr)
        res = []
        for C in (0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0):
            S = 200
            flt = (f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB"><feFlood flood-color="rgb({C*100:.4f}%,50%,0%)" result="m"/>'
                   f'<feDisplacementMap in="SourceGraphic" in2="m" scale="{S}" xChannelSelector="R" yChannelSelector="G"/></filter>')
            pg.set_content(f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{flt}</defs></svg>'
                           f'<div style="width:240px;height:20px;background:{stripes};filter:url(#f)"></div></body></html>')
            img = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
            x = 10
            v = img.getpixel((int((x + 0.5) * dpr), int(10 * dpr)))[0]
            res.append((C, round(S * (round(C * 255) / 255 - 0.5), 2), v - x))
        print(f"dpr {dpr}: (C, expected shift, observed shift)", res)
        pg.close()
    b.close()
