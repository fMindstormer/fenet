"""Measure how Chromium evaluates SVG filters on an HTML element."""
import io
import sys
from PIL import Image
from playwright.sync_api import sync_playwright

def page_html(filters, body):
    return f"""<!doctype html><html><body style="margin:0;background:#fff">
<svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>"""

def px(img, x, y):
    return img.getpixel((x, y))

TESTS = {
 # 200/255 scaled down by 16 then up by 16: 8-bit intermediates lose the low bits
 "precision": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
               '<feColorMatrix type="matrix" values="0.0625 0 0 0 0  0 0.0625 0 0 0  0 0 0.0625 0 0  0 0 0 1 0" result="a"/>'
               '<feColorMatrix in="a" type="matrix" values="16 0 0 0 0  0 16 0 0 0  0 0 16 0 0  0 0 0 1 0"/></filter>',
               '<div style="width:40px;height:40px;background:rgb(203,101,57);filter:url(#f)"></div>'),
 # arithmetic with k2+k3 < 1: what happens to colour when alpha drops
 "arith_low": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
               '<feFlood flood-color="rgb(128,128,128)" result="g"/>'
               '<feComposite in="SourceGraphic" in2="g" operator="arithmetic" k1="0" k2="0.5" k3="0.25" k4="0"/></filter>',
               '<div style="width:40px;height:40px;background:rgb(200,100,40);filter:url(#f)"></div>'),
 # k2=k3=1, k4=-0.5 on 0.5-centred values
 "arith_sum": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
               '<feFlood flood-color="rgb(160,96,128)" result="g"/>'
               '<feComposite in="SourceGraphic" in2="g" operator="arithmetic" k1="0" k2="1" k3="1" k4="-0.5"/></filter>',
               '<div style="width:40px;height:40px;background:rgb(150,140,128);filter:url(#f)"></div>'),
 # offset of 10 user units, and tile of a subregion
 "offset": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
            '<feOffset dx="10" dy="0"/></filter>',
            '<div style="width:40px;height:20px;filter:url(#f);display:flex"><div style="width:10px;height:20px;background:#f00"></div><div style="width:30px;height:20px;background:#00f"></div></div>'),
 "tile": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
          '<feOffset dx="0" dy="0" x="12" y="2" width="4" height="4" result="t"/><feTile in="t"/></filter>',
          '<div style="width:40px;height:20px;filter:url(#f);display:flex"><div style="width:10px;height:20px;background:#f00"></div><div style="width:30px;height:20px;background:#00f"></div></div>'),
 # lighten and componentTransfer table
 "lighten": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
             '<feFlood flood-color="rgb(100,200,50)" result="g"/><feBlend in="SourceGraphic" in2="g" mode="lighten"/></filter>',
             '<div style="width:40px;height:40px;background:rgb(150,140,60);filter:url(#f)"></div>'),
 "transfer": ('<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">'
              '<feComponentTransfer><feFuncR type="discrete" tableValues="0 1"/><feFuncG type="table" tableValues="0 0 1"/><feFuncB type="linear" slope="2" intercept="-0.5"/></feComponentTransfer></filter>',
              '<div style="width:40px;height:40px;background:rgb(140,200,200);filter:url(#f)"></div>'),
}

with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 2):
        ctx = b.new_context(viewport={"width": 200, "height": 100}, device_scale_factor=dpr)
        pg = ctx.new_page()
        for name, (flt, body) in TESTS.items():
            pg.set_content(page_html(flt, body))
            img = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
            if name in ("offset", "tile"):
                row = [img.getpixel((x, 5 * dpr)) for x in range(0, 40 * dpr, dpr)]
                print(f"dpr{dpr} {name}:", "".join("R" if c[0] > 200 and c[2] < 50 else "B" if c[2] > 200 and c[0] < 50 else "." for c in row))
            else:
                print(f"dpr{dpr} {name}:", px(img, 20 * dpr, 20 * dpr))
        ctx.close()
    b.close()
