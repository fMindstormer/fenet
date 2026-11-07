import io, time
from PIL import Image
from playwright.sync_api import sync_playwright

def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'

CM = lambda s: f'<feColorMatrix type="matrix" values="{s} 0 0 0 0  0 {s} 0 0 0  0 0 {s} 0 0  0 0 0 1 0"'
tests = {
 # break colour-filter fusion with an offset and with a composite between the two scalings
 "prec_offset": f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{CM(0.0625)} result="a"/><feOffset in="a" dx="0" dy="0" result="b"/>{CM(16)} in="b"/></filter>',
 "prec_comp": f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{CM(0.0625)} result="a"/><feComposite in="a" in2="a" operator="arithmetic" k2="1" k3="0" k4="0" result="b"/>{CM(16)} in="b"/></filter>',
 "prec_blend": f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{CM(0.0625)} result="a"/><feBlend in="a" in2="a" mode="lighten" result="b"/>{CM(16)} in="b"/></filter>',
}
body = '<div style="width:40px;height:40px;background:rgb(203,101,57);filter:url(#f)"></div>'
with sync_playwright() as p:
    b = p.chromium.launch()
    for dpr in (1, 1.5, 2):
        ctx = b.new_context(viewport={"width": 200, "height": 100}, device_scale_factor=dpr)
        pg = ctx.new_page()
        for name, flt in tests.items():
            pg.set_content(html(flt, body))
            img = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
            print(f"dpr{dpr} {name}:", img.getpixel((int(20 * dpr), int(20 * dpr))))
        # fractional offsets of 7 px on 7-px blocks: are block centres intact?
        flt = '<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB"><feOffset dx="7" dy="0"/></filter>'
        cells = "".join(f'<div style="width:7px;height:20px;background:{"#000" if i % 2 else "#fff"}"></div>' for i in range(8))
        pg.set_content(html(flt, f'<div style="display:flex;width:56px;filter:url(#f)">{cells}</div>'))
        img = Image.open(io.BytesIO(pg.screenshot())).convert("L")
        print(f"dpr{dpr} offset7 centres:", [img.getpixel((int((7 * i + 3.5) * dpr), int(10 * dpr))) for i in range(8)])
        # performance: a chain of 600 primitives on a 240x240 element
        prims = "".join(f'<feOffset dx="{(i % 3) - 1}" dy="0" result="o{i}"/><feComposite in="o{i}" in2="{"SourceGraphic" if i == 0 else f"c{i-1}"}" operator="arithmetic" k2="0.5" k3="0.5" result="c{i}"/>' for i in range(300))
        flt = f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{prims}</filter>'
        pg.set_content(html(flt, '<div id="d" style="width:240px;height:240px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'))
        t = time.time(); pg.screenshot(); t1 = time.time() - t
        ms = pg.evaluate("""() => new Promise(res => { const d = document.getElementById('d'); let n = 0, t0 = performance.now();
            function step() { d.style.background = n % 2 ? '#000' : '#fff'; n++; if (n < 20) requestAnimationFrame(step); else res((performance.now() - t0) / 20); }
            requestAnimationFrame(step); })""")
        print(f"dpr{dpr} 600 primitives: screenshot {t1*1000:.0f} ms, {ms:.1f} ms per frame while changing")
        ctx.close()
    b.close()
