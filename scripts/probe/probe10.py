import io, time
from PIL import Image
from playwright.sync_api import sync_playwright
def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'
def shift_flood(dx, dy, S):
    r, g = 0.5 + dx / S, 0.5 + dy / S
    return f'rgb({r*100:.4f}%,{g*100:.4f}%,50%)'
def layers(L, cell=3, S=6):
    s = "".join(f'<feFlood flood-color="{shift_flood((t % 3 - 1) * cell, (t // 3 - 1) * cell, S)}" result="map{t}"/>' for t in range(9))
    prev = "SourceGraphic"
    for l in range(L):
        acc = None
        for t in range(9):
            s += f'<feDisplacementMap in="{prev}" in2="map{t}" scale="{S}" xChannelSelector="R" yChannelSelector="G" result="L{l}o{t}"/>'
            s += f'<feColorMatrix in="L{l}o{t}" type="matrix" values="0.1 0 0 0 0.05  0 0.1 0 0 0.05  0 0 0.1 0 0.05  0 0 0 1 0" result="L{l}m{t}"/>'
            if acc is None: acc = f"L{l}m{t}"
            else:
                s += f'<feComposite in="{acc}" in2="L{l}m{t}" operator="arithmetic" k2="1" k3="1" k4="-0.05" result="L{l}a{t}"/>'; acc = f"L{l}a{t}"
        s += f'<feComponentTransfer in="{acc}" result="L{l}out"><feFuncR type="linear" slope="1"/></feComponentTransfer>'
        prev = f"L{l}out"
    return s
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 300})
    body = '<div style="width:48px;height:48px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'
    for L in (1, 2, 3, 4, 6, 8):
        pg.set_content(html(f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{layers(L)}</filter>', body))
        t = time.time()
        try:
            pg.screenshot(timeout=30000); print(f"displacement taps L={L}: {(time.time()-t)*1000:.0f} ms", flush=True)
        except Exception:
            print(f"L={L}: timeout", flush=True); break
    # correctness of a displacement shift at DPR 1, 1.25, 2: shift by +3 px
    for dpr in (1, 1.25, 2):
        pg2 = b.new_page(viewport={"width": 100, "height": 40}, device_scale_factor=dpr)
        cells = "".join(f'<div style="width:3px;height:12px;background:{"#000" if i == 5 else "#fff"}"></div>' for i in range(16))
        flt = f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB"><feFlood flood-color="{shift_flood(3, 0, 6)}" result="m"/><feDisplacementMap in="SourceGraphic" in2="m" scale="6" xChannelSelector="R" yChannelSelector="G"/></filter>'
        pg2.set_content(html(flt, f'<div style="display:flex;width:48px;filter:url(#f)">{cells}</div>'))
        img = Image.open(io.BytesIO(pg2.screenshot())).convert("L")
        row = [img.getpixel((int((x + 0.5) * dpr), int(6 * dpr))) for x in range(48)]
        print(f"dpr{dpr}: dark CSS px", [x for x, v in enumerate(row) if v < 128], "(source 15..17; +3 px read means output 12..14)")
        pg2.close()
    b.close()
