import time
from playwright.sync_api import sync_playwright
def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'
def layers(L, fan, same_bounds):
    """Each layer: `fan` convolutions of the previous layer's single result, summed by a chain of composites."""
    s, prev = "", "SourceGraphic"
    for l in range(L):
        acc = None
        for k in range(fan):
            order = "3" if same_bounds else ("3" if k % 2 else "5")
            n = int(order) ** 2
            kern = " ".join(["0.02"] * n)
            s += f'<feConvolveMatrix in="{prev}" order="{order}" kernelMatrix="{kern}" preserveAlpha="true" edgeMode="none" result="L{l}c{k}"/>'
            if acc is None: acc = f"L{l}c{k}"
            else:
                s += f'<feComposite in="{acc}" in2="L{l}c{k}" operator="arithmetic" k2="1" k3="1" result="L{l}a{k}"/>'; acc = f"L{l}a{k}"
        prev = acc
    return s
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 300})
    body = '<div style="width:64px;height:64px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'
    for same in (True, False):
        for L in (2, 3, 4, 6, 8):
            pg.set_content(html(f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{layers(L, 8, same)}</filter>', body))
            t = time.time()
            try:
                pg.screenshot(timeout=30000); print(f"same_bounds={same} L={L} fan=8: {(time.time()-t)*1000:.0f} ms", flush=True)
            except Exception:
                print(f"same_bounds={same} L={L}: timeout", flush=True); break
    b.close()
