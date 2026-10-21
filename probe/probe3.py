import io, time, sys
from playwright.sync_api import sync_playwright

def html(filters, body):
    return f'<!doctype html><html><body style="margin:0;background:#fff"><svg width="0" height="0" style="position:absolute"><defs>{filters}</defs></svg>{body}</body></html>'

BODY = '<div id="d" style="width:240px;height:240px;background:linear-gradient(90deg,#000,#fff);filter:url(#f)"></div>'

def chain(n):        # no sharing: each primitive reads only the previous one
    return "".join(f'<feOffset dx="{(i % 3) - 1}" dy="0"/>' for i in range(n))

def diamonds(n):     # each level reads the previous result twice
    s = ""
    for i in range(n):
        prev = "SourceGraphic" if i == 0 else f"c{i-1}"
        s += f'<feOffset in="{prev}" dx="1" dy="0" result="o{i}"/><feComposite in="o{i}" in2="{prev}" operator="arithmetic" k2="0.5" k3="0.5" result="c{i}"/>'
    return s

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 300})
    for name, fn, ns in [("chain", chain, [50, 200, 800]), ("diamonds", diamonds, [4, 8, 12, 16])]:
        for n in ns:
            pg.set_content(html(f'<filter id="f" x="0" y="0" width="1" height="1" color-interpolation-filters="sRGB">{fn(n)}</filter>', BODY))
            t = time.time()
            try:
                pg.screenshot(timeout=60000)
                print(f"{name} {n}: {(time.time()-t)*1000:.0f} ms", flush=True)
            except Exception as e:
                print(f"{name} {n}: timeout", flush=True)
                break
    b.close()
