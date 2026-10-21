"""Compare one intermediate result of the filter in Chromium with the emulator."""
import io, json, sys, copy
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
import train as T
from compile import build, emulate, to_svg, N

model = dict(np.load("model.npz")); alpha = json.load(open("alpha.json"))["alpha"]
g = build(model, alpha)
_, (Xv, _), _ = T.load()
x = T.batch_grid(Xv, [0], None, aug=False)[0, 0]

def emulate_until(g, x, upto):
    g2 = copy.copy(g); g2.ops = g.ops[:upto + 1]; g2.output = g.ops[upto][0]
    # reuse emulate internals: run and grab the image
    import compile as C
    W, H = g.W + 40, g.H + 8
    res = {}
    orig_output = g2.output
    # quick re-run that returns the image of the last op
    def run():
        import types
        img = {}
        src = np.zeros((H, W)); src_a = np.zeros((H, W)); src_a[:g.H, :g.W] = 1; src[:N, :N] = x
        img["SourceGraphic"] = (src, src_a)
        return img
    return g2

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 300, "height": 700})
    for upto in [int(a) for a in sys.argv[1:]]:
        g2 = copy.copy(g); g2.ops = g.ops[:upto + 1]; g2.output = g.ops[upto][0]
        name, kind, prm = g.ops[upto]
        # emulator image of that op
        import compile as C
        saved = C.emulate.__code__
        W, H = g.W + 40, g.H + 8
        # instrument: run emulate on truncated graph but read the raw image
        img = {}
        def capture(g3, xx):
            # copy of emulate's loop returning the image dict
            src = np.zeros((H, W)); src_a = np.zeros((H, W)); src_a[:g.H, :g.W] = 1.0; src[:N, :N] = xx
            store = {"SourceGraphic": (src, src_a)}
            for nm, kd, pp in g3.ops:
                gg = copy.copy(g3); gg.ops = [(nm, kd, pp)]
            return store
        # simplest: monkeypatch output extraction
        ops_backup = g.ops
        try:
            g3 = copy.copy(g); g3.ops = g.ops[:upto + 1]; g3.output = name
            g3.meta = dict(g.meta); g3.meta["scores"] = name
            # emulate returns scores row from meta['scores']; we need the full image, so re-implement quickly:
            import types
            src_code = open("compile.py").read()
        finally:
            pass
        ns = {}
        exec(src_code.split("# ------------------------------------------------------------------ SVG")[0].split("def emulate")[0], ns)
        body = "def emulate" + src_code.split("def emulate")[1].split("# ------------------------------------------------------------------ SVG")[0]
        body = body.replace("    out = img[g.output][0]\n", "    return img[g.output]\n")
        exec(body, ns)
        ev, ea = ns["emulate"](g3, x, quant=False)
        svg_filter = to_svg(g3, "dbg").replace(f'result="{name}"', f'result="{name}"')
        cells = "".join(f'<rect x="{c}" y="{r}" width="1" height="1" fill="{"#fff" if x[r, c] else "#000"}"/>' for r in range(N) for c in range(N))
        html = (f'<!doctype html><html><body style="margin:0;background:#f0f"><svg width="200" height="600" viewBox="0 0 200 600" shape-rendering="crispEdges" style="position:absolute;left:0;top:0">'
                f'<defs>{svg_filter}</defs><g filter="url(#dbg)"><rect width="{g.W}" height="{g.H}" fill="#000"/>{cells}</g></svg></body></html>')
        pg.set_content(html)
        im = np.asarray(Image.open(io.BytesIO(pg.screenshot(clip={"x": 0, "y": 0, "width": 60, "height": min(600, g.H)}))).convert("RGB")).astype(float)
        got = im[:, :, 1] / 255          # green channel (magenta background shows where transparent: G = 0)
        h, w = got.shape
        e = ev[:h, :w]
        x0, y0, rw, rh = prm["region"]
        sl = (slice(y0, min(h, y0 + rh)), slice(x0, min(w, x0 + rw)))
        d = np.abs(got[sl] - e[sl])
        print(f"op {upto} {name} {kind} region {prm['region']}: max |chrome - emulator| {d.max():.3f}, mean {d.mean():.4f}")
        if d.max() > 0.05:
            yy, xx = np.unravel_index(d.argmax(), d.shape)
            print(f"   worst at ({xx + x0},{yy + y0}): chrome {got[sl][yy, xx]:.3f} emulator {e[sl][yy, xx]:.3f}")
            print("   chrome tile row 5:", np.round(got[y0 + 5, x0:x0 + 14], 2))
            print("   emulat tile row 5:", np.round(e[y0 + 5, x0:x0 + 14], 2))
    b.close()
