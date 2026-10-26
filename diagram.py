"""One prediction, step by step: the opening example as the filters see it, from the emulator's own images.

    python diagram.py           writes site/prediction.svg (the README's copy); build_page.py puts the
                                same panels inline on the page

Panels: the drawing, the pixels after threshold and blur, layer 1 and layer 2 as the 14x14 images the
filters write (one pixel per neuron), the ten scores, and the answer.
"""
import json
from pathlib import Path
import numpy as np
from compile_mlp import N, SLOTS, build, emulate, positions
from build_page import FONT, ROOT, sample_cells

C = 8                       # px per cell in a 14x14 panel
SIDE = N * C

CAPTIONS = [
    ("Your strokes", "14 × 14 cells on the pad."),
    ("Pixels", "196 <code>:has()</code> rules copy the cells into screen pixels; a threshold and a 3 × 3 blur follow."),
    ("Layer 1: {h1} neurons", "Each is one <code>feConvolveMatrix</code>: 196 weights over the whole image, plus a bias, clamped to 0…1, written to one pixel."),
    ("Layer 2: {h2} neurons", "The same, reading layer 1's image."),
    ("Scores", "Ten more neurons, one per digit."),
    ("Answer", "<code>feMorphology</code> finds the highest score; a second filter prints it."),
]
STYLE_VARS = """
.dg-cell { fill: var(--cell); }
.dg-line { fill: none; stroke: var(--dropout-soft); stroke-width: 0.6; }
.dg-ink { fill: var(--ink); }
.dg-slot { fill: var(--dropout-soft); }
.dg-bar { fill: var(--dropout); }
.dg-mark { fill: var(--mark); }
.dg-dim { fill: var(--dropout-soft); }
.dg-text { fill: var(--muted); font: 500 9px var(--font-data); }
"""


def trace():
    model = dict(np.load(ROOT / "mlp.npz"))
    g = build(model, float(json.loads((ROOT / "alpha.json").read_text())["alpha"]))
    x = np.zeros(N * N); x[sample_cells()] = 1; x = x.reshape(N, N)
    img = {}
    winners, scores, _ = emulate(g, x, trace=img)
    blur = next(name for name, kind, _ in g.ops if kind == "conv")
    l1, l2 = (stage[-1][0] for stage in g.stages()[:2])
    h1, h2 = model["W1"].shape[1], model["W2"].shape[1]
    return dict(x=x, blur=img[blur][0][:N, :N], l1=img[l1][0][:N, :N], l2=img[l2][0][:N, :N],
                p1=positions(h1), p2=positions(h2), scores=scores, winner=winners[0], h1=h1, h2=h2)


def grid(values, where=None):
    """A 14x14 panel: every cell framed, ink by value; `where` limits it to neuron pixels (others blank)."""
    lines = "".join(f"M{i * C} 0V{SIDE}M0 {i * C}H{SIDE}" for i in range(N + 1))
    out = [f'<rect class="dg-cell" width="{SIDE}" height="{SIDE}"/><path class="dg-line" d="{lines}"/>']
    cells = where if where is not None else [(c, r) for r in range(N) for c in range(N)]
    for c, r in cells:
        v = float(values[r, c])
        if where is not None:
            out.append(f'<rect class="dg-slot" x="{c * C + 3}" y="{r * C + 3}" width="2" height="2"/>')
        if v > 0.02:
            out.append(f'<rect class="dg-ink" x="{c * C}" y="{r * C}" width="{C}" height="{C}" fill-opacity="{v:.2f}"/>')
    return "".join(out)


def bars(scores, winner):
    w = SIDE / 10
    out = []
    for k, s in enumerate(scores):
        h = max(1.0, float(s) * (SIDE - 16))
        out.append(f'<rect class="{"dg-mark" if k == winner else "dg-bar"}" x="{k * w + 2:.1f}" y="{SIDE - 14 - h:.1f}" width="{w - 4:.1f}" height="{h:.1f}"/>'
                   f'<text class="dg-text" x="{k * w + w / 2:.1f}" y="{SIDE - 2}" text-anchor="middle">{k}</text>')
    return "".join(out)


def answer(winner):
    pitch = SIDE / 8
    rows = FONT[str(winner)].split()
    x0, y0 = (SIDE - 5 * pitch) / 2, (SIDE - 7 * pitch) / 2
    return "".join(f'<circle class="{"dg-mark" if v == "1" else "dg-dim"}" cx="{x0 + (c + 0.5) * pitch:.1f}" cy="{y0 + (r + 0.5) * pitch:.1f}" r="{pitch * 0.38:.1f}"/>'
                   for r, row in enumerate(rows) for c, v in enumerate(row))


def panels(t):
    return [grid(t["x"]), grid(t["blur"]), grid(t["l1"], t["p1"]), grid(t["l2"], t["p2"]),
            bars(t["scores"], t["winner"]), answer(t["winner"])]


def html():
    """The page's version: an ordered list of panels, coloured by the page's theme."""
    t = trace()
    items = []
    for body, (head, text) in zip(panels(t), CAPTIONS):
        items.append(f'<li><svg viewBox="0 0 {SIDE} {SIDE}" width="{SIDE}" height="{SIDE}" aria-hidden="true">{body}</svg>'
                     f'<p><strong>{head.format(**t)}</strong> {text}</p></li>')
    return f'<style>{STYLE_VARS}</style><ol class="flow">{"".join(items)}</ol>'


def svg_file():
    """The README's version: one SVG, three panels to a row, with its own light and dark colours."""
    t = trace()
    light = "--cell:#fff;--dropout-soft:#d5e3f1;--ink:#18202b;--dropout:#6f9bc8;--mark:#d1362b;--muted:#586473;--paper:#fbfcfd"
    dark = "--cell:#10161e;--dropout-soft:#22334a;--ink:#e9eef4;--dropout:#5a82ad;--mark:#ff6b57;--muted:#95a2b2;--paper:#0e131a"
    pw, ph, gap = SIDE + 48, SIDE + 46, 24
    parts = []
    for k, (body, (head, _)) in enumerate(zip(panels(t), CAPTIONS)):
        x, y = 16 + (k % 3) * (pw + gap), 16 + (k // 3) * ph
        parts.append(f'<text class="dg-head" x="{x}" y="{y + 12}">{k + 1}. {head.format(**t)}</text>'
                     f'<g transform="translate({x},{y + 22})">{body}</g>')
    W, H = 32 + 3 * pw + 2 * gap, 16 + 2 * ph
    style = (f":root{{{light}}} @media (prefers-color-scheme: dark){{:root{{{dark}}}}}"
             f"{STYLE_VARS} .dg-bg{{fill:var(--paper)}} .dg-head{{fill:var(--ink);font:600 12px -apple-system,'Segoe UI',Helvetica,Arial,sans-serif}}"
             f" .dg-text{{font:500 9px ui-monospace,Menlo,Consolas,monospace}}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="One prediction: the drawing, its pixels, layer 1, layer 2, the ten scores and the answer {t["winner"]}">'
            f'<style>{style}</style><rect class="dg-bg" width="{W}" height="{H}" rx="6"/>{"".join(parts)}</svg>')


if __name__ == "__main__":
    (ROOT / "site" / "prediction.svg").write_text(svg_file())
    print("site/prediction.svg")
