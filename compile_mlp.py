"""Compile the perceptron (mlp.npz) into SVG filters, and emulate them in NumPy.

User units are device pixels (the page scales the group by 1/devicePixelRatio). Every image is
14x14. A neuron is one feConvolveMatrix: a 14x14 kernel whose target is the neuron's own pixel, so
the kernel always spans exactly the 14x14 input, and every neuron of a layer asks for the same input
region (Chromium can then reuse it). A layer's neurons are merged onto a black square.

Each layer is its own filter (#net1, #net2, #net3), applied on nested groups. Chromium's GPU raster
does not reuse a shared input: in one filter, every layer-2 neuron would compute layer 1 again, and
past about 43 of them the result is empty. A group's content is drawn once before its filter runs,
so a layer reads the one below as a finished image. The ink flag rides along in pixels no neuron
uses: CARRY in layer 1, INK in layer 2.

Output: the winning class lights a 3x3 block centred on its slot (1 + 4*(k % 4), 1 + 4*(k // 4));
the block at (13, 9) says whether anything is drawn. These are in the red channel; the green channel
holds each score as a 3x3 block around its slot.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

N = 14
SLOTS = [(1 + 4 * (k % 4), 1 + 4 * (k // 4)) for k in range(10)]
INK = (13, 9)
CARRY = (13, 6)              # free in layer 1 (192 neurons), above INK so the move is a plain offset
BLUR = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16


def positions(n):
    """Pixels for n neurons, spread over the 14x14 square (row-major on a stride that fits)."""
    allpos = [(x, y) for y in range(N) for x in range(N)]
    step = len(allpos) / n
    return [allpos[int(i * step)] for i in range(n)]


class Graph:
    def __init__(self):
        self.ops = []
        self.cuts = []           # op indices where the next filter starts

    def add(self, kind, **p):
        name = f"r{len(self.ops)}"
        self.ops.append((name, kind, p))
        return name

    def cut(self):
        """End the current filter: the next one reads its last result as SourceGraphic."""
        self.cuts.append(len(self.ops))

    def stages(self):
        bounds = [0] + self.cuts + [len(self.ops)]
        return [self.ops[a:b] for a, b in zip(bounds, bounds[1:])]


def build(m, alpha):
    W1, b1, W2, b2, W3, b3 = (m[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    H1, H2 = W1.shape[1], W2.shape[1]
    P1, P2 = positions(H1), positions(H2)
    g = Graph()
    full = (0, 0, N, N)
    # the drawn cells are 0 or 1; a threshold undoes any resampling at fractional pixel positions
    inp = g.add("threshold", src="SourceGraphic", region=full)
    blur = g.add("conv", src=inp, K=np.pad(BLUR, ((0, 11), (0, 11))), tx=1, ty=1, ox=3, oy=3, bias=0.0, region=full,
                 K3=BLUR)
    assert CARRY not in P1 and INK not in P2 and CARRY[0] <= INK[0] and CARRY[1] <= INK[1]
    # whether anything is drawn, kept in a pixel the layer-2 kernels ignore
    carry = g.add("dilate", src=inp, r=N, region=(*CARRY, 1, 1))
    prev = blur
    for W, b, P, Pin in ((W1, b1, P1, None), (W2, b2, P2, P1)):
        black = g.add("flood", color=0.0, region=full)
        units = []
        for j, (hx, hy) in enumerate(P):
            img = np.zeros((N, N))
            if Pin is None:
                img[:, :] = W[:, j].reshape(N, N)
            else:
                for i, (px, py) in enumerate(Pin):
                    img[py, px] = W[i, j]
            units.append(g.add("neuron", src=prev, img=img, x=hx, y=hy, bias=float(b[j]), region=(hx, hy, 1, 1)))
        prev = g.add("merge", srcs=[black] + units + [carry], region=full)
        g.cut()
        if Pin is None:          # layer 2 moves the flag to INK, which the score kernels ignore
            carry = g.add("offset", src=prev, dx=INK[0] - CARRY[0], dy=INK[1] - CARRY[1], region=(*INK, 1, 1))
    black = g.add("flood", color=0.0, region=full)
    scores, scores_img, score_bias = [], [], []
    for k, (sx, sy) in enumerate(SLOTS):
        img = np.zeros((N, N))
        for i, (px, py) in enumerate(P2):
            img[py, px] = alpha * W3[i, k]
        scores_img.append(img); score_bias.append(0.5 + alpha * float(b3[k]))
        scores.append(g.add("neuron", src=prev, img=img, x=sx, y=sy, bias=score_bias[k], region=(sx, sy, 1, 1)))
    sc = g.add("merge", srcs=[black] + scores, region=full)
    best = g.add("dilate", src=sc, r=N, region=full)
    # 1 + score - best is 1 exactly where the score is the maximum. (feBlend difference would do too,
    # but Chromium evaluates its inputs again, the whole network with them: five times slower.)
    near = g.add("near", a=sc, b=best, region=full)
    onehot = g.add("isone", src=near, region=full)
    # ties (two scores equal after 8-bit rounding): keep a winner only if no slot before it in reading
    # order also won. One convolution: +1 at the pixel itself, -1 at every offset where an earlier slot
    # can sit (left in the same row, anywhere in the rows above).
    taps = {(0, 0): 1.0}
    for dx in (-4, -8, -12):
        taps[(dx, 0)] = -1.0
    for dy in (-4, -8):
        for dx in range(-12, 13, 4):
            taps[(dx, dy)] = -1.0
    unique = g.add("taps", src=onehot, taps=taps, region=full)
    winner = g.add("dilate", src=unique, r=1, region=full)
    ink = g.add("offset", src=prev, dx=0, dy=0, region=(*INK, 1, 1))
    inkb = g.add("dilate", src=ink, r=1, region=(INK[0] - 1, INK[1] - 1, 3, 3))
    marks = g.add("sum", a=winner, b=inkb, region=full)
    # output: the marks in red, the scores (grown to 3x3 blocks, like the marks) in green, so a reader
    # that can only trust the middle of a block gets both from the same place
    red = g.add("chan", src=marks, keep="R", region=full)
    green = g.add("chan", src=g.add("dilate", src=sc, r=1, region=full), keep="G", region=full)
    out = g.add("sum", a=red, b=green, region=full)
    g.output, g.scores, g.flags = out, sc, marks
    return g


# ------------------------------------------------------------------ emulator
def emulate(g, x, quant=True, trace=None):
    """Run the graph on a 14x14 drawing. trace, a dict, receives every result as (value, alpha) images."""
    S = N + 4
    src = np.zeros((S, S)); src[:N, :N] = x
    alpha = np.zeros((S, S)); alpha[:N, :N] = 1
    img = {"SourceGraphic": (src, alpha)}
    q = (lambda v: np.round(v * 255) / 255) if quant else (lambda v: v)
    for name, kind, p in g.ops:
        x0, y0, w, h = p["region"]
        mask = np.zeros((S, S), bool); mask[y0:y0 + h, x0:x0 + w] = True
        if kind == "flood":
            v = np.full((S, S), p["color"]); a = np.ones((S, S))
        elif kind == "cell":
            v = np.full((S, S), float(x.ravel()[p["index"]])); a = np.ones((S, S))
        elif kind == "threshold":
            sv, sa = img[p["src"]]
            v = (sv >= 0.5).astype(float); a = sa
        elif kind == "conv":                         # the blur: 3x3, zero padding
            sv, sa = img[p["src"]]
            v = ndimage.correlate(sv, p["K3"], mode="constant") + p["bias"]
            v = q(np.clip(v, 0, 1)); a = sa
        elif kind == "neuron":
            sv, sa = img[p["src"]]
            v = np.zeros((S, S))
            v[p["y"], p["x"]] = q(np.clip((sv[:N, :N] * p["img"]).sum() + p["bias"], 0, 1))
            a = sa
        elif kind == "merge":
            v = np.zeros((S, S)); a = np.zeros((S, S))
            for s in p["srcs"]:
                sv, sa = img[s]
                v = np.where(sa > 0, sv, v); a = np.maximum(a, sa)
        elif kind == "dilate":
            sv, sa = img[p["src"]]
            k = (2 * p.get("ry", p.get("r")) + 1, 2 * p.get("rx", p.get("r")) + 1)
            v = ndimage.maximum_filter(sv, size=k, mode="constant"); a = ndimage.maximum_filter(sa, size=k, mode="constant")
        elif kind == "chan":                          # the emulator keeps one channel; read g.flags and g.scores
            v, a = img[p["src"]]
        elif kind == "near":
            av, aa = img[p["a"]]; bv, ba = img[p["b"]]
            v = np.clip(1 + av - bv, 0, 1); a = np.clip(1 + aa - ba, 0, 1)
        elif kind == "isone":
            sv, sa = img[p["src"]]
            v = (sv > 1 - 1 / 512).astype(float); a = np.ones((S, S))
        elif kind == "taps":
            sv, sa = img[p["src"]]
            v = np.zeros((S, S))
            for (dx, dy), w in p["taps"].items():
                sh = np.zeros((S, S))
                ys, yd = (slice(0, S + dy), slice(-dy, S)) if dy <= 0 else (slice(dy, S), slice(0, S - dy))
                xs, xd = (slice(0, S + dx), slice(-dx, S)) if dx <= 0 else (slice(dx, S), slice(0, S - dx))
                sh[yd, xd] = sv[ys, xs]           # sh(p) = src(p + (dx, dy))
                v += w * sh
            v = q(np.clip(v, 0, 1)); a = sa
        elif kind == "not":
            sv, sa = img[p["src"]]
            v = 1 - sv; a = sa
        elif kind == "mul":
            av, aa = img[p["a"]]; bv, ba = img[p["b"]]
            v = av * bv; a = aa * ba
        elif kind == "diff":
            av, aa = img[p["a"]]; bv, ba = img[p["b"]]
            v = np.abs(av - bv); a = np.maximum(aa, ba)
        elif kind == "iszero":
            sv, sa = img[p["src"]]
            v = (sv < 1 / 256).astype(float); a = np.ones((S, S))
        elif kind == "sum":
            av, aa = img[p["a"]]; bv, ba = img[p["b"]]
            v = np.clip(av + bv, 0, 1); a = np.clip(aa + ba, 0, 1)
        elif kind == "offset":
            sv, sa = img[p["src"]]
            v = np.zeros((S, S)); a = np.zeros((S, S))
            v[p["dy"]:, p["dx"]:] = sv[:S - p["dy"], :S - p["dx"]]; a[p["dy"]:, p["dx"]:] = sa[:S - p["dy"], :S - p["dx"]]
        else:
            raise ValueError(kind)
        img[name] = (np.where(mask, v, 0.0), np.where(mask, a, 0.0))
    if trace is not None:
        trace.update(img)
    out = img[g.flags][0]
    winners = [k for k, (sx, sy) in enumerate(SLOTS) if out[sy, sx] > 0.5]
    scores = [img[g.scores][0][sy, sx] for sx, sy in SLOTS]
    return winners, np.array(scores), out[INK[1], INK[0]] > 0.5


# ------------------------------------------------------------------ SVG
def fmt(w):
    if w == 0:
        return "0"
    s = f"{w:.4g}".replace("e-0", "e-").replace("e+0", "e")
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


def primitive(name, kind, p):
    x0, y0, w, h = p["region"]
    reg = f'x="{x0}" y="{y0}" width="{w}" height="{h}"'
    if kind == "flood":
        c = round(p["color"] * 255)
        return f'<feFlood flood-color="rgb({c},{c},{c})" {reg} result="{name}"/>'
    elif kind == "cell":
        return f'<feFlood id="m{p["index"]}" {reg} result="{name}"/>'
    elif kind == "threshold":
        return (f'<feComponentTransfer in="{p["src"]}" {reg} result="{name}"><feFuncR type="discrete" tableValues="0 1"/>'
                f'<feFuncG type="discrete" tableValues="0 1"/><feFuncB type="discrete" tableValues="0 1"/></feComponentTransfer>')
    elif kind == "conv":
        K = p["K3"][::-1, ::-1].ravel()
        return (f'<feConvolveMatrix in="{p["src"]}" order="3" kernelMatrix="{" ".join(fmt(k) for k in K)}" divisor="1" '
                f'bias="0" preserveAlpha="true" edgeMode="none" {reg} result="{name}"/>')
    elif kind == "neuron":
        K = p["img"][::-1, ::-1].ravel()        # feConvolveMatrix flips the kernel
        return (f'<feConvolveMatrix in="{p["src"]}" order="{N}" targetX="{p["x"]}" targetY="{p["y"]}" divisor="1" '
                f'bias="{fmt(p["bias"])}" preserveAlpha="true" edgeMode="none" {reg} result="{name}" '
                f'kernelMatrix="{" ".join(fmt(k) for k in K)}"/>')
    elif kind == "merge":
        nodes = "".join(f'<feMergeNode in="{s}"/>' for s in p["srcs"])
        return f'<feMerge {reg} result="{name}">{nodes}</feMerge>'
    elif kind == "dilate":
        rad = p["r"] if "r" in p else f'{p["rx"]} {p["ry"]}'
        return f'<feMorphology in="{p["src"]}" operator="dilate" radius="{rad}" {reg} result="{name}"/>'
    elif kind == "chan":
        m = {"R": "1 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 1 0", "G": "0 0 0 0 0  1 0 0 0 0  0 0 0 0 0  0 0 0 1 0"}[p["keep"]]
        return f'<feColorMatrix in="{p["src"]}" type="matrix" values="{m}" {reg} result="{name}"/>'
    elif kind == "near":
        return f'<feComposite in="{p["a"]}" in2="{p["b"]}" operator="arithmetic" k2="1" k3="-1" k4="1" {reg} result="{name}"/>'
    elif kind == "isone":
        table = " ".join(["0"] * 255 + ["1"])
        return (f'<feComponentTransfer in="{p["src"]}" {reg} result="{name}"><feFuncR type="discrete" tableValues="{table}"/>'
                f'<feFuncG type="discrete" tableValues="{table}"/><feFuncB type="discrete" tableValues="{table}"/>'
                f'<feFuncA type="linear" slope="0" intercept="1"/></feComponentTransfer>')
    elif kind == "taps":
        xs = [dx for dx, dy in p["taps"]] + [0]; ys = [dy for dx, dy in p["taps"]] + [0]
        tx, ty = -min(xs), -min(ys)
        ox, oy = max(xs) + tx + 1, max(ys) + ty + 1
        K = np.zeros(ox * oy)
        for (dx, dy), w in p["taps"].items():
            j, i = dx + tx, dy + ty
            K[(oy - 1 - i) * ox + (ox - 1 - j)] = w
        return (f'<feConvolveMatrix in="{p["src"]}" order="{ox} {oy}" targetX="{tx}" targetY="{ty}" divisor="1" bias="0" '
                f'preserveAlpha="true" edgeMode="none" {reg} result="{name}" kernelMatrix="{" ".join(fmt(k) for k in K)}"/>')
    elif kind == "not":
        return (f'<feComponentTransfer in="{p["src"]}" {reg} result="{name}"><feFuncR type="linear" slope="-1" intercept="1"/>'
                f'<feFuncG type="linear" slope="-1" intercept="1"/><feFuncB type="linear" slope="-1" intercept="1"/></feComponentTransfer>')
    elif kind == "mul":
        return f'<feComposite in="{p["a"]}" in2="{p["b"]}" operator="arithmetic" k1="1" {reg} result="{name}"/>'
    elif kind == "diff":
        return f'<feBlend in="{p["a"]}" in2="{p["b"]}" mode="difference" {reg} result="{name}"/>'
    elif kind == "iszero":
        table = " ".join(["1"] + ["0"] * 255)
        return (f'<feComponentTransfer in="{p["src"]}" {reg} result="{name}"><feFuncR type="discrete" tableValues="{table}"/>'
                f'<feFuncG type="discrete" tableValues="{table}"/><feFuncB type="discrete" tableValues="{table}"/>'
                f'<feFuncA type="linear" slope="0" intercept="1"/></feComponentTransfer>')
    elif kind == "sum":
        return f'<feComposite in="{p["a"]}" in2="{p["b"]}" operator="arithmetic" k2="1" k3="1" {reg} result="{name}"/>'
    elif kind == "offset":
        return f'<feOffset in="{p["src"]}" dx="{p["dx"]}" dy="{p["dy"]}" {reg} result="{name}"/>'
    raise ValueError(kind)


def to_svg(g, fid="net"):
    """One filter per stage, #<fid>1, #<fid>2, ..., for groups nested innermost first. A reference to the
    previous stage's last result becomes SourceGraphic; any other reference across stages is an error."""
    out, last = [], None
    for k, ops in enumerate(g.stages(), 1):
        out.append(f'<filter id="{fid}{k}" x="0" y="0" width="{N}" height="{N}" filterUnits="userSpaceOnUse" '
                   f'primitiveUnits="userSpaceOnUse" color-interpolation-filters="sRGB">')
        seen = {"SourceGraphic"} if k == 1 else set()

        def ref(r):
            if r == last:
                return "SourceGraphic"
            assert r in seen, f"{r} is not in filter {k}"
            return r
        for name, kind, p in ops:
            p = dict(p)
            for key in ("src", "a", "b"):
                if key in p:
                    p[key] = ref(p[key])
            if "srcs" in p:
                p["srcs"] = [ref(r) for r in p["srcs"]]
            out.append(primitive(name, kind, p))
            seen.add(name)
        out.append("</filter>")
        last = ops[-1][0]
    return "\n".join(out)
