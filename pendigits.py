"""UCI Pendigits as pad drawings: 10,992 digits written with a stylus by 44 people, kept as pen paths.

    python pendigits.py [heights]     validation accuracy at each digit height, e.g. 7,10,13

A digit's path is scaled to a height on the 14x14 pad (10 cells by default, MNIST's proportion:
20 of 28 pixels), centred, and drawn as a one-cell line through every cell it passes, as a mouse
drag paints the pad (replayed as real drags, 200 validation digits got the model's same answer for 198;
the differences are cells that a line only grazes).

Splits: pendigits-orig.tes, 14 writers, is test, read only by evaluate.py. pendigits-orig.tra, 30 other
writers, is split at random into train (two thirds) and validation. Its order is shuffled, so the two
share writers, and validation flatters the model on these writers' styles: it only helps pick the
checkpoint, alongside MNIST validation.

The files come from https://archive.ics.uci.edu/dataset/81 (CC BY 4.0): pendigits-orig.tra and
pendigits-orig.tes, uncompressed, in data/pendigits/.
"""
import sys
import numpy as np

N = 14
FILES = {"tra": "data/pendigits/pendigits-orig.tra", "tes": "data/pendigits/pendigits-orig.tes"}


def parse(path):
    """UNIPEN text -> [(label, [stroke, ...])], a stroke an (n, 2) array of tablet x, y (y up)."""
    out, strokes, stroke = [], None, None
    for line in open(path, encoding="latin1"):
        t = line.split()
        if not t:
            continue
        if t[0] == ".SEGMENT":
            strokes = []
            out.append((int(t[-1].strip('"')), strokes))
        elif t[0] == ".PEN_DOWN":
            stroke = []
            strokes.append(stroke)
        elif t[0] == ".PEN_UP":
            stroke = None
        elif stroke is not None and len(t) == 2 and not t[0].startswith("."):
            stroke.append((int(t[0]), int(t[1])))
    return [(label, [np.array(s, float) * (1, -1) for s in st if s]) for label, st in out if st]


def split(name):
    """train / val (from the 30 writers of .tra) or test (the 14 of .tes): [(label, strokes)]."""
    if name == "test":
        return parse(FILES["tes"])
    data = parse(FILES["tra"])
    order = np.random.default_rng(0).permutation(len(data))
    cut = 2 * len(data) // 3
    return [data[i] for i in (order[:cut] if name == "train" else order[cut:])]


def draw(strokes, height=10.0, A=np.eye(2), shift=(0.0, 0.0)):
    """Pen strokes (y down) -> 14x14 cells: scaled to `height` cells (narrower than 13), transformed
    by A about the centre, shifted, kept inside the pad, and drawn as a line sampled every 0.05 cell."""
    pts = np.concatenate(strokes)
    lo, hi = pts.min(0), pts.max(0)
    s = min(height / max(hi[1] - lo[1], 1), 13 / max(hi[0] - lo[0], 1))
    paths = [((st - (lo + hi) / 2) * s) @ A.T for st in strokes]
    allp = np.concatenate(paths)
    c = np.clip(N / 2 + np.asarray(shift), 0.02 - allp.min(0), N - 0.02 - allp.max(0))
    g = np.zeros((N, N), np.float32)
    for p in paths:
        p = p + c
        d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        t = np.arange(0, d[-1] + 0.05, 0.05)
        q = np.clip(np.c_[np.interp(t, d, p[:, 0]), np.interp(t, d, p[:, 1])].astype(int), 0, N - 1)
        g[q[:, 1], q[:, 0]] = 1
    return g


def augment(strokes, r):
    """Random size, slant, rotation and position, as train.augment does for MNIST, and gaps from a fast stroke."""
    rot = np.deg2rad(r.uniform(-14, 14))
    A = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]]) @ np.array([[1, r.uniform(-0.25, 0.25)], [0, 1]])
    g = draw(strokes, 10 * r.uniform(0.75, 1.3), A, r.uniform(-2, 2, size=2))
    if r.random() < 0.3:
        g[r.random(g.shape) < 0.06] = 0
    return g


def load(name, height=10.0):
    data = split(name)
    return np.stack([draw(st, height) for _, st in data]), np.array([label for label, _ in data])


if __name__ == "__main__":
    from train_mlp import MLP, blur
    heights = [float(h) for h in sys.argv[1].split(",")] if len(sys.argv) > 1 else [10.0]
    net = MLP(); net.p = dict(np.load("mlp.npz"))
    for h in heights:
        G, y = load("val", h)
        pred = net.predict(blur(G))
        print(f"validation, {len(y)} digits, height {h:g} cells: model correct {np.mean(pred == y):.3f}")
    if len(heights) > 1:
        sys.exit()
    print("digit    n  correct  most often read as (when wrong)")
    for d in range(10):
        k = y == d
        wrong = pred[k][pred[k] != d]
        print(f"  {d}   {k.sum():4d}  {np.mean(pred[k] == d):.3f}    {np.bincount(wrong, minlength=10).argmax() if len(wrong) else '-'}")
