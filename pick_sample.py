"""Pick the page's opening example: a thin, mouse-like stroke the model reads with a wide margin."""
import json, sys
import numpy as np
import train as T
from train_mlp import MLP, blur
from eval_thin import thin28
X, y, Xt, yt = T.load()
net = MLP(); net.p = dict(np.load("mlp.npz"))
want = int(sys.argv[1]); pick = int(sys.argv[2]) if len(sys.argv) > 2 else -1
cands = []
for i in np.where(yt == want)[0][:400]:
    g = thin28(Xt[i]); lg = net.forward(blur(g[None]))[0][0]
    s = np.sort(lg)
    if lg.argmax() == want and 18 <= g.sum() <= 30:
        cands.append((s[-1] - s[-2], i, g))
cands.sort(key=lambda c: -c[0])
for m, i, g in cands[:4]:
    print(i, f"margin {m:.1f}")
    print("\n".join("".join("#" if v else "." for v in row) for row in g))
if pick >= 0:
    g = thin28(Xt[pick])
    json.dump({"source": f"MNIST test image {pick}, skeletonized and pooled to 14x14", "label": want,
               "cells": [int(v) for v in g.ravel()]}, open("sample.json", "w"))
    print("saved", pick)
