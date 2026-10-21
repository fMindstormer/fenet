"""The test numbers, once mlp.npz, alpha.json and the page are fixed. Nothing reads its output back
into a choice: the checkpoint and the score scale come from validation (train_mlp.py, check_mlp.py).

    python evaluate.py [n]      model accuracy on all 10,000 test digits, emulated filter on the first n
"""
import json, sys
import numpy as np
import train as T
from train_mlp import MLP, blur
from compile_mlp import build, emulate
from eval_thin import thin28

m = dict(np.load("mlp.npz"))
net = MLP(); net.p = m
alpha = json.load(open("alpha.json"))["alpha"]
_, _, (Xt, yt) = T.load()
clean = T.batch_grid(Xt, range(len(Xt)), None, aug=False)[:, 0]
skel = np.stack([thin28(x) for x in Xt])
pred = net.predict(blur(clean))
print(f"test, pooled to 14x14:            {np.mean(pred == yt):.4f}")
print(f"test, skeletonized to one cell:   {np.mean(net.predict(blur(skel)) == yt):.4f}")
g = build(m, alpha)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
agree = correct = ties = 0
for i in range(n):
    w, s, ink = emulate(g, clean[i])
    ties += len(w) != 1; agree += w == [pred[i]]; correct += w == [yt[i]]
print(f"emulated filter, first {n} test digits: agrees with the model {agree}, correct {correct}, ties {ties}")
