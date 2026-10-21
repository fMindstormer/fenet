"""Score scale from the perceptron's validation scores; emulated filter vs the NumPy model on validation digits."""
import json, sys
import numpy as np
import train as T
from train_mlp import MLP, blur
from compile_mlp import build, emulate

m = dict(np.load("mlp.npz"))
net = MLP(); net.p = m
_, (Xv, yv), _ = T.load()
grids = T.batch_grid(Xv, range(len(Xv)), None, aug=False)[:, 0]
xb = blur(grids)
logits = np.concatenate([net.forward(xb[i:i + 1000])[0] for i in range(0, len(xb), 1000)])
pred = logits.argmax(1)
alpha = 0.47 / float(np.quantile(logits.max(1), 0.995))
json.dump({"alpha": alpha}, open("alpha.json", "w"))
print(f"model validation accuracy {np.mean(pred == yv):.4f}, alpha {alpha:.4f}")
g = build(m, alpha)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
agree = correct = ties = 0
for i in range(n):
    w, s, ink = emulate(g, grids[i])
    ties += len(w) != 1; agree += w == [pred[i]]; correct += w == [yv[i]]
print(f"emulated {n}: agrees with the model {agree}, correct {correct}, ties {ties}")
