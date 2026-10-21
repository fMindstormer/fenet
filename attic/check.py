"""Check the compiled graph: alpha from the model's scores, emulator vs NumPy model on test digits."""
import json
import sys
import numpy as np
import train as T
from compile import build, emulate

model = dict(np.load("model.npz"))
net = T.Net()
net.p = {k: model[k] for k in ("K1", "b1", "K2", "b2", "K3", "b3", "W", "c")}
X, y, Xt, yt = T.load()
xt = T.batch_grid(Xt, range(len(Xt)), None, aug=False)
logits = np.concatenate([net.forward(xt[i:i + 500])[0] for i in range(0, len(xt), 500)])
pred = logits.argmax(1)
alpha = 0.45 / float(np.quantile(logits.max(1), 0.995))
json.dump({"alpha": alpha}, open("alpha.json", "w"))
print(f"model test accuracy {np.mean(pred == yt):.4f}; alpha {alpha:.4f}")
g = build(model, alpha)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
agree = correct = ties = 0
for i in range(n):
    winners, scores, ink = emulate(g, xt[i, 0])
    ties += len(winners) != 1
    agree += winners == [pred[i]]
    correct += winners == [yt[i]]
print(f"emulated {n}: agrees with the model on {agree}, correct {correct}, ties/none {ties}")
