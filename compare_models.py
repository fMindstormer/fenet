import sys
import numpy as np
import train as T
from train_mlp import MLP, blur, thin_augment
from eval_thin import thin28
X, y, Xt, yt = T.load()
clean = np.stack([T.to_grid(Xt[i], 0.3) for i in range(len(Xt))])
sk = np.stack([thin28(Xt[i]) for i in range(len(Xt))])
r = np.random.default_rng(3); ta = np.stack([thin_augment(Xt[i], r) for i in range(len(Xt))])
for f in sys.argv[1:]:
    net = MLP(); net.p = dict(np.load(f))
    print(f, " ".join(f"{n} {(net.predict(blur(G)) == yt).mean():.4f}" for n, G in (("clean", clean), ("skeleton", sk), ("thin+transform", ta))))
