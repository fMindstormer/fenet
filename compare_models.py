"""Compare trained models on validation digits: python compare_models.py a.npz b.npz ..."""
import sys
import numpy as np
import train as T
from train_mlp import MLP, blur, thin_augment
from eval_thin import thin28
_, (Xv, yv), _ = T.load()
clean = np.stack([T.to_grid(Xv[i], 0.3) for i in range(len(Xv))])
sk = np.stack([thin28(Xv[i]) for i in range(len(Xv))])
r = np.random.default_rng(3); ta = np.stack([thin_augment(Xv[i], r) for i in range(len(Xv))])
for f in sys.argv[1:]:
    net = MLP(); net.p = dict(np.load(f))
    print(f, " ".join(f"{n} {(net.predict(blur(G)) == yv).mean():.4f}" for n, G in (("clean", clean), ("skeleton", sk), ("thin+transform", ta))))
