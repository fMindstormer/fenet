The earlier attempt: a small CNN laid out as tiles. Its second layer needed kernels over 256 taps,
which Chromium ignores, so the project moved to one feConvolveMatrix per neuron.
