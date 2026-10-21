# fenet

A handwritten-digit recognizer that runs inside SVG filters. The page has no JavaScript: CSS paints the
drawing, and a chain of filter primitives (mostly `feConvolveMatrix`) evaluates a small neural network
on it and prints the answer.

Live page: https://fmindstormer.github.io/fenet/

```
drawing pad (CSS :active + :hover, transition delay of 1,000,000 s)
   │ 196 :has() rules
   ▼
14 × 14 mirror, one cell = one device pixel (media queries scale it by 1/devicePixelRatio)
   │ filters #net1–#net3 — one SVG filter per layer, on nested groups around the mirror
   ▼
threshold → 3×3 blur → 192 neurons → 96 neurons → 10 scores → argmax → tie-break
   │ CSS filter on the result SVG: url(#x<ratio>) url(#render)
   ▼
dot-matrix digit + ten score bars, coloured from the page's theme
```

## How it works

**Drawing.** `.grid:active rect:hover` turns a cell dark with `transition: fill 0s`. Every other state
of a cell has `transition: fill 0s 1000000s`, so when the pointer leaves, the cell waits a million
seconds before going back. Clear is a `<label>` for a radio button: while it is pressed every cell
gets `transition: none` and snaps back. The opening example is drawn by a rule that stops matching once
the radio is checked, and a radio stays checked.

**Mirror.** For each of the 196 cells, a rule like
`.app:has(.grid:active rect:nth-child(37):hover) .mirror rect.m:nth-of-type(38)` paints the matching
cell of a 14 × 14 square inside the result SVG, with the same million-second trick. The square sits in
a group scaled by `1/devicePixelRatio` (one media query per supported ratio), so each cell is exactly
one device pixel, which is the unit `feConvolveMatrix` works in.

**Network.** One `feConvolveMatrix` per neuron. The kernel is 14 × 14 (196 taps), its target is the
neuron's own pixel, and its subregion is that one pixel, so the primitive computes a full dot product
over the previous layer's image plus a bias, and clamps it to 0…1. Each layer is merged onto a black
square, because `preserveAlpha` copies the source's alpha.

Each layer is its own filter, on nested groups with the first layer innermost. A group's content is drawn
once before its filter runs, so a layer reads the one below as a finished image. In one big filter,
Chromium's GPU raster would compute layer 1 again for every layer-2 neuron and give up (see below).
The ink flag, whether anything is drawn at all, rides along in pixels no neuron uses: (13, 6) in
layer 1, (13, 9) in layer 2.

| stage | primitives |
|---|---|
| threshold + 3×3 blur | 2 |
| hidden layer 1 (positions spread over the 14 × 14 image), ink flag | 192 + flood + 1 + merge |
| hidden layer 2, ink flag moved to (13, 9) | 96 + flood + 1 + merge |
| scores at slots (1 + 4·(k mod 4), 1 + 4·⌊k/4⌋) | 10 + flood + merge |
| argmax: `feMorphology` dilate to the max, `feComposite` 1 + score − max, lookup table "is exactly 1" | 3 |
| tie-break: one 25 × 9 convolution, +1 at the pixel, −1 wherever an earlier slot can sit | 1 |
| output: winner block and ink flag in red, each score as a 3 × 3 block in green | 8 |

Everything the display reads is a uniform 3 × 3 block of device pixels, centred on a slot: the
winner flag in red and the score in green. At fractional ratios the compositor applies the CSS filter at
a coarser scale than the content was drawn, so a single pixel blurs into its neighbours, while the
middle of a block keeps its value.

**Display.** The result SVG's CSS filter is two filters in a row. `#x<ratio>` copies the middle pixel of
each slot's block (flag and score) and of the ink block into 4 × 4 CSS-pixel blocks; there is one per
supported ratio. `#render` (shared) tiles each block over the readout and masks a 5 × 7 dot-matrix digit
(`feImage` of a data-URI SVG) and a segmented bar per class. Bar segment *i* lights when the score
reaches its level: `feComposite arithmetic` with k2 = 255, k3 = −255, k4 = 1 against a stepped ramp.
Colours come from `flood-color: var(--mark)` and friends, so the readout follows the page's light or
dark theme.

## Results

| | |
|---|---|
| network | 196 → 192 → 96 → 10, 57,322 weights, clipped-linear activations, 8-bit |
| MNIST test, pooled to 14 × 14 | 97.0 % |
| MNIST test, skeletonized to one-cell strokes | 95.0 % |
| emulator vs NumPy model | identical except ties, which the filter gives to the lower digit |
| Chromium vs NumPy model | the page prints the model's answer for 199 of 200 test digits at each of 8 zoom levels from 0.75 to 3, with GPU and with software raster (`logs_agreement.txt`). The odd one is a near-tie: Chromium's convolutions don't round exactly like the emulator |
| page size | about 710 kB, all inline except the Google Fonts stylesheet |
| time per painted cell | under 0.1 s in Chromium 156 on an Apple M4, GPU or software raster (`time_frames.py`, screenshot included) |

## Chromium facts it depends on

Found with the probes in `probe/`, in headless Chromium with software raster unless GPU raster is named
(`probe28.py`: Chromium 156, Apple M4):

- `feConvolveMatrix` works in device pixels and ignores `kernelUnitLength`.
- A kernel with more than 256 taps is ignored and the result is transparent. 196 fits.
- `edgeMode="none"` reads zero outside the input; with `preserveAlpha` the result keeps the source alpha.
- Results are stored as 8-bit premultiplied colour between primitives. Training rounds every activation
  the same way (straight-through gradients).
- With software raster, a shared input is computed once when every consumer asks for the same
  rectangle, as a layer's neurons do. GPU raster (Chrome and Arc on a Mac) computes it again for every
  consumer, and a filter whose work grows too far comes out transparent: 96 layer-2 neurons over a
  196-primitive layer 1 did, and the limit fell between 43 and 44 of them. Hence one filter per layer.
- With software raster, a shared node with consumers of different kinds can make Chromium evaluate
  everything above it again: `feBlend` was the worst (the argmax took five times longer than with
  `feComposite arithmetic`), and a second branch of score neurons cost almost as much.
- A filter sees its source only inside its own filter region.
- An element with `will-change: transform` is rasterized from a whole device pixel at a scale of
  max(1, devicePixelRatio). That makes the mirror land on pixels at fractional ratios. Below a ratio of 1
  the CSS filter on that layer runs at the screen's scale while its content is drawn at 1, so the readout
  is scaled up by 1/ratio to bring the two together.
- A CSS filter on a composited layer runs at a coarser scale than the layer's content at fractional
  ratios: one device pixel of content smears over its neighbours. Values meant to be read are 3 × 3 blocks.
- `feTile` from a subregion smaller than a device pixel reads nothing; reading a 3 × 3 CSS-pixel square
  inside a uniform 4 × 4 block works at every ratio tested.
- Gradients are dithered, so the bar ramp is drawn as solid steps.
- `visibility: hidden` does not stop a filter from painting; unsupported ratios use `display: none`.
- An SVG filter's output is its last primitive, whatever its `result` name.

## Rebuild

Needs Python 3 with NumPy, SciPy, scikit-image, Pillow and Playwright (`pip install -r requirements.txt`,
then `playwright install chromium`). MNIST comes from `mnist.pkl.gz` of
mnielsen/neural-networks-and-deep-learning, placed in `data/`. Run everything from the repository root.

```
python train_mlp.py 40 mlp.npz                 # 192 and 96 hidden units, about 6 minutes on a CPU
python check_mlp.py                            # score scale (alpha.json), emulator vs model
python pick_sample.py 5 1967                   # the opening example (sample.json)
python build_page.py                           # site/index.html and site/test.html
python browser_check_mlp.py 200 1,1.25,2      # filter vs model in Chromium, by device pixel ratio
python test_draw.py 1,1.5,2                    # real mouse strokes, Clear, the dot-matrix answer
```

The Chromium checks run full Chromium, which rasterizes on the GPU where there is one, as desktop
browsers do; `RASTER=sw` forces software raster.

`site/test.html` is the standalone page; GitHub Pages builds and serves it (`.github/workflows/pages.yml`).
`site/index.html` is the same page without a document skeleton, for hosts that add their own.
`build_page.py` and `test_draw.py` run without MNIST. The scripts in `probe/` and `attic/` import
modules from the root, so run them as `PYTHONPATH=. python probe/probe21.py`.

## Files

| file | role |
|---|---|
| `train.py` | MNIST loading, 14 × 14 pooling, augmentation (also holds the earlier CNN attempt) |
| `train_mlp.py` | the perceptron, thin-stroke augmentation, quantization-aware training |
| `compile_mlp.py` | network → filter graph, NumPy emulator of the graph, SVG writer |
| `build_page.py` | the page: drawing rules, ratio media queries, extract and render filters |
| `site/template.html` | page markup and styles |
| `check_mlp.py`, `eval_thin.py`, `compare_models.py` | accuracy and emulator checks |
| `browser_check_mlp.py`, `test_draw.py` | Chromium checks |
| `probe/` | the experiments behind the facts above, and screenshot helpers |
| `attic/` | the earlier CNN attempt, stopped by the 256-tap limit |
| `mlp.npz`, `alpha.json`, `sample.json` | trained weights, score scale, opening example |
| `facts.json` | accuracy and agreement figures quoted on the page |
| `train.log`, `logs_agreement.txt` | training run and the Chromium agreement run |

## Limits

- Chromium only. Other engines place filter pixels differently and may misread.
- The page must be shown unscaled. A frame scaled by its parent (the claude.ai artifact viewer shows pages
  at 90 %) draws it at a scale the media queries can't see, and the network reads an empty pad.
- Zoom: any device pixel ratio of 1, the 65 ratios above 1 listed by `ratios()` in `build_page.py`, and ten common ratios below 1.
  Other zoom levels show a note asking for 100 %.
- Touch: dragging does not move `:hover`, so on a touchscreen cells are tapped one at a time.
- Drawing only adds ink; Clear starts over.

## License

MIT, Fedir Yudin. MNIST by Yann LeCun, Corinna Cortes and Christopher Burges.
