<!-- Draft for https://github.com/w3c/csswg-drafts/issues/new (Filter Effects issues moved there from
w3c/fxtf-drafts). Not filed yet. -->

# [filter-effects-1] Does a filter primitive's subregion bound its input image? Engines differ for feConvolveMatrix edgeMode="none"

w3c/fxtf-drafts#324 asks how big the "input image" is whose edges `edgeMode` extends. It has been open
since 2018. Here is a concrete case where engines disagree, and where all three already agree on a
reading that the spec text doesn't say.

Spec text, [Filter primitive subregion](https://drafts.csswg.org/filter-effects-1/#FilterPrimitiveSubRegion):

> The filter region acts as a hard clip clipping rectangle on the filter primitive's input image(s).
> The filter primitive subregion acts as a hard clip clipping rectangle on the filter primitive result.

Read literally, a primitive's subregion clips only its result, and its input extends to the filter
region. Then a convolution with a small subregion should read input pixels outside that subregion, and
`edgeMode` should only matter at the filter region's edge.

Test case: a 200 × 100 filter region over green (left half) and red (right half). A convolution whose
result at x is its input at x + 50 has the subregion (0, 0, 100, 100), the green half.

```html
<svg width="200" height="100">
  <filter id="f" filterUnits="userSpaceOnUse" primitiveUnits="userSpaceOnUse" x="0" y="0" width="200" height="100">
    <feConvolveMatrix order="51 1" targetX="0" targetY="0" kernelMatrix="1 0 0 … (50 zeros)" divisor="1"
                      edgeMode="none" preserveAlpha="false" x="0" y="0" width="100" height="100"/>
  </filter>
  <g filter="url(#f)">
    <rect x="0" y="0" width="100" height="100" fill="green"/>
    <rect x="100" y="0" width="100" height="100" fill="red"/>
  </g>
</svg>
```

Columns 50–99 of the result read input columns 100–149, which are inside the filter region but outside
the subregion:

| | edgeMode="none" | edgeMode="duplicate" |
|---|---|---|
| Chromium 156 | red (reads beyond the subregion) | green (repeats the subregion's last column) |
| Firefox 157 | transparent (input ends at the subregion) | green |
| Safari (WebKit 27.2, Playwright) | transparent | green |
| literal spec text | red | red |

So for `duplicate`, every engine treats the primitive subregion as the edge of the input image, which
the text above doesn't say. For `none`, Chromium reads past it and the others don't. That matches the
Gecko description in #324: the input image rectangle is the intersection of the filter region and the
primitive subregion.

Proposal: define a primitive's input image as its input clipped to the intersection of the filter region
and the primitive's own subregion, and say that `edgeMode` extends that image. This is what Firefox and
WebKit do in both modes and what Chromium does for `duplicate`. The alternative (input clipped only to
the filter region) matches the current sentence but no engine's `duplicate`.

Tentative tests for the proposal, which Chromium fails for `none` only:
`css/filter-effects/feconvolvematrix-subregion-edgemode-none.tentative.html` and
`…-edgemode-duplicate.tentative.html` (web-platform-tests PR: link).

How I ran into it: [fenet](https://github.com/fMindstormer/fenet) runs a small neural network in SVG
filters, one `feConvolveMatrix` per neuron with a one-pixel subregion. It worked only in Chromium.
