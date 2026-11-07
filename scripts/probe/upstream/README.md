# Upstream: feConvolveMatrix and its primitive subregion

Not filed yet. To file:

1. **Tests.** Copy `wpt/*.html` into `css/filter-effects/` and `wpt/reference/*.html` into
   `css/filter-effects/reference/` of a fork of
   [web-platform-tests/wpt](https://github.com/web-platform-tests/wpt), and open a pull request. The
   `.tentative` names mark behaviour the spec doesn't settle yet. Measured with Playwright 1.64: both
   pass in Firefox 157 and WebKit 27.2; Chromium 156 passes `duplicate` and fails `none` (5,000 red
   pixels).
2. **Issue.** Open `issue.md` as a new issue on
   [w3c/csswg-drafts](https://github.com/w3c/csswg-drafts/issues/new) (Filter Effects moved there from
   w3c/fxtf-drafts, whose issue #324 asks the same question in general), with the tests' pull request
   linked where it says "link".

`scripts/probe/subregion.py` is the same difference with one-pixel subregions, as fenet hit it.
