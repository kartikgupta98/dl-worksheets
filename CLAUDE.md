# dl-worksheets

Interactive deep learning worksheets, published with GitHub Pages at
https://kartikgupta98.github.io/dl-worksheets/ straight from `main` at the repo root.
Pushing to `main` publishes. There is no build step.

Read the parent folder's CLAUDE.md first: it carries the writing and teaching rules.

## The one hard constraint

**Every worksheet is a single self-contained `.html` file.** All CSS, JavaScript,
maths and data inlined. No CDN, no `<script src>` to the network, no npm, no build.
Students open the file or the Pages URL and it works offline.

That means:
- **Maths is hand built in HTML and CSS.** No MathJax, no KaTeX. Match the existing
  fraction, subscript and matrix markup in `worksheet09_regularisation.html`.
- Training demos are plain JavaScript on a canvas or SVG. No TensorFlow.js.
- Fonts are system stacks only, declared as `--serif`, `--sans`, `--mono` in `:root`.

## House design system

Each file opens with a `:root` block of CSS variables, then styles. Copy the palette
from the most recent worksheet rather than inventing one:

- Warm paper background (`--bg:#FAF8F4`), white panels, ink `#2A2E3A`, not pure black.
- Muted accents: blue `#4E6FA3`, rose `#B4606B`, amber `#C48A3F`, green `#3C8C5A`,
  each with a `-soft` and `-tint` variant for fills.
- Serif for prose, sans for UI and labels, mono for code.
- `.wrap` at about 820px, generous top padding, radius `--r` around 10px.

## Worksheets

| File | Topics | On `main`? |
|---|---|---|
| `activations.html` | sigmoid, tanh, ReLU, saturation, dying units | yes |
| `backprop_worksheet_output_layer.html` | forward pass, loss, chain rule, gradients | yes |
| `lecture06_gradient_descent.html` | slope, learning rate, step size, convergence | yes |
| `worksheet07_momentum_nag.html` | SGD, EWMA, momentum, Nesterov | yes |
| `worksheet08_adaptive_lr.html` | AdaGrad, RMSProp, Adam, bias correction | **no, still only on `fraud-competition`** |
| `worksheet09_regularisation.html` | overfitting, L2, L1, early stopping, dropout, live training lab | yes |
| `worksheet09_cnn.html` | pixels, Sobel, convolution, ReLU, padding, stride, pooling, LeNet | yes |
| `worksheet10_architectures.html` | LeNet, AlexNet, VGG-16: diagrams, real activation maps, shapes, parameter counts (~22 MB, data inlined; rebuild with `worksheet_tools/`) | yes |

## Adding a worksheet

1. New self-contained `.html` at the repo root.
2. Add a card `<li>` to `index.html`.
3. Add a row to the README table and to the table above.
4. QC, then commit and push to `main`.

## Competition

`fraud_competition/` is the "Flagged or Fraud?" in-class Kaggle competition.
Its `HANDOFF.md` is the entry point and is kept current: read it before touching
anything there. Key points that are easy to get wrong:

- `materials/technique_cards.md` is the source. The two HTML handouts are **built**
  by `make_cards_html.py`, never edited by hand, and published on `main` as
  `flagged_or_fraud_techniques.html` (no code) and `flagged_or_fraud_techniques_code.html`
  (with closed "Hint: show code" dropdowns).
- Technique cards are hints, not copy-paste code: what to try, where it goes, which
  functions to look up, how to check it worked. Exact code lives in the instructor
  files only.
- Numbers quoted in the materials come from `results_exp2.csv`. If a knob changes,
  recalibrate rather than editing the numbers by hand.

## Current working state (2026-09-20)

Checked out on branch `fraud-competition`, which is ahead of `main` with the
competition work and worksheet 08, and behind on worksheet 09. Uncommitted here:
modified `README.md` and `index.html`, untracked `worksheet09_regularisation.html`.
Worksheet 08 has never been merged to `main`, so it is not live on the site.
