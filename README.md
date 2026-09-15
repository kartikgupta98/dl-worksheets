# Deep Learning: Interactive Worksheets

Self-contained, browser-only worksheets for a deep learning course. Each worksheet is a
single HTML file with all CSS, JavaScript and math inlined: no build step, no
dependencies, no network calls at runtime.

**Live site:** https://kartikgupta98.github.io/dl-worksheets/

## Contents

| Worksheet | File | Topics |
|---|---|---|
| Activation Functions | [`activations.html`](activations.html) | sigmoid, tanh, ReLU, saturation, dying units |
| Backpropagation 03: Output Layer | [`backprop_worksheet_output_layer.html`](backprop_worksheet_output_layer.html) | forward pass, loss, chain rule, weight gradients |
| Lecture 06: Gradient Descent | [`lecture06_gradient_descent.html`](lecture06_gradient_descent.html) | slope, learning rate, step size, convergence |
| Worksheet 07: Momentum & NAG | [`worksheet07_momentum_nag.html`](worksheet07_momentum_nag.html) | SGD, EWMA, momentum, Nesterov |
| Worksheet 09: Regularisation | [`worksheet09_regularisation.html`](worksheet09_regularisation.html) | overfitting, L2, L1, early stopping, dropout, live training lab |

## Kaggle competition: Flagged or Fraud?

Handouts for the in-class Kaggle competition, built from `fraud_competition/materials/technique_cards.md`
on the `fraud-competition` branch. Unlike the worksheets, these pages load their fonts from Google Fonts
(offline they fall back to system fonts).

| Page | File | Contents |
|---|---|---|
| Technique Cards | [`flagged_or_fraud_techniques.html`](flagged_or_fraud_techniques.html) | what to try, where it goes in the notebook, how to check it worked |
| Technique Cards with Code | [`flagged_or_fraud_techniques_code.html`](flagged_or_fraud_techniques_code.html) | the same cards, each with a closed code hint |

## Running locally

Because every file is self-contained, you can just open one in a browser:

```bash
start activations.html          # Windows
```

Or serve the folder if you prefer real URLs:

```bash
python -m http.server 8000
```

## Deploying

Deployment is GitHub Pages, served straight from `main` at the repository root.
Pushing to `main` publishes; there is no build.

To (re)enable it: **Settings → Pages → Source: Deploy from a branch → `main` / `/ (root)`**.

## Adding a worksheet

1. Drop the new self-contained `.html` file in the repository root.
2. Add a matching `<li>` card to `index.html`.
3. Add a row to the table above.
4. Commit and push.

## License

Not yet licensed. Without a license file, default copyright applies and others may
not reuse the material. For teaching content, **CC BY 4.0** is the usual choice;
add it as `LICENSE` when you're ready.
