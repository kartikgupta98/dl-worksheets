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
| Worksheet 08: Adaptive Learning Rates | [`worksheet08_adaptive_lr.html`](worksheet08_adaptive_lr.html) | AdaGrad, RMSProp, Adam, bias correction |

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
