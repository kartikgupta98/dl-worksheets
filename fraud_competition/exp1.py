"""First calibration grid: ladder, leave-one-out, regularisation options, starter variants, null cards."""
KNOBS = {"sharpness": 3.6}
SEEDS = [0, 1, 2]

LADDER = [
    ("L01 +scale", dict(scale=True)),
    ("L02 +log skewed", dict(log=True)),
    ("L03 +one-hot", dict(encode="onehot")),
    ("L04 +median & missing flags", dict(impute="median_flag")),
    ("L05 +chargeback -1 flag", dict(placeholder=True)),
    ("L06 +adam", dict(optimizer="adam", lr=1e-3)),
    ("L07 +100 epochs", dict(epochs=100)),
    ("L08 +2x128 network", dict(layers=(128, 128))),
    ("L09 +dropout 0.3", dict(dropout=0.3)),
    ("L10 +drop noise cols", dict(drop_noise=True)),
    ("L11 +amount vs usual", dict(fe_ratio=True)),
    ("L12 +hour sin/cos", dict(fe_hour=True)),
    ("L13 +reduce lr", dict(reduce_lr=True)),
]

CONFIGS = {"L00 starter": {}}
cum = {}
for name, step in LADDER:
    cum = {**cum, **step}
    CONFIGS[name] = dict(cum)
FULL = dict(cum)

# full pipeline with one step undone
UNDO = {
    "scale": dict(scale=False), "log": dict(log=False), "onehot->codes": dict(encode="codes"),
    "onehot->drop": dict(encode="drop"), "impute": dict(impute="zero"), "placeholder": dict(placeholder=False),
    "adam": dict(optimizer="sgd", lr=0.01), "epochs": dict(epochs=10), "network": dict(layers=(4,)),
    "dropout": dict(dropout=0.0), "noise": dict(drop_noise=False), "fe_ratio": dict(fe_ratio=False),
    "fe_hour": dict(fe_hour=False), "reduce_lr": dict(reduce_lr=False),
}
for k, v in UNDO.items():
    CONFIGS[f"F- {k}"] = {**FULL, **v}

# regularisation options, each on the overfitting big network
BIG = CONFIGS["L08 +2x128 network"]
for k, v in {
    "dropout 0.3": dict(dropout=0.3), "dropout 0.5": dict(dropout=0.5),
    "l2 1e-4": dict(l2=1e-4), "l2 1e-3": dict(l2=1e-3), "l1 1e-4": dict(l1=1e-4),
    "early stop": dict(early_stopping=True), "drop noise": dict(drop_noise=True),
    "batchnorm": dict(batchnorm=True), "dropout+early": dict(dropout=0.3, early_stopping=True),
    "dropout+l2+early": dict(dropout=0.3, l2=1e-4, early_stopping=True),
}.items():
    CONFIGS[f"R {k}"] = {**BIG, **v}

# starter variants (the plain starter is dead at ~50%)
CONFIGS["S adam unscaled"] = dict(optimizer="adam", lr=1e-3)
CONFIGS["S sgd lr1e-3 unscaled"] = dict(lr=1e-3)
CONFIGS["S log only, no scale"] = dict(log=True)

# cards that should not help, on the full pipeline
for k, v in {
    "batchnorm": dict(batchnorm=True), "sigmoid act": dict(activation="sigmoid"),
    "5 layers": dict(layers=(128,) * 5), "batch 256": dict(batch_size=256),
    "early stop": dict(early_stopping=True), "l1 1e-4": dict(l1=1e-4),
}.items():
    CONFIGS[f"N {k}"] = {**FULL, **v}
