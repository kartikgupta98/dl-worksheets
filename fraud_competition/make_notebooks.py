"""
Builds the notebooks from one description of the starter plus the technique cards.

    python make_notebooks.py

materials/starter_notebook.ipynb              for students (Google Colab, with train.csv / test.csv uploaded)
materials/instructor_solution_notebook.ipynb  every helpful card applied, in the instructor guide's order
materials/instructor_card_solutions.md        the exact change each card makes to the notebook

The starter matches exp2.START (columns, zero fill, network, optimizer, epochs, split), and
each card's code mirrors the matching pipeline.py switch, so the notebooks land on the
calibrated scores. test_cards.py runs them and checks.
"""
import difflib
import json
from pathlib import Path

import exp2
from pipeline import CATEGORICAL, NOISE, SKEWED, STARTER

HERE = Path(__file__).parent
S = {**STARTER, **exp2.START}

# statements inside the preprocessing loop run in the same order as pipeline.preprocess
LOOP_ORDER = ["A8", "E1", "E3", "E2", "A3", "A7"]
SPLIT_ORDER = ["A6", "A2"]  # fill blanks, then scale
CALLBACK_ORDER = ["D4", "B4"]


def q(values):
    return json.dumps(values)


def starter_state():
    return dict(
        imports=["import numpy as np", "import pandas as pd", "import matplotlib.pyplot as plt", "import keras",
                 "from sklearn.model_selection import train_test_split"],
        loop={}, all_numeric=False, drop_noise=False, encode=None, fill_zero=True, after_split={},
        hidden=list(S["layers"]), activation=S["activation"], batchnorm=False, dropout=S["dropout"],
        regularizer=None, optimizer=S["optimizer"], lr=S["lr"], epochs=S["epochs"],
        batch_size=S["batch_size"], callbacks={}, seeds=1,
    )


# ---- technique cards: each one changes the notebook description ----

def a1(nb):
    nb["all_numeric"] = True


def a2(nb):
    nb["imports"].append("from sklearn.preprocessing import StandardScaler")
    nb["after_split"]["A2"] = ["scaler = StandardScaler().fit(X_train)", "X_train = scaler.transform(X_train)",
                               "X_val = scaler.transform(X_val)", "X_test = scaler.transform(X_test)"]


def a3(nb):
    nb["loop"]["A3"] = [f"for col in {q(SKEWED)}:", "    df[col] = np.log1p(df[col])"]


def a4(nb):
    nb["encode"] = "onehot"


def a5(nb):
    nb["encode"] = "codes"


def a6(nb):
    nb["fill_zero"] = False
    nb["after_split"]["A6"] = ["medians = X_train.median()", "X_train = X_train.fillna(medians)",
                               "X_val = X_val.fillna(medians)", "X_test = X_test.fillna(medians)"]


def a7(nb):
    nb["loop"]["A7"] = ["for col in train.columns[train.isna().any()]:",
                        '    df[col + "_missing"] = df[col].isna().astype(int)']


def a8(nb):
    nb["loop"]["A8"] = ['df["never_chargeback"] = (df["days_since_last_chargeback"] == -1).astype(int)',
                        'df["days_since_last_chargeback"] = np.log1p(df["days_since_last_chargeback"].replace(-1, np.nan))']


def a9(nb):
    nb["drop_noise"] = True


def b1(nb):
    nb["optimizer"], nb["lr"] = "adam", 0.001


def b2(nb):
    nb["epochs"] = 100


def b3(nb):
    nb["batch_size"] = 256


def b4(nb):
    nb["callbacks"]["B4"] = 'reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)'


def c1(nb):
    nb["hidden"] = [128, 128]


def c2(nb):
    nb["hidden"] = [128] * 5


def c3(nb):
    nb["activation"] = "sigmoid"


def c4(nb):
    nb["batchnorm"] = True


def d1(nb):
    nb["dropout"] = 0.3


def d2(nb):
    nb["regularizer"] = "keras.regularizers.L2(1e-3)"


def d3(nb):
    nb["regularizer"] = "keras.regularizers.L1(1e-4)"


def d4(nb):
    nb["callbacks"]["D4"] = ('early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, '
                             'restore_best_weights=True)')


def e1(nb):
    nb["loop"]["E1"] = ['df["amount_vs_usual"] = np.log(df["amount_inr"] / df["customer_avg_spend_90d"])']


def e2(nb):
    nb["loop"]["E2"] = ['df["hour_sin"] = np.sin(2 * np.pi * df["transaction_hour"] / 24)',
                        'df["hour_cos"] = np.cos(2 * np.pi * df["transaction_hour"] / 24)']


def e3(nb):
    nb["loop"]["E3"] = ['df["amount_is_round"] = (df["amount_inr"] % 500 == 0).astype(int)']


def f1(nb):
    nb["seeds"] = 3


CARDS = {
    "A1": ("Use more columns", a1), "A2": ("Scale the inputs", a2), "A3": ("Log-transform skewed columns", a3),
    "A4": ("One-hot encode text columns", a4), "A5": ("Label-encode text columns", a5),
    "A6": ("Fill blanks with the median", a6), "A7": ("Add \"was missing\" flags", a7),
    "A8": ("Handle special codes", a8), "A9": ("Drop columns that cannot carry signal", a9),
    "B1": ("Change the optimizer or learning rate", b1), "B2": ("Train for more epochs", b2),
    "B3": ("Change the batch size", b3), "B4": ("Lower the learning rate when progress stalls", b4),
    "C1": ("A wider or deeper network", c1), "C2": ("A very deep network", c2),
    "C3": ("A different activation", c3), "C4": ("Batch normalization", c4),
    "D1": ("Dropout", d1), "D2": ("L2 penalty", d2), "D3": ("L1 penalty", d3), "D4": ("Early stopping", d4),
    "E1": ("Compare the amount with the customer's usual spend", e1), "E2": ("Treat the hour as a circle", e2),
    "E3": ("Look at the amounts themselves", e3), "F1": ("Average several models", f1),
}

# the instructor guide's path: (calibration config in results_exp2.csv, cards added at that step)
LADDER = [
    ("L00 starter", []), ("L01 +scale", ["A2"]), ("L02 +all numeric columns", ["A1"]),
    ("L03 +log skewed", ["A3"]), ("L04 +one-hot", ["A4"]), ("L05 +median & missing flags", ["A6", "A7"]),
    ("L06 +chargeback -1 flag", ["A8"]), ("L07 +adam", ["B1"]), ("L08 +100 epochs", ["B2"]),
    ("L09 +2x128 network", ["C1"]), ("L10 +dropout 0.3", ["D1"]), ("L11 +drop noise cols", ["A9"]),
    ("L12 +amount vs usual", ["E1"]), ("L13 +hour sin/cos", ["E2"]), ("L14 +round amounts", ["E3"]),
    ("L15 +reduce lr", ["B4"]),
]
FULL = LADDER[-1][0]
# (calibration config, ladder point it starts from, cards added)
EXTRA = [
    ("F- one-hot->codes", FULL, ["A5"]), ("N batch 256", FULL, ["B3"]), ("N 5 layers", FULL, ["C2"]),
    ("N sigmoid act", FULL, ["C3"]), ("N batchnorm", FULL, ["C4"]), ("R l2 1e-3", "L09 +2x128 network", ["D2"]),
    ("N l1 1e-4", FULL, ["D3"]), ("N early stop", FULL, ["D4"]), ("T 3-seed average", FULL, ["F1"]),
    ("R early stop", "L09 +2x128 network", ["D4"]), ("N dropout on small net", "L08 +100 epochs", ["D1"]),
]


def cards_up_to(config):
    names = [name for name, _ in LADDER]
    return [c for _, cards in LADDER[:names.index(config) + 1] for c in cards]


# ---- rendering ----

def indent(lines, n=4):
    return [" " * n + line if line else line for line in lines]


def features_lines(nb):
    if nb["all_numeric"]:
        lines = ['FEATURES = train.select_dtypes("number").columns.drop("is_fraud").tolist()']
    else:
        lines = ["FEATURES = [", *[f'    "{c}",' for c in S["numeric"]], "]"]
    if nb["drop_noise"]:
        lines.append(f"FEATURES = [col for col in FEATURES if col not in {q(NOISE)}]")
    fill = ".fillna(0)" if nb["fill_zero"] else ""
    lines.append("")
    if nb["encode"] is None:
        lines += [f"X = train[FEATURES]{fill}", f"X_test = test[FEATURES]{fill}"]
    elif nb["encode"] == "onehot":
        lines += [f"TEXT = {q(CATEGORICAL)}",
                  f"X = pd.get_dummies(train[FEATURES + TEXT], columns=TEXT, dtype=float){fill}",
                  f"X_test = pd.get_dummies(test[FEATURES + TEXT], columns=TEXT, dtype=float){fill}",
                  "X_test = X_test.reindex(columns=X.columns, fill_value=0)"]
    else:
        lines += [f"TEXT = {q(CATEGORICAL)}",
                  f"X = train[FEATURES + TEXT]{fill or '.copy()'}",
                  f"X_test = test[FEATURES + TEXT]{fill or '.copy()'}",
                  "for col in TEXT:",
                  "    categories = sorted(train[col].unique())",
                  "    X[col] = pd.Categorical(X[col], categories=categories).codes",
                  "    X_test[col] = pd.Categorical(X_test[col], categories=categories).codes"]
    return lines + ['y = train["is_fraud"]']


def model_lines(nb, seed):
    reg = f", kernel_regularizer={nb['regularizer']}" if nb["regularizer"] else ""
    layers = []
    for width in nb["hidden"]:
        layers.append(f'keras.layers.Dense({width}, activation="{nb["activation"]}"{reg}),')
        if nb["batchnorm"]:
            layers.append("keras.layers.BatchNormalization(),")
        if nb["dropout"]:
            layers.append(f"keras.layers.Dropout({nb['dropout']}),")
    optimizer = {"sgd": "SGD", "adam": "Adam"}[nb["optimizer"]]
    return [f"keras.utils.set_random_seed({seed})", "",
            "model = keras.Sequential([",
            "    keras.layers.Input(shape=(X_train.shape[1],)),",
            *indent(layers),
            '    keras.layers.Dense(1, activation="sigmoid"),',
            "])",
            f"model.compile(optimizer=keras.optimizers.{optimizer}(learning_rate={nb['lr']}), "
            'loss="binary_crossentropy", metrics=["accuracy"])']


def fit_lines(nb, lhs="history = ", verbose=2):
    defs = [nb["callbacks"][k] for k in CALLBACK_ORDER if k in nb["callbacks"]]
    names = [d.split(" = ")[0] for d in defs]
    callbacks = f", callbacks=[{', '.join(names)}]" if names else ""
    return defs + [f"{lhs}model.fit(X_train, y_train, validation_data=(X_val, y_val),",
                   " " * len(f"{lhs}model.fit(") +
                   f"epochs={nb['epochs']}, batch_size={nb['batch_size']}{callbacks}, verbose={verbose})"]


def submission_lines(nb):
    if nb["seeds"] == 1:
        lines = ["pred = (model.predict(X_test) > 0.5).astype(int).ravel()"]
    else:
        lines = ["probs = []",
                 f"for seed in {list(range(nb['seeds']))}:",
                 *indent(model_lines(nb, "seed") + fit_lines(nb, lhs="", verbose=0) +
                         ["probs.append(model.predict(X_test).ravel())"]),
                 "pred = (np.mean(probs, axis=0) > 0.5).astype(int)"]
    return lines + ['submission = pd.DataFrame({"transaction_id": test["transaction_id"], "is_fraud": pred})',
                    'submission.to_csv("submission.csv", index=False)']


CURVES = """plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="validation")
plt.title("Loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="validation")
plt.title("Accuracy")
plt.xlabel("epoch")
plt.legend()
plt.show()"""


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(lines):
    text = lines if isinstance(lines, str) else "\n".join(lines)
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build(cards=(), title="Flagged or Fraud? Starter notebook"):
    nb = starter_state()
    for c in cards:
        CARDS[c][1](nb)
    loop = [line for k in LOOP_ORDER for line in nb["loop"].get(k, [])] or ["pass"]
    after = [line for k in SPLIT_ORDER for line in nb["after_split"].get(k, [])]
    split = ["X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)"]
    cells = [
        md(f"# {title}"),
        code(nb["imports"]),
        md("## Load data"),
        code(['train = pd.read_csv("train.csv")', 'test = pd.read_csv("test.csv")']),
        code("train.head()"),
        code("train.describe()"),
        code("train.isna().sum()"),
        md("## Preprocessing"),
        code(["for df in [train, test]:", *indent(loop)]),
        md("## Features"),
        code(features_lines(nb)),
        md("## Train / validation split"),
        code(split + ([""] + after if after else [])),
        md("## Model"),
        code(model_lines(nb, 0) + ["model.summary()"]),
        md("## Training"),
        code(fit_lines(nb)),
        md("## Loss and accuracy curves"),
        code(CURVES),
        md("## Submission"),
        code(submission_lines(nb)),
    ]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


def changes(before, after):
    out, heading = [], ""
    for a, b in zip(before["cells"], after["cells"]):
        src_a, src_b = "".join(a["source"]), "".join(b["source"])
        if b["cell_type"] == "markdown":
            heading = src_b[3:] if src_b.startswith("## ") else "Imports"
            continue
        if src_a != src_b:
            diff = list(difflib.unified_diff(src_a.splitlines(), src_b.splitlines(), lineterm="", n=1))[2:]
            out += [f"**{heading}**", "```diff", *diff, "```", ""]
    return out


def solutions_md():
    lines = ["# Technique card solutions (instructor only)", "",
             "The change each card makes to the starter notebook, in the instructor guide's order, each on top of",
             "the previous ones. Lines starting with `+` are added, `-` removed. For help in class: students do not",
             "need to match this code, only the idea. `instructor_solution_notebook.ipynb` has every helpful card",
             "applied; `card_test_results.csv` has the score each step reached when run.", ""]
    for config, cards in LADDER[1:]:
        before, after = build(cards_up_to(config)[:-len(cards)]), build(cards_up_to(config))
        lines += [f"## {' + '.join(f'{c}. {CARDS[c][0]}' for c in cards)}", "", *changes(before, after)]
    lines += ["## Cards that do not help, or replace another card", ""]
    shown = {c for _, cards in LADDER for c in cards}
    for config, base, cards in EXTRA:
        if cards[0] in shown:
            continue
        shown.add(cards[0])
        where = "the full solution" if base == FULL else f"step `{base}`"
        lines += [f"### {cards[0]}. {CARDS[cards[0]][0]} (on {where})", "",
                  *changes(build(cards_up_to(base)), build(cards_up_to(base) + cards))]
    return "\n".join(lines)


if __name__ == "__main__":
    out = HERE / "materials"
    out.mkdir(exist_ok=True)
    (out / "starter_notebook.ipynb").write_text(json.dumps(build(), indent=1))
    (out / "instructor_solution_notebook.ipynb").write_text(
        json.dumps(build(cards_up_to(FULL), title="Flagged or Fraud? Instructor solution"), indent=1))
    (out / "instructor_card_solutions.md").write_text(solutions_md())
    print("wrote starter_notebook.ipynb, instructor_solution_notebook.ipynb, instructor_card_solutions.md")
