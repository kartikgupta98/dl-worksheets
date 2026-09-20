"""
One configurable pipeline that can reproduce any point on the students' path,
from the starter notebook to a fully tuned model. Every step is a switch in
the config, so the calibration can turn steps on and off and measure them.
"""
import os

# JAX compiles the training step, about 15x faster than torch on CPU for these small networks
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IDS = ["transaction_id", "terminal_id"]
CATEGORICAL = ["merchant_category", "channel", "card_type", "city_tier"]
SKEWED = ["amount_inr", "customer_avg_spend_90d", "card_age_days",
          "distance_from_home_km", "txn_count_last_1h"]
NOISE = ["batch_number", "acquirer_code", "pos_software_version"]
TARGET = "is_fraud"

# the starter notebook
STARTER = dict(
    numeric=None,         # None = every numeric column, or a list of numeric columns to use
    log=False,            # log1p on skewed columns
    encode="drop",        # drop | codes | onehot
    impute="zero",        # zero | median | median_flag
    placeholder=False,    # -1 in days_since_last_chargeback -> flag
    drop_noise=False,
    fe_ratio=False,       # log(amount / usual spend)
    fe_hour=False,        # sin / cos of hour
    fe_round=False,       # amount is a round figure (multiple of 500)
    scale=False,
    optimizer="sgd",
    lr=0.01,
    epochs=10,
    batch_size=32,
    layers=(4,),
    activation="relu",
    batchnorm=False,
    dropout=0.0,
    l2=0.0,
    l1=0.0,
    early_stopping=False,
    patience=15,
    reduce_lr=False,
    n_seeds=1,            # average predictions of this many seeds
)


def preprocess(train, others, cfg):
    """Fit every transformation on `train` only, apply to all frames."""
    frames = [train.copy()] + [o.copy() for o in others]

    for d in frames:
        d.drop(columns=[c for c in IDS + [TARGET] if c in d.columns], inplace=True)
        if cfg["numeric"] is not None:
            d.drop(columns=[c for c in d.columns if c not in list(cfg["numeric"]) + CATEGORICAL], inplace=True)
        if cfg["placeholder"] and "days_since_last_chargeback" in d:
            never = d["days_since_last_chargeback"] == -1
            d["never_chargeback"] = never.astype(float)
            d.loc[never, "days_since_last_chargeback"] = np.nan
        if cfg["fe_ratio"] and {"amount_inr", "customer_avg_spend_90d"} <= set(d.columns):
            d["amount_vs_usual"] = np.log(d["amount_inr"] / d["customer_avg_spend_90d"])
        if cfg["fe_round"] and "amount_inr" in d:
            d["amount_is_round"] = (d["amount_inr"] % 500 == 0).astype(float)
        if cfg["fe_hour"] and "transaction_hour" in d:
            d["hour_sin"] = np.sin(2 * np.pi * d["transaction_hour"] / 24)
            d["hour_cos"] = np.cos(2 * np.pi * d["transaction_hour"] / 24)
        if cfg["log"]:
            cols = SKEWED + (["days_since_last_chargeback"] if cfg["placeholder"] else [])
            for c in [c for c in cols if c in d]:
                d[c] = np.log1p(d[c].clip(lower=0) if c != "days_since_last_chargeback" else d[c])
        if cfg["drop_noise"]:
            d.drop(columns=[c for c in NOISE if c in d], inplace=True)

    base = frames[0]
    if cfg["impute"] == "median_flag":
        for c in [c for c in base.columns if base[c].isna().any()]:
            for d in frames:
                d[c + "_missing"] = d[c].isna().astype(float)
    if cfg["impute"] == "zero":
        fill = {c: 0.0 for c in base.columns}
    else:
        fill = base.median(numeric_only=True).to_dict()
    for d in frames:
        d.fillna(value=fill, inplace=True)

    if cfg["encode"] == "drop":
        for d in frames:
            d.drop(columns=CATEGORICAL, inplace=True)
    elif cfg["encode"] == "codes":
        for c in CATEGORICAL:
            cats = sorted(base[c].unique())
            for d in frames:
                d[c] = pd.Categorical(d[c], categories=cats).codes.astype(float)
    else:
        frames = [pd.get_dummies(d, columns=CATEGORICAL, dtype=float) for d in frames]
        cols = frames[0].columns
        frames = [d.reindex(columns=cols, fill_value=0.0) for d in frames]

    arrays = [d.to_numpy(dtype="float32") for d in frames]
    if cfg["scale"]:
        scaler = StandardScaler().fit(arrays[0])
        arrays = [scaler.transform(a).astype("float32") for a in arrays]
    return arrays, list(frames[0].columns)


def build_model(n_inputs, cfg):
    reg = None
    if cfg["l2"] or cfg["l1"]:
        reg = keras.regularizers.L1L2(l1=cfg["l1"], l2=cfg["l2"])
    layers = [keras.layers.Input(shape=(n_inputs,))]
    for width in cfg["layers"]:
        layers.append(keras.layers.Dense(width, activation=cfg["activation"], kernel_regularizer=reg))
        if cfg["batchnorm"]:
            layers.append(keras.layers.BatchNormalization())
        if cfg["dropout"]:
            layers.append(keras.layers.Dropout(cfg["dropout"]))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.Sequential(layers)
    opt = {"sgd": keras.optimizers.SGD, "adam": keras.optimizers.Adam}[cfg["optimizer"]]
    model.compile(optimizer=opt(learning_rate=cfg["lr"]), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run(train_df, test_df, cfg, seed=0, split_seed=42, return_probs=False):
    """Train the way a student would: own train/val split, then score on the hidden test set."""
    cfg = {**STARTER, **cfg}
    tr, va = train_test_split(train_df, test_size=0.2, random_state=split_seed, stratify=train_df[TARGET])
    (Xtr, Xva, Xte), _ = preprocess(tr, [va, test_df], cfg)
    ytr, yva, yte = tr[TARGET].to_numpy("float32"), va[TARGET].to_numpy("float32"), test_df[TARGET].to_numpy()

    probs, info = [], {}
    for s in range(cfg["n_seeds"]):
        keras.utils.set_random_seed(seed * 100 + s)
        model = build_model(Xtr.shape[1], cfg)
        callbacks = []
        if cfg["early_stopping"]:
            callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                                           restore_best_weights=True))
        if cfg["reduce_lr"]:
            callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5))
        hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=cfg["epochs"],
                         batch_size=cfg["batch_size"], callbacks=callbacks, verbose=0)
        p = model.predict(Xte, verbose=0, batch_size=4096).ravel()
        probs.append(p)
        if s == 0:
            h = hist.history
            info = dict(train_acc=h["accuracy"][-1], val_acc=h["val_accuracy"][-1],
                        best_val_acc=max(h["val_accuracy"]), epochs_run=len(h["loss"]),
                        nan_loss=bool(np.isnan(h["loss"][-1])))
    p = np.nan_to_num(np.mean(probs, axis=0), nan=0.0)
    info["test_acc"] = float(((p > 0.5).astype(int) == yte).mean())
    return (info, p) if return_probs else info
