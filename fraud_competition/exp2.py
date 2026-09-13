"""
Second calibration grid, on the rebalanced generator.

Same views as exp1 (ladder, full-minus-one, regularisation options, starter
variants, cards that should not help), built from START and STEPS so the grid
follows any change to the starter or to the order of steps.
REF maps a config to (reference config, sign): gain = sign * (test - reference test).
"""
from pipeline import STARTER

KNOBS = dict(
    # weak category effects, so one-hot encoding is a normal-sized step
    merchant=0.2,
    channel={"in_store": -0.1, "contactless": -0.05, "online": 0.1, "atm": 0.05},
    card_type={"debit": 0.0, "credit": 0.1, "prepaid": 0.4},
    new_card=0.4, new_card_online=1.5,
    amount_above_usual=0.4, amount_online=0.8, big_amount_risky_merchant=1.0, far_big_amount=1.0,
    amount_tiny=0.3, card_testing=1.8,
    night=0.5, night_peak_hour=0.5, night_velocity=1.5, night_risky_merchant=1.0, night_international=2.2,
    international=0.8, intl_in_person=-1.0, distance_in_person=1.2,
    device_blocked=1.2, failed_pin=0.3, atm_failed_pin=1.5, velocity=0.4,
    # informative blanks, so median fill + missing flags is a real step
    location_hidden=1.6, location_hidden_rate=0.10, distance_missing=0.03,
    profile_reset=1.4, profile_reset_rate=0.06,
    usual_spend_sigma=1.6, outlier_rate=0.008,
    # round cash-out amounts: a feature only hand engineering finds
    round_amount=1.2, round_amount_rate=0.10, whole_rupee_rate=0.5,
    # sign flips: a 4-unit network cannot cover eight of them, a big regularised one can
    flips={"age*velocity": -0.5, "age*amount": -0.5, "trust*amount": -0.5, "merchant*amount": 0.5,
           "merchant*velocity": 0.5, "age*merchant": -0.5, "far*amount": 0.5, "trust*velocity": -0.5},
    sharpness=3.531,  # best possible accuracy 92%
)
N_TRAIN = 20_000
SEEDS = [0, 1, 2]
# the starter notebook uses a few small-valued numeric columns, so its first submission is not dead
SMALL_NUMERIC = ["transaction_hour", "is_international", "failed_pin_attempts_24h", "txn_count_last_1h",
                 "merchant_risk_score", "device_trust_score"]
START = dict(numeric=SMALL_NUMERIC)  # changes to pipeline.STARTER

STEPS = [
    ("scale", dict(scale=True)),
    ("all numeric columns", dict(numeric=None)),
    ("log skewed", dict(log=True)),
    ("one-hot", dict(encode="onehot")),
    ("median & missing flags", dict(impute="median_flag")),
    ("chargeback -1 flag", dict(placeholder=True)),
    ("adam", dict(optimizer="adam", lr=1e-3)),
    ("100 epochs", dict(epochs=100)),
    ("2x128 network", dict(layers=(128, 128))),
    ("dropout 0.3", dict(dropout=0.3)),
    ("drop noise cols", dict(drop_noise=True)),
    ("amount vs usual", dict(fe_ratio=True)),
    ("hour sin/cos", dict(fe_hour=True)),
    ("round amounts", dict(fe_round=True)),
    ("reduce lr", dict(reduce_lr=True)),
]
NETWORK_STEP = "2x128 network"
# the same steps in another order (configs "M.."): grow the network before training longer
ALT_ORDER = ["scale", "all numeric columns", "log skewed", "one-hot", "median & missing flags",
             "chargeback -1 flag", "adam", "2x128 network", "100 epochs", "dropout 0.3", "drop noise cols",
             "amount vs usual", "hour sin/cos", "round amounts", "reduce lr"]

# each on the overfitting big network
REGULARISE = {
    "dropout 0.3": dict(dropout=0.3), "dropout 0.5": dict(dropout=0.5),
    "l2 1e-4": dict(l2=1e-4), "l2 1e-3": dict(l2=1e-3), "l1 1e-4": dict(l1=1e-4),
    "early stop": dict(early_stopping=True), "drop noise": dict(drop_noise=True),
    "dropout+early": dict(dropout=0.3, early_stopping=True),
    "dropout+l2+early": dict(dropout=0.3, l2=1e-4, early_stopping=True),
}
# on the full pipeline: none of these should help
NULL = {
    "batchnorm": dict(batchnorm=True), "sigmoid act": dict(activation="sigmoid"),
    "5 layers": dict(layers=(128,) * 5), "batch 256": dict(batch_size=256),
    "early stop": dict(early_stopping=True), "l1 1e-4": dict(l1=1e-4),
}
STARTERS = {
    "adam": dict(optimizer="adam", lr=1e-3), "adam 1x16": dict(optimizer="adam", lr=1e-3, layers=(16,)),
    "sgd lr1e-3": dict(lr=1e-3), "adam +scale": dict(optimizer="adam", lr=1e-3, scale=True),
    "adam +card age": dict(optimizer="adam", lr=1e-3, numeric=SMALL_NUMERIC + ["card_age_days"]),
    "adam -txn count": dict(optimizer="adam", lr=1e-3,
                            numeric=[c for c in SMALL_NUMERIC if c != "txn_count_last_1h"]),
    "adam +card age -txn count": dict(optimizer="adam", lr=1e-3,
                                      numeric=[c for c in SMALL_NUMERIC if c != "txn_count_last_1h"]
                                      + ["card_age_days"]),
}


def build(start=None):
    start = {**STARTER, **START, **(start or {})}
    configs, ref, after = {"L00 starter": dict(start)}, {}, {}
    cum, prev = dict(start), "L00 starter"
    for name, step in STEPS:
        if all(cum[k] == v for k, v in step.items()):
            continue  # already in the starter
        cum = {**cum, **step}
        after[name] = f"L{len(after) + 1:02d} +{name}"
        configs[after[name]], ref[after[name]] = dict(cum), (prev, 1)
        prev = after[name]
    full, steps = prev, dict(STEPS)

    for name in after:
        configs[f"F- {name}"], ref[f"F- {name}"] = {**cum, **{k: start[k] for k in steps[name]}}, (full, -1)
    configs["F- one-hot->codes"], ref["F- one-hot->codes"] = {**cum, "encode": "codes"}, (full, -1)
    configs["F- network & dropout"], ref["F- network & dropout"] = (
        {**cum, "layers": start["layers"], "dropout": start["dropout"]}, (full, -1))

    prev_alt, cum_alt, steps_by_name = "L00 starter", dict(start), dict(STEPS)
    for i, name in enumerate([n for n in ALT_ORDER if n in after], 1):
        cum_alt = {**cum_alt, **steps_by_name[name]}
        same = next((n for n, c in configs.items() if n.startswith("L") and c == cum_alt), None)
        if same:  # identical to a point on the main ladder: reuse it
            prev_alt = same
            continue
        configs[f"M{i:02d} +{name}"], ref[f"M{i:02d} +{name}"] = dict(cum_alt), (prev_alt, 1)
        prev_alt = f"M{i:02d} +{name}"

    big = after[NETWORK_STEP]
    for name, v in REGULARISE.items():
        configs[f"R {name}"], ref[f"R {name}"] = {**configs[big], **v}, (big, 1)
    small = ref[big][0]
    configs["N dropout on small net"], ref["N dropout on small net"] = {**configs[small], "dropout": 0.3}, (small, 1)
    for name, v in NULL.items():
        configs[f"N {name}"], ref[f"N {name}"] = {**cum, **v}, (full, 1)
    for name, v in STARTERS.items():
        configs[f"S {name}"], ref[f"S {name}"] = {**start, **v}, ("L00 starter", 1)
    configs["T 3-seed average"], ref["T 3-seed average"] = {**cum, "n_seeds": 3}, (full, 1)
    return configs, ref


CONFIGS, REF = build()
