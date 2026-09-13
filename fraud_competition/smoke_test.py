"""
Quick check that a new machine reproduces the calibration numbers (a minute or two).

    python smoke_test.py

Compares seed 0 of a few configs with the same runs in results_exp2.csv.
Differences up to about 1 point are normal across machines and Keras backends.
"""
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import exp2
from calibrate import N_TEST
from generate_data import generate
from pipeline import run

df, _ = generate(exp2.N_TRAIN + N_TEST, seed=2026, knobs=exp2.KNOBS)
train, test = df.iloc[:exp2.N_TRAIN].reset_index(drop=True), df.iloc[exp2.N_TRAIN:].reset_index(drop=True)
recorded = pd.read_csv("results_exp2.csv").query("seed == 0").set_index("config")["test_acc"]

for name in ["L00 starter", "L01 +scale", "L07 +adam"]:
    t = time.time()
    acc = run(train, test, exp2.CONFIGS[name], seed=0)["test_acc"]
    print(f"{name:12s} test {acc:.4f}   recorded {recorded[name]:.4f}   {time.time() - t:.0f}s")
