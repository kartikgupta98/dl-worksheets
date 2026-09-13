"""
Quick check that a new machine reproduces the calibration numbers (about a minute).

    python smoke_test.py
"""
import time
import warnings

warnings.filterwarnings("ignore")

import exp1
from generate_data import generate
from pipeline import run

df, _ = generate(40_000, seed=2026, knobs=exp1.KNOBS)
train, test = df.iloc[:10_000].reset_index(drop=True), df.iloc[10_000:].reset_index(drop=True)

# test accuracy measured on the Windows laptop, seed 0
EXPECTED = {"L00 starter": 0.4959, "L01 +scale": 0.6852, "L06 +adam": 0.8670}
for name, expected in EXPECTED.items():
    t = time.time()
    acc = run(train, test, exp1.CONFIGS[name], seed=0)["test_acc"]
    print(f"{name:12s} test {acc:.4f}   Windows {expected:.4f}   {time.time() - t:.0f}s")
