"""
Writes the files for the Kaggle Community Competition, and scores submissions locally.

    python export_kaggle.py                      # write the competition files
    python export_kaggle.py score my_sub.csv     # score a submission like Kaggle would

kaggle/    goes to Kaggle's Data tab: train.csv, test.csv, sample_submission.csv
private/   solution.csv is uploaded as the hidden solution. Never share it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from generate_data import generate

# final generator settings, set from the calibration results
KNOBS = {"sharpness": 3.6}
SEED = 2026
N_TRAIN, N_TEST = 10_000, 30_000
PUBLIC_FRACTION = 0.30

HERE = Path(__file__).parent
PUBLIC_DIR, PRIVATE_DIR = HERE / "kaggle", HERE / "private"
ID, TARGET = "transaction_id", "is_fraud"


def export():
    df, p = generate(N_TRAIN + N_TEST, seed=SEED, knobs=KNOBS)
    train, test = df.iloc[:N_TRAIN], df.iloc[N_TRAIN:].reset_index(drop=True)
    p_test = p[N_TRAIN:]

    rng = np.random.default_rng(SEED + 1)
    usage = np.where(rng.random(N_TEST) < PUBLIC_FRACTION, "Public", "Private")

    PUBLIC_DIR.mkdir(exist_ok=True)
    PRIVATE_DIR.mkdir(exist_ok=True)
    train.to_csv(PUBLIC_DIR / "train.csv", index=False)
    test.drop(columns=[TARGET]).to_csv(PUBLIC_DIR / "test.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: 0}).to_csv(PUBLIC_DIR / "sample_submission.csv", index=False)
    pd.DataFrame({ID: test[ID], TARGET: test[TARGET], "Usage": usage}).to_csv(
        PRIVATE_DIR / "solution.csv", index=False)

    best = np.maximum(p_test, 1 - p_test)
    print(f"train {train.shape}, test {test.shape}, fraud rate train {train[TARGET].mean():.3f}")
    print(f"public rows {(usage == 'Public').sum()}, private rows {(usage == 'Private').sum()}")
    print(f"best possible accuracy: public {best[usage == 'Public'].mean():.4f}, "
          f"private {best[usage == 'Private'].mean():.4f}")


def score(path):
    sol = pd.read_csv(PRIVATE_DIR / "solution.csv")
    sub = pd.read_csv(path)
    assert list(sub.columns) == [ID, TARGET], f"columns must be {[ID, TARGET]}, got {list(sub.columns)}"
    assert len(sub) == len(sol), f"expected {len(sol)} rows, got {len(sub)}"
    assert set(sub[TARGET].unique()) <= {0, 1}, "is_fraud must be 0 or 1, not probabilities"
    m = sol.merge(sub, on=ID, suffixes=("_true", "_pred"), validate="one_to_one")
    assert len(m) == len(sol), "transaction_id values do not match test.csv"
    correct = m[f"{TARGET}_true"] == m[f"{TARGET}_pred"]
    for part in ["Public", "Private"]:
        print(f"{part:8s} accuracy {correct[m['Usage'] == part].mean():.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "score":
        score(sys.argv[2])
    else:
        export()
