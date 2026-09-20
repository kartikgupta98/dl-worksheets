"""
Checks that the technique cards, applied to the starter notebook as in the instructor
solutions, reach the calibrated scores.

    python test_cards.py [workers]

Every case in make_notebooks.LADDER and EXTRA is built as a notebook, run with train.csv
and test.csv next to it (as in Colab), and its submission.csv is scored on the whole hidden
test set. The score is compared with the same config in results_exp2.csv (seed 0, and the
3-seed mean). Results go to card_test_results.csv.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import pandas as pd

HERE = Path(__file__).parent


def run_case(case):
    import nbformat
    from nbclient import NotebookClient
    from make_notebooks import build, cards_up_to

    config, base, added = case
    cards = cards_up_to(base) + added
    t = time.time()
    try:
        with tempfile.TemporaryDirectory() as d:
            for f in ["train.csv", "test.csv"]:
                os.symlink(HERE / "kaggle" / f, Path(d) / f)
            nb = nbformat.reads(json.dumps(build(cards)), as_version=4)
            NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": d}}).execute()
            sub = pd.read_csv(Path(d) / "submission.csv")
        sol = pd.read_csv(HERE / "private" / "solution.csv")
        m = sol.merge(sub, on="transaction_id", suffixes=("", "_pred"), validate="one_to_one")
        acc = float((m["is_fraud"] == m["is_fraud_pred"]).mean())
        return dict(config=config, cards=" ".join(added) or "starter", notebook=acc, seconds=round(time.time() - t))
    except Exception:
        return dict(config=config, cards=" ".join(added), error=traceback.format_exc()[-800:])


def main():
    from make_notebooks import EXTRA, LADDER

    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    cases = [(config, config, []) if not cards else (config, LADDER[i - 1][0], cards)
             for i, (config, cards) in enumerate(LADDER)] + EXTRA
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(run_case, cases))
    for r in rows:
        if "error" in r:
            print("ERROR", r["config"], r["error"])
    rows = [r for r in rows if "error" not in r]
    if not rows:
        sys.exit("no notebook ran")

    rec = pd.read_csv(HERE / "results_exp2.csv")
    res = pd.DataFrame(rows)
    res["calibration_seed0"] = [rec.query("config == @c and seed == 0")["test_acc"].iloc[0] for c in res["config"]]
    res["calibration_mean"] = [rec.query("config == @c")["test_acc"].mean() for c in res["config"]]
    res["diff_seed0"] = res["notebook"] - res["calibration_seed0"]
    for c in ["notebook", "calibration_seed0", "calibration_mean", "diff_seed0"]:
        res[c] = (100 * res[c]).round(1)
    res.to_csv(HERE / "card_test_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
