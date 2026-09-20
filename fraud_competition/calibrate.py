"""
Runs many (config, seed) training jobs in parallel and writes a results table.

usage: python calibrate.py <experiment_module> [n_workers]
The experiment module defines KNOBS (generator settings) and CONFIGS
(name -> pipeline config) and optionally SEEDS and N_TRAIN.

Results are appended to results_<experiment>.csv as they finish, and jobs
already in that file are skipped, so an interrupted run can be restarted.
"""
import importlib
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("KERAS_BACKEND", "jax")
# one CPU thread per worker process
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

N_TEST = 30_000
_DATA = {}


def _init(knobs, n_train=10_000):
    import warnings
    warnings.filterwarnings("ignore")
    if os.environ["KERAS_BACKEND"] == "torch":
        import torch
        torch.set_num_threads(1)
    import pipeline  # noqa: F401  (import keras once per worker, up front)
    from generate_data import generate
    df, _ = generate(n_train + N_TEST, seed=2026, knobs=knobs)
    _DATA["train"], _DATA["test"] = df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


def _job(args):
    name, cfg, seed = args
    from pipeline import run
    t = time.time()
    try:
        info = run(_DATA["train"], _DATA["test"], cfg, seed=seed)
        return dict(config=name, seed=seed, seconds=round(time.time() - t, 1), **info)
    except Exception:
        return dict(config=name, seed=seed, error=traceback.format_exc(limit=3))


def summarise(res, order):
    import pandas as pd
    summary = (res.groupby("config")
               .agg(test_mean=("test_acc", "mean"), test_std=("test_acc", "std"),
                    train=("train_acc", "mean"), val=("val_acc", "mean"),
                    epochs=("epochs_run", "mean"), nan=("nan_loss", "sum"), n=("seed", "count"))
               .reindex([o for o in order if o in set(res["config"])]))
    pd.set_option("display.width", 200)
    return summary.round(4).to_string()


def main():
    import pandas as pd
    exp_name = sys.argv[1]
    exp = importlib.import_module(exp_name)
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    seeds = getattr(exp, "SEEDS", [0, 1, 2])
    out = f"results_{exp_name}.csv"

    done = set()
    if os.path.exists(out):
        prev = pd.read_csv(out)
        done = set(zip(prev["config"], prev["seed"]))
    jobs = [(name, cfg, s) for name, cfg in exp.CONFIGS.items() for s in seeds if (name, s) not in done]
    print(f"{len(jobs)} jobs to run ({len(done)} already done), {workers} workers", flush=True)

    with Pool(workers, initializer=_init, initargs=(exp.KNOBS, getattr(exp, "N_TRAIN", 10_000))) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            if "error" in r:
                print(f"[{i}/{len(jobs)}] {r['config']} seed {r['seed']}: ERROR\n{r['error']}", flush=True)
                continue
            pd.DataFrame([r]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
            print(f"[{i}/{len(jobs)}] {r['config']} seed {r['seed']}: test {r['test_acc']:.4f} ({r['seconds']}s)",
                  flush=True)

    print(summarise(pd.read_csv(out), list(exp.CONFIGS)))


if __name__ == "__main__":
    main()
