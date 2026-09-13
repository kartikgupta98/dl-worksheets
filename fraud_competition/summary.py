"""
Print the results table of an experiment, finished or not.

    python summary.py exp2

If the experiment defines REF, a gain column (accuracy points) compares each config
with its reference: the previous ladder step, or for "F-" configs what the full
pipeline loses without that step.
"""
import importlib
import sys

import pandas as pd

from calibrate import summarise

name = sys.argv[1]
exp = importlib.import_module(name)
res = pd.read_csv(f"results_{name}.csv")
table = summarise(res, list(exp.CONFIGS))
ref = getattr(exp, "REF", None)
if not ref:
    print(table)
    sys.exit()

g = res.groupby("config").agg(test=("test_acc", "mean"), sd=("test_acc", "std"), train=("train_acc", "mean"),
                              epochs=("epochs_run", "mean"), n=("seed", "count"))
g = g.reindex([c for c in exp.CONFIGS if c in g.index])
g["gain"] = [ref[c][1] * (g.at[c, "test"] - g.at[ref[c][0], "test"]) if c in ref and ref[c][0] in g.index
             else float("nan") for c in g.index]
for c in ["test", "sd", "train", "gain"]:
    g[c] = (100 * g[c]).round(1)
g["epochs"] = g["epochs"].round(0)
pd.set_option("display.width", 200)
print(g.to_string())
