"""
Print the results table of an experiment, finished or not.

    python summary.py exp1
"""
import importlib
import sys

import pandas as pd

from calibrate import summarise

name = sys.argv[1]
print(summarise(pd.read_csv(f"results_{name}.csv"), list(importlib.import_module(name).CONFIGS)))
