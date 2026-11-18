"""
Trainable-parameter counts per model over all (seed, fold) cells.

Tuned models select their size per outer fold (PCA dim, conv layers, hidden width), so a
single count is misleading; this reports median, min and max over the cells. Non-parametric
models (SVM, RF) report no count.

Reads  results/oof/<model>/seed<S>_fold<F>.meta.json  (model_meta.n_params)
Writes results/param_counts.csv

Usage:
  python scripts/param_counts.py
"""

from __future__ import annotations

import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
R = V3_DIR / "results"


def main() -> None:
    counts: dict[str, list] = defaultdict(list)
    for p in sorted(glob.glob(str(R / "oof" / "*" / "*.meta.json"))):
        m = json.loads(Path(p).read_text())
        counts[m["model"]].append(m.get("model_meta", {}).get("n_params"))
    rows = []
    for model, v in sorted(counts.items()):
        have = [int(x) for x in v if x is not None]
        row = {"model": model, "n_cells": len(v), "n_cells_with_count": len(have)}
        if have:
            row.update(n_params_median=float(np.median(have)), n_params_min=min(have), n_params_max=max(have))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(R / "param_counts.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
