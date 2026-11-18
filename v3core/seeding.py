"""
Deterministic seeding.

One top-level seed per CV repeat. Every stochastic component gets its own seed,
derived from (seed, fold, component name) via numpy SeedSequence, so components
are independent of each other and of execution order or parallelism.
"""

from __future__ import annotations

import os
import random
import zlib

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python's and NumPy's global RNGs (belt and braces; v3 code passes
    explicit seeds/Generators everywhere and never relies on global state)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def component_seed(seed: int, fold: int, name: str) -> int:
    ss = np.random.SeedSequence([int(seed), int(fold), zlib.crc32(name.encode("utf-8"))])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def component_rng(seed: int, fold: int, name: str) -> np.random.Generator:
    return np.random.default_rng(component_seed(seed, fold, name))
