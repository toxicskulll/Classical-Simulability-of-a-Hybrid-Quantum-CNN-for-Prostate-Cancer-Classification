"""Load the v3 YAML config and resolve paths relative to the repository root."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

V3_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = V3_DIR / "configs" / "champion.yaml"
RESULTS_DIR = V3_DIR / "results"

REQUIRED_SECTIONS = (
    "data", "cv", "features", "preprocessing", "hqcnn",
    "baselines", "clinical_arms", "models", "summary",
)


def v3_path(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else V3_DIR / p


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config. `extends: other.yaml` (relative to this file) deep-merges this
    file on top of the other; lists are replaced, not concatenated."""
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in cfg:
        base = load_config(path.parent / cfg.pop("extends"))
        base.pop("_path", None)
        cfg = _deep_merge(base, cfg)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"{path}: missing config sections {missing}")
    cfg = copy.deepcopy(cfg)
    cfg["_path"] = str(path)
    return cfg
