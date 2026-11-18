"""Load the frozen v3 lesion table and build patient-grouped folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import v3_path

# Any feature name containing one of these is post-biopsy / label information.
LEAK_TOKENS = ("gleason", "isup", "clinsig", "cspca", "grade_group", "label", "leak_")


class LeakageError(AssertionError):
    pass


def assert_no_leakage(feature_names, allow: tuple[str, ...] = ()) -> None:
    """Raise if any model-input column looks like post-biopsy information."""
    bad = [
        c for c in feature_names
        if c not in allow and any(t in str(c).lower() for t in LEAK_TOKENS)
    ]
    if bad:
        raise LeakageError(f"post-biopsy / label columns at model input: {bad}")


@dataclass(frozen=True)
class Cohort:
    frame: pd.DataFrame           # full lesion table (incl. leak_ reference columns)
    y: np.ndarray
    groups: np.ndarray            # ProxID per lesion
    radiomics_cols: list[str]
    clinical_cols: list[str]      # pre-biopsy only

    def matrix(self, cols: list[str]) -> np.ndarray:
        return self.frame[cols].to_numpy(dtype=np.float64)


def load_cohort(cfg: dict) -> Cohort:
    dcfg = cfg["data"]
    df = pd.read_csv(v3_path(dcfg["lesions_csv"]))
    if df.duplicated(["ProxID", "fid"]).any():
        raise ValueError("lesion table has duplicate (ProxID, fid) rows")
    if len(df) != dcfg["expected_n_lesions"]:
        raise ValueError(f"expected {dcfg['expected_n_lesions']} lesions, got {len(df)}")
    if df["ProxID"].nunique() != dcfg["expected_n_patients"]:
        raise ValueError(
            f"expected {dcfg['expected_n_patients']} patients, got {df['ProxID'].nunique()}"
        )

    mods = tuple(cfg["features"]["radiomics_modalities"])
    radiomics = [c for c in df.columns if c.split("_", 1)[0] in mods]
    clinical = list(cfg["features"]["clinical"])
    missing = [c for c in clinical if c not in df.columns]
    if missing:
        raise ValueError(f"clinical columns missing from lesion table: {missing}")
    assert_no_leakage(radiomics + clinical)
    if df[radiomics].isna().any().any():
        raise ValueError("radiomics contain NaN")

    return Cohort(
        frame=df,
        y=df["label"].to_numpy(dtype=int),
        groups=df["ProxID"].astype(str).to_numpy(),
        radiomics_cols=radiomics,
        clinical_cols=clinical,
    )


def make_folds(cohort: Cohort, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold by ProxID; asserts no patient crosses train/test."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr, te in cv.split(np.zeros(len(cohort.y)), cohort.y, cohort.groups):
        if set(cohort.groups[tr]) & set(cohort.groups[te]):
            raise AssertionError("patient appears in both train and test")
        if cohort.y[te].sum() == 0:
            raise AssertionError("test fold without positives")
        folds.append((np.sort(tr), np.sort(te)))
    covered = np.sort(np.concatenate([te for _, te in folds]))
    if not np.array_equal(covered, np.arange(len(cohort.y))):
        raise AssertionError("folds do not partition the cohort")
    return folds
