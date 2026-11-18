"""Shared helpers for Phase 2 analyses (real-data inputs, config variants, tables)."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .data import load_cohort, make_folds
from .preprocess import build_fold
from .seeding import component_seed


def hqcnn_variant(cfg: dict, **overrides) -> dict:
    h = copy.deepcopy(cfg["hqcnn"])
    h.update(overrides)
    return h


def real_inputs(cfg: dict, pca_dim: int | None = None, seed: int = 0, fold: int = 0) -> dict:
    """PCA inputs from a real Phase 1 fold, angle-scaled exactly as HQCNN.fit does
    (min-max on the resampled training matrix, test clipped to [0, pi])."""
    pcfg = dict(cfg["preprocessing"])
    if pca_dim is not None:
        pcfg["pca_dim"] = int(pca_dim)
    cohort = load_cohort(cfg)
    tr, te = make_folds(cohort, cfg["cv"]["n_splits"], seed)[fold]
    fd = build_fold(cohort, tr, te, pcfg, resample_seed=component_seed(seed, fold, "resample"))
    lo, hi = fd.X_fit.min(axis=0), fd.X_fit.max(axis=0)
    a, b = cfg["hqcnn"]["angle_range"]
    span = np.where(hi > lo, hi - lo, 1.0)
    scale = lambda X: a + np.clip((X - lo) / span, 0.0, 1.0) * (b - a)  # noqa: E731
    return {"fold_data": fd, "train": scale(fd.X_fit), "test": scale(fd.X_test),
            "y_test": fd.y_test, "n_features": fd.X_fit.shape[1]}


def write_tex_table(path: Path, caption: str, label: str, header: list[str],
                    rows: list[list[str]], note: str | None = None) -> None:
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\small", rf"\begin{{tabular}}{{l{'c' * (len(header) - 1)}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [rf"\par\smallskip\footnotesize {note}"]
    lines += [r"\end{table}", ""]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def half_cut_entropy(state: np.ndarray, n_qubits: int) -> float:
    """Von Neumann entropy (bits) of wires 0..n/2-1 for a pure n-qubit state."""
    k = n_qubits // 2
    m = np.asarray(state).reshape(2 ** k, 2 ** (n_qubits - k))
    s = np.linalg.svd(m, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-(s * np.log2(s)).sum())


def bipartition_entropy(state: np.ndarray, n_qubits: int, wires_a) -> float:
    """Von Neumann entropy (bits) of the subsystem `wires_a` of a pure n-qubit state
    (PennyLane ordering: wire 0 is the most significant index)."""
    wires_a = list(wires_a)
    rest = [w for w in range(n_qubits) if w not in wires_a]
    psi = np.asarray(state).reshape([2] * n_qubits).transpose(wires_a + rest)
    m = psi.reshape(2 ** len(wires_a), 2 ** len(rest))
    s = np.linalg.svd(m, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-(s * np.log2(s)).sum())


CUTS = {
    "contiguous": lambda n: list(range(n // 2)),        # wires 0..n/2-1 vs the rest
    "interleaved": lambda n: list(range(0, n, 2)),      # even wires vs odd wires
}
