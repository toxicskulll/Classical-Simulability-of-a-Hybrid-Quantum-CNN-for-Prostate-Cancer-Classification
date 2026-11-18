"""
Phase 3 step 2 (hardware cost): two-qubit gate count after routing onto a LINEAR
nearest-neighbour device, counted analytically from the pair structure (no Qiskit).

Native two-qubit gate: CNOT. Decompositions: conv block = 2 CNOT; CRZ = 2 CNOT;
CRX = 2 CNOT (so one pool = 4 CNOT); SWAP = 3 CNOT.

Layout: logical qubit i on physical site i (the tree's natural order). Level-0 pairs
(0,1), (2,3), ... are adjacent. At level l >= 1 the partners of a pair sit 2^l sites apart
(the discarded qubits in between stay on the chain), so one partner is moved next to the
other with 2^l - 1 SWAPs; all conv and pool ops of that pair then act on adjacent sites.

Tree SWAP count (n = 2^L):   S(n) = sum_{l=1}^{L-1} (n / 2^{l+1}) (2^l - 1)
                                  = (n/2)(L - 1) - (n/2)(1/2 + ... + 1/2^{L-1})
                                  = (n/2)(log2 n - 2) + 1           [n >= 2]

A greedy router (move the first qubit of each op towards its partner, keep the new
layout) replays the actual op sequence to check the formula and to count brickwork
circuits, whose offset pairs also become non-adjacent after level 0.

Writes results/phase3/routing/linear_routing.csv and .tex.

Usage:
  python scripts/linear_routing.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from v3core.config import RESULTS_DIR, load_config  # noqa: E402
from v3core.hqcnn import qcnn_plan  # noqa: E402
from v3core.phase2 import write_tex_table  # noqa: E402

OUT = RESULTS_DIR / "phase3" / "routing"
CNOT_PER = {"conv": 2, "pool": 4}
SIZES = [4, 8, 16, 32, 64, 128, 256]


def analytic_tree_swaps(n: int) -> int:
    L = int(math.log2(n))
    return sum((n // 2 ** (l + 1)) * (2 ** l - 1) for l in range(1, L))


def route(ops, n: int) -> tuple[int, int]:
    """Greedy linear routing; returns (#SWAPs, #native 2q gates without SWAPs)."""
    pos = list(range(n))            # logical -> physical
    site = list(range(n))           # physical -> logical
    swaps, native = 0, 0
    for kind, (a, b) in ops:
        while abs(pos[a] - pos[b]) > 1:
            step = 1 if pos[b] > pos[a] else -1
            p, q = pos[a], pos[a] + step
            la, lb = site[p], site[q]
            site[p], site[q] = lb, la
            pos[la], pos[lb] = q, p
            swaps += 1
        native += CNOT_PER[kind]
    return swaps, native


def main() -> None:
    cfg = load_config()
    cl = int(cfg["hqcnn"]["conv_layers"])
    rows = []
    for n in SIZES:
        for topo in ("tree", "brickwork"):
            ops, _ = qcnn_plan(n, cl, "second", topo)
            swaps, native = route(ops, n)
            row = {"n_qubits": n, "topology": topo, "conv_layers": cl, "native_cnot": native,
                   "swaps": swaps, "routed_cnot": native + 3 * swaps,
                   "overhead_factor": (native + 3 * swaps) / native}
            if topo == "tree":
                row["swaps_analytic"] = analytic_tree_swaps(n)
                row["closed_form"] = int(n / 2 * (math.log2(n) - 2) + 1) if n >= 4 else 0
                assert row["swaps_analytic"] == swaps == row["closed_form"], (n, swaps, row)
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "linear_routing.csv", index=False)
    t = df[df.topology == "tree"].set_index("n_qubits")
    b = df[df.topology == "brickwork"].set_index("n_qubits")
    write_tex_table(
        OUT / "linear_routing.tex",
        f"Two-qubit (CNOT) gate count of the tree HQCNN ({cl} conv layers per level) on an all-to-all "
        "versus a linear nearest-neighbour device. Conv block = 2 CNOT, CRZ = CRX = 2 CNOT, SWAP = 3 CNOT. "
        "Tree SWAPs follow $S(n)=\\tfrac{n}{2}(\\log_2 n-2)+1$.",
        "tab:v3_routing",
        ["Qubits", "CNOT (all-to-all)", "SWAPs (linear)", "CNOT (linear)", "Overhead", "Brickwork CNOT (linear)"],
        [[str(n), str(int(t.loc[n, "native_cnot"])), str(int(t.loc[n, "swaps"])), str(int(t.loc[n, "routed_cnot"])),
          f"{t.loc[n, 'overhead_factor']:.2f}$\\times$", str(int(b.loc[n, "routed_cnot"]))] for n in SIZES],
    )
    print(df.to_string(index=False))
    print(f"  wrote {OUT / 'linear_routing.csv'} and .tex (tree SWAP formula verified by the router)")


if __name__ == "__main__":
    main()
