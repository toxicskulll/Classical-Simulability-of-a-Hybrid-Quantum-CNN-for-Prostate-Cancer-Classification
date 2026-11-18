"""
Numerically verify docs/simulability.md section 4 (ABD limitation 2):

  (a) each tree node Phi(rho_a, rho_b) = Tr_a[U (rho_a x rho_b) U^dag] is bilinear in Bloch
      coordinates: r~'_i = sum_jk M_ijk r~a_j r~b_k with r~ = (1, r) and
      M_ijk = 1/4 Tr[(I x sigma_i) U (sigma_j x sigma_k) U^dag], M real, M_0jk = d_j0 d_k0;
  (b) the whole tree evaluated by this Bloch-vector recursion equals twin_expval;
  (c) the encoding RY(x)|0> has Bloch vector (sin x, 0, cos x).

Writes results/phase2/bilinear_check.json.

Usage:
  python scripts/verify_bilinear.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

V3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V3_DIR))

from v3core.config import RESULTS_DIR, load_config  # noqa: E402
from v3core.hqcnn import CONV_PARAMS, n_quantum_params, qcnn_plan  # noqa: E402
from v3core.twin import conv_unitary, encoded_qubit_states, pool_unitary, twin_expval  # noqa: E402

SIG = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]
SIG = [s.astype(complex) for s in SIG]


def node_tensor(U: np.ndarray) -> np.ndarray:
    M = np.empty((4, 4, 4), dtype=complex)
    for i in range(4):
        Oi = np.kron(SIG[0], SIG[i])
        for j in range(4):
            for k in range(4):
                M[i, j, k] = 0.25 * np.trace(Oi @ U @ np.kron(SIG[j], SIG[k]) @ U.conj().T)
    return M


def rho_from_bloch(r):
    return 0.5 * (SIG[0] + r[0] * SIG[1] + r[1] * SIG[2] + r[2] * SIG[3])


def bloch(rho):
    return np.real([np.trace(rho @ s) for s in SIG[1:]])


def random_ball(rng, n):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * rng.uniform(0, 1, (n, 1)) ** (1 / 3)


def pair_unitaries(q, ops, pooling_gate):
    """(src, sink) -> merged conv+pool unitary, in the twin's order."""
    us, i = {}, 0
    for kind, pair in ops:
        if kind == "conv":
            u = conv_unitary(q[i: i + CONV_PARAMS])
            i += CONV_PARAMS
        else:
            u = pool_unitary(q[i: i + 2], pooling_gate)
            i += 2
        us[pair] = u @ us.get(pair, np.eye(4, dtype=complex))
    return us


def main() -> None:
    cfg = load_config()
    h = cfg["hqcnn"]
    n, cl, gates = int(h["n_qubits"]), int(h["conv_layers"]), list(h["reupload_gates"])
    nf = int(cfg["preprocessing"]["pca_dim"])
    ops, ro = qcnn_plan(n, cl, "second", "tree")
    n_q = n_quantum_params(ops, h["pooling_gate"])
    rng = np.random.default_rng(11)
    res = {}

    # (a) node-level bilinearity
    max_imag = max_trace_row = max_node_err = 0.0
    for _ in range(200):
        q = rng.uniform(-np.pi, np.pi, 6 * cl + 2)
        U = np.eye(4, dtype=complex)
        for L in range(cl):
            U = conv_unitary(q[6 * L: 6 * L + 6]) @ U
        U = pool_unitary(q[6 * cl:], h["pooling_gate"]) @ U
        M = node_tensor(np.asarray(U))
        max_imag = max(max_imag, float(np.abs(M.imag).max()))
        target = np.zeros((4, 4)); target[0, 0] = 1.0                            # noqa: E702
        max_trace_row = max(max_trace_row, float(np.abs(M[0].real - target).max()))
        ra, rb = random_ball(rng, 20), random_ball(rng, 20)
        for a, b in zip(ra, rb):
            j = (U @ np.kron(rho_from_bloch(a), rho_from_bloch(b)) @ U.conj().T).reshape(2, 2, 2, 2)
            out = bloch(j[0, :, 0, :] + j[1, :, 1, :])
            pred = np.einsum("ijk,j,k->i", M.real, np.r_[1.0, a], np.r_[1.0, b])[1:]
            max_node_err = max(max_node_err, float(np.abs(out - pred).max()))
    res.update(node_draws=200, node_state_pairs=4000, max_imag_M=max_imag,
               max_dev_trace_row=max_trace_row, max_node_bloch_err=max_node_err)
    print(f"  (a) 200 random nodes x 20 random mixed-state pairs: max|Bloch err| = {max_node_err:.1e}, "
          f"max|Im M| = {max_imag:.1e}, max|M_0jk - d_j0 d_k0| = {max_trace_row:.1e}")

    # (b) whole-tree Bloch recursion vs twin_expval
    max_tree_err = 0.0
    for _ in range(1000):
        x = rng.uniform(0, np.pi, (1, nf))
        q = rng.uniform(-np.pi, np.pi, n_q)
        psis = encoded_qubit_states(x, n, gates)
        state = {k: np.r_[1.0, bloch(np.outer(p[0], p[0].conj()))] for k, p in enumerate(psis)}
        for (a, b), U in pair_unitaries(q, ops, h["pooling_gate"]).items():
            state[b] = np.einsum("ijk,j,k->i", node_tensor(np.asarray(U)).real, state[a], state[b])
            del state[a]
        z_bloch = state[ro][3]
        z_twin = float(np.asarray(twin_expval(x, q, n, cl, gates))[0])
        max_tree_err = max(max_tree_err, abs(z_bloch - z_twin))
    res.update(tree_draws=1000, max_tree_err=max_tree_err)
    print(f"  (b) 1000 random draws, 8q/3L/12 features: max|<Z>_bloch - <Z>_twin| = {max_tree_err:.1e}")

    # (c) encoding
    xs = rng.uniform(0, np.pi, 100)
    psi = encoded_qubit_states(xs[:, None], 1, ["RY"])[0]
    enc_err = max(float(np.abs(bloch(np.outer(p, p.conj())) - [np.sin(x), 0, np.cos(x)]).max())
                  for p, x in zip(psi, xs))
    res["max_encoding_err"] = enc_err
    print(f"  (c) RY(x)|0> Bloch vector vs (sin x, 0, cos x): max|err| = {enc_err:.1e}")

    assert max(max_node_err, max_imag, max_trace_row, max_tree_err, enc_err) < 1e-12
    out = RESULTS_DIR / "phase2" / "bilinear_check.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  all bilinear checks passed -> {out}")


if __name__ == "__main__":
    main()
