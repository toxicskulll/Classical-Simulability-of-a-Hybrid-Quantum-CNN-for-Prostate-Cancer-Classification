"""
Classical twin of the tree HQCNN: exact, O(n_qubits), no statevector.

Why it is exact (details in docs/simulability.md):
  * the encoding is a product state: every qubit gets only single-qubit gates;
  * at each level, conv blocks and the pool act only inside disjoint pairs, so
    different pairs never interact within a level;
  * after pooling, the discarded qubit is never touched again and is not
    measured, so it can be traced out without changing <Z> on the read-out;
  * so the survivors are always in a product of (mixed) single-qubit states,
    and each pair needs only a 4x4 density matrix.

Cost: one 4x4 conjugation per conv/pool op, i.e. O(n) for n qubits, vs O(2^n)
memory for a statevector. Written in pennylane.numpy (autograd), so gradients
come from the same autodiff machinery as the HQCNN.
"""

from __future__ import annotations

import numpy as onp
from pennylane import numpy as np

from .hqcnn import CONV_PARAMS, pool_n_params, qcnn_plan

Z = onp.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)


# ── gate matrices (autograd-friendly) ───────────────────────────────────────
def _ry(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c, -s], [s, c]]) + 0j


def _rz(t):
    return np.array([[np.exp(-0.5j * t), 0.0 * t], [0.0 * t, np.exp(0.5j * t)]])


def _rx(t):
    c, s = np.cos(t / 2.0), np.sin(t / 2.0)
    return np.array([[c + 0j, -1j * s], [-1j * s, c + 0j]])


GATE_FN = {"RY": _ry, "RZ": _rz, "RX": _rx}


# Only the champion's pooling settings have been verified against the statevector
# (scripts/twin_equivalence.py). Everything else is refused rather than trusted.
VERIFIED_POOLING_GATES = ("crx_crz",)
VERIFIED_SURVIVORS = ("second",)
VERIFIED_ENCODING_GATES = ("RY", "RZ", "RX")


def _require_verified(pooling_gate: str, survivor: str, gates) -> None:
    if pooling_gate not in VERIFIED_POOLING_GATES:
        raise NotImplementedError(
            f"twin not verified for pooling_gate={pooling_gate!r}; verified: {VERIFIED_POOLING_GATES}")
    if survivor not in VERIFIED_SURVIVORS:
        raise NotImplementedError(
            f"twin not verified for pooling_survivor={survivor!r}; verified: {VERIFIED_SURVIVORS}")
    bad = [g for g in gates if g not in VERIFIED_ENCODING_GATES]
    if bad:
        raise NotImplementedError(f"twin not verified for encoding gates {bad}")

CNOT_AB = onp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
CNOT_BA = onp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex)
P0 = onp.array([[1, 0], [0, 0]], dtype=complex)
P1 = onp.array([[0, 0], [0, 1]], dtype=complex)
I2 = onp.eye(2, dtype=complex)


def _controlled(u):
    """|0><0| x I + |1><1| x U  (control = first qubit of the pair)."""
    return np.kron(P0, I2) + np.kron(P1, u)


def conv_unitary(p):
    u = np.kron(_ry(p[0]), _ry(p[1]))
    u = CNOT_AB @ u
    u = np.kron(_ry(p[2]), _ry(p[3])) @ u
    u = CNOT_BA @ u
    return np.kron(_rz(p[4]), _rz(p[5])) @ u


def pool_unitary(p, gate: str):
    if gate == "crx_crz":
        return _controlled(_rx(p[1])) @ _controlled(_rz(p[0]))
    if gate == "cry":
        return _controlled(_ry(p[1])) @ _controlled(_ry(p[0]))
    if gate == "cr":
        return _controlled(_rz(p[0]))
    raise ValueError(f"unknown pooling_gate: {gate}")


# ── simulation ──────────────────────────────────────────────────────────────
def encoded_qubit_states(Xs, n_qubits: int, gates: list[str]):
    """Per-qubit pure states after encoding: list of (B, 2) arrays."""
    B, n_feat = Xs.shape
    states = []
    for q in range(n_qubits):
        psi = np.array(onp.tile(onp.array([1.0, 0.0], dtype=complex), (B, 1)))
        for r, gname in enumerate(gates):
            i = r * n_qubits + q
            if i >= n_feat:
                break
            t = Xs[:, i]
            if gname == "RY":
                c, s = np.cos(t / 2.0), np.sin(t / 2.0)
                psi = np.stack([c * psi[:, 0] - s * psi[:, 1], s * psi[:, 0] + c * psi[:, 1]], axis=1)
            elif gname == "RZ":
                psi = np.stack([np.exp(-0.5j * t) * psi[:, 0], np.exp(0.5j * t) * psi[:, 1]], axis=1)
            elif gname == "RX":
                c, s = np.cos(t / 2.0), np.sin(t / 2.0)
                psi = np.stack([c * psi[:, 0] - 1j * s * psi[:, 1], -1j * s * psi[:, 0] + c * psi[:, 1]], axis=1)
            else:
                raise ValueError(gname)
        states.append(psi)
    return states


def twin_expval(Xs, q, n_qubits: int, conv_layers: int, gates: list[str],
                pooling_gate: str = "crx_crz", survivor: str = "second"):
    """<Z> on the read-out qubit for a batch of scaled inputs Xs (B, n_feat).

    Mirrors v3core.hqcnn exactly for conv_topology='tree'."""
    _require_verified(pooling_gate, survivor, gates)
    ops, readout = qcnn_plan(n_qubits, conv_layers, survivor, topology="tree")
    pool_n = pool_n_params(pooling_gate)

    psis = encoded_qubit_states(Xs, n_qubits, gates)
    rho = {q_: np.einsum("bi,bj->bij", psis[q_], np.conj(psis[q_])) for q_ in range(n_qubits)}

    # group ops into per-pair programmes, preserving order within each pair
    idx, i = {}, 0
    level_ops = []            # list of lists of (kind, pair, param_slice)
    current, seen_pool = [], False
    for kind, pair in ops:
        n_p = CONV_PARAMS if kind == "conv" else pool_n
        if kind == "conv" and seen_pool:
            level_ops.append(current)
            current, seen_pool = [], False
        current.append((kind, pair, slice(i, i + n_p)))
        seen_pool = seen_pool or kind == "pool"
        i += n_p
    level_ops.append(current)

    for level in level_ops:
        pairs = []
        for _, pair, _ in level:
            if pair not in pairs:
                pairs.append(pair)
        for a, b in pairs:
            u_total = np.eye(4) + 0j
            for kind, pair, sl in level:
                if pair != (a, b):
                    continue
                u = conv_unitary(q[sl]) if kind == "conv" else pool_unitary(q[sl], pooling_gate)
                u_total = u @ u_total
            joint = np.einsum("bij,bkl->bikjl", rho[a], rho[b]).reshape(-1, 4, 4)
            joint = np.einsum("ij,bjk,lk->bil", u_total, joint, np.conj(u_total))
            j = joint.reshape(-1, 2, 2, 2, 2)
            # partial trace written out (autograd's einsum VJP rejects repeated indices)
            if survivor == "second":
                rho[b] = j[:, 0, :, 0, :] + j[:, 1, :, 1, :]
            else:
                rho[a] = j[:, :, 0, :, 0] + j[:, :, 1, :, 1]
            del rho[b if survivor == "first" else a]
    r = rho[readout]
    return np.real(r[:, 0, 0] - r[:, 1, 1])


def product_state_expval(Xs, q, ops, readout: int, n_qubits: int, gates: list[str],
                         pooling_gate: str = "crx_crz", survivor: str = "second"):
    """The twin's algorithm applied to ANY pairwise circuit (e.g. brickwork).

    Consecutive ops are merged while they act on one fixed set of disjoint pairs (a
    "phase"). When an op touches a qubit that belongs to a different pair in the
    current phase, the phase is flushed: each pair's 4x4 state is evolved, then pooled
    pairs keep the survivor's marginal and un-pooled pairs are replaced by the product
    of their two marginals.

    For the tree, every phase ends in a pool, so no correlation is ever dropped and the
    result is exact (== twin_expval). For brickwork, the offset layer starts a new phase
    on pairs that are still correlated, so marginalising discards correlations and the
    result is only a product-state (mean-field) approximation.

    Returns (<Z> per input, n_lossy_flushes): the number of un-pooled pair states that
    were replaced by products of marginals.
    """
    _require_verified(pooling_gate, survivor, gates)
    pool_n = pool_n_params(pooling_gate)
    psis = encoded_qubit_states(Xs, n_qubits, gates)
    rho = {k: np.einsum("bi,bj->bij", psis[k], np.conj(psis[k])) for k in range(n_qubits)}
    phase: dict[tuple, dict] = {}      # pair -> {"u": 4x4, "pooled": bool}
    owner: dict[int, tuple] = {}       # qubit -> pair in current phase
    lossy = 0

    def flush():
        nonlocal lossy
        for (a, b), st in phase.items():
            joint = np.einsum("bij,bkl->bikjl", rho[a], rho[b]).reshape(-1, 4, 4)
            joint = np.einsum("ij,bjk,lk->bil", st["u"], joint, np.conj(st["u"]))
            j = joint.reshape(-1, 2, 2, 2, 2)
            rb = j[:, 0, :, 0, :] + j[:, 1, :, 1, :]          # marginal of b
            ra = j[:, :, 0, :, 0] + j[:, :, 1, :, 1]          # marginal of a
            if st["pooled"]:
                keep, drop = (b, a) if survivor == "second" else (a, b)
                rho[keep] = rb if keep == b else ra
                del rho[drop]
            else:
                rho[a], rho[b] = ra, rb
                lossy += 1
        phase.clear()
        owner.clear()

    i = 0
    for kind, pair in ops:
        n_p = CONV_PARAMS if kind == "conv" else pool_n
        u = conv_unitary(q[i: i + n_p]) if kind == "conv" else pool_unitary(q[i: i + n_p], pooling_gate)
        i += n_p
        a, b = pair
        if any(owner.get(w, pair) != pair for w in pair) or \
                (pair in phase and phase[pair]["pooled"]):
            flush()
        st = phase.setdefault(pair, {"u": np.eye(4) + 0j, "pooled": False})
        st["u"] = u @ st["u"]
        st["pooled"] = st["pooled"] or kind == "pool"
        owner[a] = owner[b] = pair
    flush()
    r = rho[readout]
    return np.real(r[:, 0, 0] - r[:, 1, 1]), lossy
