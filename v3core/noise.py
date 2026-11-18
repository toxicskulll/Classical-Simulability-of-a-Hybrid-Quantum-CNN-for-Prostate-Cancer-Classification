"""
Phase 3 step 2: local gate noise on the tree HQCNN, simulated exactly by the twin.

Noise model (applied after EVERY gate, including the encoding rotations):
  1-qubit gate on q      : depolarising(p1) on q, then amplitude damping(gamma = ad_ratio*p1) on q
  2-qubit gate on (a, b) : two-qubit depolarising(p2) on (a, b), then amplitude damping
                           (gamma = ad_ratio*p2) on a and on b
  depolarising(p), 1q    : rho -> (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)   [PennyLane convention]
  depolarising(p), 2q    : rho -> (1-p) rho + p/15 sum_{P != II} P rho P
  amplitude damping(g)   : K0 = [[1,0],[0,sqrt(1-g)]], K1 = [[0,sqrt(g)],[0,0]]

Local channels keep the tree argument intact (they act inside a pair), so this is exact
and O(n). Checked against PennyLane default.mixed in scripts/noise_study.py --verify.
Shot noise: <Z> estimated from N single-shot +/-1 outcomes, i.e. 2*Binomial(N, (1+<Z>)/2)/N - 1.
"""

from __future__ import annotations

import numpy as onp
from pennylane import numpy as np

from .hqcnn import CONV_PARAMS, pool_n_params, qcnn_plan
from .twin import _controlled, _require_verified, _rx, _ry, _rz, CNOT_AB, CNOT_BA

I2 = onp.eye(2, dtype=complex)
I4 = onp.eye(4, dtype=complex)


def _conj(u, rho):
    return np.einsum("ij,bjk,lk->bil", u, rho, np.conj(u))


# ── single-qubit states (encoding stage) ─────────────────────────────────────
def dep1_single(rho, p):
    """1q depolarising on a batch of 2x2 states: (1 - 4p/3) rho + (2p/3) Tr(rho) I."""
    if p == 0:
        return rho
    tr = rho[:, 0, 0] + rho[:, 1, 1]
    return (1 - 4 * p / 3) * rho + (2 * p / 3) * np.einsum("b,ij->bij", tr, I2)


def ad_single(rho, g):
    if g == 0:
        return rho
    k0 = onp.array([[1, 0], [0, onp.sqrt(1 - g)]], dtype=complex)
    k1 = onp.array([[0, onp.sqrt(g)], [0, 0]], dtype=complex)
    return _conj(k0, rho) + _conj(k1, rho)


# ── pair states (4x4, qubit a = first factor) ────────────────────────────────
def _embed(k, which):
    """k (x) I or I (x) k; np.kron keeps autograd tracing when k depends on parameters."""
    return np.kron(k, I2) if which == 0 else np.kron(I2, k)


def dep1_pair(rho, p, which):
    """1q depolarising on qubit `which` (0 = a, 1 = b) of a pair:
    (1 - 4p/3) rho + (2p/3) I_q (x) Tr_q(rho)."""
    if p == 0:
        return rho
    j = rho.reshape(-1, 2, 2, 2, 2)
    if which == 0:
        red = j[:, 0, :, 0, :] + j[:, 1, :, 1, :]                        # Tr_a -> rho_b
        mixed = np.einsum("ij,bkl->bikjl", I2, red).reshape(-1, 4, 4)
    else:
        red = j[:, :, 0, :, 0] + j[:, :, 1, :, 1]                        # Tr_b -> rho_a
        mixed = np.einsum("bij,kl->bikjl", red, I2).reshape(-1, 4, 4)
    return (1 - 4 * p / 3) * rho + (2 * p / 3) * mixed


def dep2_pair(rho, p):
    """2q depolarising: (1 - 16p/15) rho + (4p/15) Tr(rho) I4, since sum_P P rho P = 4 Tr(rho) I4."""
    if p == 0:
        return rho
    tr = rho[:, 0, 0] + rho[:, 1, 1] + rho[:, 2, 2] + rho[:, 3, 3]
    return (1 - 16 * p / 15) * rho + (4 * p / 15) * np.einsum("b,ij->bij", tr, I4)


def ad_pair(rho, g, which):
    if g == 0:
        return rho
    k0 = _embed(onp.array([[1, 0], [0, onp.sqrt(1 - g)]], dtype=complex), which)
    k1 = _embed(onp.array([[0, onp.sqrt(g)], [0, 0]], dtype=complex), which)
    return _conj(k0, rho) + _conj(k1, rho)


def noise_1q_pair(rho, which, p1, ad_ratio):
    return ad_pair(dep1_pair(rho, p1, which), ad_ratio * p1, which)


def noise_2q_pair(rho, p2, ad_ratio):
    rho = dep2_pair(rho, p2)
    return ad_pair(ad_pair(rho, ad_ratio * p2, 0), ad_ratio * p2, 1)


# ── noisy encoding + tree ───────────────────────────────────────────────────
def noisy_encoded_states(Xs, n_qubits: int, gates: list[str], p1: float, ad_ratio: float):
    B, n_feat = Xs.shape
    rho0 = onp.zeros((B, 2, 2), dtype=complex)
    rho0[:, 0, 0] = 1.0
    out = []
    for q in range(n_qubits):
        rho = np.array(rho0)
        for r, gname in enumerate(gates):
            i = r * n_qubits + q
            if i >= n_feat:
                break
            t = Xs[:, i]
            c, s = np.cos(t / 2.0), np.sin(t / 2.0)
            if gname == "RY":
                u = np.stack([np.stack([c, -s], 1), np.stack([s, c], 1)], 1) + 0j
            elif gname == "RZ":
                e = np.exp(-0.5j * t)
                u = np.stack([np.stack([e, 0 * e], 1), np.stack([0 * e, np.conj(e)], 1)], 1)
            elif gname == "RX":
                u = np.stack([np.stack([c + 0j, -1j * s], 1), np.stack([-1j * s, c + 0j], 1)], 1)
            else:
                raise ValueError(gname)
            rho = np.einsum("bij,bjk,blk->bil", u, rho, np.conj(u))
            rho = ad_single(dep1_single(rho, p1), ad_ratio * p1)
        out.append(rho)
    return out


def noisy_tree_expval(Xs, q, n_qubits: int, conv_layers: int, gates: list[str],
                      p1: float = 0.0, p2: float = 0.0, ad_ratio: float = 1.0,
                      pooling_gate: str = "crx_crz", survivor: str = "second"):
    """<Z> on the read-out with local depolarising + amplitude damping after every gate."""
    _require_verified(pooling_gate, survivor, gates)
    ops, readout = qcnn_plan(n_qubits, conv_layers, survivor, "tree")
    pool_n = pool_n_params(pooling_gate)
    rho = dict(enumerate(noisy_encoded_states(Xs, n_qubits, gates, p1, ad_ratio)))
    joint, i = {}, 0

    def g1(r, u, which):                               # 1q gate on a pair + its noise
        return noise_1q_pair(_conj(_embed(u, which), r), which, p1, ad_ratio)

    def g2(r, u):                                       # 2q gate + its noise
        return noise_2q_pair(_conj(u, r), p2, ad_ratio)

    for kind, (a, b) in ops:
        if (a, b) not in joint:
            joint[(a, b)] = np.einsum("bij,bkl->bikjl", rho[a], rho[b]).reshape(-1, 4, 4)
        r = joint[(a, b)]
        if kind == "conv":
            p = q[i: i + CONV_PARAMS]
            i += CONV_PARAMS
            r = g1(g1(r, _ry(p[0]), 0), _ry(p[1]), 1)
            r = g2(r, CNOT_AB)
            r = g1(g1(r, _ry(p[2]), 0), _ry(p[3]), 1)
            r = g2(r, CNOT_BA)
            r = g1(g1(r, _rz(p[4]), 0), _rz(p[5]), 1)
            joint[(a, b)] = r
        else:
            p = q[i: i + pool_n]
            i += pool_n
            r = g2(r, _controlled(_rz(p[0])))
            r = g2(r, _controlled(_rx(p[1])))
            j = r.reshape(-1, 2, 2, 2, 2)
            rho[b] = j[:, 0, :, 0, :] + j[:, 1, :, 1, :]
            del rho[a], joint[(a, b)]
    rr = rho[readout]
    return np.real(rr[:, 0, 0] - rr[:, 1, 1])


def shot_estimate(z: onp.ndarray, shots: int, rng: onp.random.Generator) -> onp.ndarray:
    p_plus = onp.clip((1.0 + onp.asarray(z, dtype=float)) / 2.0, 0.0, 1.0)
    return 2.0 * rng.binomial(shots, p_plus) / shots - 1.0
