"""
Phase 2C mechanism-ablation arms.

Every arm trains on the SAME resampled PCA-12 matrices as the HQCNN (identical
input_digest) and, unless noted, with the HQCNN's own training recipe: focal loss
(gamma from config), Adam (lr), n_epochs, batch_size, balanced class weights, mini-batch
order from an explicit Generator, and an F1 threshold on training-fold probabilities.

Arms
  no_entanglement      HQCNN circuit with CNOTs removed and CRZ/CRX replaced by uncontrolled
                       RZ/RX on the sink (PennyLane statevector). Floor: no information can
                       reach the read-out qubit from the other qubits.
  dephased_control     HQCNN circuit with every control qubit fully dephased immediately
                       before each controlled gate (CNOT, CRZ, CRX): classically controlled
                       gates, so information flows but the state stays separable. Simulated
                       exactly by the density-matrix twin (tree_channel_expval).
  frozen_random        random circuit parameters frozen; only the affine head (a, b) trains.
  frozen_random_multiZ same frozen circuit, <Z> on all 8 qubits -> Phase 1 LR head.
  brickwork_8q         HQCNN with conv_topology=brickwork (PennyLane statevector).
  classical_tree_net   same tree, same fixed Bloch-vector leaves as the twin, but each node
                       is a free tanh map R^6 -> R^3 (no unitarity), linear read-out.
  relaxed_bilinear_tree same tree and leaves; each node is the twin's bilinear Bloch map
                       r_out = M (r~_a (x) r~_b) with the unitarity constraint removed
                       (rank-R CP factorisation of M to keep ~150 parameters).
  mlp_matched          12 -> 10 -> 1 tanh MLP (141 params) trained with the HQCNN recipe.
  mlp_matched_sklearn  same architecture, sklearn MLPClassifier (secondary row).
"""

from __future__ import annotations

import copy
import time

import numpy as onp
import pennylane as qml
from pennylane import numpy as np

from .hqcnn import HQCNN, CONV_PARAMS, n_quantum_params, pool_n_params, qcnn_plan, weighted_loss
from .preprocess import FoldData
from .seeding import component_rng, component_seed
from .twin import (_controlled, _require_verified, _rx, _ry, _rz, CNOT_AB, CNOT_BA,
                   encoded_qubit_states, twin_expval)

Z2 = onp.array([[1, 0], [0, -1]], dtype=complex)
I2 = onp.eye(2, dtype=complex)
ZA = onp.kron(Z2, I2)          # Z on the first qubit of a pair
ZB = onp.kron(I2, Z2)          # Z on the second qubit of a pair


# ── shared pieces ────────────────────────────────────────────────────────────
class AngleScaler:
    """Min-max to angle_range fitted on the training matrix (exactly as HQCNN)."""

    def __init__(self, angle_range):
        self.a, self.b = angle_range

    def fit(self, X):
        self.lo, self.hi = X.min(axis=0), X.max(axis=0)
        return self

    def transform(self, X):
        span = onp.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        return self.a + onp.clip((X - self.lo) / span, 0.0, 1.0) * (self.b - self.a)


def class_weights(y):
    counts = onp.bincount(y.astype(int), minlength=2)
    return onp.where(y == 1, len(y) / (2.0 * counts[1]), len(y) / (2.0 * counts[0]))


def train_adam(logit_fn, params0, X, y, rng, h: dict):
    """The HQCNN training loop (v3core.hqcnn.HQCNN.fit) for an arbitrary logit_fn(p, Xb)."""
    params = np.array(onp.asarray(params0, dtype=float), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=float(h["lr"]))
    w_all = class_weights(y) if h.get("class_weight") == "balanced" else onp.ones(len(y))
    bs, n = int(h["batch_size"]), len(y)
    gamma = float(h.get("focal_gamma", 2.0))
    wd = float(h.get("weight_decay", 0.0))       # optional L2 on all trainable params
    hist = []
    for _ in range(int(h["n_epochs"])):
        order = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, bs):
            bi = order[s: s + bs]
            Xb = X[bi]
            yb = np.array(y[bi], requires_grad=False)
            wb = np.array(w_all[bi], requires_grad=False)

            if wd > 0:
                def cost(p, Xb=Xb, yb=yb, wb=wb):
                    return weighted_loss(logit_fn(p, Xb), yb, wb, h["loss"], gamma) + wd * np.sum(p ** 2)
            else:  # identical graph to the HQCNN loop when no L2 is requested
                def cost(p, Xb=Xb, yb=yb, wb=wb):
                    return weighted_loss(logit_fn(p, Xb), yb, wb, h["loss"], gamma)
            params, c = opt.step_and_cost(cost, params)
            tot += float(c) * len(bi)
        hist.append(tot / n)
    return onp.asarray(params, dtype=float), hist


def sigmoid(z):
    return 1.0 / (1.0 + onp.exp(-z))


def bloch_leaves(Xs, n_qubits: int, gates: list[str]):
    """Fixed leaf states: Bloch vectors of the twin's encoded single-qubit states."""
    psis = encoded_qubit_states(Xs, n_qubits, gates)
    out = []
    for psi in psis:
        a, b = psi[:, 0], psi[:, 1]
        rx = 2.0 * np.real(np.conj(a) * b)
        ry = 2.0 * np.imag(np.conj(a) * b)
        rz = np.abs(a) ** 2 - np.abs(b) ** 2
        out.append(np.stack([rx, ry, rz], axis=1))
    return out


def tree_pairs(n_qubits: int):
    """Pool pairs level by level: [(src, sink), ...] in circuit order, and read-out."""
    ops, ro = qcnn_plan(n_qubits, 1, "second", "tree")
    return [p for t, p in ops if t == "pool"], ro


def result(prob_train, prob_test, h, meta):
    return {"prob_train": onp.asarray(prob_train), "prob_test": onp.asarray(prob_test),
            "threshold_strategy": h["threshold"], "meta": meta}


# ── dephased-control twin ────────────────────────────────────────────────────
def _dephase(rho, Zq):
    return 0.5 * (rho + np.einsum("ij,bjk,kl->bil", Zq, rho, Zq))


def _conj(u, rho):
    return np.einsum("ij,bjk,lk->bil", u, rho, np.conj(u))


def pair_negativity(rho4) -> onp.ndarray:
    """Negativity of each 2-qubit state in a batch (PPT is exact for two qubits)."""
    r = onp.asarray(rho4).reshape(-1, 2, 2, 2, 2).transpose(0, 1, 4, 3, 2).reshape(-1, 4, 4)
    ev = onp.linalg.eigvalsh(0.5 * (r + onp.conj(onp.transpose(r, (0, 2, 1)))))
    return onp.clip(-ev, 0, None).sum(axis=1)


def tree_channel_expval(Xs, q, n_qubits: int, conv_layers: int, gates: list[str],
                        pooling_gate: str = "crx_crz", survivor: str = "second",
                        dephase_controls: bool = False, track_negativity: bool = False):
    """Density-matrix tree simulation, applying each gate of each pair in order.

    dephase_controls=True fully dephases the control qubit right before every CNOT, CRZ
    and CRX (a classically controlled gate), so every pair state stays separable. With
    dephase_controls=False this equals twin_expval (checked in check_phase2_arms.py)."""
    _require_verified(pooling_gate, survivor, gates)
    ops, readout = qcnn_plan(n_qubits, conv_layers, survivor, "tree")
    pool_n = pool_n_params(pooling_gate)
    psis = encoded_qubit_states(Xs, n_qubits, gates)
    rho = {k: np.einsum("bi,bj->bij", psis[k], np.conj(psis[k])) for k in range(n_qubits)}
    joint, i, max_neg = {}, 0, 0.0
    for kind, (a, b) in ops:
        if (a, b) not in joint:
            joint[(a, b)] = np.einsum("bij,bkl->bikjl", rho[a], rho[b]).reshape(-1, 4, 4)
        r = joint[(a, b)]
        if kind == "conv":
            p = q[i: i + CONV_PARAMS]
            i += CONV_PARAMS
            r = _conj(np.kron(_ry(p[0]), _ry(p[1])), r)
            if dephase_controls:
                r = _dephase(r, ZA)
            r = _conj(CNOT_AB, r)
            r = _conj(np.kron(_ry(p[2]), _ry(p[3])), r)
            if dephase_controls:
                r = _dephase(r, ZB)
            r = _conj(CNOT_BA, r)
            r = _conj(np.kron(_rz(p[4]), _rz(p[5])), r)
            joint[(a, b)] = r
        else:
            p = q[i: i + pool_n]
            i += pool_n
            if dephase_controls:
                r = _dephase(r, ZA)
            r = _conj(_controlled(_rz(p[0])), r)
            if dephase_controls:
                r = _dephase(r, ZA)
            r = _conj(_controlled(_rx(p[1])), r)
            if track_negativity:
                max_neg = max(max_neg, float(pair_negativity(r).max()))
            j = r.reshape(-1, 2, 2, 2, 2)
            rho[b] = j[:, 0, :, 0, :] + j[:, 1, :, 1, :]
            del rho[a], joint[(a, b)]
    rr = rho[readout]
    z = np.real(rr[:, 0, 0] - rr[:, 1, 1])
    return (z, max_neg) if track_negativity else z


# ── arm fitters: fn(fd, cfg, seed, fold) -> result dict ──────────────────────
def _arm_cfg(cfg: dict, name: str) -> dict:
    h = copy.deepcopy(cfg["hqcnn"])
    h.update(cfg.get("phase2_arms", {}).get(name, {}).get("hqcnn_overrides", {}))
    return h


def fit_hqcnn_variant(name: str):
    """PennyLane HQCNN with config overrides (no_entanglement, brickwork_8q)."""
    def fit(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
        h = _arm_cfg(cfg, name)
        model = HQCNN(h)
        t0 = time.perf_counter()
        model.fit(fd.X_fit, fd.y_fit, component_rng(seed, fold, name))
        return result(model.predict_proba(fd.X_train), model.predict_proba(fd.X_test), h, {
            "fit_seconds": time.perf_counter() - t0, "n_params": model.n_params,
            "n_quantum_params": model.n_q, "loss_history": model.loss_history_,
            "params": model.params_.tolist(), "hqcnn_overrides": {k: h[k] for k in
                cfg["phase2_arms"][name].get("hqcnn_overrides", {})},
        })
    return fit


def fit_dephased_control(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    h = _arm_cfg(cfg, "dephased_control")
    n, cl, gates = int(h["n_qubits"]), int(h["conv_layers"]), list(h["reupload_gates"])
    ops, _ = qcnn_plan(n, cl, h["pooling_survivor"], "tree")
    n_q = n_quantum_params(ops, h["pooling_gate"])
    sc = AngleScaler(h["angle_range"]).fit(fd.X_fit)
    rng = component_rng(seed, fold, "dephased_control")
    p0 = onp.concatenate([rng.uniform(-onp.pi, onp.pi, n_q), [1.0, 0.0]])

    def logit(p, Xb):
        z = tree_channel_expval(np.array(Xb, requires_grad=False), p[:n_q], n, cl, gates,
                                h["pooling_gate"], h["pooling_survivor"], dephase_controls=True)
        return p[-2] * z + p[-1]
    t0 = time.perf_counter()
    params, hist = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), rng, h)
    prob = lambda X: sigmoid(onp.asarray(logit(np.array(params), sc.transform(X))))  # noqa: E731
    _, neg = tree_channel_expval(sc.transform(fd.X_test), params[:n_q], n, cl, gates,
                                 dephase_controls=True, track_negativity=True)
    return result(prob(fd.X_train), prob(fd.X_test), h, {
        "fit_seconds": time.perf_counter() - t0, "n_params": n_q + 2, "n_quantum_params": n_q,
        "loss_history": hist, "params": params.tolist(), "max_pair_negativity_test": neg,
        "simulator": "density-matrix twin (tree_channel_expval)"})


def _frozen_circuit(fd: FoldData, cfg: dict, seed: int, fold: int):
    """Shared by frozen_random and frozen_random_multiZ: same random circuit per (seed, fold)."""
    h = copy.deepcopy(cfg["hqcnn"])
    model = HQCNN(h)
    q = component_rng(seed, fold, "frozen_random").uniform(-onp.pi, onp.pi, model.n_q)
    sc = AngleScaler(h["angle_range"]).fit(fd.X_fit)
    return h, model, np.array(q, requires_grad=False), sc


def fit_frozen_random(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    h, model, q, sc = _frozen_circuit(fd, cfg, seed, fold)
    z = {k: onp.asarray(model._z(sc.transform(X), q)) for k, X in
         (("fit", fd.X_fit), ("train", fd.X_train), ("test", fd.X_test))}
    rng = component_rng(seed, fold, "frozen_random:head")
    t0 = time.perf_counter()
    params, hist = train_adam(lambda p, zb: p[0] * zb + p[1], [1.0, 0.0], z["fit"],
                              fd.y_fit.astype(float), rng, h)
    return result(sigmoid(params[0] * z["train"] + params[1]), sigmoid(params[0] * z["test"] + params[1]),
                  h, {"fit_seconds": time.perf_counter() - t0, "n_params": 2, "n_trained_params": 2,
                      "n_frozen_params": int(model.n_q), "head": params.tolist(), "loss_history": hist})


def fit_frozen_random_multiz(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from .hqcnn import build_qnode
    h, model, q, sc = _frozen_circuit(fd, cfg, seed, fold)
    circ = build_qnode(model.n_qubits, model.ops, model.readout, list(h["reupload_gates"]),
                       h["pooling_gate"], h["simulator"]["device"], h["simulator"]["diff_method"],
                       measure="all_z")
    feats = lambda X: onp.stack([onp.asarray(v) for v in circ(np.array(sc.transform(X).T), q)], axis=1)  # noqa: E731
    F = {k: feats(X) for k, X in (("fit", fd.X_fit), ("train", fd.X_train), ("test", fd.X_test))}
    p = cfg["baselines"]["lr"]
    est = make_pipeline(StandardScaler(), LogisticRegression(
        C=p["C"], max_iter=p["max_iter"], random_state=component_seed(seed, fold, "frozen_random_multiZ")))
    t0 = time.perf_counter()
    est.fit(F["fit"], fd.y_fit)
    return {"prob_train": est.predict_proba(F["train"])[:, 1], "prob_test": est.predict_proba(F["test"])[:, 1],
            "threshold_strategy": cfg["baselines"]["threshold"],
            "meta": {"fit_seconds": time.perf_counter() - t0, "n_params": model.n_qubits + 1,
                     "n_trained_params": model.n_qubits + 1, "n_frozen_params": int(model.n_q)}}


def _tree_leaves(fd, h):
    sc = AngleScaler(h["angle_range"]).fit(fd.X_fit)
    n, gates = int(h["n_qubits"]), list(h["reupload_gates"])
    return sc, n, gates


def fit_classical_tree_net(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    """Same tree + same fixed Bloch leaves; node h_b <- tanh(W [h_a; h_b] + c), W: 3x6."""
    h = copy.deepcopy(cfg["hqcnn"])
    sc, n, gates = _tree_leaves(fd, h)
    pairs, ro = tree_pairs(n)
    per_node = 3 * 6 + 3
    rng = component_rng(seed, fold, "classical_tree_net")
    p0 = []
    for _ in pairs:
        p0 += list(rng.normal(0.0, onp.sqrt(1.0 / 6.0), 18)) + [0.0] * 3
    p0 += list(rng.normal(0.0, onp.sqrt(1.0 / 3.0), 3)) + [0.0]

    def logit(p, Xb):
        state = dict(enumerate(bloch_leaves(np.array(Xb, requires_grad=False), n, gates)))
        for k, (a, b) in enumerate(pairs):
            W = np.reshape(p[k * per_node: k * per_node + 18], (3, 6))
            c = p[k * per_node + 18: (k + 1) * per_node]
            state[b] = np.tanh(np.concatenate([state[a], state[b]], axis=1) @ W.T + c)
        w = p[len(pairs) * per_node: len(pairs) * per_node + 3]
        return state[ro] @ w + p[-1]
    t0 = time.perf_counter()
    params, hist = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), rng, h)
    prob = lambda X: sigmoid(onp.asarray(logit(np.array(params), sc.transform(X))))  # noqa: E731
    return result(prob(fd.X_train), prob(fd.X_test), h, {
        "fit_seconds": time.perf_counter() - t0, "n_params": len(p0), "loss_history": hist,
        "params": params.tolist()})


def fit_relaxed_bilinear_tree(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    """Twin's bilinear Bloch node without the unitarity constraint:
    r_out_i = sum_r U[i,r] (V[:,r] . r~_a) (W[:,r] . r~_b), r~ = (1, r); read-out a*z + b."""
    h = copy.deepcopy(cfg["hqcnn"])
    R = int(cfg["phase2_arms"]["relaxed_bilinear_tree"].get("rank", 2))
    sc, n, gates = _tree_leaves(fd, h)
    pairs, ro = tree_pairs(n)
    per_node = 3 * R + 4 * R + 4 * R
    rng = component_rng(seed, fold, "relaxed_bilinear_tree")
    p0 = list(rng.normal(0.0, 0.5, per_node * len(pairs))) + [1.0, 0.0]

    def logit(p, Xb):
        leaves = bloch_leaves(np.array(Xb, requires_grad=False), n, gates)
        ones = np.ones((leaves[0].shape[0], 1))
        state = {k: np.concatenate([ones, v], axis=1) for k, v in enumerate(leaves)}
        for k, (a, b) in enumerate(pairs):
            o = k * per_node
            U = np.reshape(p[o: o + 3 * R], (3, R))
            V = np.reshape(p[o + 3 * R: o + 7 * R], (4, R))
            W = np.reshape(p[o + 7 * R: o + 11 * R], (4, R))
            r_out = ((state[a] @ V) * (state[b] @ W)) @ U.T
            state[b] = np.concatenate([ones, r_out], axis=1)
        return p[-2] * state[ro][:, 3] + p[-1]
    t0 = time.perf_counter()
    params, hist = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), rng, h)
    prob = lambda X: sigmoid(onp.asarray(logit(np.array(params), sc.transform(X))))  # noqa: E731
    return result(prob(fd.X_train), prob(fd.X_test), h, {
        "fit_seconds": time.perf_counter() - t0, "n_params": len(p0), "rank": R,
        "loss_history": hist, "params": params.tolist()})


def fit_mlp_matched(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    """12 -> H -> 1 tanh MLP on standardised PCA inputs, HQCNN training recipe."""
    from sklearn.preprocessing import StandardScaler
    h = copy.deepcopy(cfg["hqcnn"])
    H = int(cfg["phase2_arms"]["mlp_matched"]["hidden"])
    d = fd.X_fit.shape[1]
    sc = StandardScaler().fit(fd.X_fit)
    rng = component_rng(seed, fold, "mlp_matched")
    p0 = list(rng.normal(0.0, onp.sqrt(1.0 / d), d * H)) + [0.0] * H + \
        list(rng.normal(0.0, onp.sqrt(1.0 / H), H)) + [0.0]

    def logit(p, Xb):
        W1 = np.reshape(p[: d * H], (d, H))
        b1 = p[d * H: d * H + H]
        w2 = p[d * H + H: d * H + 2 * H]
        return np.tanh(Xb @ W1 + b1) @ w2 + p[-1]
    t0 = time.perf_counter()
    params, hist = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), rng, h)
    prob = lambda X: sigmoid(onp.asarray(logit(np.array(params), sc.transform(X))))  # noqa: E731
    return result(prob(fd.X_train), prob(fd.X_test), h, {
        "fit_seconds": time.perf_counter() - t0, "n_params": len(p0), "loss_history": hist})


def fit_mlp_matched_sklearn(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    H = int(cfg["phase2_arms"]["mlp_matched"]["hidden"])
    p = cfg["baselines"]["mlp"]
    est = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(H,), activation="tanh", alpha=p["alpha"], max_iter=p["max_iter"],
        random_state=component_seed(seed, fold, "mlp_matched_sklearn")))
    t0 = time.perf_counter()
    est.fit(fd.X_fit, fd.y_fit)
    mlp = est.steps[-1][1]
    n_params = int(sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_))
    return {"prob_train": est.predict_proba(fd.X_train)[:, 1], "prob_test": est.predict_proba(fd.X_test)[:, 1],
            "threshold_strategy": cfg["baselines"]["threshold"],
            "meta": {"fit_seconds": time.perf_counter() - t0, "n_params": n_params}}


def fit_hqcnn_twin(fd: FoldData, cfg: dict, seed: int, fold: int) -> dict:
    """The champion HQCNN trained through the exact twin instead of PennyLane.
    Same init/shuffle stream ("hqcnn") and loop as fit_hqcnn; reproduces it to 4.8e-11
    (check_phase2_arms.py --trainer-check) and is ~1.7x faster. Tree topology only."""
    h = copy.deepcopy(cfg["hqcnn"])
    if h.get("conv_topology", "tree") != "tree" or h.get("entangling", "full") != "full":
        raise NotImplementedError("hqcnn_twin is exact only for the full-entangling tree")
    n, cl, gates = int(h["n_qubits"]), int(h["conv_layers"]), list(h["reupload_gates"])
    ops, _ = qcnn_plan(n, cl, h["pooling_survivor"], "tree")
    n_q = n_quantum_params(ops, h["pooling_gate"])
    sc = AngleScaler(h["angle_range"]).fit(fd.X_fit)
    rng = component_rng(seed, fold, "hqcnn")
    p0 = onp.concatenate([rng.uniform(-onp.pi, onp.pi, n_q), [1.0, 0.0]])

    def logit(p, Xb):
        z = twin_expval(np.array(Xb, requires_grad=False), p[:n_q], n, cl, gates,
                        h["pooling_gate"], h["pooling_survivor"])
        return p[-2] * z + p[-1]
    t0 = time.perf_counter()
    params, hist = train_adam(logit, p0, sc.transform(fd.X_fit), fd.y_fit.astype(float), rng, h)
    prob = lambda X: sigmoid(onp.asarray(logit(np.array(params), sc.transform(X))))  # noqa: E731
    return result(prob(fd.X_train), prob(fd.X_test), h, {
        "fit_seconds": time.perf_counter() - t0, "n_params": n_q + 2, "n_quantum_params": n_q,
        "loss_history": hist, "params": params.tolist(), "scaler_lo": sc.lo.tolist(),
        "scaler_hi": sc.hi.tolist(), "simulator": "exact twin (twin_expval)"})


ARM_REGISTRY = {
    "hqcnn_twin": fit_hqcnn_twin,
    "no_entanglement": fit_hqcnn_variant("no_entanglement"),
    "dephased_control": fit_dephased_control,
    "frozen_random": fit_frozen_random,
    "frozen_random_multiZ": fit_frozen_random_multiz,
    "brickwork_8q": fit_hqcnn_variant("brickwork_8q"),
    "classical_tree_net": fit_classical_tree_net,
    "relaxed_bilinear_tree": fit_relaxed_bilinear_tree,
    "mlp_matched": fit_mlp_matched,
    "mlp_matched_sklearn": fit_mlp_matched_sklearn,
}
