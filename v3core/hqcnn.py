"""
Hybrid quantum convolutional neural network (HQCNN), v3.

Circuit (Cong et al.-style QCNN with a classical affine read-out):

  encoding   features loaded in rounds of n_qubits; round r applies gate
             reupload_gates[r] (default RY, RZ, RX, RY). 12 features on 8 qubits:
             RY(x1..x8) on q0..q7, then RZ(x9..x12) on q0..q3.
  conv       per disjoint neighbouring pair (a, b), repeated conv_layers times:
             RY(t1)xRY(t2), CNOT(a->b), RY(t3)xRY(t4), CNOT(b->a), RZ(t5)xRZ(t6)
             conv_topology='brickwork' adds one offset conv layer on (a1,a2), (a3,a4), ...
             before each pooling level (breaks the tree; see docs/simulability.md)
  pooling    CRZ then CRX from source to sink; the survivor (default: second
             qubit of each pair) carries on: 8 -> 4 -> 2 -> 1
  readout    <Z> on the surviving qubit; logit = a * <Z> + b; p = sigmoid(logit)

Training: Adam on focal/BCE/MSE loss with class weights. Mini-batch order and
parameter initialisation come ONLY from the Generator passed to fit().
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

CONV_PARAMS = 6
GATES = {"RY": qml.RY, "RZ": qml.RZ, "RX": qml.RX}


# ── circuit structure ────────────────────────────────────────────────────────
def qcnn_plan(n_qubits: int, conv_layers: int, survivor: str = "second",
              topology: str = "tree"):
    """(op, wires) schedule and the read-out qubit."""
    if survivor not in ("first", "second"):
        raise ValueError(f"unknown pooling_survivor: {survivor}")
    if topology not in ("tree", "brickwork"):
        raise ValueError(f"unknown conv_topology: {topology}")
    ops, active = [], list(range(n_qubits))
    while len(active) > 1:
        pairs = [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
        for _ in range(conv_layers):
            ops.extend(("conv", p) for p in pairs)
        if topology == "brickwork":
            # one offset conv layer (a1,a2), (a3,a4), ... couples neighbouring pairs
            # before pooling, so the circuit is no longer a tree
            ops.extend(("conv", (active[i], active[i + 1]))
                       for i in range(1, len(active) - 1, 2))
        nxt = [p[0] if survivor == "first" else p[1] for p in pairs]
        ops.extend(("pool", p) for p in pairs)
        if len(active) % 2:
            nxt.append(active[-1])
        active = nxt
    return ops, active[0]


def pool_n_params(gate: str) -> int:
    return {"crx_crz": 2, "cry": 2, "cr": 1}[gate]


def n_quantum_params(ops, pooling_gate: str) -> int:
    return sum(CONV_PARAMS if t == "conv" else pool_n_params(pooling_gate) for t, _ in ops)


def encode(x, n_qubits: int, gates: list[str]) -> None:
    """x has shape (n_features,) or (n_features, batch) for parameter broadcasting."""
    n = x.shape[0]
    if n > len(gates) * n_qubits:
        raise ValueError(f"{n} features exceed {len(gates)} re-upload rounds x {n_qubits} qubits")
    for r, gname in enumerate(gates):
        start = r * n_qubits
        if start >= n:
            break
        for q in range(n_qubits):
            if start + q < n:
                GATES[gname](x[start + q], wires=q)


def conv_block(p, wires, entangling: str = "full") -> None:
    """entangling='none' drops both CNOTs (same 6 parameters, no two-qubit gate)."""
    a, b = wires
    qml.RY(p[0], wires=a)
    qml.RY(p[1], wires=b)
    if entangling == "full":
        qml.CNOT(wires=[a, b])
    qml.RY(p[2], wires=a)
    qml.RY(p[3], wires=b)
    if entangling == "full":
        qml.CNOT(wires=[b, a])
    qml.RZ(p[4], wires=a)
    qml.RZ(p[5], wires=b)


def pool_block(p, wires, gate: str, entangling: str = "full") -> None:
    """entangling='none' replaces CRZ/CRX (src->sink) by uncontrolled RZ/RX on the sink."""
    src, sink = wires
    if entangling == "none":
        if gate != "crx_crz":
            raise NotImplementedError("entangling='none' is defined for crx_crz pooling only")
        qml.RZ(p[0], wires=sink)
        qml.RX(p[1], wires=sink)
        return
    if gate == "crx_crz":
        qml.CRZ(p[0], wires=[src, sink])
        qml.CRX(p[1], wires=[src, sink])
    elif gate == "cry":
        qml.CRY(p[0], wires=[src, sink])
        qml.CRY(p[1], wires=[src, sink])
    elif gate == "cr":
        qml.CRZ(p[0], wires=[src, sink])
    else:
        raise ValueError(f"unknown pooling_gate: {gate}")


def build_qnode(n_qubits: int, ops, readout: int, gates: list[str], pooling_gate: str,
                device: str, diff_method: str, measure: str = "expval",
                entangling: str = "full"):
    """measure='expval' -> <Z> on the read-out; 'state' -> pre-readout statevector;
    'all_z' -> <Z> on every qubit at the end of the circuit."""
    dev = qml.device(device, wires=n_qubits)
    pool_n = pool_n_params(pooling_gate)

    @qml.qnode(dev, interface="autograd", diff_method=diff_method)
    def circuit(x, q):
        encode(x, n_qubits, gates)
        i = 0
        for t, w in ops:
            if t == "conv":
                conv_block(q[i: i + CONV_PARAMS], w, entangling)
                i += CONV_PARAMS
            else:
                pool_block(q[i: i + pool_n], w, pooling_gate, entangling)
                i += pool_n
        if measure == "state":
            return qml.state()
        if measure == "all_z":
            return [qml.expval(qml.PauliZ(k)) for k in range(n_qubits)]
        return qml.expval(qml.PauliZ(readout))

    return circuit


# ── loss ─────────────────────────────────────────────────────────────────────
def weighted_loss(raw, y, w, loss: str, gamma: float):
    if loss == "mse":
        return pnp.mean(w * (raw - pnp.where(y == 1, 1.0, -1.0)) ** 2)
    p = pnp.clip(1.0 / (1.0 + pnp.exp(-raw)), 1e-7, 1.0 - 1e-7)
    if loss == "bce":
        return pnp.mean(w * -(y * pnp.log(p) + (1.0 - y) * pnp.log(1.0 - p)))
    if loss == "focal":
        p_t = pnp.where(y == 1, p, 1.0 - p)
        return pnp.mean(w * -((1.0 - p_t) ** gamma) * pnp.log(p_t))
    raise ValueError(f"unknown loss: {loss}")


# ── estimator ────────────────────────────────────────────────────────────────
class HQCNN:
    def __init__(self, hcfg: dict):
        self.cfg = dict(hcfg)
        if self.cfg["encoding"] != "angle":
            raise ValueError("v3 HQCNN supports angle encoding only")
        sim = self.cfg["simulator"]
        self.n_qubits = int(self.cfg["n_qubits"])
        self.topology = self.cfg.get("conv_topology", "tree")
        self.ops, self.readout = qcnn_plan(
            self.n_qubits, int(self.cfg["conv_layers"]), self.cfg["pooling_survivor"],
            self.topology,
        )
        self.n_q = n_quantum_params(self.ops, self.cfg["pooling_gate"])
        self.broadcast = bool(sim.get("broadcast", False))
        self.entangling = self.cfg.get("entangling", "full")
        self.circuit = build_qnode(
            self.n_qubits, self.ops, self.readout, list(self.cfg["reupload_gates"]),
            self.cfg["pooling_gate"], sim["device"], sim["diff_method"],
            entangling=self.entangling,
        )

    @property
    def n_params(self) -> int:
        return self.n_q + 2

    # angle scaling fitted on the training data passed to fit()
    def _fit_scaler(self, X: np.ndarray) -> None:
        self.lo_, self.hi_ = X.min(axis=0), X.max(axis=0)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        a, b = self.cfg["angle_range"]
        if self.cfg["angle_scaling"] == "arctan":
            return np.arctan(X) + np.pi / 2.0
        if self.cfg["angle_scaling"] != "minmax":
            raise ValueError(f"unknown angle_scaling: {self.cfg['angle_scaling']}")
        span = np.where(self.hi_ > self.lo_, self.hi_ - self.lo_, 1.0)
        Z = np.clip((X - self.lo_) / span, 0.0, 1.0)
        return a + Z * (b - a)

    def _z(self, Xs: np.ndarray, q) -> pnp.ndarray:
        if self.broadcast:
            return self.circuit(pnp.array(Xs.T, requires_grad=False), q)
        return pnp.stack([self.circuit(pnp.array(x, requires_grad=False), q) for x in Xs])

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "HQCNN":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._fit_scaler(X)
        Xs = self._scale(X)

        counts = np.bincount(y.astype(int), minlength=2)
        cw = {c: len(y) / (2.0 * counts[c]) for c in (0, 1)} \
            if self.cfg.get("class_weight") == "balanced" else {0: 1.0, 1: 1.0}
        w_all = np.where(y == 1, cw[1], cw[0])

        init_q = rng.uniform(-np.pi, np.pi, self.n_q)
        params = pnp.array(np.concatenate([init_q, [1.0, 0.0]]), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=float(self.cfg["lr"]))
        loss, gamma = self.cfg["loss"], float(self.cfg.get("focal_gamma", 2.0))
        bs, n = int(self.cfg["batch_size"]), len(y)
        n_q = self.n_q

        self.loss_history_ = []
        for _ in range(int(self.cfg["n_epochs"])):
            order = rng.permutation(n)
            tot = 0.0
            for s in range(0, n, bs):
                bi = order[s: s + bs]
                Xb = Xs[bi]
                yb = pnp.array(y[bi], requires_grad=False)
                wb = pnp.array(w_all[bi], requires_grad=False)

                def cost(p, Xb=Xb, yb=yb, wb=wb):
                    raw = p[-2] * self._z(Xb, p[:n_q]) + p[-1]
                    return weighted_loss(raw, yb, wb, loss, gamma)

                params, c = opt.step_and_cost(cost, params)
                tot += float(c) * len(bi)
            self.loss_history_.append(tot / n)
        self.params_ = np.asarray(params, dtype=np.float64)
        return self

    def decision_function(self, X: np.ndarray, chunk: int = 256) -> np.ndarray:
        Xs = self._scale(np.asarray(X, dtype=np.float64))
        q = pnp.array(self.params_[: self.n_q], requires_grad=False)
        z = np.concatenate([
            np.asarray(self._z(Xs[i: i + chunk], q), dtype=np.float64).reshape(-1)
            for i in range(0, len(Xs), chunk)
        ])
        return self.params_[-2] * z + self.params_[-1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.decision_function(X)))
