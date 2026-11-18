# Why the tree HQCNN is classically simulable in O(n)

This note explains why the champion HQCNN can be simulated classically at a cost that
grows linearly with the number of qubits. It also lists which design changes break the
argument and which don't. The code is `v3core/twin.py`; the evidence is
`scripts/twin_equivalence.py`, `scripts/twin_trained_check.py`, `scripts/twin_scaling.py`
and `scripts/brickwork_analysis.py`. Numbers are in [Evidence](#evidence).

## 1. The circuit

For n qubits (n = 8 in the champion):

1. **Encoding.** Each qubit gets only single-qubit rotations: RY(x_q), then optionally
   RZ(x_{q+n}), RX(…), … when there are more features than qubits. The input state is a
   **product state** ρ⁽⁰⁾ = ρ_0 ⊗ ρ_1 ⊗ … ⊗ ρ_{n−1}.
2. **Level ℓ** acts on the active register A_ℓ (A_0 = all qubits):
   - the active qubits are split into disjoint neighbouring pairs (a, b);
   - each pair gets `conv_layers` conv blocks U_conv (RY⊗RY, CNOT, RY⊗RY, CNOT, RZ⊗RZ)
     and then the pool U_pool (CRZ, CRX from a to b);
   - the survivor b goes on to A_{ℓ+1}; the other qubit a is **never acted on again**.
3. **Read-out.** ⟨Z⟩ on the last survivor r, then logit = α⟨Z⟩ + β.

## 2. The argument

**Claim.** Just before level ℓ, the reduced state of the active register is a product of
single-qubit density matrices: ρ_{A_ℓ} = ⊗_{s ∈ A_ℓ} ρ_s.

**Base case.** ℓ = 0: the encoded state is a product state by construction.

**Inductive step.** Level ℓ applies V_ℓ = ⊗_{(a,b)} U_{ab}, a tensor product of unitaries on
disjoint pairs. Because the input to the level is a product over qubits, the output is a
product over pairs:

    V_ℓ (⊗_s ρ_s) V_ℓ† = ⊗_{(a,b)} U_{ab} (ρ_a ⊗ ρ_b) U_{ab}†.

Each pair's two-qubit state can be entangled internally, but different pairs are not.
Tracing out the discarded qubit a of every pair then leaves a product over survivors:
ρ_{A_{ℓ+1}} = ⊗_b ρ'_b, with ρ'_b = Tr_a[U_{ab}(ρ_a ⊗ ρ_b)U_{ab}†].

**Why tracing out is allowed.** A discarded qubit a is not acted on by any later gate
and not measured. For any later unitary W on the other qubits and any observable O on
them, Tr[(I_a ⊗ O)(I_a ⊗ W) ρ (I_a ⊗ W)†] = Tr[O W Tr_a(ρ) W†]. So ⟨Z_r⟩ depends only on
the reduced state of the survivors, and each qubit can be traced out right after its
pool.

**Result.** ⟨Z_r⟩ = Tr[Z ρ_r], where ρ_r comes out of a **binary tree of node maps**

    ρ_b' = Φ_{ab}(ρ_a, ρ_b) = Tr_a[U_{ab} (ρ_a ⊗ ρ_b) U_{ab}†],

each needing only a 4×4 matrix. This is exactly what `twin_expval` computes.

## 3. Cost

| | Classical twin | Statevector (PennyLane) |
|---|---|---|
| Memory | O(n) 2×2 matrices (O(1) per pair in flight) | O(2ⁿ) amplitudes |
| Time per forward | O(n · L) 4×4 products (L = conv layers) | O(#gates · 2ⁿ) = O(n L 2ⁿ) |
| Gradient | same autodiff pass, O(n L) | adjoint/backprop, O(n L 2ⁿ) |

The tree has n − 1 internal nodes (pools) and L conv blocks per pair, so the twin does
(L + 1)(n − 1) small conjugations. Nothing in it grows exponentially.

## 4. The equivalent classical model

In Bloch coordinates each node map is **bilinear**. Write ρ = ½(I + r·σ) and
r̃ = (1, r) ∈ ℝ⁴. Then r̃'_b = M_{ab}(r̃_a ⊗ r̃_b) for a fixed real 4 × 16 matrix M_{ab},
which is set by the node's circuit parameters (6L + 2 of them; 20 for L = 3).
Encoding maps each feature to a point on the Bloch sphere; RY(x) gives
r = (sin x, 0, cos x).

So the HQCNN is exactly a **classical, depth-log₂(n) binary tree of bilinear maps on
3-dimensional states**, followed by an affine map and a sigmoid. Equivalently, ⟨Z⟩ is a
multilinear polynomial in the n Bloch vectors whose coefficient tensor is a tree tensor
network with bond dimension 4 (the Pauli basis of one qubit). The quantum circuit only
restricts which bilinear maps can occur (those that come from 4×4 unitaries); it adds no
computational power beyond this classical model. The Phase 2C baseline
`classical_tree_net` tests the same topology with unconstrained nonlinear nodes.

**Checked numerically** (`scripts/verify_bilinear.py`, `results/phase2/bilinear_check.json`):
- For 200 random node unitaries × 20 random mixed-state pairs, the explicit tensor
  M_ijk = ¼ Tr[(I⊗σ_i) U (σ_j⊗σ_k) U†] reproduces the node output to 5.6 × 10⁻¹⁶.
- M is real (|Im M| ≤ 5.6 × 10⁻¹⁷) and trace-preserving (|M_0jk − δ_j0 δ_k0| ≤ 1.1 × 10⁻¹⁵).
- Over 1,000 random draws, the whole-tree Bloch recursion equals `twin_expval` to
  6.7 × 10⁻¹⁶.
- The RY encoding gives r = (sin x, 0, cos x) to 1.7 × 10⁻¹⁶.

The Phase 2C arm `relaxed_bilinear_tree` keeps this bilinear node form but drops the
unitarity constraint.

## 5. Which changes break the argument, and which don't

The argument needs three things: (i) a product input state, (ii) gates within a level
confined to disjoint pairs whose members then merge, and (iii) no gate or observable
touching a discarded qubit later.

| Change | Breaks it? | Why |
|---|---|---|
| **Brickwork / offset conv layer** (conv on (a₁,a₂), (a₃,a₄), … before pooling) | **Yes** | Couples qubits in different pairs, so the survivors are no longer in a product state. The twin would need joint states of growing clusters. `brickwork_analysis.py` shows the product-state shortcut fails: its error is about as large as the spread of ⟨Z⟩. |
| **Mid-tree interactions**: gates between different subtrees before they merge, gates on already-discarded qubits, feedback from lower levels | **Yes** | Condition (ii) or (iii) fails; the causal cone of the read-out is no longer a tree. |
| **Non-local read-out** (e.g. ⟨Z_a Z_b⟩ across subtrees, or measuring discarded qubits) | **Yes** | The observable spans several branches. |
| **Amplitude encoding** | **Yes, for this argument** | The input is not a product state (condition (i)). But amplitude-encoding d features uses log₂ d qubits, so a statevector simulation costs O(d · #gates): polynomial in the input size, which is not a route to advantage. |
| **Entangling feature maps** (e.g. ZZ feature map across all qubits) | **Yes** | Condition (i) fails. |
| More conv layers per pair, other pair unitaries, other pooling gates | No | Still disjoint pairs; only the 4×4 node unitary changes. |
| Data re-uploading onto survivors between levels | No | Single-qubit gates keep the product form. |
| Local noise (single-qubit or within-pair channels), finite shots | No | The twin already propagates density matrices; Kraus operators on a pair are one more 4×4 map. Shot noise is sampling from the exact ⟨Z⟩. |
| Mid-circuit measurement of discarded qubits with classical feed-forward to their sink | No | It is a local channel on the pair. |

Breaking the tree only removes **this** O(n) algorithm. It does not make the model hard to
simulate classically. Shallow brickwork circuits on 12–16 qubits are still easy for
statevector or tensor-network methods.

Related work puts this result in context:
- QCNNs have no barren plateaus: gradient variance decays only polynomially, because each
  parameter's causal cone is small [1].
- Cerezo et al. argue that the structure giving provable absence of barren plateaus
  usually also makes the loss classically simulable, or estimable with a classical
  surrogate after limited quantum data collection [2].
- Bermejo et al. show that standard QCNNs are *effectively* classically simulable. At
  random initialisation they act only on low-bodyness measurements of their input, and
  on common benchmark datasets a classical surrogate (Pauli propagation, tensor networks,
  classical shadows) matches or beats them [3].

Our result is narrower but stronger for this particular model: because the tree acts on
a product-state angle encoding, the simulation is *exact* and O(n) for every parameter
setting, including the trained ones. No low-bodyness truncation or surrogate training is
needed.

**References** (checked against arXiv and the publisher on 2026-09-23)

1. A. Pesah, M. Cerezo, S. Wang, T. Volkoff, A. T. Sornborger, P. J. Coles, "Absence of
   barren plateaus in quantum convolutional neural networks," *Phys. Rev. X* **11**,
   041011 (2021). doi:10.1103/PhysRevX.11.041011. arXiv:2011.02966.
2. M. Cerezo, M. Larocca, D. García-Martín, N. L. Diaz, P. Braccia, E. Fontana,
   M. S. Rudolph, P. Bermejo, A. Ijaz, S. Thanasilp, E. R. Anschuetz, Z. Holmes, "Does
   provable absence of barren plateaus imply classical simulability?" *Nat. Commun.*
   **16**, 7907 (2025). doi:10.1038/s41467-025-63099-6. arXiv:2312.09121.
3. P. Bermejo, P. Braccia, M. S. Rudolph, Z. Holmes, L. Cincio, M. Cerezo, "Quantum
   convolutional neural networks are effectively classically simulable," *PRX Quantum*
   **7**, 020304 (2026). doi:10.1103/8qt9-72ts. arXiv:2408.12739.

## 6. What this means for the paper

- The champion HQCNN cannot provide a computational advantage: an exact classical model
  with O(n) cost exists and reproduces its outputs to machine precision (Section 7).
- Any performance difference against classical baselines is about **inductive bias**
  (a bilinear tree over Bloch-sphere-encoded features). It is not about quantum
  resources. Phase 1 found no
  performance advantage either.
- Scaling to more qubits is free classically. On hardware, the circuit is shallow
  (depth 53 at n = 8). Its first-level two-qubit gates act on nearest neighbours, but later
  levels couple qubits that are further apart. On a linear nearest-neighbour device this
  needs S(n) = n/2 (log2 n - 2) + 1 SWAPs: 70 to 85 CNOTs at n = 8, and an overhead
  of 1.90 at 256 qubits (paper Section VIII-E; `results/phase3/routing/linear_routing.csv`).
  So it is NISQ-friendly. That is not the same as useful.

## 7. Evidence

All numbers come from `results/phase2/`. Machine: AMD Ryzen 7 6800H (8 cores), 15 GB RAM,
PennyLane 0.38.

**Exact equivalence on random inputs** (`twin/equivalence.csv`). 1,600 random
(input ∈ [0, π], parameters ∈ U(−π, π)) draws over 7 configurations: 4–16 qubits, 1–3
conv layers, 4–40 features (including RZ/RX re-upload rounds and an odd-width level at
6 qubits). The twin matches PennyLane `default.qubit`:

| | max \|twin − statevector\| |
|---|---|
| ⟨Z⟩ | 1.1 × 10⁻¹⁴ |
| ∂⟨Z⟩/∂θ (all circuit parameters) | 6.8 × 10⁻¹⁵ |

The champion configuration alone (8 qubits, 3 layers, 12 features, 1,000 draws):
2.0 × 10⁻¹⁵ and 1.9 × 10⁻¹⁵.

**Exact equivalence on the trained Phase 1 models** (`twin/trained_check.csv`). All 25
Phase 1 folds were refitted from their seeds. The refit reproduces the stored Phase 1
OOF probabilities to 5.0 × 10⁻¹¹, which is the precision of the 10-digit CSV. Evaluating
the twin with those trained weights gives the same 1,635 OOF probabilities: within
1.3 × 10⁻¹⁵ of the refit and 5.0 × 10⁻¹¹ of the stored Phase 1 values. **The trained
HQCNN and its classical twin are the same function.**

**Cost** (`twin/scaling.csv`, `figures/twin_scaling.pdf`). One forward pass, 3 conv
layers, one feature per qubit:

| n | twin | lightning.qubit | default.qubit |
|---|---|---|---|
| 8 | 0.042 s | 0.020 s | 0.070 s |
| 16 | 0.088 s | 0.088 s | 0.65 s |
| 24 | 0.14 s | 29 s | 151 s |
| 26 | 0.15 s | 146 s | stopped (projected > 300 s) |
| 28 | 0.16 s | stopped (projected > 300 s) | — |
| 256 | 1.6 s | — | — |

The twin grows linearly, at about 6 ms per qubit (Python-level; a vectorised
implementation would be far faster). The statevector simulators grow by about 2× per
added qubit and would also run out of memory near 30 qubits on this 15 GB machine. Below
about 16 qubits the C++ simulator is faster only because of the twin's Python overhead.

**The shortcut fails once the tree is broken** (`brickwork/twin_mismatch.csv`, 200
draws per size). Applying the twin's product-state propagation to the brickwork circuit:

| n | tree: max \|err\| | brickwork: mean \|err\| (max) | spread of ⟨Z⟩ (SD) |
|---|---|---|---|
| 4 | 1.4e-15 | 0.133 (0.70) | 0.28 |
| 8 | 1.4e-15 | 0.096 (0.58) | 0.16 |
| 12 | 3.5e-15 | 0.073 (0.38) | 0.11 |

The brickwork error is about as large as the signal. Dropping correlations between pairs
is not a small approximation.

**Entanglement** (`brickwork/entropy.csv`, `figures/entanglement_entropy.pdf`). Von
Neumann entropy (bits) of the pre-read-out state across two bipartitions, computed on the
**same** states. The contiguous cut is wires 0…n/2−1 vs the rest; the interleaved cut is
even vs odd wires, which splits every level-0 pair. States are 200 random
initialisations with real PCA inputs ("rand"), or the 66 test-fold inputs at seed 0 /
fold 0 trained weights ("trained").

| n | Tree contiguous (rand / trained) | Tree interleaved | Brick contiguous | Brick interleaved | Haar (Page) | Max |
|---|---|---|---|---|---|---|
| 4 | 0.87 / 0.96 | 0.92 / 1.28 | 1.11 / 1.45 | 1.11 / 1.35 | 1.28 | 2 |
| 8 | 1.21 / 1.22 | 1.87 / 1.37 | 2.11 / 1.92 | 2.40 / 2.17 | 3.28 | 4 |
| 12 | 1.31 / 1.13 | 2.95 / 2.65 | 2.69 / 2.50 | 3.66 / 2.90 | 5.28 | 6 |

How to read this:
- Across the contiguous cut, which follows a subtree boundary, the tree's entanglement
  stays around 1–1.5 bits for all n, because only the top of the tree crosses it.
- Across the interleaved cut, which cuts every level-0 pair, the tree's entanglement
  **grows with n** (0.9 → 2.9 bits at 4 → 12 qubits), still below the Haar value.
- So the tree is not a low-entanglement circuit, and its entanglement depends strongly on
  the cut. Brickwork is higher across both cuts. Training lowers entanglement slightly in
  most cases.
- **Entanglement is not what decides simulability here.** The tree is exactly O(n)
  simulable even though it creates about 3 bits of entanglement across the interleaved
  cut at 12 qubits. What makes it simulable is *where* that entanglement sits: inside
  pairs that are traced out before they could interact with other branches. This is why
  the entanglement ablation in step C (dephased control) is the relevant test for the
  role of entanglement, not these entropy values.
