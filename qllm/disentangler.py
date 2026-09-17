"""Tensor-network disentanglers: quantum circuits that make a layer compressible.

Implements the encoding of *Quantum Large Language Models via Tensor Network
Disentanglers* (Aizpurua et al.): a weight matrix ``W`` is rewritten as

    W  ~=  U  MPO_new  V^T ,

where ``U`` and ``V`` are **variational quantum circuits** (brickwalls of
few-qubit real-orthogonal gates) acting on the output and input indices, and
``MPO_new`` is the residual tensor network that stays classical. The circuits
absorb entanglement out of the operator, so ``MPO_new`` tolerates a much smaller
bond dimension than the MPO of ``W`` itself -- the paper truncates all the way to
bond dimension one at a perplexity cost below 0.3%.

Index embedding
---------------
A circuit acts on qubits, so each dimension is embedded into the smallest power
of two that holds it: ``d -> 2**ceil(log2 d)`` (the paper's (576, 192) -> (10, 8)
qubits). ``W`` is zero-padded into that box; the padding is exact (the extra rows
and columns never carry signal) but it *does* enlarge the residual MPO, which is
why :func:`disentangle` also supports ``depth=0`` -- the circuits are then the
identity and the run measures the padding overhead alone.

Quantum circuits (PennyLane)
----------------------------
The disentanglers are genuine **PennyLane** circuits: each few-qubit gate is a
``qml.QubitUnitary`` placed on its brickwall wires, and the operator a circuit
applies is realized by ``qml.matrix`` (see :func:`circuit_unitary`). Every
full-circuit application here -- forming ``MPO_new = U^T W_pad V`` and rebuilding
the layer in :func:`hybrid_weight` -- runs through that PennyLane matrix, and the
gate list can be handed to hardware/transpilation via :func:`circuit_ops`.

Training (the paper's explicit disentangling algorithm)
-------------------------------------------------------
The parameters are the gate unitaries, trained by the environment / SVD sweep of
Appendix A -- *not* by gradient descent, parameter-shift, or a device. The
objective is the disentangling accuracy in its error form,

    minimize_{U, V}  || U^T W V  -  T_chi(U^T W V) ||_F ,

optimized by alternation:

1. **Target step** -- ``M <- T_chi(U^T W V)``: the truncation is the best bond-chi
   approximation of the current disentangled operator.
2. **Gate sweeps** -- with ``M`` fixed, maximize the overlap ``<W, U M V^T>``.
   The overlap is *linear* in every individual gate, so for gate ``g`` it reads
   ``<g, E_g>`` with ``E_g`` the environment tensor (the network with ``g``
   removed). The paper's update sets the gate to the polar factor of the
   environment, ``E_g = A S B -> g <- A B`` (Eq. A4), the exact maximizer of
   ``<g, E_g>`` over the orthogonal group.

Because ``U`` and ``V`` are orthogonal, ``||W - U M V^T|| = ||U^T W V - M||``, so
step 1 and step 2 both decrease the *same* error and the iteration is monotone --
:func:`disentangle` asserts this and reports the retained weight per sweep. The
environment sweep applies one gate at a time to the operator (the quantum gate's
local action on two qubits); the closed-form SVD update makes each gate optimal
given the rest.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pennylane as qml
import torch

from .compactifai import (
    MPOPlan,
    _deinterleave_perm,
    _interleave_perm,
    mpo_param_count,
)
from .qubit_mpo import make_plan, plan_bond_entropy, plan_compress


# --------------------------------------------------------------------------- #
# Qubit embedding
# --------------------------------------------------------------------------- #
def n_qubits_for(dim: int) -> int:
    """Qubits needed to embed a dimension: ``ceil(log2 dim)`` (0 for dim <= 1)."""
    return max(0, math.ceil(math.log2(max(1, int(dim)))))


def pad_to_qubits(weight: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Zero-pad ``W`` into the smallest power-of-two box, return (W_pad, n_out, n_in)."""
    d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
    n_out, n_in = n_qubits_for(d_out), n_qubits_for(d_in)
    padded = torch.zeros(1 << n_out, 1 << n_in, dtype=torch.float32)
    padded[:d_out, :d_in] = weight.detach().to(torch.float32).cpu()
    return padded, n_out, n_in


# --------------------------------------------------------------------------- #
# Circuit ansatz
# --------------------------------------------------------------------------- #
@dataclass
class Gate:
    """One few-qubit gate: a real orthogonal matrix on qubits ``[start, start+k)``.

    Realized as a ``qml.QubitUnitary`` on :attr:`wires`; ``matrix`` is the current
    unitary, the trainable parameter the environment-SVD sweep updates in place.
    """
    start: int
    k: int
    matrix: torch.Tensor

    @property
    def wires(self) -> list[int]:
        return list(range(self.start, self.start + self.k))

    def op(self) -> "qml.operation.Operator":
        """This gate as a PennyLane operation (for simulation / transpilation)."""
        return qml.QubitUnitary(
            self.matrix.detach().to(torch.float64).cpu().numpy(),
            wires=self.wires, unitary_check=False)


def gate_positions(n_qubits: int, gate_size: int, depth: int) -> list[tuple[int, int]]:
    """``(start, k)`` of every gate of a brickwall, in application order.

    Layers alternate between offset 0 and offset ``k // 2`` so that consecutive
    layers straddle each other's boundaries -- the standard brickwall that lets
    correlations spread across the whole register. A gate as wide as the register
    (``k >= n``) already covers everything and stacking more of them only
    multiplies orthogonal matrices together, so that case collapses to a single
    gate (the paper's "10qU, 8qV, L=1" configuration).
    """
    if n_qubits <= 0 or depth <= 0 or gate_size <= 0:
        return []
    k = min(gate_size, n_qubits)
    if k >= n_qubits:
        return [(0, n_qubits)]                 # one gate spans the register
    positions = []
    for layer in range(depth):
        offset = 0 if layer % 2 == 0 else k // 2
        start = offset
        while start + k <= n_qubits:
            positions.append((start, k))
            start += k
    return positions


def gate_param_count(k: int, counting: str = "manifold") -> int:
    """Variational parameters carried by one real ``k``-qubit gate.

    ``manifold`` (default) counts the dimension of the orthogonal group,
    ``dim O(2^k) = 2^(k-1) (2^k - 1)`` -- the number of independent angles the
    gate actually has, and hence the number of quantum parameters to be stored /
    trained. ``entries`` counts the ``4^k`` matrix entries instead, which is the
    right figure if the gate is kept as a dense classical tensor.
    """
    dim = 1 << k
    if counting == "entries":
        return dim * dim
    return dim * (dim - 1) // 2


def circuit_param_count(n_qubits: int, gate_size: int, depth: int,
                        counting: str = "manifold") -> int:
    """Total variational parameters of a brickwall circuit -- the ``Q(D)`` term."""
    return sum(gate_param_count(k, counting)
               for _s, k in gate_positions(n_qubits, gate_size, depth))


def quantum_param_count(n_out_qubits: int, n_in_qubits: int, gate_size: int,
                        depth: int, counting: str = "manifold") -> int:
    """``Q(D)``: parameters of both disentangling circuits ``U`` and ``V``."""
    return (circuit_param_count(n_out_qubits, gate_size, depth, counting)
            + circuit_param_count(n_in_qubits, gate_size, depth, counting))


def _identity_gate(k: int) -> torch.Tensor:
    return torch.eye(1 << k, dtype=torch.float32)


def _random_gate(k: int, generator: torch.Generator | None) -> torch.Tensor:
    a = torch.randn(1 << k, 1 << k, dtype=torch.float32, generator=generator)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)   # Haar on O(2^k)


def build_circuit(n_qubits: int, gate_size: int, depth: int, init: str = "identity",
                  generator: torch.Generator | None = None) -> list[Gate]:
    """Brickwall of real orthogonal gates, initialised to identity or Haar-random.

    ``identity`` makes the circuit a no-op, so the hybrid layer starts exactly at
    the classical (padded) MPO and the first sweep can only improve on it -- the
    paper's "the hybrid model departs from the classical baseline".
    """
    gates = []
    for start, k in gate_positions(n_qubits, gate_size, depth):
        mat = _identity_gate(k) if init == "identity" else _random_gate(k, generator)
        gates.append(Gate(start=start, k=k, matrix=mat))
    return gates


# --------------------------------------------------------------------------- #
# Applying gates / circuits -- realized by PennyLane
# --------------------------------------------------------------------------- #
def circuit_ops(gates: list[Gate]) -> list["qml.operation.Operator"]:
    """The brickwall as a list of PennyLane operations (queue these in a QNode)."""
    return [g.op() for g in gates]


def circuit_unitary(gates: list[Gate], n_qubits: int) -> torch.Tensor:
    """The register unitary of the brickwall, composed by PennyLane.

    Each :class:`Gate` becomes a ``qml.QubitUnitary`` on its wires and ``qml.matrix``
    contracts them into the ``2^n x 2^n`` operator the circuit applies -- wire 0 is
    the most significant bit, matching the qubit embedding. This is how the
    disentangler circuits are *run*: the full-circuit applications below and the
    reconstruction in :func:`hybrid_weight` all go through this matrix.

    The gate matrices are passed as torch tensors (PennyLane's torch interface),
    so ``qml.matrix`` keeps the autograd graph -- when the gates depend on
    trainable angles the returned unitary is differentiable, which is how the
    gradient scheme optimizes the circuit directly through PennyLane.
    """
    if not gates:
        return torch.eye(1 << n_qubits, dtype=torch.float32)

    def _qfunc():
        # Construct each gate inside the circuit so PennyLane queues it in order,
        # keeping the torch tensor (do not detach) so gradients flow through.
        for g in gates:
            qml.QubitUnitary(g.matrix.to(torch.float64), wires=g.wires,
                             unitary_check=False)

    u = qml.matrix(_qfunc, wire_order=list(range(n_qubits)))()
    if not torch.is_tensor(u):                       # numpy fallback (no torch inputs)
        u = torch.as_tensor(np.asarray(u))
    if torch.is_complex(u):
        u = u.real
    return u.to(torch.float32)


def apply_gate(tensor: torch.Tensor, gate: torch.Tensor, start: int, k: int,
               n_qubits: int) -> torch.Tensor:
    """Apply one ``k``-qubit gate's unitary to the row index of ``tensor``.

    The local action of the quantum gate: qubit ``0`` is the most significant bit,
    so a contiguous block of qubits is a contiguous stride of the flat index and
    the gate is one reshape into ``(left, 2^k, rest)`` and a batched matmul -- no
    copy of the big tensor and, crucially, no ``2^n x 2^n`` register unitary.
    Differentiable in the gate matrix.
    """
    left = 1 << start
    mid = 1 << k
    view = tensor.reshape(left, mid, -1)
    out = torch.matmul(gate, view)
    return out.reshape(1 << n_qubits, -1)


def apply_circuit(gates: list[Gate], tensor: torch.Tensor, n_qubits: int,
                  transpose: bool = False) -> torch.Tensor:
    """``C X`` (or ``C^T X``): the circuit applied to the row index, gate by gate.

    Gates are applied one at a time (the way a state-vector simulator runs a
    circuit -- PennyLane's own ``default.qubit`` never builds the full register
    unitary either), so cost is ``O(#gates)`` local contractions, not the
    exponential ``qml.matrix`` composition. Differentiable in the gate matrices,
    so this carries the gradient scheme's autograd too. Use :func:`circuit_unitary`
    when the explicit ``2^n x 2^n`` operator is actually wanted (export, small
    circuits, verification).
    """
    out = tensor
    for g in (reversed(gates) if transpose else gates):
        mat = g.matrix.T if transpose else g.matrix
        out = apply_gate(out, mat, g.start, g.k, n_qubits)
    return out


def apply_right(gates: list[Gate], tensor: torch.Tensor, n_qubits: int,
                transpose: bool = False) -> torch.Tensor:
    """``X C^T`` (or ``X C``): the circuit acting on the *column* index of ``X``."""
    moved = apply_circuit(gates, tensor.transpose(0, 1).contiguous(), n_qubits,
                          transpose=transpose)
    return moved.transpose(0, 1).contiguous()


def _environment(left: torch.Tensor, right: torch.Tensor, start: int, k: int) -> torch.Tensor:
    """``E[a,b] = sum_{l,c} L[l,a,c] R[l,b,c]`` -- the gradient w.r.t. the gate."""
    nl = 1 << start
    mid = 1 << k
    lr = left.reshape(nl, mid, -1)
    rr = right.reshape(nl, mid, -1)
    return torch.einsum("lac,lbc->ab", lr, rr)


def _polar(matrix: torch.Tensor) -> torch.Tensor:
    """Paper's Eq. (A4) gate update: the polar factor of the environment.

    For ``E = A S B`` (SVD) the maximizer of ``<g, E>`` over the orthogonal group
    is ``A B`` -- the paper writes the conjugate ``B^dagger A^dagger`` for the
    gate as it enters ``U^dagger``; here the gate enters as ``U`` so the factor is
    ``A B``. This is a closed-form optimal step, not a gradient update.
    """
    p, _s, q = torch.linalg.svd(matrix, full_matrices=False)
    return p @ q


def sweep_circuit(gates: list[Gate], target: torch.Tensor, source: torch.Tensor,
                  n_qubits: int) -> float:
    """One environment sweep maximizing ``<target, C source>`` over the gates.

    Updates ``gates`` in place (back to front) and returns the overlap reached.
    The running partial products need no storage: ``R`` walks *down* by applying
    the old gate transposed (the gates are orthogonal, so that inverts them), and
    ``L`` walks down by applying each freshly updated gate transposed.
    """
    if not gates:
        return float((target * source).sum())
    right = apply_circuit(gates, source, n_qubits)     # R = C source
    left = target                                     # L = target
    for gate in reversed(gates):
        old = gate.matrix
        right = apply_gate(right, old.T, gate.start, gate.k, n_qubits)
        env = _environment(left, right, gate.start, gate.k)
        gate.matrix = _polar(env)
        left = apply_gate(left, gate.matrix.T, gate.start, gate.k, n_qubits)
    return float((left * source).sum())               # <C^T target, source>


# --------------------------------------------------------------------------- #
# Disentangling a weight matrix
# --------------------------------------------------------------------------- #
@dataclass
class DisentangleResult:
    """A disentangled layer: two circuits plus the residual operator's MPO plan."""
    u_gates: list[Gate]
    v_gates: list[Gate]
    n_out_qubits: int
    n_in_qubits: int
    gate_size: int
    depth: int
    shape: tuple[int, int]                 # original (d_out, d_in)
    padded_shape: tuple[int, int]
    plan: MPOPlan                          # MPO plan (cached SVD) of MPO_new
    operator: torch.Tensor                 # MPO_new = U^T W_pad V, dense
    quantum_params: int
    target_chi: int
    accuracy: float                        # paper Eq. (4), vs. the chi-target
    entropy: float                         # bond entropy of MPO_new (nats)
    retained: float                        # ||T_chi(U^T W V)|| / ||W||
    retained_classical: float              # ||T_chi(W_pad)|| / ||W|| (no circuits)
    sweeps_run: int
    seconds: float
    optimizer: str = "explicit"            # 'explicit' (env-SVD) or 'gradient'
    history: list[float] = field(default_factory=list)

    @property
    def max_chi(self) -> int:
        return self.plan.max_chi


def _interleaved(matrix: torch.Tensor, plan: MPOPlan) -> torch.Tensor:
    """Reshape into the matrix whose SVD is the MPO bond of a 2-site plan."""
    o, i = plan.out_dims, plan.in_dims
    return (matrix.reshape(o[0], o[1], i[0], i[1])
            .permute(0, 2, 1, 3)
            .reshape(o[0] * i[0], o[1] * i[1]))


def bond_entropy(matrix: torch.Tensor, plan: MPOPlan) -> float:
    """Mean von Neumann entropy of the MPO bonds of ``matrix`` (nats)."""
    if plan.n_sites != 2:
        return float("nan")
    s = torch.linalg.svdvals(_interleaved(matrix, plan))
    p = (s ** 2)
    total = float(p.sum())
    if total <= 0:
        return 0.0
    p = p / total
    p = p[p > 1e-30]
    return float(-(p * p.log()).sum())


# --------------------------------------------------------------------------- #
# Gradient training: optimize the gate angles directly (the "implicit" scheme)
# --------------------------------------------------------------------------- #
# A real orthogonal ``k``-qubit gate is parameterized by its Lie-algebra angles:
# ``g(theta) = expm(A(theta))`` with ``A`` skew-symmetric, ``theta`` filling its
# strict upper triangle. That is ``dim so(2^k) = 2^(k-1)(2^k-1)`` angles per gate
# -- exactly ``quantum_param_count`` -- and ``g`` is orthogonal for any theta, so
# unconstrained gradient descent on ``theta`` stays on the gate manifold.
def _skew(theta: torch.Tensor, dim: int) -> torch.Tensor:
    idx = torch.triu_indices(dim, dim, offset=1)
    a = theta.new_zeros(dim, dim)
    a[idx[0], idx[1]] = theta
    return a - a.T


def _gate_from_angles(theta: torch.Tensor, k: int) -> torch.Tensor:
    """Orthogonal ``2^k x 2^k`` gate ``expm(skew(theta))`` -- differentiable in theta."""
    return torch.linalg.matrix_exp(_skew(theta, 1 << k))


def _gates_from_angles(thetas: list[torch.Tensor],
                       positions: list[tuple[int, int]],
                       bases: list[torch.Tensor] | None = None) -> list[Gate]:
    """Build :class:`Gate` objects (differentiable matrices) from angle tensors.

    Without ``bases`` each gate is ``expm(skew(theta))`` (identity at theta=0).
    With ``bases`` each gate is ``base @ expm(skew(theta))`` -- a differentiable
    refinement multiplying a fixed orthogonal base gate, so ``theta=0`` reproduces
    ``base`` exactly. This is how the gradient stage warm-starts from the explicit
    sweep's gates (the same parameterization :class:`~qllm.hybrid_heal.HybridAdapter`
    uses to heal circuits).
    """
    if bases is None:
        return [Gate(s, k, _gate_from_angles(t, k))
                for t, (s, k) in zip(thetas, positions)]
    return [Gate(s, k, b @ _gate_from_angles(t, k))
            for t, (s, k), b in zip(thetas, positions, bases)]


def disentangle_loss(current: torch.Tensor, out_dims: list[int], in_dims: list[int],
                     n_sites: int, chi: int) -> torch.Tensor:
    """Differentiable disentangling objective: mean discarded weight over MPO bonds.

    Interleaves ``current`` into the MPO chain of the tensorization and, at every
    bond cut, measures the Frobenius weight a rank-``chi`` truncation discards
    (``1 - sum(top-chi sigma^2)/sum(sigma^2)``, from ``svdvals`` so it is
    differentiable). Zero means every bond already fits in ``chi`` -- the
    product-operator target of the paper. This is the loss gradient descent
    minimizes; it matches the error the explicit sweep drives down.
    """
    perm = _interleave_perm(n_sites)
    chain = current.reshape(*out_dims, *in_dims).permute(*perm).contiguous()
    site = [out_dims[k] * in_dims[k] for k in range(n_sites)]
    losses = []
    for cut in range(1, n_sites):
        left = math.prod(site[:cut])
        mat = chain.reshape(left, -1)
        s2 = torch.linalg.svdvals(mat) ** 2
        total = s2.sum().clamp_min(1e-30)
        losses.append(1.0 - s2[:chi].sum() / total)
    if not losses:
        return current.new_zeros(())
    return torch.stack(losses).mean()


GRADIENT_OBJECTIVES = ("disentangle-loss", "relative-error")


def _truncated_dense(current: torch.Tensor, out_dims: list[int], in_dims: list[int],
                     n_sites: int, chi: int) -> torch.Tensor:
    """Differentiable ``T_chi(current)``: the rank-``chi`` MPO truncation, contracted.

    A grad-enabled copy of :func:`~qllm.compactifai.mpo_factors` followed by
    :func:`~qllm.compactifai.contract_factors` (both ``@torch.no_grad``): the same
    left-to-right sequential truncated SVD the evaluation path uses, so this
    reproduces ``plan_compress(current, plan, chi)`` exactly while keeping the
    autograd graph back to ``current`` (and thence the gate angles).
    """
    tensor = current.reshape(*out_dims, *in_dims).permute(*_interleave_perm(n_sites))
    tensor = tensor.contiguous()
    factors: list[torch.Tensor] = []
    bond_left = 1
    mat = tensor.reshape(out_dims[0] * in_dims[0], -1)
    for k in range(n_sites - 1):
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r = max(1, min(chi, int(s.numel())))
        factors.append(u[:, :r].reshape(bond_left, out_dims[k], in_dims[k], r))
        mat = s[:r].unsqueeze(1) * vh[:r]
        bond_left = r
        if k + 1 < n_sites - 1:
            mat = mat.reshape(bond_left * out_dims[k + 1] * in_dims[k + 1], -1)
    factors.append(mat.reshape(bond_left, out_dims[-1], in_dims[-1], 1))
    dense = factors[0]
    for k in range(1, n_sites):
        dense = torch.tensordot(dense, factors[k], dims=([dense.ndim - 1], [0]))
    dense = dense.squeeze(0).squeeze(-1).permute(*_deinterleave_perm(n_sites)).contiguous()
    return dense.reshape(math.prod(out_dims), math.prod(in_dims))


def relative_error_loss(current: torch.Tensor, out_dims: list[int], in_dims: list[int],
                        n_sites: int, chi: int) -> torch.Tensor:
    """Differentiable reconstruction error ``||current - T_chi(current)|| / ||current||``.

    Because ``U``/``V`` are orthogonal, ``||W_pad - U T_chi(U^T W_pad V) V^T||``
    equals ``||current - T_chi(current)||`` with ``current = U^T W_pad V``; dividing
    by ``||current|| = ||W_pad||`` gives the padded relative error, a tight upper
    bound on the reported (cropped) one. Minimizing it directly targets the
    compressed layer's error at bond ``chi`` -- unlike :func:`disentangle_loss`,
    which minimizes the *mean per-bond* discarded weight, a proxy that can move the
    joint reconstruction error the wrong way on a multi-site MPO.
    """
    recon = _truncated_dense(current, out_dims, in_dims, n_sites, chi)
    denom = torch.linalg.norm(current).clamp_min(1e-30)
    return torch.linalg.norm(current - recon) / denom


def _train_gradient(padded, n_out, n_in, gate_size, depth, plan_pad, target_chi,
                    steps, lr, init, seed, log, base_u=None, base_v=None,
                    fast=False, objective="disentangle-loss"):
    """Optimize the gate angles by Adam to minimize the chosen ``objective``.

    Returns ``(u_gates, v_gates, steps_run, history)`` with the trained gates as
    :class:`Gate` objects (PennyLane ``QubitUnitary`` under the hood).

    ``objective`` picks the loss: ``disentangle-loss`` (default) minimizes the
    mean per-bond rank-``target_chi`` discarded weight (:func:`disentangle_loss`);
    ``relative-error`` minimizes the joint reconstruction error at bond
    ``target_chi`` (:func:`relative_error_loss`), i.e. the quantity the compressed
    layer actually incurs, evaluated at ``target_chi`` (pair with
    ``disentangle_target_per_chi`` to target each served chi').

    ``base_u`` / ``base_v`` warm-start the optimization from an existing circuit
    (e.g. the explicit sweep's result): each trainable gate becomes
    ``base @ expm(skew(theta))`` with ``theta`` initialised to zero, so step 0
    reproduces the base circuit exactly and Adam only *refines* it. Without them
    the gates are ``expm(skew(theta))`` from an identity (or Haar, ``init=random``)
    start, the from-scratch behaviour.

    ``fast`` forms the disentangled operator ``U^T W V`` with the pure-torch
    :func:`apply_circuit`/:func:`apply_right` local contractions (``O(#gates)``,
    no ``2^n x 2^n`` register unitary) instead of composing it through PennyLane's
    ``qml.matrix``. Both are differentiable and give the same operator; the torch
    path is several times faster per step and much lighter on memory for wide
    registers. The default keeps the PennyLane path (``qml.matrix``).
    """
    pos_u = gate_positions(n_out, gate_size, depth)
    pos_v = gate_positions(n_in, gate_size, depth)
    gen = torch.Generator().manual_seed(seed)
    refine = base_u is not None                      # warm-start from base gates
    bases_u = [g.matrix for g in base_u] if refine else None
    bases_v = [g.matrix for g in base_v] if refine else None

    def _init(positions):
        out = []
        for _s, k in positions:
            m = (1 << k) * ((1 << k) - 1) // 2
            if refine or init != "random":
                out.append(torch.zeros(m, requires_grad=True))  # start at base/identity
            else:
                out.append((0.1 * torch.randn(m, generator=gen)).requires_grad_(True))
        return out

    theta_u, theta_v = _init(pos_u), _init(pos_v)
    params = theta_u + theta_v
    history: list[float] = []
    if not params:                                   # depth 0: no gates to train
        return (base_u or []), (base_v or []), 0, history

    out_dims, in_dims, n_sites = plan_pad.out_dims, plan_pad.in_dims, plan_pad.n_sites
    opt = torch.optim.Adam(params, lr=lr)
    for step in range(max(1, steps)):
        opt.zero_grad()
        # Build the disentangled operator U^T W V from the current angles. The
        # gate matrices carry autograd either way; ``fast`` chooses how they are
        # contracted into ``current``.
        ug = _gates_from_angles(theta_u, pos_u, bases_u)
        vg = _gates_from_angles(theta_v, pos_v, bases_v)
        if fast:
            # Pure-torch local contractions -- O(#gates), no register unitary.
            current = apply_right(vg, apply_circuit(ug, padded, n_out, transpose=True),
                                  n_in, transpose=True)
        else:
            # THROUGH PennyLane (qml.matrix, torch interface) -- autograd flows
            # from the loss back to the angles via the PennyLane circuit itself.
            u = circuit_unitary(ug, n_out)
            v = circuit_unitary(vg, n_in)
            current = u.transpose(0, 1) @ padded @ v
        if objective == "relative-error":
            loss = relative_error_loss(current, out_dims, in_dims, n_sites, target_chi)
        else:
            loss = disentangle_loss(current, out_dims, in_dims, n_sites, target_chi)
        loss.backward()
        opt.step()
        history.append(1.0 - float(loss.detach()))
        if log and (step == 0 or (step + 1) % max(1, steps // 5) == 0):
            print(f"      gd step {step + 1:>4}/{steps}  loss={float(loss):.6f}")

    with torch.no_grad():
        u_gates = [Gate(g.start, g.k, g.matrix.detach())
                   for g in _gates_from_angles(theta_u, pos_u, bases_u)]
        v_gates = [Gate(g.start, g.k, g.matrix.detach())
                   for g in _gates_from_angles(theta_v, pos_v, bases_v)]
    return u_gates, v_gates, max(1, steps), history


def _build_result(weight, padded, u_gates, v_gates, n_out, n_in, gate_size, depth,
                  target_chi, plan_fn, param_counting, retained_classical, target_ref,
                  ref_norm, norm, sweeps_run, t0, history, optimizer) -> DisentangleResult:
    """Assemble a :class:`DisentangleResult` from trained gates (shared by both schemes)."""
    current = apply_right(v_gates,
                          apply_circuit(u_gates, padded, n_out, transpose=True),
                          n_in, transpose=True)
    plan = plan_fn(current)
    final_target, _ = plan_compress(current, plan, target_chi)
    retained = float(torch.linalg.norm(final_target)) / norm if norm else 0.0
    accuracy = (float((target_ref * current).sum()) / (norm * ref_norm)
                if norm and ref_norm else float("nan"))
    return DisentangleResult(
        u_gates=u_gates, v_gates=v_gates, n_out_qubits=n_out, n_in_qubits=n_in,
        gate_size=gate_size, depth=depth,
        shape=(int(weight.shape[0]), int(weight.shape[1])),
        padded_shape=(1 << n_out, 1 << n_in), plan=plan, operator=current,
        quantum_params=quantum_param_count(n_out, n_in, gate_size, depth,
                                           param_counting),
        target_chi=target_chi, accuracy=accuracy,
        entropy=plan_bond_entropy(current, plan), retained=retained,
        retained_classical=retained_classical, sweeps_run=sweeps_run,
        seconds=time.perf_counter() - t0, optimizer=optimizer,
        history=list(history) + [retained],
    )


def _disentangle_once(weight: torch.Tensor, gate_size: int, depth: int,
                      target_chi: int = 1, n_sites: int = 2, sweeps: int = 12,
                      tol: float = 1e-6, init: str = "identity", seed: int = 0,
                      param_counting: str = "manifold",
                      target_mode: str = "adaptive", tensorization: str = "balanced",
                      qubit_align: str = "msb", optimizer: str = "explicit",
                      gd_steps: int = 200, gd_lr: float = 0.05,
                      fast_gradient: bool = False,
                      gradient_objective: str = "disentangle-loss",
                      log: bool = False) -> DisentangleResult:
    """Disentangle ``W`` into ``U MPO_new V^T`` with brickwall circuits of depth ``D``.

    ``target_chi`` is the bond dimension the circuits are asked to squeeze the
    operator into (the paper uses 1). The returned plan caches the SVD of
    ``MPO_new``, so the residual can afterwards be truncated to *any* bond
    dimension for free -- which is how the hybrid sweep reuses one optimization
    across its whole chi' grid (the paper's Table I).

    ``target_mode`` selects the explicit objective:

    * ``adaptive`` (default) re-truncates the *current* disentangled operator at
      every iteration, which minimizes the quantity that actually decides the
      compressed layer's error, ``||U^T W V - T_chi(U^T W V)||``.
    * ``fixed`` freezes the target at ``T_chi(W_pad)``, the truncation of the
      original operator -- the literal objective behind the paper's Eq. (4)
      disentangling accuracy, kept for comparison with its figures.

    ``optimizer`` selects the training scheme:

    * ``explicit`` (default) -- the paper's environment / SVD sweep (Appendix A):
      each gate is set to the polar factor of its environment, a closed-form
      optimal step, alternating U/V sweeps to convergence.
    * ``gradient`` -- optimize the gate angles directly by Adam on
      :func:`disentangle_loss` (``gd_steps`` steps at ``gd_lr``). Same objective,
      an implicit iterative solver instead of the closed-form update.
    * ``explicit+gradient`` -- run the explicit sweep first, then refine its gates
      with the gradient optimizer warm-started from them (``gd_steps`` Adam steps,
      each gate ``base @ expm(skew(theta))`` with ``theta=0`` at start, so the
      polish begins exactly at the explicit result). The polish monotonically
      descends :func:`disentangle_loss`, its own objective. Note that loss is the
      *mean per-bond* rank-``target_chi`` discarded weight, which coincides with the
      reported ``retained``/``accuracy`` only for a single-bond (balanced) plan; for
      the multi-site qubit MPO it is a proxy, so the polish can move those headline
      metrics either way even as its own loss falls. Use with ``balanced``, or
      compare against ``explicit`` and keep the better, when ``retained`` is what you
      optimize.
    """
    t0 = time.perf_counter()
    padded, n_out, n_in = pad_to_qubits(weight)
    norm = float(torch.linalg.norm(padded))

    def _plan(op):
        return make_plan(op, tensorization, n_sites, svd_cache=True, align=qubit_align)

    # Classical reference: the same truncation with the circuits switched off.
    plan_pad = _plan(padded)
    trunc_pad, _ = plan_compress(padded, plan_pad, target_chi)
    retained_classical = float(torch.linalg.norm(trunc_pad)) / norm if norm else 0.0
    target_ref = trunc_pad                       # MPO_target of the paper's Eq. (4)
    ref_norm = float(torch.linalg.norm(target_ref))

    if optimizer == "gradient":
        u_gates, v_gates, steps_run, history = _train_gradient(
            padded, n_out, n_in, gate_size, depth, plan_pad, target_chi,
            gd_steps, gd_lr, init, seed, log, fast=fast_gradient,
            objective=gradient_objective)
        return _build_result(
            weight, padded, u_gates, v_gates, n_out, n_in, gate_size, depth,
            target_chi, _plan, param_counting, retained_classical, target_ref,
            ref_norm, norm, steps_run, t0, history, "gradient")

    # --- explicit environment / SVD sweep (paper Appendix A) ---
    generator = torch.Generator().manual_seed(seed)
    u_gates = build_circuit(n_out, gate_size, depth, init, generator)
    v_gates = build_circuit(n_in, gate_size, depth, init, generator)

    current = padded                             # U^T W V, starts at W (identity gates)
    history: list[float] = []
    score = -math.inf
    sweeps_run = 0
    for sweep in range(max(1, sweeps)):
        # Where the circuits stand now: the weight the truncation keeps.
        plan = _plan(current)
        truncated, _ = plan_compress(current, plan, target_chi)
        retained = float(torch.linalg.norm(truncated)) / norm if norm else 0.0
        history.append(retained)
        if not u_gates and not v_gates:
            break                                # depth 0: nothing to optimize
        # The paper's Eq. (4) freezes the target at the ORIGINAL operator's
        # truncation; the default re-truncates the current one, which is what
        # the compressed layer's error actually depends on.
        target = target_ref if target_mode == "fixed" else truncated
        # U sweep: maximize <W_pad, U M V^T>.
        sweep_circuit(u_gates, padded, apply_right(v_gates, target, n_in), n_out)
        # V sweep: the same objective transposed, <W_pad^T, V (U M)^T>.
        new_score = sweep_circuit(
            v_gates, padded.T.contiguous(),
            apply_circuit(u_gates, target, n_out).T.contiguous(), n_in)
        # Refresh the disentangled operator: MPO_new = U^T W_pad V.
        current = apply_right(v_gates, apply_circuit(u_gates, padded, n_out,
                                                     transpose=True),
                              n_in, transpose=True)
        sweeps_run = sweep + 1
        if log:
            print(f"      sweep {sweep + 1:>2}: retained {retained:.6f}  "
                  f"overlap {new_score:.6f}")
        if new_score - score < tol * max(1.0, abs(score)):
            break                                # the gate sweeps have converged
        score = new_score

    label = "explicit"
    # Optional implicit (gradient) polish, warm-started from the explicit gates:
    # Adam refines base @ expm(skew(theta)) with theta=0 at start (step 0 reproduces
    # the sweep) and monotonically descends disentangle_loss. That per-bond loss is
    # only a proxy for retained/accuracy on the multi-site qubit MPO, so the polish
    # can move those either way; on a single-bond (balanced) plan it aligns.
    if optimizer == "explicit+gradient" and (u_gates or v_gates):
        u_gates, v_gates, gd_run, gd_hist = _train_gradient(
            padded, n_out, n_in, gate_size, depth, plan_pad, target_chi,
            gd_steps, gd_lr, init, seed, log, base_u=u_gates, base_v=v_gates,
            fast=fast_gradient, objective=gradient_objective)
        history = history + gd_hist
        label = "explicit+gradient"

    return _build_result(
        weight, padded, u_gates, v_gates, n_out, n_in, gate_size, depth,
        target_chi, _plan, param_counting, retained_classical, target_ref,
        ref_norm, norm, sweeps_run, t0, history, label)


def hybrid_weight(result: DisentangleResult, chi: int) -> tuple[torch.Tensor, int]:
    """Rebuild the layer from the hybrid representation at bond dimension ``chi``.

    Returns ``(W', C(chi))``: the dense weight the compressed model would apply,
    ``W' = (U T_chi(MPO_new) V^T)`` cropped back to the original shape, and the
    classical parameter count of the stored MPO tensors. At ``chi`` equal to the
    plan's full rank this reproduces ``W`` exactly, because ``U`` and ``V`` are
    orthogonal and the padding carries no signal.
    """
    truncated, params = plan_compress(result.operator, result.plan, chi)
    dense = apply_right(result.v_gates,
                        apply_circuit(result.u_gates, truncated, result.n_out_qubits),
                        result.n_in_qubits)
    d_out, d_in = result.shape
    return dense[:d_out, :d_in].contiguous(), params


def classical_param_count(result: DisentangleResult, chi: int) -> int:
    """``C(chi')`` for the hybrid layer: the MPO stored over the *padded* indices."""
    return mpo_param_count(result.plan.out_dims, result.plan.in_dims, chi)


def disentangle(weight: torch.Tensor, gate_size: int, depth: int,
                target_chi: int = 1, n_sites: int = 2, sweeps: int = 12,
                tol: float = 1e-6, init: str = "identity", seed: int = 0,
                param_counting: str = "manifold", target_mode: str = "adaptive",
                tensorization: str = "balanced", qubit_align: str = "msb",
                optimizer: str = "explicit", gd_steps: int = 200, gd_lr: float = 0.05,
                restarts: int = 1, fast_gradient: bool = False,
                gradient_objective: str = "disentangle-loss",
                log: bool = False) -> DisentangleResult:
    """Disentangle ``W``, keeping the best of ``restarts`` initializations.

    The optimization is not convex, so the gates can settle in different optima.
    ``restarts`` runs it again from random initializations (one fresh seed each)
    and keeps whichever run retains the most weight at the target bond dimension;
    ``restarts=1`` is a single run from ``init``. ``optimizer`` picks the training
    scheme: ``explicit`` (the paper's environment-SVD sweep, default) or
    ``gradient`` (Adam on the gate angles); both minimize the same objective.
    ``fast_gradient`` evaluates the gradient scheme's loss with the pure-torch
    ``apply_circuit`` contractions instead of PennyLane's ``qml.matrix`` (same
    result, faster per step); it has no effect on the ``explicit`` optimizer.
    ``gradient_objective`` picks what the gradient scheme minimizes:
    ``disentangle-loss`` (default, mean per-bond discarded weight) or
    ``relative-error`` (the joint reconstruction error at ``target_chi``); it also
    has no effect on ``explicit``.
    """
    def _run(this_init, this_seed):
        return _disentangle_once(
            weight, gate_size, depth, target_chi, n_sites, sweeps, tol, this_init,
            this_seed, param_counting, target_mode, tensorization, qubit_align,
            optimizer, gd_steps, gd_lr, fast_gradient, gradient_objective, log)

    best = _run(init, seed)
    for extra in range(1, max(1, restarts)):
        candidate = _run("random", seed + extra)
        if candidate.retained > best.retained:
            best = candidate
    return best
