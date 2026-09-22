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
from torch.utils.checkpoint import checkpoint as _ckpt

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

    ``groups`` optionally constrains the gate to be **block diagonal**: a partition
    of ``range(2^k)`` into index sets, with the matrix orthogonal inside each set
    and zero between them (indices in no group stay on the identity). ``ties``
    labels the groups, and groups sharing a label share one matrix. That is exactly
    a uniformly-controlled (multiplexed) rotation, and it is how an ansatz built
    from a known index pairing -- RoPE's ``(i, i + d_head/2)`` -- is expressed
    without inventing a new gate type: the structure lives in the matrix, so
    application, the register unitary and the sweep all work unchanged.
    """
    start: int
    k: int
    matrix: torch.Tensor
    groups: tuple[tuple[int, ...], ...] | None = None
    ties: tuple[int, ...] | None = None

    def param_count(self, counting: str = "manifold") -> int:
        """Variational parameters this gate actually carries, structure included."""
        if self.groups is None:
            return gate_param_count(self.k, counting)
        seen, total = set(), 0
        for gi, grp in enumerate(self.groups):
            lab = self.ties[gi] if self.ties is not None else gi
            if lab in seen:
                continue                      # a tied group is not a new parameter
            seen.add(lab)
            n = len(grp)
            total += n * n if counting == "entries" else n * (n - 1) // 2
        return total

    @property
    def wires(self) -> list[int]:
        return list(range(self.start, self.start + self.k))

    def op(self) -> "qml.operation.Operator":
        """This gate as a PennyLane operation (for simulation / transpilation)."""
        return qml.QubitUnitary(
            self.matrix.detach().to(torch.float64).cpu().numpy(),
            wires=self.wires, unitary_check=False)


ANSATZE = ("brickwall", "head-block", "rope-pair",
           "adjacent-pair", "random-pair")
PAIR_ANSATZE = ("rope-pair", "adjacent-pair", "random-pair")


def head_block_positions(n_qubits: int, head_dim: int, gate_size: int,
                         depth: int) -> list[tuple[int, int]]:
    """``(start, k)`` for the head-aligned ansatz: one dense gate on the head index.

    With a power-of-two ``head_dim`` the qubit factorization already lines up with
    attention heads: the low ``log2(head_dim)`` qubits index the dimension *within*
    a head and the remaining high qubits index *which* head. Because qubit 0 is the
    most significant bit, those head-index qubits are a contiguous block at the
    start of the register, so "dense on the head index" is just an ordinary gate
    ``(0, head_bits)`` -- no new gate type, and every downstream consumer (parameter
    counting, the environment sweep, the gradient) works unchanged.

    The structure diagnostic (``analyze_structure.py``) found the mutual information
    of q_proj/o_proj concentrated 11-45x on exactly these qubits, which is what this
    ansatz is built to exploit: ``dim SO(2^head_bits)`` parameters instead of a
    brickwall's depth-times-width.

    ``gate_size`` and ``depth`` optionally add a brickwall over the *within-head*
    qubits on top; ``depth <= 0`` or ``gate_size <= 0`` leaves them untouched.
    """
    if n_qubits <= 0 or head_dim <= 1:
        return []
    within = int(round(math.log2(head_dim)))
    head_bits = n_qubits - within
    positions: list[tuple[int, int]] = []
    if head_bits >= 2:                       # a 1-qubit "block" has no SO(2) freedom
        positions.append((0, head_bits))
    elif head_bits < 0:                      # head_dim wider than the register
        return []
    if gate_size > 0 and depth > 0 and within >= 2:
        for s, k in gate_positions(within, gate_size, depth):
            positions.append((s + head_bits, k))
    return positions


def rope_pair_groups(n_qubits: int, head_dim: int, share_heads: bool = True,
                     pairing: str = "rope", seed: int = 0):
    """Index groups of the RoPE pairing: ``(i, i + head_dim/2)`` inside each head.

    RoPE is the one place in these weights where the index ordering means something
    rather than being an artefact of writing a row number in binary. It rotates
    coordinate ``i`` of a head against coordinate ``i + head_dim/2``, with an angle
    that depends on ``i`` and -- importantly -- is the *same for every head*. So the
    faithful ansatz is one ``SO(2)`` per frequency, shared across heads
    (``share_heads``), which costs ``head_dim/2`` parameters for the whole register
    no matter how wide it is. ``share_heads=False`` gives each head its own angles,
    at ``2^n / 2`` parameters.

    Returns ``(groups, ties)``, or ``(None, None)`` when the register does not
    factor into heads of this size. The register is padded to a power of two, so the
    trailing heads are all-zero; they contribute nothing to a tied environment,
    which is a further reason to prefer sharing.
    """
    dim = 1 << n_qubits
    if head_dim < 2 or head_dim % 2 or n_qubits <= 0 or dim % head_dim:
        return None, None
    half = head_dim // 2
    if pairing == "random":
        # The matched null. Same number of groups, same group size, same tie
        # structure -- only WHICH coordinates are paired is destroyed. A gain that
        # survives this is a gain from a rotation of that shape, not from RoPE.
        gen = torch.Generator().manual_seed(seed)
        order = [torch.randperm(head_dim, generator=gen).tolist()
                 for _ in range(dim // head_dim)]
    groups, ties = [], []
    for head in range(dim // head_dim):
        base = head * head_dim
        for freq in range(half):
            if pairing == "rope":
                a, b = freq, freq + half           # RoPE: i against i + d_head/2
            elif pairing == "adjacent":
                a, b = 2 * freq, 2 * freq + 1      # control: neighbouring coords
            elif pairing == "random":
                a, b = order[head][2 * freq], order[head][2 * freq + 1]
            else:                                   # pragma: no cover
                raise ValueError(f"unknown pairing {pairing!r}")
            groups.append((base + a, base + b))
            ties.append(freq if share_heads else head * half + freq)
    return tuple(groups), tuple(ties)


def gate_structures(n_qubits: int, gate_size: int, depth: int,
                    ansatz: str = "brickwall", head_dim: int = 0,
                    share_heads: bool = True, side: str = "out",
                    pair_seed: int = 0):
    """``(start, k, groups, ties)`` for every gate, in application order.

    The layout view :func:`gate_positions` returns the first two fields; this adds
    the block structure that a multiplexed ansatz needs.

    ``rope-pair`` puts one register-wide multiplexed rotation first, then the
    ordinary brickwall (if ``gate_size``/``depth`` ask for one) on top of it.
    ``side`` matters: RoPE acts on the *output* of q_proj / k_proj, so the pairing
    is real for ``U`` and meaningless for ``V``, which falls back to the brickwall.
    """
    if ansatz not in PAIR_ANSATZE:
        return [(st, k, None, None) for st, k
                in gate_positions(n_qubits, gate_size, depth, ansatz, head_dim)]
    plain = [(st, k, None, None) for st, k
             in gate_positions(n_qubits, gate_size, depth)]
    if side not in ("out", "both"):
        return plain
    groups, ties = rope_pair_groups(n_qubits, head_dim, share_heads,
                                    ansatz.split("-")[0], pair_seed)
    if not groups:
        return plain
    return [(0, n_qubits, groups, ties)] + plain


def gate_positions(n_qubits: int, gate_size: int, depth: int,
                   ansatz: str = "brickwall", head_dim: int = 0) -> list[tuple[int, int]]:
    """``(start, k)`` of every gate of a brickwall, in application order.

    Layers alternate between offset 0 and offset ``k // 2`` so that consecutive
    layers straddle each other's boundaries -- the standard brickwall that lets
    correlations spread across the whole register. A gate as wide as the register
    (``k >= n``) already covers everything and stacking more of them only
    multiplies orthogonal matrices together, so that case collapses to a single
    gate (the paper's "10qU, 8qV, L=1" configuration).

    ``ansatz="head-block"`` instead lays the gates out along the attention-head
    split -- see :func:`head_block_positions` -- using ``head_dim``.
    """
    if ansatz == "head-block":
        return head_block_positions(n_qubits, head_dim, gate_size, depth)
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
                        counting: str = "manifold", ansatz: str = "brickwall",
                        head_dim: int = 0, share_heads: bool = True,
                        side: str = "out", pair_seed: int = 0) -> int:
    """Total variational parameters of the circuit -- the ``Q(D)`` term."""
    return sum(Gate(start=st, k=k, matrix=_EMPTY, groups=gr,
                    ties=ti).param_count(counting)
               for st, k, gr, ti in gate_structures(n_qubits, gate_size, depth,
                                                    ansatz, head_dim, share_heads,
                                                    side, pair_seed))


def quantum_param_count(n_out_qubits: int, n_in_qubits: int, gate_size: int,
                        depth: int, counting: str = "manifold",
                        ansatz: str = "brickwall", head_dim: int = 0,
                        share_heads: bool = True, rope_side: str = "out") -> int:
    """``Q(D)``: parameters of both disentangling circuits ``U`` and ``V``.

    ``U`` and ``V`` are counted on their own sides, because ``rope-pair`` is a
    statement about the output index and does not apply to the input one.
    """
    return (circuit_param_count(n_out_qubits, gate_size, depth, counting, ansatz,
                                head_dim, share_heads, "out")
            + circuit_param_count(n_in_qubits, gate_size, depth, counting, ansatz,
                                  head_dim, share_heads,
                                  "out" if rope_side == "both" else "in"))


_EMPTY = torch.zeros(0)          # placeholder for count-only Gate objects


def _identity_gate(k: int) -> torch.Tensor:
    return torch.eye(1 << k, dtype=torch.float32)


def _random_gate(k: int, generator: torch.Generator | None) -> torch.Tensor:
    a = torch.randn(1 << k, 1 << k, dtype=torch.float32, generator=generator)
    q, r = torch.linalg.qr(a)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)   # Haar on O(2^k)


def _scatter_blocks(k: int, groups, ties, block_by_tie) -> torch.Tensor:
    """Assemble a block-diagonal orthogonal from one matrix per tie label.

    Indices covered by no group keep the identity, so the result is orthogonal
    whatever partition is handed in.
    """
    out = torch.eye(1 << k, dtype=torch.float32)
    for gi, grp in enumerate(groups):
        lab = ties[gi] if ties is not None else gi
        idx = torch.as_tensor(grp, dtype=torch.long)
        out[idx.unsqueeze(1), idx.unsqueeze(0)] = block_by_tie[lab].to(out.dtype)
    return out


def _structured_random(k: int, groups, ties,
                       generator: torch.Generator | None) -> torch.Tensor:
    blocks = {}
    for gi, grp in enumerate(groups):
        lab = ties[gi] if ties is not None else gi
        if lab not in blocks:
            n = len(grp)
            a = torch.randn(n, n, dtype=torch.float32, generator=generator)
            q, r = torch.linalg.qr(a)
            blocks[lab] = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    return _scatter_blocks(k, groups, ties, blocks)


def build_circuit(n_qubits: int, gate_size: int, depth: int, init: str = "identity",
                  generator: torch.Generator | None = None,
                  ansatz: str = "brickwall", head_dim: int = 0,
                  share_heads: bool = True, side: str = "out",
                  pair_seed: int = 0) -> list[Gate]:
    """Brickwall of real orthogonal gates, initialised to identity or Haar-random.

    ``identity`` makes the circuit a no-op, so the hybrid layer starts exactly at
    the classical (padded) MPO and the first sweep can only improve on it -- the
    paper's "the hybrid model departs from the classical baseline".
    """
    gates = []
    for start, k, groups, ties in gate_structures(n_qubits, gate_size, depth,
                                                  ansatz, head_dim, share_heads,
                                                  side, pair_seed):
        if groups is None:
            mat = _identity_gate(k) if init == "identity" else _random_gate(k, generator)
        else:
            mat = (_identity_gate(k) if init == "identity"
                   else _structured_random(k, groups, ties, generator))
        gates.append(Gate(start=start, k=k, matrix=mat, groups=groups, ties=ties))
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


def _ckpt_segments(n_gates: int, threshold: int = 32) -> int | None:
    """Number of gradient-checkpoint segments for a chain of ``n_gates`` gates.

    ``None`` (no checkpointing) below ``threshold`` gates -- a shallow circuit's
    graph is cheap, so the recompute isn't worth it. Above it, ``~sqrt(n_gates)``
    segments minimize peak memory (``O(sqrt(n_gates))`` retained) for one extra
    forward pass.
    """
    if n_gates <= threshold:
        return None
    return max(2, round(n_gates ** 0.5))


def apply_circuit(gates: list[Gate], tensor: torch.Tensor, n_qubits: int,
                  transpose: bool = False, checkpoint_segments: int | None = None) -> torch.Tensor:
    """``C X`` (or ``C^T X``): the circuit applied to the row index, gate by gate.

    Gates are applied one at a time (the way a state-vector simulator runs a
    circuit -- PennyLane's own ``default.qubit`` never builds the full register
    unitary either), so cost is ``O(#gates)`` local contractions, not the
    exponential ``qml.matrix`` composition. Differentiable in the gate matrices,
    so this carries the gradient scheme's autograd too. Use :func:`circuit_unitary`
    when the explicit ``2^n x 2^n`` operator is actually wanted (export, small
    circuits, verification).

    ``checkpoint_segments`` enables gradient checkpointing when building an autograd
    graph: the chain is applied in that many contiguous segments, and only the
    segment boundaries are kept for backward -- the intra-segment activations are
    recomputed. Without it the retained graph is ``O(#gates)``, so a deep circuit's
    training memory grows linearly with depth and OOMs; with ``~sqrt(#gates)``
    segments it grows as ``~sqrt(depth)`` for one extra forward pass. It is a no-op
    (identical result, no recompute) when there is no grad to record.
    """
    seq = list(reversed(gates)) if transpose else list(gates)

    def _run(sub, x):
        for g in sub:
            mat = g.matrix.T if transpose else g.matrix
            x = apply_gate(x, mat, g.start, g.k, n_qubits)
        return x

    if not checkpoint_segments or len(seq) <= 1 or not torch.is_grad_enabled():
        return _run(seq, tensor)
    # Gradient checkpointing: store only each segment boundary, recompute the rest in
    # backward. use_reentrant=False is required -- the tensors that need gradients (the
    # gate matrices) are captured in the closure, not passed as inputs, and only the
    # non-reentrant checkpoint tracks those.
    n_seg = max(1, min(int(checkpoint_segments), len(seq)))
    bounds = [round(i * len(seq) / n_seg) for i in range(n_seg + 1)]
    out = tensor
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            out = _ckpt(lambda x, _a=a, _b=b: _run(seq[_a:_b], x), out,
                        use_reentrant=False)
    return out


def apply_right(gates: list[Gate], tensor: torch.Tensor, n_qubits: int,
                transpose: bool = False, checkpoint_segments: int | None = None) -> torch.Tensor:
    """``X C^T`` (or ``X C``): the circuit acting on the *column* index of ``X``."""
    moved = apply_circuit(gates, tensor.transpose(0, 1).contiguous(), n_qubits,
                          transpose=transpose, checkpoint_segments=checkpoint_segments)
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


def _polar_structured(env: torch.Tensor, groups, ties) -> torch.Tensor:
    """Polar update restricted to a block-diagonal (and optionally tied) gate.

    Still closed form, and still *exact*. With the off-block entries of ``g`` pinned
    to zero, ``<g, E>`` = ``sum_G tr(g_G^T E[G,G])``, so the groups decouple; tying
    a set of groups to one matrix ``R`` turns their terms into
    ``tr(R^T sum_G E[G,G])``. Each tie class is therefore maximised by the polar
    factor of its *summed* diagonal sub-block -- one small SVD per class, no
    iteration and no projection step that could lose optimality.
    """
    acc: dict[int, torch.Tensor] = {}
    for gi, grp in enumerate(groups):
        lab = ties[gi] if ties is not None else gi
        idx = torch.as_tensor(grp, dtype=torch.long)
        sub = env[idx.unsqueeze(1), idx.unsqueeze(0)]
        acc[lab] = sub if lab not in acc else acc[lab] + sub
    k = int(round(math.log2(env.shape[0])))
    return _scatter_blocks(k, groups, ties, {c: _polar(m) for c, m in acc.items()})


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
        gate.matrix = (_polar(env) if gate.groups is None
                       else _polar_structured(env, gate.groups, gate.ties))
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


def structured_angle_count(groups, ties) -> int:
    """Angles a block-diagonal gate carries: ``dim SO(|G|)`` per distinct tie class."""
    seen, total = set(), 0
    for gi, grp in enumerate(groups):
        lab = ties[gi] if ties is not None else gi
        if lab in seen:
            continue
        seen.add(lab)
        total += len(grp) * (len(grp) - 1) // 2
    return total


def _structured_from_angles(theta: torch.Tensor, k: int, groups, ties) -> torch.Tensor:
    """Differentiable block-diagonal orthogonal from one angle block per tie class.

    The constraint has to live in the *parameterization*, not in a projection after
    the fact: ``expm(skew(theta))`` over the whole register would train
    ``dim SO(2^k)`` angles and leave the block structure only approximately intact.
    Here each tie class owns ``dim SO(|G|)`` angles, its block is
    ``expm(skew(.))``, and the blocks are scattered into an identity -- so the gate
    is exactly orthogonal, exactly block diagonal, and carries exactly the
    parameters :meth:`Gate.param_count` charges it for, at every point of the
    optimisation rather than only at the end.

    Written without a Python loop over groups: a pair ansatz on a 10-qubit register
    has 512 of them and this runs inside every Adam step.
    """
    sizes = {len(g) for g in groups}
    if len(sizes) != 1:                       # pragma: no cover - pairings are uniform
        raise ValueError(f"block sizes must be uniform, got {sorted(sizes)}")
    n = sizes.pop()
    labels = sorted({(ties[gi] if ties is not None else gi)
                     for gi in range(len(groups))})
    per = n * (n - 1) // 2
    blocks = torch.stack([_ortho_from_angles(theta[i * per:(i + 1) * per], n)
                          for i in range(len(labels))])          # (classes, n, n)
    pos = {lab: i for i, lab in enumerate(labels)}
    take = torch.tensor([pos[ties[gi] if ties is not None else gi]
                         for gi in range(len(groups))], dtype=torch.long)
    mats = blocks[take]                                          # (groups, n, n)
    idx = torch.tensor(groups, dtype=torch.long)                 # (groups, n)
    rows = idx.unsqueeze(2).expand(-1, -1, n).reshape(-1)
    cols = idx.unsqueeze(1).expand(-1, n, -1).reshape(-1)
    eye = torch.eye(1 << k, dtype=blocks.dtype)
    return eye.index_put((rows, cols), mats.reshape(-1))


def _ortho_from_angles(theta: torch.Tensor, n: int) -> torch.Tensor:
    """``expm(skew(theta))`` for an arbitrary ``n``, not just a power of two."""
    a = torch.zeros(n, n, dtype=theta.dtype)
    iu = torch.triu_indices(n, n, offset=1)
    a = a.index_put((iu[0], iu[1]), theta)
    return torch.linalg.matrix_exp(a - a.T)


def _gates_from_angles(thetas: list[torch.Tensor],
                       positions: list[tuple[int, int]],
                       bases: list[torch.Tensor] | None = None,
                       structures: list | None = None) -> list[Gate]:
    """Build :class:`Gate` objects (differentiable matrices) from angle tensors.

    Without ``bases`` each gate is ``expm(skew(theta))`` (identity at theta=0).
    With ``bases`` each gate is ``base @ expm(skew(theta))`` -- a differentiable
    refinement multiplying a fixed orthogonal base gate, so ``theta=0`` reproduces
    ``base`` exactly. This is how the gradient stage warm-starts from the explicit
    sweep's gates (the same parameterization :class:`~qllm.hybrid_heal.HybridAdapter`
    uses to heal circuits).
    """
    structures = structures or [(None, None)] * len(positions)
    out = []
    for i, (t, (st, k)) in enumerate(zip(thetas, positions)):
        groups, ties = structures[i]
        # A structured gate stays structured under warm start too: the base is
        # block diagonal on the same partition and the refinement is built on that
        # same partition, so the product is as well.
        mat = (_structured_from_angles(t, k, groups, ties) if groups is not None
               else _gate_from_angles(t, k))
        if bases is not None:
            mat = bases[i] @ mat
        out.append(Gate(st, k, mat, groups=groups, ties=ties))
    return out


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


def activation_error_loss(current, ug, vg, n_out, n_in, out_dims, in_dims,
                          n_sites, chi, weight, cov, denom):
    """Differentiable ``sqrt(tr[D H D^T] / tr[W H W^T])`` for the *cropped* layer.

    The Frobenius objectives can be evaluated on the disentangled operator alone,
    because ``U``/``V`` are orthogonal and the norm does not see them. The
    activation-weighted error can not: ``H`` sits between the factors, so the
    reconstruction has to be carried back through the circuits and cropped to the
    original shape before it is scored. That costs two extra circuit applications
    per step and is the price of optimising the quantity that actually matters.
    """
    recon = _truncated_dense(current, out_dims, in_dims, n_sites, chi)
    dense = apply_right(vg, apply_circuit(ug, recon, n_out), n_in)
    d = weight - dense[:weight.shape[0], :weight.shape[1]]
    num = ((d @ cov) * d).sum()
    return torch.sqrt(torch.clamp(num, min=0.0) / denom)


def _train_gradient(padded, n_out, n_in, gate_size, depth, plan_pad, target_chi,
                    steps, lr, init, seed, log, base_u=None, base_v=None,
                    fast=False, objective="disentangle-loss",
                    ansatz="brickwall", head_dim=0, share_heads=True,
                    rope_side="out", pair_seed=0, cov=None, weight=None,
                    train_side="both"):
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
    if objective == "activation-mse" and (cov is None or weight is None):
        raise ValueError("objective='activation-mse' needs both `cov` (H) and "
                         "`weight` (the uncropped W); without them there is "
                         "nothing to weight the error by.")
    struct_u = gate_structures(n_out, gate_size, depth, ansatz, head_dim,
                               share_heads, "out", pair_seed)
    struct_v = gate_structures(n_in, gate_size, depth, ansatz, head_dim,
                               share_heads, "out" if rope_side == "both" else "in",
                               pair_seed)
    pos_u = [(st, k) for st, k, _g, _t in struct_u]
    pos_v = [(st, k) for st, k, _g, _t in struct_v]
    sgr_u = [(g, t) for _s, _k, g, t in struct_u]
    sgr_v = [(g, t) for _s, _k, g, t in struct_v]
    gen = torch.Generator().manual_seed(seed)
    refine = base_u is not None                      # warm-start from base gates
    bases_u = [g.matrix for g in base_u] if refine else None
    bases_v = [g.matrix for g in base_v] if refine else None

    def _init(positions, structures, trainable):
        out = []
        for (_s, k), (groups, ties) in zip(positions, structures):
            # A structured gate is sized by its own partition, not by the register:
            # 32 angles for a tied pairing where the unconstrained gate would want
            # dim SO(2^k). This is the whole point of the constrained parameterization.
            m = (structured_angle_count(groups, ties) if groups is not None
                 else (1 << k) * ((1 << k) - 1) // 2)
            if refine or init != "random":
                t = torch.zeros(m)                     # start at base/identity
            else:
                t = 0.1 * torch.randn(m, generator=gen)
            out.append(t.requires_grad_(trainable))
        return out

    # train_side freezes one circuit. With the H-weighted objective the U update is
    # still a closed-form polar problem (the U-dependent quadratic cancels under the
    # trace) while the V one is not, so "train V only, on top of the explicit sweep"
    # is the cheap mixed scheme -- worth measuring against training both.
    theta_u = _init(pos_u, sgr_u, train_side in ("both", "u"))
    theta_v = _init(pos_v, sgr_v, train_side in ("both", "v"))
    params = [t for t in theta_u + theta_v if t.requires_grad]
    history: list[float] = []
    if not params:                                   # depth 0: no gates to train
        return (base_u or []), (base_v or []), 0, history

    out_dims, in_dims, n_sites = plan_pad.out_dims, plan_pad.in_dims, plan_pad.n_sites
    act_denom = None
    if objective == "activation-mse":
        w64 = weight.to(torch.float64)
        act_denom = float(((w64 @ cov.to(torch.float64)) * w64).sum())
        if not act_denom > 0:
            raise ValueError("tr[W H W^T] is not positive; H is degenerate here.")
        act_denom = torch.tensor(act_denom, dtype=weight.dtype)
    opt = torch.optim.Adam(params, lr=lr)
    # Keep the best (lowest-loss) *finite* iterate and restore it at the end. The
    # differentiable SVD in the relative-error objective is ill-conditioned near
    # degenerate singular values and can hand Adam a non-finite gradient, which
    # then poisons the angles (and the next forward SVD). Snapshotting the best
    # iterate and stopping on the first non-finite step makes the optimizer
    # monotone-safe: it can only improve on the start (the explicit gates, for
    # explicit+gradient), never diverge into NaNs.
    best_loss = math.inf
    best_u = [t.detach().clone() for t in theta_u]
    best_v = [t.detach().clone() for t in theta_v]
    for step in range(max(1, steps)):
        opt.zero_grad()
        # Build the disentangled operator U^T W V from the current angles. The
        # gate matrices carry autograd either way; ``fast`` chooses how they are
        # contracted into ``current``.
        ug = _gates_from_angles(theta_u, pos_u, bases_u, sgr_u)
        vg = _gates_from_angles(theta_v, pos_v, bases_v, sgr_v)
        if fast:
            # Pure-torch local contractions -- O(#gates), no register unitary. For a
            # deep circuit the retained autograd graph (one activation per gate) is
            # what makes training memory grow with depth and OOM; checkpoint the chain
            # in ~sqrt(#gates) segments so it grows as ~sqrt(depth) instead. Shallow
            # circuits (few gates) skip it -- no benefit, and it avoids the recompute.
            seg_u = _ckpt_segments(len(ug))
            seg_v = _ckpt_segments(len(vg))
            current = apply_right(vg, apply_circuit(ug, padded, n_out, transpose=True,
                                                    checkpoint_segments=seg_u),
                                  n_in, transpose=True, checkpoint_segments=seg_v)
        else:
            # THROUGH PennyLane (qml.matrix, torch interface) -- autograd flows
            # from the loss back to the angles via the PennyLane circuit itself.
            u = circuit_unitary(ug, n_out)
            v = circuit_unitary(vg, n_in)
            current = u.transpose(0, 1) @ padded @ v
        if not torch.isfinite(current).all():
            break                                    # angles diverged; keep best
        try:
            if objective == "activation-mse":
                loss = activation_error_loss(current, ug, vg, n_out, n_in, out_dims,
                                             in_dims, n_sites, target_chi,
                                             weight, cov, act_denom)
            elif objective == "relative-error":
                loss = relative_error_loss(current, out_dims, in_dims, n_sites, target_chi)
            else:
                loss = disentangle_loss(current, out_dims, in_dims, n_sites, target_chi)
            lval = float(loss.detach())
            if not math.isfinite(lval):
                break
            if lval < best_loss:                     # snapshot the best-so-far
                best_loss = lval
                best_u = [t.detach().clone() for t in theta_u]
                best_v = [t.detach().clone() for t in theta_v]
            loss.backward()
        except RuntimeError:                         # SVD non-convergence -> keep best
            break
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):
            break                                    # non-finite gradient; keep best
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        opt.step()
        history.append(1.0 - lval)
        if log and (step == 0 or (step + 1) % max(1, steps // 5) == 0):
            print(f"      gd step {step + 1:>4}/{steps}  loss={lval:.6f}")

    with torch.no_grad():                            # restore the best iterate
        for t, b in zip(theta_u, best_u):
            t.copy_(b)
        for t, b in zip(theta_v, best_v):
            t.copy_(b)
        # Keep groups/ties on the returned gates: Gate.param_count reads them, so
        # dropping them here would silently price a 32-angle pairing at dim SO(2^k).
        u_gates = [Gate(g.start, g.k, g.matrix.detach(), g.groups, g.ties)
                   for g in _gates_from_angles(theta_u, pos_u, bases_u, sgr_u)]
        v_gates = [Gate(g.start, g.k, g.matrix.detach(), g.groups, g.ties)
                   for g in _gates_from_angles(theta_v, pos_v, bases_v, sgr_v)]
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
        # Counted from the gates that were actually built, not re-derived from
        # (gate_size, depth) -- so it stays correct for any ansatz layout.
        quantum_params=sum(g.param_count(param_counting)
                           for g in list(u_gates) + list(v_gates)),
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
                      ansatz: str = "brickwall", head_dim: int = 0,
                      share_heads: bool = True, rope_side: str = "out",
                      pair_seed: int = 0, cov=None, train_side: str = "both",
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
            objective=gradient_objective, ansatz=ansatz, head_dim=head_dim,
            share_heads=share_heads, rope_side=rope_side, pair_seed=pair_seed,
            cov=cov, weight=weight, train_side=train_side)
        return _build_result(
            weight, padded, u_gates, v_gates, n_out, n_in, gate_size, depth,
            target_chi, _plan, param_counting, retained_classical, target_ref,
            ref_norm, norm, steps_run, t0, history, "gradient")

    # --- explicit environment / SVD sweep (paper Appendix A) ---
    generator = torch.Generator().manual_seed(seed)
    u_gates = build_circuit(n_out, gate_size, depth, init, generator, ansatz,
                            head_dim, share_heads, "out", pair_seed)
    v_gates = build_circuit(n_in, gate_size, depth, init, generator, ansatz,
                            head_dim, share_heads,
                            "out" if rope_side == "both" else "in", pair_seed)

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
            fast=fast_gradient, objective=gradient_objective,
            ansatz=ansatz, head_dim=head_dim, share_heads=share_heads,
            rope_side=rope_side, pair_seed=pair_seed, cov=cov, weight=weight,
            train_side=train_side)
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
                ansatz: str = "brickwall", head_dim: int = 0,
                share_heads: bool = True, rope_side: str = "out",
                pair_seed: int = 0, cov=None, train_side: str = "both",
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
            optimizer, gd_steps, gd_lr, fast_gradient, gradient_objective,
            ansatz, head_dim, share_heads, rope_side, pair_seed, cov,
            train_side, log)

    best = _run(init, seed)
    for extra in range(1, max(1, restarts)):
        candidate = _run("random", seed + extra)
        if candidate.retained > best.retained:
            best = candidate
    return best
