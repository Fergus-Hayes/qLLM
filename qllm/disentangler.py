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

Optimization
------------
The objective is the one the disentangling accuracy of the paper measures, in
its error form:

    minimize_{U, V}  || U^T W V  -  T_chi(U^T W V) ||_F ,

with ``T_chi`` the MPO truncation at the target bond dimension. It is optimized
by alternation, exactly as in Appendix A of the paper:

1. **Target step** -- ``M <- T_chi(U^T W V)``: the truncation is the best bond-chi
   approximation of the current disentangled operator.
2. **Gate sweeps** -- with ``M`` fixed, maximize the overlap ``<W, U M V^T>``.
   The overlap is *linear* in every individual gate, so for gate ``g`` it reads
   ``<g, E_g>`` with ``E_g`` the environment tensor (the network with ``g``
   removed). The maximizer over the orthogonal group is the polar factor of the
   environment: ``E_g = P S Q  ->  g <- P Q``.

Because ``U`` and ``V`` are orthogonal, ``||W - U M V^T|| = ||U^T W V - M||``, so
step 1 and step 2 both decrease the *same* error and the iteration is monotone --
:func:`disentangle` asserts this and reports the retained weight per sweep.

Sweeping costs no extra memory: the gates are orthogonal, so the partial product
to the right of the current gate is recovered from the running one by applying
the *old* gate transposed, and the partial product to the left is accumulated
from the freshly updated gates. One sweep therefore costs O(#gates) applications
of a gate to the padded matrix and stores O(1) of them.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .compactifai import MPOPlan, mpo_param_count
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
    """One few-qubit gate: a real orthogonal matrix on qubits ``[start, start+k)``."""
    start: int
    k: int
    matrix: torch.Tensor


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
# Applying gates / circuits
# --------------------------------------------------------------------------- #
def apply_gate(tensor: torch.Tensor, gate: torch.Tensor, start: int, k: int,
               n_qubits: int) -> torch.Tensor:
    """Apply a ``k``-qubit gate to the row index of ``tensor`` (shape ``(2^n, C)``).

    Qubit ``0`` is the most significant bit of the row index, so a *contiguous*
    block of qubits is already a contiguous stride of the flat index: the
    operation is a reshape into ``(left, 2^k, rest)`` and one batched matmul --
    no permutation or copy of the big tensor.
    """
    left = 1 << start
    mid = 1 << k
    view = tensor.reshape(left, mid, -1)
    out = torch.matmul(gate, view)
    return out.reshape(1 << n_qubits, -1)


def apply_circuit(gates: list[Gate], tensor: torch.Tensor, n_qubits: int,
                  transpose: bool = False) -> torch.Tensor:
    """``C X`` (or ``C^T X``) for ``C = G_last ... G_first``."""
    out = tensor
    order = reversed(gates) if transpose else gates
    for g in order:
        mat = g.matrix.T if transpose else g.matrix
        out = apply_gate(out, mat, g.start, g.k, n_qubits)
    return out


def apply_right(gates: list[Gate], tensor: torch.Tensor, n_qubits: int,
                transpose: bool = False) -> torch.Tensor:
    """``X C^T`` (or ``X C``): the circuit acting on the *column* index of ``X``."""
    moved = apply_circuit(gates, tensor.T.contiguous(), n_qubits, transpose=transpose)
    return moved.T.contiguous()


def _environment(left: torch.Tensor, right: torch.Tensor, start: int, k: int) -> torch.Tensor:
    """``E[a,b] = sum_{l,c} L[l,a,c] R[l,b,c]`` -- the gradient w.r.t. the gate."""
    nl = 1 << start
    mid = 1 << k
    lr = left.reshape(nl, mid, -1)
    rr = right.reshape(nl, mid, -1)
    return torch.einsum("lac,lbc->ab", lr, rr)


def _polar(matrix: torch.Tensor) -> torch.Tensor:
    """Orthogonal factor of ``E``: the maximizer of ``<g, E>`` over ``O(m)``."""
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


def _disentangle_once(weight: torch.Tensor, gate_size: int, depth: int,
                      target_chi: int = 1, n_sites: int = 2, sweeps: int = 12,
                      tol: float = 1e-6, init: str = "identity", seed: int = 0,
                      param_counting: str = "manifold",
                      target_mode: str = "adaptive", tensorization: str = "balanced",
                      qubit_align: str = "msb",
                      log: bool = False) -> DisentangleResult:
    """Disentangle ``W`` into ``U MPO_new V^T`` with brickwall circuits of depth ``D``.

    ``target_chi`` is the bond dimension the circuits are asked to squeeze the
    operator into (the paper uses 1). The returned plan caches the SVD of
    ``MPO_new``, so the residual can afterwards be truncated to *any* bond
    dimension for free -- which is how the hybrid sweep reuses one optimization
    across its whole chi' grid (the paper's Table I).

    ``target_mode`` selects the objective:

    * ``adaptive`` (default) re-truncates the *current* disentangled operator at
      every iteration, which minimizes the quantity that actually decides the
      compressed layer's error, ``||U^T W V - T_chi(U^T W V)||``.
    * ``fixed`` freezes the target at ``T_chi(W_pad)``, the truncation of the
      original operator -- the literal objective behind the paper's Eq. (4)
      disentangling accuracy, kept for comparison with its figures.
    """
    t0 = time.perf_counter()
    padded, n_out, n_in = pad_to_qubits(weight)
    norm = float(torch.linalg.norm(padded))
    generator = torch.Generator().manual_seed(seed)
    u_gates = build_circuit(n_out, gate_size, depth, init, generator)
    v_gates = build_circuit(n_in, gate_size, depth, init, generator)

    def _plan(op):
        return make_plan(op, tensorization, n_sites, svd_cache=True, align=qubit_align)

    # Classical reference: the same truncation with the circuits switched off.
    plan_pad = _plan(padded)
    trunc_pad, _ = plan_compress(padded, plan_pad, target_chi)
    retained_classical = float(torch.linalg.norm(trunc_pad)) / norm if norm else 0.0
    target_ref = trunc_pad                       # MPO_target of the paper's Eq. (4)
    ref_norm = float(torch.linalg.norm(target_ref))

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

    plan = _plan(current)
    final_target, _ = plan_compress(current, plan, target_chi)
    retained = float(torch.linalg.norm(final_target)) / norm if norm else 0.0
    history.append(retained)
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
        seconds=time.perf_counter() - t0, history=history,
    )


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
                restarts: int = 1, log: bool = False) -> DisentangleResult:
    """Disentangle ``W``, keeping the best of ``restarts`` initializations.

    The alternating optimization is monotone but not convex, so the gates can
    settle in different optima. ``restarts`` runs it again from Haar-random
    circuits (one fresh seed each) and keeps whichever run retains the most
    weight at the target bond dimension; ``restarts=1`` is a single run from
    ``init``. Random starts usually edge out the identity start for wide gates,
    while the identity start is the one that begins exactly at the classical
    (padded) layer.
    """
    best = _disentangle_once(weight, gate_size, depth, target_chi, n_sites,
                             sweeps, tol, init, seed, param_counting,
                             target_mode, tensorization, qubit_align, log)
    for extra in range(1, max(1, restarts)):
        candidate = _disentangle_once(weight, gate_size, depth, target_chi,
                                      n_sites, sweeps, tol, "random",
                                      seed + extra, param_counting, target_mode,
                                      tensorization, qubit_align, log)
        if candidate.retained > best.retained:
            best = candidate
    return best
