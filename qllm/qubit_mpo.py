"""Qubit-level MPO tensorization -- the geometry of arXiv:2410.17397 / CompactifAI.

The 2-site MPO in :mod:`qllm.compactifai` splits each matrix index into two
coarse balanced factors (e.g. ``576 -> [24, 24]``). This module provides the
*fine* tensorization the papers actually use: each dimension is embedded in
qubits, ``d -> 2**ceil(log2 d)`` (zero-padded), and split into ``log2`` legs of
dimension two. The MPO then has one site per qubit position.

Where the two indices need a different number of qubits (the paper's ``(576,
192)`` layer is ``(10, 8)`` qubits), the surplus qubits of the longer index form
single-leg sites -- ``(out_leg, 1)`` or ``(1, in_leg)`` -- at the tail of the
chain (``align="msb"``, the default: the leading qubits pair up, most-significant
to most-significant). This is exactly the structure that stores **36** parameters
at bond dimension 1 and **132** at bond dimension 2 for that layer, reproducing
the paper's Table I.

The construction is a drop-in for both methods:

* **classical** -- build the plan on the original weight; the truncated MPO is
  contracted back and cropped to the original shape, so the compressed layer is
  a pure tensor network with the paper's parameter counts.
* **hybrid** -- build the plan on the disentangled (already power-of-two)
  operator; the crop is a no-op and the disentangling circuits wrap the
  truncated residual.

Because a qubit MPO has ``max(n_out, n_in)`` sites (up to ~10 here) it needs that
many sequential SVDs and has no single-SVD fast path, so a sweep recomputes the
factors per bond dimension rather than reusing one cached SVD (the 2-site
``balanced`` tensorization keeps that optimization). The two share the same
plan interface -- ``out_dims``, ``in_dims``, ``n_sites``, ``dense_params``,
``max_chi`` -- so ``mpo_param_count``, ``mpo_bond_dims`` and ``full_rank_chi``
apply unchanged to either.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .compactifai import (
    build_plan,
    compress_weight,
    contract_factors,
    mpo_factors,
    mpo_param_count,
)


def n_qubits_for(dim: int) -> int:
    """Qubits needed to embed a dimension: ``ceil(log2 dim)`` (0 for dim <= 1)."""
    return max(0, math.ceil(math.log2(max(1, int(dim)))))


def qubit_mpo_dims(d_out: int, d_in: int, align: str = "msb") -> tuple[list[int], list[int]]:
    """Per-site output/input legs for the qubit MPO of a ``(d_out, d_in)`` matrix.

    Both lists have length ``max(n_out, n_in)``. The shorter index is padded with
    dim-1 legs so each site is a pair ``(out_leg, in_leg)``. ``align="msb"`` puts
    the fillers at the tail (pair the leading/most-significant qubits);
    ``align="lsb"`` puts them at the head (pair the trailing qubits).
    """
    n_out, n_in = n_qubits_for(d_out), n_qubits_for(d_in)
    n = max(n_out, n_in, 1)
    out = [2] * n_out
    inn = [2] * n_in
    if align == "lsb":
        out = [1] * (n - n_out) + out
        inn = [1] * (n - n_in) + inn
    else:  # "msb"
        out = out + [1] * (n - n_out)
        inn = inn + [1] * (n - n_in)
    return out, inn


@dataclass
class QubitMPOPlan:
    """A qubit-level MPO plan, interface-compatible with :class:`MPOPlan`.

    ``out_dims``/``in_dims`` are the per-site legs (dim 2, or dim 1 fillers);
    ``orig_shape`` is the matrix's true shape and ``pad_shape`` its power-of-two
    box. ``dense_params`` is the *original* parameter count (what you would store
    otherwise), so ``max_chi`` is the largest bond that still beats it.
    """
    out_dims: list[int]
    in_dims: list[int]
    n_sites: int
    orig_shape: tuple[int, int]
    pad_shape: tuple[int, int]
    dense_params: int
    max_chi: int
    align: str = "msb"
    cached: bool = field(default=False)   # no single-SVD fast path for N sites


def qubit_max_useful_chi(out_dims: list[int], in_dims: list[int], dense: int) -> int:
    """Largest bond dimension whose MPO still stores fewer params than ``dense``."""
    lo, hi, best = 1, max(1, dense), 0
    while lo <= hi:                                  # param count is monotone in chi
        mid = (lo + hi) // 2
        if mpo_param_count(out_dims, in_dims, mid) < dense:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def build_qubit_plan(weight: torch.Tensor, align: str = "msb") -> QubitMPOPlan:
    """Factorize a weight matrix into a qubit-level MPO plan (no SVD yet)."""
    d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
    out_dims, in_dims = qubit_mpo_dims(d_out, d_in, align)
    pad_shape = (math.prod(out_dims), math.prod(in_dims))
    dense = d_out * d_in
    return QubitMPOPlan(
        out_dims=out_dims, in_dims=in_dims, n_sites=len(out_dims),
        orig_shape=(d_out, d_in), pad_shape=pad_shape, dense_params=dense,
        max_chi=qubit_max_useful_chi(out_dims, in_dims, dense), align=align,
    )


def _pad(weight: torch.Tensor, pad_shape: tuple[int, int]) -> torch.Tensor:
    """Zero-pad ``weight`` into its power-of-two box (a no-op if already there)."""
    w = weight.detach().to(torch.float32).cpu()
    if tuple(w.shape) == tuple(pad_shape):
        return w
    out = torch.zeros(*pad_shape, dtype=torch.float32)
    out[: w.shape[0], : w.shape[1]] = w
    return out


@torch.no_grad()
def compress_weight_qubit(weight: torch.Tensor, plan: QubitMPOPlan,
                          chi: int) -> tuple[torch.Tensor, int]:
    """Truncate to bond ``chi`` and return (dense recon cropped to orig, params).

    The weight is padded to the power-of-two box, decomposed by the shared
    sequential-SVD MPO builder, contracted back, and cropped to the original
    shape -- the padding carries no signal, so the crop is exact at full rank.
    """
    w = _pad(weight, plan.pad_shape)
    factors = mpo_factors(w, plan, chi)
    dense = contract_factors(factors, plan)
    d_out, d_in = plan.orig_shape
    params = mpo_param_count(plan.out_dims, plan.in_dims, chi)
    return dense[:d_out, :d_in].contiguous(), params


@torch.no_grad()
def bond_entropy_qubit(weight: torch.Tensor, plan: QubitMPOPlan) -> float:
    """Mean von Neumann entropy over the MPO bonds of ``weight`` (nats).

    Computed from the untruncated left-to-right sequential SVD: the singular
    values at each cut are the Schmidt spectrum of that bond.
    """
    w = _pad(weight, plan.pad_shape)
    out_dims, in_dims, n = plan.out_dims, plan.in_dims, plan.n_sites
    from .compactifai import _interleave_perm
    tensor = w.reshape(*out_dims, *in_dims).permute(*_interleave_perm(n)).contiguous()
    mat = tensor.reshape(out_dims[0] * in_dims[0], -1)
    bond_left, entropies = 1, []
    for k in range(n - 1):
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        p = (s ** 2)
        total = float(p.sum())
        if total > 0:
            p = p / total
            p = p[p > 1e-30]
            entropies.append(float(-(p * p.log()).sum()))
        mat = s.unsqueeze(1) * vh
        bond_left = int(s.numel())
        if k + 1 < n - 1:
            mat = mat.reshape(bond_left * out_dims[k + 1] * in_dims[k + 1], -1)
    return sum(entropies) / len(entropies) if entropies else 0.0


# --------------------------------------------------------------------------- #
# Dispatch: one interface, either tensorization
# --------------------------------------------------------------------------- #
def make_plan(weight: torch.Tensor, tensorization: str, mpo_sites: int,
              svd_cache: bool = True, align: str = "msb"):
    """Build a ``balanced`` (2-site, cached) or ``qubit`` plan for ``weight``."""
    if tensorization == "qubit":
        return build_qubit_plan(weight, align)
    return build_plan(weight, mpo_sites, cache=svd_cache)


def plan_compress(weight: torch.Tensor, plan, chi: int) -> tuple[torch.Tensor, int]:
    """Compress ``weight`` with whichever plan type it is."""
    if isinstance(plan, QubitMPOPlan):
        return compress_weight_qubit(weight, plan, chi)
    return compress_weight(weight, plan, chi)


def plan_bond_entropy(weight: torch.Tensor, plan) -> float:
    if isinstance(plan, QubitMPOPlan):
        return bond_entropy_qubit(weight, plan)
    from .disentangler import bond_entropy   # 2-site entropy lives with the plan's use
    return bond_entropy(weight, plan)
