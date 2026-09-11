"""CompactifAI-style tensor-network (MPO) compression of LLM weight matrices.

Implements the compression method of *CompactifAI: Extreme Compression of Large
Language Models using Quantum-Inspired Tensor Networks* (Tomut et al.): the
weight matrices of the Self-Attention and MLP layers of each decoder block are
replaced by Matrix Product Operators (MPOs) obtained by reshaping the matrix
indices and performing sequential SVDs, retaining only the largest ``chi``
singular values at each step. The bond dimension ``chi`` controls the degree of
truncation -- and hence of compression -- of the correlations in the layer.

For a weight matrix ``W`` of shape ``(d_out, d_in)`` and an ``N``-site MPO, the
dimensions are factorized as ``d_out = o_1...o_N`` and ``d_in = i_1...i_N``, the
tensor is permuted to interleave the pairs ``(o_k, i_k)``, and ``N-1``
sequential truncated SVDs produce ``N`` tensors. The parameter count is the sum
of the tensor sizes, e.g. for the paper's 216x216 / 3-site example
``2 * 36 chi + 36 chi^2``.

To measure the effect on perplexity, the truncated MPO is contracted back into a
dense matrix and written into the model: the contraction of the truncated MPO is
exactly the effective operator the compressed model applies, so perplexity is
identical to running the factorized layers directly, while the *parameter count*
reported is that of the stored MPO tensors (not the reconstruction).

Note: this module implements the compression + evaluation only. The paper's
subsequent "healing" (brief retraining) stage is not performed, so the reported
perplexities are the pre-healing values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Index bookkeeping
# --------------------------------------------------------------------------- #
def balanced_factorization(n: int, parts: int) -> list[int]:
    """Factor ``n`` into ``parts`` positive integers as close to equal as possible.

    Greedy: repeatedly pull out the divisor nearest to the ideal geometric mean
    ``n ** (1/remaining)``. Falls back to 1s for primes.
    """
    if parts <= 1:
        return [n]
    target = round(n ** (1.0 / parts))
    best, best_err = 1, None
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d:
            continue
        for cand in (d, n // d):
            err = abs(cand - target)
            if best_err is None or err < best_err:
                best, best_err = cand, err
    return [best] + balanced_factorization(n // best, parts - 1)


def mpo_dims(d_out: int, d_in: int, n_sites: int) -> tuple[list[int], list[int]]:
    """Factorize the output/input dimensions into ``n_sites`` factors each."""
    return (balanced_factorization(d_out, n_sites),
            balanced_factorization(d_in, n_sites))


def _interleave_perm(n_sites: int) -> list[int]:
    """Permutation taking (o_1..o_N, i_1..i_N) -> (o_1,i_1, ..., o_N,i_N)."""
    perm = []
    for k in range(n_sites):
        perm += [k, n_sites + k]
    return perm


def _deinterleave_perm(n_sites: int) -> list[int]:
    """Inverse of :func:`_interleave_perm`."""
    return [2 * k for k in range(n_sites)] + [2 * k + 1 for k in range(n_sites)]


def mpo_bond_dims(out_dims: list[int], in_dims: list[int], chi: int) -> list[int]:
    """Actual bond dimensions after truncating to ``chi`` at every cut.

    A bond can never exceed the rank of the matrix across that cut, which is
    ``min(prod of site dims on the left, prod on the right)``.
    """
    n = len(out_dims)
    site = [out_dims[k] * in_dims[k] for k in range(n)]
    bonds = []
    for cut in range(1, n):
        left = math.prod(site[:cut])
        right = math.prod(site[cut:])
        bonds.append(min(chi, left, right))
    return bonds


def mpo_param_count(out_dims: list[int], in_dims: list[int], chi: int) -> int:
    """Number of parameters stored by the MPO tensors at bond dimension ``chi``."""
    n = len(out_dims)
    site = [out_dims[k] * in_dims[k] for k in range(n)]
    bonds = [1] + mpo_bond_dims(out_dims, in_dims, chi) + [1]
    return sum(bonds[k] * site[k] * bonds[k + 1] for k in range(n))


def max_useful_chi(out_dims: list[int], in_dims: list[int]) -> int:
    """Largest chi that still stores fewer parameters than the dense matrix."""
    dense = math.prod(out_dims) * math.prod(in_dims)
    lo, hi = 1, max(1, dense)
    best = 0
    # mpo_param_count is monotonically non-decreasing in chi -> binary search.
    while lo <= hi:
        mid = (lo + hi) // 2
        if mpo_param_count(out_dims, in_dims, mid) < dense:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best


def log_spaced_ints(lo: int, hi: int, count: int) -> list[int]:
    """Logarithmically spaced, de-duplicated integers in ``[lo, hi]``."""
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi <= lo or count <= 1:
        return [hi]
    ratio = hi / lo
    vals = {int(round(lo * ratio ** (i / (count - 1)))) for i in range(count)}
    return sorted(v for v in vals if lo <= v <= hi)


# --------------------------------------------------------------------------- #
# MPO decomposition
# --------------------------------------------------------------------------- #
@dataclass
class MPOPlan:
    """Per-layer decomposition plan plus (optionally) a cached leading SVD."""
    out_dims: list[int]
    in_dims: list[int]
    n_sites: int
    dense_params: int
    max_chi: int
    # Cache for the 2-site fast path: W_perm = U diag(S) Vh
    u: torch.Tensor | None = None
    s: torch.Tensor | None = None
    vh: torch.Tensor | None = None

    @property
    def cached(self) -> bool:
        return self.u is not None


@torch.no_grad()
def build_plan(weight: torch.Tensor, n_sites: int, cache: bool = True) -> MPOPlan:
    """Factorize dimensions and, for a 2-site MPO, precompute the single SVD.

    A 2-site MPO needs exactly one SVD, so caching ``(U, S, Vh)`` lets every bond
    dimension in a sweep be produced by truncation alone -- no further SVDs.
    """
    d_out, d_in = int(weight.shape[0]), int(weight.shape[1])
    out_dims, in_dims = mpo_dims(d_out, d_in, n_sites)
    plan = MPOPlan(
        out_dims=out_dims,
        in_dims=in_dims,
        n_sites=n_sites,
        dense_params=d_out * d_in,
        max_chi=max_useful_chi(out_dims, in_dims),
    )
    if cache and n_sites == 2:
        w = weight.detach().to(torch.float32).cpu()
        perm = w.reshape(*out_dims, *in_dims).permute(*_interleave_perm(2))
        mat = perm.reshape(out_dims[0] * in_dims[0], out_dims[1] * in_dims[1])
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        plan.u, plan.s, plan.vh = u, s, vh
    return plan


@torch.no_grad()
def mpo_factors(weight: torch.Tensor, plan: MPOPlan, chi: int) -> list[torch.Tensor]:
    """Sequential truncated SVDs producing the MPO tensors (general N sites).

    Each tensor has shape ``(bond_left, o_k, i_k, bond_right)``.
    """
    n = plan.n_sites
    out_dims, in_dims = plan.out_dims, plan.in_dims
    w = weight.detach().to(torch.float32).cpu()
    tensor = w.reshape(*out_dims, *in_dims).permute(*_interleave_perm(n)).contiguous()

    factors: list[torch.Tensor] = []
    bond_left = 1
    mat = tensor.reshape(out_dims[0] * in_dims[0], -1)
    for k in range(n - 1):
        u, s, vh = torch.linalg.svd(mat, full_matrices=False)
        r = max(1, min(chi, int(s.numel())))
        factors.append(u[:, :r].reshape(bond_left, out_dims[k], in_dims[k], r))
        mat = s[:r].unsqueeze(1) * vh[:r]          # (r, rest)
        bond_left = r
        if k + 1 < n - 1:
            mat = mat.reshape(bond_left * out_dims[k + 1] * in_dims[k + 1], -1)
    factors.append(mat.reshape(bond_left, out_dims[-1], in_dims[-1], 1))
    return factors


@torch.no_grad()
def contract_factors(factors: list[torch.Tensor], plan: MPOPlan) -> torch.Tensor:
    """Contract MPO tensors back into a dense ``(d_out, d_in)`` matrix."""
    n = plan.n_sites
    tensor = factors[0]
    for k in range(1, n):
        tensor = torch.tensordot(tensor, factors[k], dims=([tensor.ndim - 1], [0]))
    tensor = tensor.squeeze(0).squeeze(-1)         # (o_1,i_1,...,o_N,i_N)
    tensor = tensor.permute(*_deinterleave_perm(n)).contiguous()
    return tensor.reshape(math.prod(plan.out_dims), math.prod(plan.in_dims))


@torch.no_grad()
def compress_weight(weight: torch.Tensor, plan: MPOPlan, chi: int) -> tuple[torch.Tensor, int]:
    """Return (dense reconstruction of the truncated MPO, stored MPO parameters).

    Uses the cached SVD when available (2-site MPO), otherwise runs the general
    sequential-SVD path.
    """
    params = mpo_param_count(plan.out_dims, plan.in_dims, chi)
    if plan.cached:
        r = max(1, min(chi, int(plan.s.numel())))
        approx = (plan.u[:, :r] * plan.s[:r]) @ plan.vh[:r]       # (o1*i1, o2*i2)
        approx = approx.reshape(plan.out_dims[0], plan.in_dims[0],
                                plan.out_dims[1], plan.in_dims[1])
        approx = approx.permute(*_deinterleave_perm(2)).contiguous()
        approx = approx.reshape(math.prod(plan.out_dims), math.prod(plan.in_dims))
        return approx, params
    factors = mpo_factors(weight, plan, chi)
    return contract_factors(factors, plan), params


def relative_error(original: torch.Tensor, approx: torch.Tensor) -> float:
    """Frobenius relative reconstruction error ``||W - W'||_F / ||W||_F``."""
    o = original.detach().to(torch.float32).cpu()
    denom = float(torch.linalg.norm(o))
    if denom == 0.0:
        return 0.0
    return float(torch.linalg.norm(o - approx.to(torch.float32).cpu()) / denom)
