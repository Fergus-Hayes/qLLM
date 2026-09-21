"""Activation statistics: compress for the function ``x -> Wx``, not for ``W``.

The pipeline's objective is the Frobenius reconstruction error ``||W - W'||_F``,
which treats every input direction as equally important. What actually matters is
the layer's *output* on the inputs that occur:

    E_x ||Wx - W'x||^2  =  tr[(W - W') H (W - W')^T],     H = E[x x^T]

so the whole of the input distribution enters through one ``d_in x d_in`` second
moment matrix. Weighting by ``H`` is the standard activation-aware move behind
GPTQ / SparseGPT (Hessian), AWQ, and the whitened SVD of SVD-LLM / ASVD.

Two things are worth being precise about, because they decide what is implementable:

* **Whitening is an exact reduction only for the plain low-rank class.**
  ``{W' : rank <= r}`` is closed under right multiplication by an invertible
  matrix, so "truncate ``W H^(1/2)``, then right-multiply by ``H^(-1/2)``" is
  exactly optimal (:func:`low_rank_whitened`). The MPO-with-bond-chi class is *not*
  closed under that operation, so the same two-step recipe does not carry over --
  a weighted objective has to be optimized directly there.
* **Scoring needs no matrix inverse or square root.** ``tr[D H D^T]`` is evaluated
  directly, so :func:`output_relative_error` is exact and needs no damping; only
  the whitened low-rank construction needs ``H^(-1/2)``, and that one damps.
"""

from __future__ import annotations

import math

import torch


def input_covariance(model, param_names, batches, device: str = "cpu",
                     dtype: torch.dtype = torch.float64, progress=None):
    """``H = E[x x^T]`` for the input of each named weight, via forward hooks.

    ``param_names`` are parameter names (``...weight``); the hook is placed on the
    module that owns each one. Returns ``(H_by_name, token_counts)`` with ``H``
    normalised by the number of rows (tokens) seen.
    """
    mods, acc, cnt, handles = {}, {}, {}, []
    for name in param_names:
        path = name[:-len(".weight")] if name.endswith(".weight") else name
        mods[name] = model.get_submodule(path)
        acc[name], cnt[name] = None, 0

    def _make(name):
        def hook(_mod, inp, _out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).to(dtype)
            gram = x.T @ x
            acc[name] = gram if acc[name] is None else acc[name] + gram
            cnt[name] += int(x.shape[0])
        return hook

    for name, mod in mods.items():
        handles.append(mod.register_forward_hook(_make(name)))
    try:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(batches):
                model(batch.to(device))
                if progress:
                    progress(i + 1, len(batches))
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()
    return ({n: acc[n] / max(1, cnt[n]) for n in param_names if acc[n] is not None},
            {n: cnt[n] for n in param_names})


def output_relative_error(weight: torch.Tensor, approx: torch.Tensor,
                          cov: torch.Tensor) -> float:
    """``sqrt(tr[D H D^T] / tr[W H W^T])`` with ``D = W - W'`` -- the relative error
    of the layer's *output* over the input distribution ``H``.

    Evaluated straight from ``H``: no square root, no inverse, no damping, so it is
    exact even when ``H`` is singular (which it will be -- activations are highly
    anisotropic).
    """
    w = weight.detach().to(torch.float64).cpu()
    d = w - approx.detach().to(torch.float64).cpu()
    h = cov.detach().to(torch.float64).cpu()
    num = float(((d @ h) * d).sum())
    den = float(((w @ h) * w).sum())
    if den <= 0:
        return float("nan")
    return math.sqrt(max(num, 0.0) / den)


def covariance_spectrum(cov: torch.Tensor) -> dict:
    """Effective-rank summary of ``H`` -- how many input directions actually carry
    energy. A small effective rank is what would make the ``x -> Wx`` problem
    intrinsically lower dimensional than reconstructing ``W``."""
    ev = torch.linalg.eigvalsh(cov.detach().to(torch.float64).cpu()).clamp_min(0.0)
    ev, _ = torch.sort(ev, descending=True)
    total = float(ev.sum())
    if total <= 0:
        return {"dim": int(ev.numel()), "stable_rank": float("nan")}
    frac = torch.cumsum(ev, 0) / total
    p = ev / total
    nz = p[p > 0]
    return {
        "dim": int(ev.numel()),
        "stable_rank": total / float(ev[0]),                  # tr(H) / lambda_max
        "rank90": int((frac < 0.90).sum()) + 1,
        "rank99": int((frac < 0.99).sum()) + 1,
        "participation": float(1.0 / (p ** 2).sum()),         # inverse Simpson
        "entropy_rank": float(torch.exp(-(nz * nz.log()).sum())),
        "lambda_max": float(ev[0]),
        "lambda_min": float(ev[-1]),
    }


def _sqrt_and_inv(cov: torch.Tensor, damp: float = 1e-6):
    """``(H^(1/2), H^(-1/2))`` from the eigendecomposition, with relative damping."""
    h = cov.detach().to(torch.float64).cpu()
    ev, q = torch.linalg.eigh(h)
    ev = ev.clamp_min(0.0)
    lam = damp * float(ev.max()) if float(ev.max()) > 0 else damp
    root = torch.sqrt(ev + lam)
    return (q * root) @ q.T, (q * (1.0 / root)) @ q.T


def low_rank_frobenius(weight: torch.Tensor, rank: int) -> torch.Tensor:
    """Best rank-``r`` approximation of ``W`` in Frobenius norm (plain truncated SVD)."""
    w = weight.detach().to(torch.float64).cpu()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    r = max(1, min(int(rank), s.numel()))
    return (u[:, :r] * s[:r]) @ vh[:r]


def low_rank_whitened(weight: torch.Tensor, cov: torch.Tensor, rank: int,
                      damp: float = 1e-6) -> torch.Tensor:
    """Best rank-``r`` approximation of ``W`` under the ``H``-weighted error.

    Exact for this class: rank is preserved by right multiplication with an
    invertible matrix, so truncating ``W H^(1/2)`` and mapping back with
    ``H^(-1/2)`` minimizes ``tr[(W - W') H (W - W')^T]`` over all rank-``r`` ``W'``.
    This is the SVD-LLM / ASVD whitening trick, and it is the natural reference any
    activation-aware scheme has to beat.
    """
    s_mat, s_inv = _sqrt_and_inv(cov, damp)
    w = weight.detach().to(torch.float64).cpu()
    return low_rank_frobenius(w @ s_mat, rank) @ s_inv


# --------------------------------------------------------------------------- #
# Stabilizer structure ("magic")
# --------------------------------------------------------------------------- #
# Bond entropy cannot see Clifford structure: a random stabilizer state has
# near-maximal entanglement across any cut yet needs ZERO continuous parameters.
# The stabilizer Renyi entropy closes that blind spot. For a pure state, the Pauli
# spectrum Xi_P = <psi|P|psi>^2 / d is a probability distribution; it is flat over
# the d elements of the stabilizer group for a stabilizer state (giving M2 = 0) and
# spread over all 4^n Paulis for a generic state (giving M2 > 0).
def _pauli_coefficients(rho: torch.Tensor) -> torch.Tensor:
    """All 4^n Pauli coefficients of a 2^n x 2^n matrix, by fast transform.

    ``rho = sum_P c_P P``. The change of basis factorizes over qubits, so this is
    n applications of one 4x4 map rather than a sum over 4^n Paulis -- O(d^2 log d)
    instead of O(4^n d).
    """
    d = rho.shape[0]
    n = int(round(math.log2(d)))
    # [m00, m01, m10, m11] -> [c_I, c_X, c_Y, c_Z]
    t = torch.tensor([[0.5, 0, 0, 0.5],
                      [0, 0.5, 0.5, 0],
                      [0, 0.5j, -0.5j, 0],
                      [0.5, 0, 0, -0.5]], dtype=torch.complex128)
    x = rho.to(torch.complex128).reshape([2] * (2 * n))
    x = x.permute(*[i for q in range(n) for i in (q, n + q)]).contiguous()
    x = x.reshape([4] * n)
    for ax in range(n):
        x = torch.movedim(x, ax, 0).reshape(4, -1)
        x = (t @ x).reshape([4] + [4] * (n - 1))
        x = torch.movedim(x, 0, ax)
    return x.reshape(-1)


def stabilizer_renyi_entropy(vec: torch.Tensor) -> float:
    """``M2`` of a real vector, in bits: 0 for a stabilizer state, larger for generic.

    The vector is cropped to the largest power-of-two prefix, as elsewhere. This is
    the quantity that decides whether a Clifford-like (zero continuous parameter)
    ansatz could encode the vector -- something entanglement entropy provably
    cannot detect.
    """
    q = int(math.floor(math.log2(vec.numel())))
    v = vec[: 1 << q].to(torch.float64)
    nrm = float(torch.linalg.norm(v))
    if nrm <= 0:
        return float("nan")
    v = v / nrm
    d = v.numel()
    c = _pauli_coefficients(torch.outer(v, v))
    xi = d * (c.abs() ** 2)                     # sums to 1 for a pure state
    s = float((xi ** 2).sum())
    if s <= 0:
        return float("nan")
    return float(-math.log2(s) - math.log2(d))


def magic_matched_null(vec: torch.Tensor, reps: int = 8, seed: int = 0) -> float:
    """Mean ``M2`` of vectors with the SAME magnitude profile, randomly arranged.

    A Haar null is the wrong control for magic. A spike is a computational basis
    state, hence a stabilizer state with ``M2 = 0``, so *any* sparse vector scores
    low for reasons that have nothing to do with Clifford structure -- a random
    4-sparse vector reads 0.10x of Haar. Permuting the observed magnitudes and
    randomizing signs holds sparsity fixed and destroys everything else, so the
    ratio against this null isolates genuine stabilizer structure: ~1 means the
    magic is fully explained by the magnitude profile, and well below 1 means
    there is Clifford structure beyond it.
    """
    q = int(math.floor(math.log2(vec.numel())))
    v = vec[: 1 << q].to(torch.float64)
    mags = v.abs()
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(max(1, reps)):
        perm = torch.randperm(mags.numel(), generator=g)
        sign = torch.where(torch.rand(mags.numel(), generator=g) < 0.5, -1.0, 1.0)
        vals.append(stabilizer_renyi_entropy(mags[perm] * sign.to(torch.float64)))
    return sum(vals) / len(vals)


# --------------------------------------------------------------------------- #
# Sparse + whitened low-rank
# --------------------------------------------------------------------------- #
# Low-rank is the wrong model for a matrix whose energy sits in a few coordinates:
# every diagnostic so far said the dominant whitened subspace is generic in
# entanglement and in magic but *sparse*, and sparsity is the one structure a
# factorization cannot express. ``W' = L + S`` with ``L`` rank r and ``S`` row-sparse
# is the class that can hold both, and it is purely classical.
def sparse_lowrank_params(rows: int, cols: int, rank: int, nnz_per_row: int,
                          value_bits: int = 16, index_bits: int | None = None) -> float:
    """Parameter-equivalent cost of ``L + S``, charging sparse indices honestly.

    A nonzero is not one number: it is a value plus the index saying where it goes.
    Counting only values would give sparsity a free ride exactly the way counting a
    quantum gate as free would, so each nonzero costs ``1 + index_bits/value_bits``
    parameter-equivalents. With a per-row support the index is a column id, so
    ``index_bits = ceil(log2(cols))`` (10 bits against 16-bit values at cols=576,
    i.e. 1.625x per nonzero). The low-rank part carries no indices.
    """
    if index_bits is None:
        index_bits = max(1, int(math.ceil(math.log2(max(2, cols)))))
    nnz = rows * max(0, int(nnz_per_row))
    return rank * (rows + cols) + nnz * (1.0 + index_bits / float(value_bits))


def _row_sparse_fit(resid: torch.Tensor, cov: torch.Tensor, nnz_per_row: int,
                    damp: float = 1e-6, refit: bool = True) -> torch.Tensor:
    """Best row-sparse ``S`` for ``min ||(R - S) H^(1/2)||_F``, support then values.

    Support is picked by the diagonal saliency ``|R_ij| * sqrt(H_jj)`` -- the leading
    term of the weighted objective, and the same criterion SparseGPT/OBS use. On that
    fixed support the values are then solved *exactly*: stationarity of
    ``(r - s) H (r - s)^T`` gives ``H[O,O] s_O^T = (R H)_O^T`` per row, which is a
    small dense system because the support is small. A row keeps the refit only if it
    measurably lowers that row's own error, so the step can never be worse than
    plain magnitude values.
    """
    m, n = resid.shape
    k = max(0, min(int(nnz_per_row), n))
    out = torch.zeros_like(resid)
    if k == 0:
        return out
    diag = torch.diagonal(cov).clamp_min(0.0).sqrt()
    idx = torch.topk(resid.abs() * diag, k, dim=1).indices           # (m, k)
    raw = torch.gather(resid, 1, idx)
    if not refit:
        return out.scatter(1, idx, raw)

    rhs = torch.gather(resid @ cov, 1, idx)                          # (m, k)
    lam = damp * float(torch.diagonal(cov).max().clamp_min(0.0))
    eye = torch.eye(k, dtype=cov.dtype) * (lam if lam > 0 else damp)
    fit = torch.empty_like(raw)
    blk = max(1, int(2e6 // max(1, k * k)))                          # bound the gather
    for b in range(0, m, blk):
        ii = idx[b:b + blk]
        sub = cov[ii.unsqueeze(2), ii.unsqueeze(1)] + eye            # (B, k, k)
        try:
            sol = torch.linalg.solve(sub, rhs[b:b + blk].unsqueeze(2))
        except Exception:                                            # noqa: BLE001
            sol = torch.linalg.lstsq(sub, rhs[b:b + blk].unsqueeze(2)).solution
        fit[b:b + blk] = sol.squeeze(2)

    # Per-row acceptance: keep whichever of {refit, raw} actually lowers that row.
    err = {}
    for tag, vals in (("fit", fit), ("raw", raw)):
        d = resid - torch.zeros_like(resid).scatter(1, idx, vals)
        err[tag] = ((d @ cov) * d).sum(1)
    keep = (err["fit"] <= err["raw"]).unsqueeze(1)
    return out.scatter(1, idx, torch.where(keep, fit, raw))


def sparse_lowrank_whitened(weight: torch.Tensor, cov: torch.Tensor, rank: int,
                            nnz_per_row: int, damp: float = 1e-6, iters: int = 3,
                            refit: bool = True):
    """``W ~= L + S``, ``L`` rank-``r`` whitened, ``S`` row-sparse, under the H-weighted error.

    Alternating minimisation. Each half-step is the exact optimum of its own block
    (:func:`low_rank_whitened` for ``L`` given ``S``, :func:`_row_sparse_fit`'s
    solve for the values of ``S`` given ``L``); only the support selection is a
    heuristic, so every intermediate is scored and the best is returned. ``S`` starts
    at zero, which makes the first candidate exactly :func:`low_rank_whitened` --
    the result therefore can never be worse than the whitened low-rank reference.

    Returns ``(approx, info)`` where ``info`` carries the achieved weighted error and
    which iteration produced it.
    """
    w = weight.detach().to(torch.float64).cpu()
    h = cov.detach().to(torch.float64).cpu()
    m, n = int(w.shape[0]), int(w.shape[1])
    r = max(0, min(int(rank), min(m, n)))
    k = max(0, min(int(nnz_per_row), n))

    def score(a):
        d = w - a
        return float(((d @ h) * d).sum())

    best, best_err, best_it = None, float("inf"), -1
    s = torch.zeros_like(w)
    for it in range(max(1, int(iters))):
        lo = low_rank_whitened(w - s, h, r, damp) if r > 0 else torch.zeros_like(w)
        for cand in (lo + s,):
            e = score(cand)
            if e < best_err:
                best, best_err, best_it = cand, e, it
        if k == 0:
            break
        s = _row_sparse_fit(w - lo, h, k, damp, refit)
        cand = lo + s
        e = score(cand)
        if e < best_err:
            best, best_err, best_it = cand, e, it
    return best, {"weighted_err_sq": best_err, "iter": best_it,
                  "rank": r, "nnz_per_row": k}
