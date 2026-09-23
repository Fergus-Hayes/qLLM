"""A deliberately diverse library of layer perturbations, and proxy metrics for them.

The question Phase A asks is which per-layer metric predicts perplexity. That can
only be answered over perturbations that arrive by DIFFERENT mechanisms. Sweep one
compression family's one knob and a metric can look excellent merely because both
it and perplexity move monotonically with that knob -- the study would be measuring
the knob, not the metric.

So the library spans MPO, hybrid, plain and whitened low-rank, row-sparse,
sparse-plus-low-rank, and -- the control that makes the rest interpretable -- a
STRUCTURELESS random perturbation at matched Frobenius error. If a metric ranks a
structured and a random perturbation of equal magnitude the same while their
perplexity damage differs, that metric is blind to what matters.
"""

from __future__ import annotations

import math

import torch

from .activation_stats import (low_rank_frobenius, low_rank_whitened,
                               sparse_lowrank_whitened)
from .compactifai import log_spaced_ints, relative_error
from .qubit_mpo import make_plan, plan_compress


def _ranks(W, n):
    return sorted(set(log_spaced_ints(1, min(W.shape), n)))


def perturbations(W, cov=None, n_per_family=6, seed=0, hybrid=None, progress=None):
    """Yield ``(family, knob, W')`` over every family the library covers.

    ``hybrid`` is an optional ``(ansatz_name, kwargs)`` pair; it is the only family
    that needs an optimiser, so it is opt-in.
    """
    W = W.detach().float()
    m, n = W.shape
    gen = torch.Generator().manual_seed(seed)

    plan = make_plan(W, "qubit", 0)
    for chi in sorted(set(log_spaced_ints(1, plan.max_chi, n_per_family))):
        yield "mpo", chi, plan_compress(W, plan, chi)[0]

    for r in _ranks(W, n_per_family):
        yield "low-rank", r, low_rank_frobenius(W, r).float()
        if cov is not None:
            yield "low-rank-whitened", r, low_rank_whitened(W, cov, r).float()

    if cov is not None:
        for k in sorted(set(log_spaced_ints(1, max(2, n // 2), n_per_family))):
            yield "sparse", k, sparse_lowrank_whitened(W, cov, 0, k)[0].float()
        for r in _ranks(W, max(2, n_per_family // 2)):
            for k in (1, max(2, n // 16)):
                yield ("sparse+low-rank", f"r{r}k{k}",
                       sparse_lowrank_whitened(W, cov, r, k)[0].float())

    # The structureless control: same Frobenius error, no structure whatsoever.
    # Its job is to break any metric that only sees error magnitude.
    nrm = float(torch.linalg.norm(W))
    for target in (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)[:n_per_family]:
        N = torch.randn(m, n, generator=gen)
        N = N / float(torch.linalg.norm(N)) * nrm * target
        yield "random", round(target, 3), (W + N)

    if hybrid is not None:
        from .disentangler import disentangle, hybrid_weight
        name, kw = hybrid
        for chi in sorted(set(log_spaced_ints(1, plan.max_chi, n_per_family))):
            r = disentangle(W, target_chi=chi, n_sites=2, tensorization="qubit", **kw)
            yield f"hybrid:{name}", chi, hybrid_weight(r, chi)[0].float()
            if progress:
                progress(name, chi)


# --------------------------------------------------------------------------- #
# Proxy metrics
# --------------------------------------------------------------------------- #
# A note on one metric that is NOT here. "The layer's output error sampled over
# real activations", E||Wx - W'x|| / E||Wx||, is not a separate metric: with
# H = E[x x^T] it equals the H-weighted error exactly, by definition of H. It was
# in the design and is dropped as redundant rather than reported twice.
def frobenius(W, Wp, **_):
    return float(relative_error(W, Wp))


def activation(W, Wp, cov=None, **_):
    """``sqrt(tr[D H D^T] / tr[W H W^T])`` -- input-side weighting."""
    from .activation_stats import output_relative_error
    return output_relative_error(W, Wp, cov)


def gradient_weighted(W, Wp, X=None, G=None, **_):
    """RMS first-order loss perturbation, relative to the layer's own contribution.

    ``H`` says which INPUT directions carry energy. It says nothing about which
    OUTPUT directions the rest of the network is sensitive to, and an output error
    in a direction the next layer ignores costs nothing. This closes that side:
    with ``g = dL/dy`` captured per token, ``g . (D x)`` is the first-order change
    in the loss that this layer's error causes, and its RMS over tokens is a
    magnitude that cannot cancel.

    Computed from sampled ``(x, g)`` pairs rather than from a factorised
    ``E[g g^T]``, so it needs no independence assumption between ``x`` and ``g``.
    """
    D = (W - Wp).double()
    num = ((X @ D.T) * G).sum(1)                 # g . (D x) per token
    den = ((X @ W.double().T) * G).sum(1)
    rms_d = float(torch.sqrt((num ** 2).mean()))
    rms_w = float(torch.sqrt((den ** 2).mean()))
    return rms_d / rms_w if rms_w > 0 else float("nan")


def predicted_dnll(W, Wp, X=None, G=None, **_):
    """First-order predicted change in mean NLL: ``mean_t g_t . (D x_t)``.

    Signed and unnormalised on purpose. Unlike every other metric here this is not
    a relative error but a PREDICTION in the units of the thing being measured, so
    it can be checked against the measured perplexity change rather than merely
    correlated with it: ``d(ppl) ~ ppl_0 * d(NLL)``.

    The difference runs ``Wp - W``, the direction the weight actually moves, so a
    positive value predicts a loss INCREASE. Taking it the other way round is not a
    cosmetic sign slip: it would put every damaging perturbation on the negative
    half of the calibration plot, where the log-log check silently discards it.
    """
    D = (Wp - W).double()
    return float(((X @ D.T) * G).sum(1).mean())


METRICS = {
    "frobenius": frobenius,
    "activation": activation,
    "activation_val": activation,          # same function, held-out H supplied
    "grad_weighted": gradient_weighted,
    "pred_dnll": predicted_dnll,
}


# --------------------------------------------------------------------------- #
def capture_xg(model, param_names, batches, device="cpu", max_tokens=4096, seed=0):
    """Sampled ``(x, g)`` pairs per layer: inputs, and the loss gradient at outputs.

    ``x`` comes from a forward hook, ``g = dL/dy`` from a backward hook on the same
    module, so the two are token-aligned by construction.

    The backward is taken on the SUMMED loss, not the mean, so ``g`` carries no
    1/N factor. Getting that wrong would scale every prediction by the batch size
    and quietly destroy the calibration check, which is the one test here that
    compares a prediction to a measurement rather than ranking two columns.

    One further factor, small but not zero: the layer sees every token position,
    while the loss scores only ``T - 1`` of them per sequence. So ``mean_t`` over
    positions recovers the sum divided by the position count, not by the scored
    count, and ``g`` is rescaled by their ratio at the end. At a 512-token window
    that is 0.2%; at a short window, or with ragged batches, it is not.

    Tokens are subsampled per batch to cap memory: a layer of width 576 over 16k
    tokens is ~37 MB per side, per layer.
    """
    mods, xs, gs, handles = {}, {}, {}, []
    for name in param_names:
        path = name[:-len(".weight")] if name.endswith(".weight") else name
        mods[name] = model.get_submodule(path)
        xs[name], gs[name] = [], []

    def _fwd(name):
        def hook(_m, inp, _out):
            x = inp[0].detach()
            xs[name].append(x.reshape(-1, x.shape[-1]).double().cpu())
        return hook

    def _bwd(name):
        def hook(_m, _gi, go):
            g = go[0].detach()
            gs[name].append(g.reshape(-1, g.shape[-1]).double().cpu())
        return hook

    for name, mod in mods.items():
        handles.append(mod.register_forward_hook(_fwd(name)))
        handles.append(mod.register_full_backward_hook(_bwd(name)))
    n_pos = n_scored_all = 0
    try:
        model.eval()
        for batch in batches:
            ids = batch.to(device)
            model.zero_grad(set_to_none=True)
            out = model(ids, labels=ids)
            n_scored = ids.shape[0] * (ids.shape[1] - 1)
            (out.loss * n_scored).backward()        # sum-NLL gradients
            n_pos += ids.shape[0] * ids.shape[1]
            n_scored_all += n_scored
        model.zero_grad(set_to_none=True)
    finally:
        for h in handles:
            h.remove()

    gen = torch.Generator().manual_seed(seed)
    out = {}
    for name in param_names:
        if not xs[name] or not gs[name]:
            continue
        X, G = torch.cat(xs[name]), torch.cat(gs[name])
        n = min(X.shape[0], G.shape[0])
        X, G = X[:n], G[:n]
        if n > max_tokens:                          # cap memory, keep it unbiased
            idx = torch.randperm(n, generator=gen)[:max_tokens]
            X, G = X[idx], G[idx]
        # positions-per-scored-token, so that a mean over sampled POSITIONS is an
        # unbiased estimate of the change in mean NLL over SCORED tokens.
        if n_scored_all:
            G = G * (n_pos / n_scored_all)
        out[name] = (X.contiguous(), G.contiguous())
    return out
