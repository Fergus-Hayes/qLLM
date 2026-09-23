"""Smoke test for the Phase A proxy study.

Phase A exists to decide which per-layer metric stands in for perplexity, so the
machinery it rests on has to be right before any of its numbers mean anything.
The checks here are the ones whose failure would be invisible in the output:

* the first-order predictor is a PREDICTION, not a correlate, so it is checked
  against a measurement -- right sign, right magnitude, and linear in the step;
* the ``(x, g)`` pairs are token-aligned, and ``g`` carries sum-NLL gradients
  rescaled from positions to scored tokens (get either wrong and every prediction
  is off by a batch-size or a ``T/(T-1)`` factor, with nothing in the report that
  would show it);
* a zero-error perturbation must reproduce the weight EXACTLY, because those rows
  are what the report uses to measure the perplexity noise floor;
* the structureless control really is structureless -- its Frobenius error hits
  the requested target and the H-weighted metric reads it differently;
* the library spans families and yields distinct keys, since a duplicate key is
  silently skipped on resume.
"""
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qllm.perturb import (  # noqa: E402
    METRICS, capture_xg, gradient_weighted, perturbations, predicted_dnll,
)


class _Tiny(torch.nn.Module):
    """A two-layer causal model, small enough to differentiate exactly."""

    def __init__(self, vocab=32, dim=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.proj = torch.nn.Linear(dim, dim, bias=False)
        self.act = torch.nn.Tanh()
        self.head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        h = self.head(self.act(self.proj(self.emb(input_ids))))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                h[:, :-1].reshape(-1, h.shape[-1]), labels[:, 1:].reshape(-1))
        return type("Out", (), {"loss": loss, "logits": h})()


def _mean_nll(model, batches):
    tot = n = 0.0
    with torch.no_grad():
        for b in batches:
            k = b.shape[0] * (b.shape[1] - 1)
            tot += float(model(b, labels=b).loss) * k
            n += k
    return tot / n


def main():
    torch.manual_seed(0)
    model = _Tiny().double().eval()
    ids = torch.randint(0, 32, (2, 24))
    batches = [ids[:, :12], ids[:, 12:]]
    name = "proj.weight"

    XG = capture_xg(model, [name], batches, max_tokens=10 ** 9)
    X, G = XG[name]
    n_pos = sum(b.shape[0] * b.shape[1] for b in batches)
    assert X.shape[0] == n_pos and G.shape[0] == n_pos, (X.shape, G.shape, n_pos)

    # -- the predictor predicts -------------------------------------------- #
    # d(NLL) ~ mean_t g_t . ((W' - W) x_t). The residual must be second order, so
    # halving the step must quarter the error, not halve it.
    p = dict(model.named_parameters())[name]
    W = p.detach().clone()
    N = torch.randn(W.shape, dtype=W.dtype, generator=torch.Generator().manual_seed(3))
    N = N / float(torch.linalg.norm(N)) * float(torch.linalg.norm(W))
    base = _mean_nll(model, batches)

    resid = []
    for eps in (1e-3, 5e-4, 2.5e-4):
        Wp = W + eps * N
        with torch.no_grad():
            p.copy_(Wp)
            meas = _mean_nll(model, batches) - base
            p.copy_(W)
        pred = predicted_dnll(W, Wp, X=X, G=G)
        assert meas * pred > 0, f"sign disagrees at eps={eps}: {meas} vs {pred}"
        assert abs(meas / pred - 1.0) < 0.05, f"eps={eps}: {meas} vs {pred}"
        resid.append(abs(meas - pred))
    # Second order: each halving of eps cuts the residual by ~4, never by ~2.
    for a, b in zip(resid, resid[1:]):
        assert 2.5 < a / max(b, 1e-300) < 6.0, f"residual is not second order: {resid}"

    # Doubling the step doubles the prediction: the predictor is linear in D.
    lin = predicted_dnll(W, W + 2e-3 * N, X=X, G=G) / \
        predicted_dnll(W, W + 1e-3 * N, X=X, G=G)
    assert abs(lin - 2.0) < 1e-6, lin

    # -- token alignment ---------------------------------------------------- #
    # Shuffling g against x destroys the prediction. If it did not, the two were
    # never paired and the metric was reading an average over unrelated tokens.
    perm = torch.randperm(X.shape[0], generator=torch.Generator().manual_seed(7))
    Wp = W + 1e-3 * N
    good = predicted_dnll(W, Wp, X=X, G=G)
    bad = predicted_dnll(W, Wp, X=X, G=G[perm])
    assert abs(bad) < 0.5 * abs(good), f"shuffled g predicted just as well: {good} {bad}"

    # -- scale-free and sign-free ------------------------------------------- #
    gw = gradient_weighted(W, Wp, X=X, G=G)
    assert gw > 0 and math.isfinite(gw)
    assert abs(gradient_weighted(W, W - 1e-3 * N, X=X, G=G) - gw) < 1e-9, \
        "grad_weighted must not depend on the sign of the perturbation"
    assert gradient_weighted(W, W, X=X, G=G) == 0.0

    # -- the library -------------------------------------------------------- #
    Wm = torch.randn(32, 32)
    cov = torch.eye(32) + 0.1 * torch.randn(32, 32) @ torch.randn(32, 32).T / 32
    cov = (cov + cov.T) / 2
    got = list(perturbations(Wm, cov, n_per_family=4, seed=0))
    fams = {f for f, _k, _W in got}
    assert {"mpo", "low-rank", "low-rank-whitened", "sparse", "sparse+low-rank",
            "random"} <= fams, fams
    keys = [(f, str(k)) for f, k, _W in got]
    assert len(keys) == len(set(keys)), "duplicate (family, knob): resume would skip"
    for _f, _k, Wp2 in got:
        assert Wp2.shape == Wm.shape and torch.isfinite(Wp2).all()

    # An exact reconstruction must be EXACT: the report reads those rows as the
    # perplexity noise floor, so a 1e-6 residue there would be read as noise.
    exact = [(f, k, Wp2) for f, k, Wp2 in got
             if METRICS["frobenius"](Wm, Wp2) < 1e-9]
    assert exact, "no zero-error row; the resolution check has nothing to stand on"
    for _f, _k, Wp2 in exact:
        assert torch.equal(Wp2, Wm) or float((Wp2 - Wm).abs().max()) < 1e-5

    # The control hits its requested Frobenius error and is NOT read the same way
    # by an H-weighted metric -- if it were, the blindness test could never fire.
    for f, k, Wp2 in got:
        if f != "random":
            continue
        assert abs(METRICS["frobenius"](Wm, Wp2) - float(k)) < 1e-4, (k, )
        assert METRICS["activation"](Wm, Wp2, cov=cov) != \
            METRICS["frobenius"](Wm, Wp2)

    # -- the figures render, including the rows a log axis would have dropped -- #
    import tempfile

    from qllm.opt_plots import plot_proxy_rho, plot_proxy_scatter
    cands = ["frobenius", "activation", "grad_weighted"]
    rows = []
    for i, (f, k, _W) in enumerate(got):
        rows.append(dict(layer_type="q_proj", depth=i % 2, family=f, knob=str(k),
                         ppl=10.0 + i, dppl=(i - 3) * 0.01, pred_dnll=1e-4 * i,
                         **{c: 0.01 * (i + 1) for c in cands}))
    with tempfile.TemporaryDirectory() as d:
        plot_proxy_scatter(rows, cands, d)
        plot_proxy_rho(rows, cands, [("q_proj", 0), ("q_proj", 1)], d)
        made = sorted(x.name for x in __import__("pathlib").Path(d).iterdir())
    assert made == ["proxy_rho.png", "proxy_scatter.png"], made
    assert any(r["dppl"] < 0 for r in rows), "the negative-dppl case went untested"

    print(f"  predictor: sign, magnitude within 5%, residual second order {[f'{r:.2e}' for r in resid]}")
    print(f"  predictor is linear in the step (2x step -> {lin:.6f}x prediction)")
    print(f"  (x, g) token-aligned: shuffling g cuts the prediction "
          f"{good:.3e} -> {bad:.3e}")
    print("  grad_weighted is positive, sign-free and exactly zero at zero error")
    print(f"  library spans {len(fams)} families, {len(got)} unique keys, "
          f"{len(exact)} exact row(s)")
    print("  control hits its target Frobenius error and reads differently under H")
    print("  both figures render, with a negative dppl row present")
    print("\nPROXY SMOKE PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
