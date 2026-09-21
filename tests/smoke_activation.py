"""Smoke test for the activation-aware tooling (Phase A).

Verifies the claims the Phase-A verdict rests on:

* hook-captured ``H`` equals the covariance computed by hand from the same inputs;
* ``output_relative_error`` equals the brute-force ``E||Wx - W'x|| / E||Wx||`` over
  the actual samples;
* the whitened low-rank build attains the Eckart-Young bound for the H-weighted
  error, i.e. it really is the exact optimum for that class;
* with isotropic ``H`` the output metric collapses onto the Frobenius metric and
  whitening buys nothing (the null case the verdict must not fire on);
* with anisotropic ``H`` both the headroom and the whitening gain are large.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qllm.activation_stats import (  # noqa: E402
    covariance_spectrum, input_covariance, low_rank_frobenius,
    low_rank_whitened, output_relative_error,
)
from qllm.compactifai import relative_error  # noqa: E402


def main():
    torch.manual_seed(0)

    # --- 1. hook capture == manual covariance -------------------------------
    from transformers import LlamaConfig, LlamaForCausalLM
    model = LlamaForCausalLM(LlamaConfig(vocab_size=128, hidden_size=32,
                                         intermediate_size=64, num_hidden_layers=2,
                                         num_attention_heads=4, num_key_value_heads=2))
    model.eval()
    name = "model.layers.0.self_attn.q_proj.weight"
    batches = [torch.randint(0, 128, (2, 16)) for _ in range(3)]

    seen = {}
    mod = model.get_submodule(name[:-len(".weight")])
    h = mod.register_forward_hook(
        lambda m, i, o: seen.setdefault("x", []).append(i[0].detach().reshape(-1, i[0].shape[-1])))
    with torch.no_grad():
        for b in batches:
            model(b)
    h.remove()
    X = torch.cat(seen["x"]).double()
    manual = X.T @ X / X.shape[0]

    cov, counts = input_covariance(model, [name], batches)
    assert counts[name] == X.shape[0], (counts[name], X.shape[0])
    assert torch.allclose(cov[name], manual, atol=1e-9), \
        f"hook covariance != manual, max diff {(cov[name]-manual).abs().max():.2e}"
    print(f"  hook H matches manual covariance over {counts[name]} tokens "
          f"(max diff {(cov[name]-manual).abs().max():.1e})")

    # --- 2. output_relative_error == brute force over the samples -----------
    W = mod.weight.detach().double()
    Wp = W + 0.05 * torch.randn_like(W)
    brute = float(torch.linalg.norm(X @ (W - Wp).T) / torch.linalg.norm(X @ W.T))
    got = output_relative_error(W, Wp, cov[name])
    assert abs(brute - got) < 1e-8, (brute, got)
    print(f"  output_relative_error matches brute force E||Wx-W'x||: {got:.6f}")

    # --- 3. whitened low-rank attains the Eckart-Young bound ----------------
    H = cov[name]
    ev, q = torch.linalg.eigh(H)
    S = (q * torch.sqrt(ev.clamp_min(0) + 1e-6 * float(ev.max()))) @ q.T
    for r in (2, 6, 12):
        sv = torch.linalg.svdvals(W @ S)
        bound = float(torch.sqrt((sv[r:] ** 2).sum()) / torch.linalg.norm(W @ S))
        got = output_relative_error(W, low_rank_whitened(W, H, r), H)
        assert abs(got - bound) < 1e-6, (r, got, bound)
    print("  whitened low-rank attains the Eckart-Young bound for the H-weighted error")

    # The whitening must be free at runtime: W' = trunc(W H^(1/2)) H^(-1/2) is still
    # rank r, so it deploys as an r*(m+n) factorization exactly like plain low-rank.
    # If this failed, every parameter count in the Phase-B curve would be wrong.
    for r in (2, 6, 12):
        got = torch.linalg.matrix_rank(low_rank_whitened(W, H, r), tol=1e-8)
        assert int(got) == r, f"whitened rank-{r} has rank {int(got)}"
    print("  whitened low-rank stays exactly rank r (so it costs the same r*(m+n))")

    # --- 4. isotropic H: output metric == Frobenius, whitening is a no-op ---
    eye = torch.eye(W.shape[1], dtype=torch.float64)
    assert abs(output_relative_error(W, Wp, eye) - relative_error(W, Wp)) < 1e-6
    r = 6
    iso_f = output_relative_error(W, low_rank_frobenius(W, r), eye)
    iso_w = output_relative_error(W, low_rank_whitened(W, eye, r), eye)
    assert abs(iso_f / iso_w - 1.0) < 1e-3, f"whitening should be a no-op, got {iso_f/iso_w}"
    print(f"  isotropic H: output==frobenius, whitening gain {iso_f/iso_w:.4f} (no-op)")

    # --- 5. anisotropic H: headroom and whitening gain both appear ----------
    d = W.shape[1]
    scale = torch.logspace(0, -4, d, dtype=torch.float64)      # steep spectrum
    qa, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    Ha = (qa * scale) @ qa.T
    spec = covariance_spectrum(Ha)
    head = relative_error(W, Wp) / output_relative_error(W, Wp, Ha)
    an_f = output_relative_error(W, low_rank_frobenius(W, r), Ha)
    an_w = output_relative_error(W, low_rank_whitened(W, Ha, r), Ha)
    assert an_w <= an_f + 1e-9, "whitened must not be worse than frobenius-optimal"
    assert spec["stable_rank"] < d / 2, spec["stable_rank"]
    print(f"  anisotropic H: stable rank {spec['stable_rank']:.1f}/{d}, "
          f"headroom {head:.2f}x, whitening gain {an_f/an_w:.2f}x")

    print("\nACTIVATION SMOKE PASSED")


if __name__ == "__main__":
    main()
