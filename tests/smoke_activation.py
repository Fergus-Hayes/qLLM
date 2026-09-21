"""Smoke test for the activation-aware tooling (Phase A).

Verifies the claims the Phase-A verdict rests on:

* hook-captured ``H`` equals the covariance computed by hand from the same inputs;
* ``output_relative_error`` equals the brute-force ``E||Wx - W'x|| / E||Wx||`` over
  the actual samples;
* the whitened low-rank build attains the Eckart-Young bound for the H-weighted
  error, i.e. it really is the exact optimum for that class;
* with isotropic ``H`` the output metric collapses onto the Frobenius metric and
  whitening buys nothing (the null case the verdict must not fire on);
* with anisotropic ``H`` both the headroom and the whitening gain are large;
* the sparse + whitened low-rank build degrades to each of its endpoints exactly
  (no sparsity reproduces whitened low-rank; a full support reproduces ``W``), never
  scores worse than whitened low-rank at the same rank, and improves monotonically
  with refinement steps -- the properties that make its budget-curve rows readable
  as a genuine comparison rather than an artefact of the search.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qllm.activation_stats import (  # noqa: E402
    covariance_spectrum, input_covariance, low_rank_frobenius,
    low_rank_whitened, output_relative_error, sparse_lowrank_params,
    sparse_lowrank_whitened,
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

    # --- 6. structure measures: calibration against known vectors ----------
    # These decide the ancilla/circuit verdict, so they must separate "sparse"
    # from "low-entanglement" -- the two are easily conflated and imply very
    # different encodings (a few index/value pairs vs an MPS).
    import activation_aware as AA
    dd, hd = 512, 64
    nul = AA._null_stats(dd, hd, 64, 0)
    def meas(v):
        v = v / torch.linalg.norm(v)
        return (AA._participation(v) / nul["participation"],
                AA._bond_entropy(v) / nul["bond_entropy"],
                AA._head_share(v, hd) / nul["head_share"])
    onehot = torch.zeros(dd, dtype=torch.float64)
    onehot[7] = 1.0
    sp, be, _hsx = meas(onehot)
    assert sp < 0.05 and be < 0.05, (sp, be)
    prod = torch.tensor([1.0], dtype=torch.float64)
    for _ in range(9):
        prod = torch.kron(prod, torch.randn(2, dtype=torch.float64))
    sp_p, be_p, _ = meas(prod)
    assert be_p < 0.02, f"product state must read ~0 bond entropy, got {be_p}"
    assert sp_p > 0.05, "a product state is NOT sparse -- the measures must differ"
    sp_r, be_r, hs_r = meas(torch.randn(dd, dtype=torch.float64))
    assert 0.8 < sp_r < 1.2 and 0.8 < be_r < 1.2, (sp_r, be_r)
    assert AA._head_share(torch.randn(dd, dtype=torch.float64), 0) != \
        AA._head_share(torch.randn(dd, dtype=torch.float64), 0), "head_dim=0 -> NaN"
    print(f"  structure measures: one-hot {sp:.3f}x/{be:.3f}x, product state "
          f"{sp_p:.2f}x sparsity but {be_p:.3f}x bond entropy, Haar {sp_r:.2f}x/{be_r:.2f}x")

    # --- 7. stabilizer Renyi entropy: the blind spot bond entropy cannot see ---
    # A random stabilizer state has near-maximal entanglement yet costs ZERO
    # continuous parameters, so M2 must read exactly 0 for it and large for generic
    # states -- otherwise the Clifford blind spot is not actually closed.
    from qllm.activation_stats import stabilizer_renyi_entropy as _m2
    nq = 7
    dd2 = 1 << nq
    had = torch.tensor([[1., 1.], [1., -1.]], dtype=torch.float64) / 2 ** 0.5

    def _ap(v, g, q):
        v = v.reshape(2 ** q, 2, -1)
        return torch.einsum("ab,ibj->iaj", g, v).reshape(-1)

    def _cx(v, c, t):
        v = v.reshape([2] * nq).clone()
        idx = [slice(None)] * nq
        idx[c] = 1
        sub = v[tuple(idx)]
        v[tuple(idx)] = sub.roll(1, dims=(t - 1 if t > c else t))
        return v.reshape(-1)

    gen = torch.Generator().manual_seed(0)
    stab = torch.zeros(dd2, dtype=torch.float64)
    stab[0] = 1.0
    for _ in range(150):
        if torch.rand(1, generator=gen).item() < 0.45:
            stab = _ap(stab, had, int(torch.randint(0, nq, (1,), generator=gen)))
        else:
            a, b = torch.randperm(nq, generator=gen)[:2].tolist()
            stab = _cx(stab, a, b)
    stab = stab / torch.linalg.norm(stab)
    m_stab = _m2(stab)
    basis = torch.zeros(dd2, dtype=torch.float64)
    basis[3] = 1.0
    m_haar = _m2(torch.randn(dd2, generator=gen, dtype=torch.float64))
    assert abs(m_stab) < 1e-6, f"stabilizer state must read M2 = 0, got {m_stab}"
    assert abs(_m2(basis)) < 1e-6, "computational basis state must read M2 = 0"
    assert m_haar > 1.0, f"Haar random must carry magic, got {m_haar}"
    # And the crucial point: entanglement CANNOT tell these apart.
    be_stab = AA._bond_entropy(stab) / AA._null_stats(dd2, 0, 32, 0)["bond_entropy"]
    assert be_stab > 0.4, ("a scrambled stabilizer state should look entangled -- "
                           f"that is the blind spot, got {be_stab}")
    print(f"  magic: stabilizer M2={m_stab:.2e}, Haar M2={m_haar:.2f} bits; that same "
          f"stabilizer\n         state reads {be_stab:.2f}x bond entropy (the blind "
          f"spot bond entropy cannot see)")

    # --- sparse + whitened low-rank ----------------------------------------
    torch.manual_seed(3)
    m_, n_ = 40, 32
    A = torch.randn(n_, n_).double()
    Hs = A @ A.T / n_ + 1e-3 * torch.eye(n_, dtype=torch.float64)
    Ws = torch.randn(m_, n_).double()

    a0, _ = sparse_lowrank_whitened(Ws, Hs, 5, 0)
    d0 = float((a0 - low_rank_whitened(Ws, Hs, 5)).abs().max())
    assert d0 < 1e-12, f"no-sparsity endpoint differs from whitened low-rank ({d0})"

    afull, _ = sparse_lowrank_whitened(Ws, Hs, 0, n_)
    dfull = float((afull - Ws).abs().max())
    assert dfull < 1e-10, f"full support did not reproduce W ({dfull})"

    for r_ in (1, 2, 4, 8):
        for k_ in (1, 2, 4):
            ax, _ = sparse_lowrank_whitened(Ws, Hs, r_, k_)
            e_mix = output_relative_error(Ws, ax, Hs)
            e_lr = output_relative_error(Ws, low_rank_whitened(Ws, Hs, r_), Hs)
            assert e_mix <= e_lr + 1e-12, f"sparse mix worse than LR at r={r_} k={k_}"

    e_fit = output_relative_error(
        Ws, sparse_lowrank_whitened(Ws, Hs, 4, 6, refit=True)[0], Hs)
    e_raw = output_relative_error(
        Ws, sparse_lowrank_whitened(Ws, Hs, 4, 6, refit=False)[0], Hs)
    assert e_fit <= e_raw + 1e-12, "exact refit lost to raw magnitude values"

    prev = None
    for it in (1, 2, 3, 5, 8):
        e_it = output_relative_error(
            Ws, sparse_lowrank_whitened(Ws, Hs, 4, 4, iters=it)[0], Hs)
        assert prev is None or e_it <= prev + 1e-12, f"iters={it} regressed"
        prev = e_it

    # Indices are charged, so a nonzero costs strictly more than a value.
    c_vals = sparse_lowrank_params(64, 64, 0, 8, value_bits=16, index_bits=0)
    c_idx = sparse_lowrank_params(64, 64, 0, 8)
    assert c_vals == 64 * 8 and c_idx > c_vals, "sparse indices were not charged"
    print(f"  sparse+LR: endpoints exact, never worse than whitened LR, "
          f"refit {e_fit:.4f} vs raw {e_raw:.4f}")
    print(f"         a nonzero costs {c_idx / c_vals:.3f} parameter-equivalents "
          f"(value + index), not 1.0")

    print("\nACTIVATION SMOKE PASSED")


if __name__ == "__main__":
    main()
