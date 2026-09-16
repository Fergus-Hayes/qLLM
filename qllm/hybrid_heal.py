"""Healing for the hybrid (circuits + MPO) layer, via the model's own LM loss.

CompactifAI healing (``compactifai_heal.py``) retrains the two MPO tensors of a
truncated layer against the language-model loss while every other layer stays
dense. Here we do the same for the *hybrid* layer -- the disentangling circuits
``U``/``V`` plus the chi'-bonded MPO core -- with two granularities:

* ``core`` -- train only the MPO core (the chi' bond), the circuits frozen. This
  is the direct analogue of CompactifAI healing, with ``C(chi')`` trainable
  parameters.
* ``full`` -- train the MPO core **and** the circuit gates. The gates stay
  genuine orthogonal ``k``-qubit gates (they are optimized on the Lie algebra,
  ``g = g0 . expm(skew(theta))`` with ``theta`` initialised to zero), so the
  count of trainable circuit parameters is exactly ``Q(D)`` and the healed layer
  still costs ``M* = C(chi') + Q(D)``. This gives the starved small-chi' core the
  extra task-trainable degrees of freedom the circuits carry.

Both start exactly at the cold (un-healed) reconstruction, so step 0 reproduces
``hybrid_weight(result, chi)`` and healing can only improve on it. The effective
weight is recomputed from the trainable pieces on every forward, differentiably,
exactly as :func:`~qllm.disentangler.hybrid_weight` builds it:
``W' = crop( U . contract(core) . V^T )``.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .compactifai import _deinterleave_perm, mpo_factors
from .compactifai_heal import _resolve_module, make_heal_batches  # noqa: F401  (re-exported)
from .disentangler import (
    DisentangleResult,
    Gate,
    _gate_from_angles,
    apply_circuit,
    apply_right,
)

HEAL_MODES = ("core", "full")


def _angle_dim(k: int) -> int:
    """Number of ``so(2^k)`` angles carried by one ``k``-qubit gate."""
    d = 1 << k
    return d * (d - 1) // 2


def _contract_core(factors: list[torch.Tensor], plan) -> torch.Tensor:
    """Differentiable MPO contraction -> dense ``(prod out, prod in)`` matrix.

    Mirrors :func:`qllm.compactifai.contract_factors`, but without its
    ``@torch.no_grad()`` guard so gradients reach the (trainable) core tensors.
    """
    n = plan.n_sites
    tensor = factors[0]
    for k in range(1, n):
        tensor = torch.tensordot(tensor, factors[k], dims=([tensor.ndim - 1], [0]))
    tensor = tensor.squeeze(0).squeeze(-1)
    tensor = tensor.permute(*_deinterleave_perm(n)).contiguous()
    return tensor.reshape(math.prod(plan.out_dims), math.prod(plan.in_dims))


class HybridAdapter(nn.Module):
    """Trainable stand-in for a hybrid-compressed layer (circuits + MPO core).

    The effective weight ``W' = crop(U . contract(core) . V^T)`` has the same
    shape and orientation as the wrapped module's ``weight`` (``nn.Linear`` weight
    ``[out, in]``, forward ``x W^T``; GPT-2 ``Conv1D`` weight ``[in, out]``,
    forward ``x W``), so the adapter is a drop-in replacement. At construction it
    reproduces ``hybrid_weight(result, chi)`` exactly.
    """

    def __init__(self, module: nn.Module, result: DisentangleResult, chi: int,
                 mode: str):
        super().__init__()
        if mode not in HEAL_MODES:
            raise ValueError(f"mode must be one of {HEAL_MODES}, got {mode!r}")
        self.mode = mode
        self.plan = result.plan
        self.n_out_qubits = result.n_out_qubits
        self.n_in_qubits = result.n_in_qubits
        self.d_out, self.d_in = result.shape
        self.weight_shape = tuple(int(d) for d in module.weight.shape)
        self.is_conv1d = type(module).__name__ == "Conv1D"
        device = module.weight.device

        # MPO core: the chi'-bonded factors of the disentangled operator. Always
        # trainable; its element count is exactly C(chi').
        factors = mpo_factors(result.operator, self.plan, chi)
        self.core = nn.ParameterList(
            nn.Parameter(f.detach().to(device=device, dtype=torch.float32))
            for f in factors)

        # Circuit gates: base orthogonal matrices from the disentangler, kept
        # fixed in ``core`` mode; in ``full`` mode multiplied by expm(skew(theta))
        # with trainable ``theta`` (initialised to zero -> starts at the base gate).
        self._u_pos = [(g.start, g.k) for g in result.u_gates]
        self._v_pos = [(g.start, g.k) for g in result.v_gates]
        self._u_base = [g.matrix.detach().to(device=device, dtype=torch.float32)
                        for g in result.u_gates]
        self._v_base = [g.matrix.detach().to(device=device, dtype=torch.float32)
                        for g in result.v_gates]
        if mode == "full":
            self.theta_u = nn.ParameterList(
                nn.Parameter(torch.zeros(_angle_dim(k), device=device))
                for _s, k in self._u_pos)
            self.theta_v = nn.ParameterList(
                nn.Parameter(torch.zeros(_angle_dim(k), device=device))
                for _s, k in self._v_pos)
        else:
            self.theta_u = self.theta_v = None

        bias = getattr(module, "bias", None)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    # -- circuit assembly -------------------------------------------------- #
    def _gates(self, base, pos, thetas) -> list[Gate]:
        if self.mode == "core" or thetas is None:
            return [Gate(s, k, m) for m, (s, k) in zip(base, pos)]
        return [Gate(s, k, m @ _gate_from_angles(t, k))
                for m, (s, k), t in zip(base, pos, thetas)]

    def effective_weight(self) -> torch.Tensor:
        truncated = _contract_core(list(self.core), self.plan)     # padded, differentiable
        u_gates = self._gates(self._u_base, self._u_pos, self.theta_u)
        v_gates = self._gates(self._v_base, self._v_pos, self.theta_v)
        dense = apply_right(v_gates,
                            apply_circuit(u_gates, truncated, self.n_out_qubits),
                            self.n_in_qubits)
        return dense[:self.d_out, :self.d_in].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        if self.is_conv1d:
            out = x.reshape(-1, x.shape[-1]) @ w
            if bias is not None:
                out = out + bias
            return out.reshape(*x.shape[:-1], w.shape[1])
        return nn.functional.linear(x, w, bias)

    # -- bookkeeping ------------------------------------------------------- #
    def trainable(self) -> list[nn.Parameter]:
        params = list(self.core)
        if self.mode == "full":
            params = params + list(self.theta_u) + list(self.theta_v)
        return params

    @property
    def n_params(self) -> int:
        """Stored parameter count: ``C(chi')`` (core) or ``M*`` (full)."""
        core = sum(int(p.numel()) for p in self.core)
        if self.mode == "core":
            return core
        angles = sum(int(t.numel()) for t in self.theta_u) + \
            sum(int(t.numel()) for t in self.theta_v)
        return core + angles


def heal_hybrid(
    model,
    param_name: str,
    result: DisentangleResult,
    chi: int,
    mode: str,
    heal_batches: list[torch.Tensor],
    device: str,
    steps: int,
    lr: float,
    log: bool = False,
) -> tuple[torch.Tensor, int, float]:
    """Swap one layer to a trainable :class:`HybridAdapter`, heal it, return its weight.

    Every other model parameter is frozen; only the adapter's trainable pieces
    (the MPO core, plus the circuit angles in ``full`` mode) are optimized by Adam
    to minimize the LM loss on ``heal_batches``. Returns
    ``(healed_dense_weight, n_params, final_loss)``; the original module is always
    restored before returning.
    """
    parent, attr, original = _resolve_module(model, param_name)
    adapter = HybridAdapter(original, result, chi, mode).to(device)
    setattr(parent, attr, adapter)

    frozen = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in frozen:
        p.requires_grad_(False)
    trainable = adapter.trainable()
    for p in trainable:
        p.requires_grad_(True)

    opt = torch.optim.Adam(trainable, lr=lr)
    was_training = model.training
    model.eval()
    final_loss = float("nan")
    try:
        step = 0
        while step < steps:
            for batch in heal_batches:
                if step >= steps:
                    break
                ids = batch.to(device)
                opt.zero_grad(set_to_none=True)
                loss = model(ids, labels=ids).loss
                loss.backward()
                opt.step()
                final_loss = float(loss.detach())
                step += 1
                if log and (step == 1 or step % max(1, steps // 4) == 0 or step == steps):
                    print(f"        heal[{mode}] step {step}/{steps}  loss={final_loss:.4f}")
        with torch.no_grad():
            healed = adapter.effective_weight().detach().to("cpu", torch.float32)
        return healed, adapter.n_params, final_loss
    finally:
        setattr(parent, attr, original)
        for p, req in frozen:
            p.requires_grad_(req)
        if was_training:
            model.train()
