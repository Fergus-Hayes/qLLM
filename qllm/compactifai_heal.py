"""Healing: brief retraining of an MPO-compressed layer (CompactifAI, Tomut et al.).

After a weight matrix is truncated to an MPO, a short retraining ("healing")
recovers much of the accuracy lost to the local, layer-by-layer truncation. For
the per-layer profile we heal *one layer at a time* while every other layer stays
at its original dense weights, and crucially we heal **in the low-rank MPO
parameterization** -- the two MPO tensors are the only trainable parameters, so
the layer stays compressed (its parameter count is unchanged by healing).

Only the 2-site MPO is supported for healing (the default), because it factors
the permuted weight into exactly two matrices ``A @ B`` that map cleanly onto two
trainable tensors.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .compactifai import MPOPlan, _deinterleave_perm, mpo_param_count


class MPOLinear(nn.Module):
    """Trainable 2-site MPO standing in for a linear projection.

    The effective weight ``W = deinterleave(A @ B)`` has the SAME shape and
    orientation as the wrapped module's ``weight`` (the SVD was taken from that
    exact tensor), so this works for both ``nn.Linear`` (weight ``[out, in]``,
    forward ``x Wᵀ``) and GPT-2 style ``Conv1D`` (weight ``[in, out]``, forward
    ``x W``). ``A`` and ``B`` are trainable, initialised from the truncated SVD
    so that at step 0 the layer reproduces the pre-heal compressed weight exactly.
    """

    def __init__(self, module: nn.Module, plan: MPOPlan, chi: int):
        super().__init__()
        if plan.n_sites != 2 or not plan.cached:
            raise ValueError("MPOLinear needs a cached 2-site plan (mpo_sites=2, svd_cache on).")
        w = module.weight
        self.weight_shape = tuple(int(d) for d in w.shape)   # original orientation
        self.out_dims = list(plan.out_dims)
        self.in_dims = list(plan.in_dims)
        r = max(1, min(int(chi), int(plan.s.numel())))
        self.chi = r
        self.n_params = mpo_param_count(plan.out_dims, plan.in_dims, r)
        # GPT-2 Conv1D: forward is x @ W (+b); nn.Linear: x @ Wᵀ (+b).
        self.is_conv1d = type(module).__name__ == "Conv1D"

        device = w.device
        sqrt_s = torch.sqrt(plan.s[:r].to(torch.float32))
        A = plan.u[:, :r].to(torch.float32) * sqrt_s        # balanced: A = U√S
        B = sqrt_s.unsqueeze(1) * plan.vh[:r].to(torch.float32)  # B = √S Vh
        self.A = nn.Parameter(A.to(device=device))
        self.B = nn.Parameter(B.to(device=device))

        bias = getattr(module, "bias", None)
        if bias is not None:
            self.register_buffer("bias", bias.detach().clone())
        else:
            self.bias = None

    def effective_weight(self) -> torch.Tensor:
        d0, d1 = self.out_dims, self.in_dims
        approx = (self.A @ self.B).reshape(d0[0], d1[0], d0[1], d1[1])
        approx = approx.permute(*_deinterleave_perm(2)).contiguous()
        return approx.reshape(*self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight().to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        if self.is_conv1d:
            out = x.reshape(-1, x.shape[-1]) @ w
            if bias is not None:
                out = out + bias
            return out.reshape(*x.shape[:-1], w.shape[1])
        return nn.functional.linear(x, w, bias)


def _resolve_module(model, param_name: str):
    """Return (parent_module, attr_name, module) for a ``...<attr>.weight`` name."""
    module_path = param_name[:-len(".weight")] if param_name.endswith(".weight") else param_name
    parts = module_path.split(".")
    parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
    attr = parts[-1]
    return parent, attr, getattr(parent, attr)


def heal_layer(
    model,
    param_name: str,
    plan: MPOPlan,
    chi: int,
    heal_batches: list[torch.Tensor],
    device: str,
    steps: int,
    lr: float,
    log: bool = False,
) -> tuple[torch.Tensor, int, float]:
    """Swap one layer to a trainable MPO, fine-tune it, return its healed weight.

    Every other parameter in the model is frozen; only the swapped layer's two
    MPO tensors are optimised (Adam) to minimise the LM loss on ``heal_batches``.
    Returns ``(healed_dense_weight, n_mpo_params, final_loss)``. The caller is
    responsible for restoring the original module afterwards (see
    :func:`healed_layer` for a context-managed version).
    """
    parent, attr, original = _resolve_module(model, param_name)
    adapter = MPOLinear(original, plan, chi).to(device)
    setattr(parent, attr, adapter)

    # Freeze the whole model, then train only the adapter's MPO tensors.
    frozen = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in frozen:
        p.requires_grad_(False)
    adapter.A.requires_grad_(True)
    adapter.B.requires_grad_(True)

    opt = torch.optim.Adam([adapter.A, adapter.B], lr=lr)
    was_training = model.training
    model.eval()                                   # no dropout; only A,B learn
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
                    print(f"        heal step {step}/{steps}  loss={final_loss:.4f}")
        with torch.no_grad():
            healed = adapter.effective_weight().detach().to("cpu")
        return healed, adapter.n_params, final_loss
    finally:
        setattr(parent, attr, original)            # always restore the dense layer
        for p, req in frozen:
            p.requires_grad_(req)
        if was_training:
            model.train()


def make_heal_batches(input_ids: torch.Tensor, window: int, batch_size: int,
                      max_tokens: int | None) -> list[torch.Tensor]:
    """Slice a token stream into non-overlapping [batch, window] training batches."""
    seq_len = input_ids.size(1)
    if max_tokens:
        seq_len = min(seq_len, max_tokens)
    windows = [input_ids[0, b:b + window] for b in range(0, seq_len - 1, window)
               if b + 2 <= seq_len]
    windows = [w for w in windows if w.numel() >= 2]
    batches = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i:i + batch_size]
        if len({w.numel() for w in chunk}) == 1:   # equal length -> stackable
            batches.append(torch.stack(chunk))
    return batches
