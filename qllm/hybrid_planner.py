"""Offline feasibility planner: which (k, D) can possibly beat the classical MPO?

Whether the hybrid layer wins at a perplexity budget depends on measured
perplexity -- but whether it *can* win is settled beforehand by arithmetic, from
the layer's shape alone:

    C_pad(chi') + w Q(D, k)   <   C(chi)

``C_pad`` is larger than ``C`` for the same bond dimension (the qubit embedding
rounds each index up to a power of two), and ``Q`` grows as ``4^k`` per gate. So
for every ``(k, D)`` there is a hard verdict before a single token is scored:

* the **quantum surcharge** ``w Q`` expressed in bond-dimension units -- how much
  of the classical budget the circuits eat before they store anything;
* ``chi'_max``, the largest residual bond dimension still affordable;
* a **veto** when even ``chi' = 1`` costs more than the classical layer, in which
  case no perplexity result can rescue that configuration.

Running this first prunes the sweep grid to the configurations that are worth
spending perplexity evaluations on.
"""

from __future__ import annotations


from .compactifai import mpo_dims, mpo_param_count
from .disentangler import n_qubits_for, quantum_param_count


def _dims(d_out: int, d_in: int, n_sites: int):
    return mpo_dims(d_out, d_in, n_sites)


def classical_cost(d_out: int, d_in: int, chi: int, n_sites: int = 2) -> int:
    """``C(chi)``: MPO parameters for the layer as it is."""
    out_dims, in_dims = _dims(d_out, d_in, n_sites)
    return mpo_param_count(out_dims, in_dims, chi)


def padded_cost(d_out: int, d_in: int, chi: int, n_sites: int = 2) -> int:
    """``C_pad(chi')``: MPO parameters over the qubit-padded indices."""
    p_out, p_in = 1 << n_qubits_for(d_out), 1 << n_qubits_for(d_in)
    out_dims, in_dims = _dims(p_out, p_in, n_sites)
    return mpo_param_count(out_dims, in_dims, chi)


def max_affordable_chi(d_out: int, d_in: int, ceiling: float, q_cost: float,
                       n_sites: int = 2) -> int:
    """Largest ``chi'`` with ``C_pad(chi') + q_cost <= ceiling`` (0 = none)."""
    if q_cost >= ceiling:
        return 0
    chi, best = 1, 0
    while True:
        if padded_cost(d_out, d_in, chi, n_sites) + q_cost <= ceiling:
            best = chi
            chi *= 2
        else:
            break
        if chi > max(d_out, d_in) * 4:
            break
    lo, hi = best, min(chi, max(d_out, d_in) * 4)
    while lo < hi:                                  # refine between the powers
        mid = (lo + hi + 1) // 2
        if padded_cost(d_out, d_in, mid, n_sites) + q_cost <= ceiling:
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_rows(shapes: list[tuple[str, int, int]], gate_sizes: list[int],
              depths: list[int], reference_chi: int | None = None,
              q_weight: float = 1.0, n_sites: int = 2,
              counting: str = "manifold") -> list[dict]:
    """One verdict row per (layer shape, k, D).

    ``reference_chi`` is the classical bond dimension the hybrid has to undercut;
    by default the budget is the *dense* layer itself, i.e. the loosest possible
    target ("does this configuration compress at all?").
    """
    rows = []
    for label, d_out, d_in in shapes:
        dense = d_out * d_in
        ceiling = (classical_cost(d_out, d_in, reference_chi, n_sites)
                   if reference_chi else float(dense))
        n_out, n_in = n_qubits_for(d_out), n_qubits_for(d_in)
        for k in gate_sizes:
            for depth in depths:
                k_eff = max(n_out, n_in) if k == 0 else k
                q = quantum_param_count(n_out, n_in, k_eff, depth, counting)
                q_cost = q_weight * q
                chi_max = max_affordable_chi(d_out, d_in, ceiling, q_cost, n_sites)
                # Per-unit-chi slope of the padded MPO: how many bond dimensions
                # the surcharge costs.
                slope = max(1, padded_cost(d_out, d_in, 2, n_sites)
                            - padded_cost(d_out, d_in, 1, n_sites))
                rows.append(dict(
                    layer=label, d_out=d_out, d_in=d_in, dense=dense,
                    n_out_qubits=n_out, n_in_qubits=n_in, gate_size=k_eff,
                    circuit_depth=depth, q_params=q, q_cost=q_cost,
                    ceiling=ceiling, chi_max=chi_max,
                    surcharge_in_chi=q_cost / slope,
                    vetoed=chi_max < 1,
                ))
    return rows


def print_plan(rows: list[dict], reference_chi: int | None, q_weight: float) -> None:
    target = (f"C(chi={reference_chi})" if reference_chi else "the dense layer")
    print("\n" + "=" * 96)
    print(f"FEASIBILITY PLAN -- can the hybrid layer undercut {target}?  "
          f"(w = {q_weight:g})")
    print("=" * 96)
    print("chi'max = largest residual bond dimension still affordable once the "
          "circuits are paid for;")
    print("surcharge = that payment expressed in bond dimensions of the padded "
          "MPO. VETO = not even chi'=1 fits.")
    current = None
    for r in rows:
        if r["layer"] != current:
            current = r["layer"]
            print(f"\n{r['layer']}  {r['d_out']}x{r['d_in']} = {r['dense']:,} dense "
                  f"params -> {r['n_out_qubits']}q x {r['n_in_qubits']}q register, "
                  f"budget {r['ceiling']:,.0f} params")
            print(f"    {'k':>3} {'D':>4} {'Q(D)':>12} {'w.Q':>12} "
                  f"{'surcharge':>11} {'chi_max':>9}  verdict")
        verdict = "VETO" if r["vetoed"] else ""
        print(f"    {r['gate_size']:>3} {r['circuit_depth']:>4} {r['q_params']:>12,} "
              f"{r['q_cost']:>12,.0f} {r['surcharge_in_chi']:>11.2f} "
              f"{r['chi_max']:>9}  {verdict}")


def shapes_from_model(model, config) -> list[tuple[str, int, int]]:
    """Distinct (layer_type, d_out, d_in) shapes among the eligible layers."""
    from .compactifai_sweep import select_layers
    from .layer_analysis import parse_layer_info

    seen, shapes = set(), []
    for name, param in select_layers(model, config):
        layer_type, _depth = parse_layer_info(name)
        key = (layer_type, int(param.shape[0]), int(param.shape[1]))
        if key not in seen:
            seen.add(key)
            shapes.append(key)
    return shapes


def parse_shape(text: str) -> tuple[str, int, int]:
    """``'576x192'`` or ``'v_proj:576x192'`` -> ``(label, d_out, d_in)``."""
    label, _, dims = text.rpartition(":")
    rows, _, cols = dims.lower().partition("x")
    if not rows or not cols:
        raise ValueError(f"Cannot read a shape from {text!r}; use e.g. 576x192.")
    return (label or dims, int(rows), int(cols))
