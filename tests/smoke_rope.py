"""Smoke test for the RoPE-pairing ansatz.

Checks the properties the experiment's verdict depends on:

* the pairing partitions the register exactly once, so the gate is a genuine
  block-diagonal orthogonal and not an overlapping mess;
* the structured polar update is the *exact* maximiser over the feasible set, so
  the sweep is still closed-form optimal and a weak result cannot be blamed on a
  bad optimiser;
* the three pairings (rope / adjacent / random) cost exactly the same, which is
  what makes ``random-pair`` a matched null rather than a cheaper alternative;
* tying across heads really costs ``head_dim/2`` and untying costs ``dim/2``;
* identity init leaves the circuit a no-op, so the hybrid starts at the classical
  MPO and any gain is a gain;
* the gradient path keeps the gate inside its blocks and at its own parameter
  count, so a pairing can be trained by Adam without silently becoming a dense
  register-wide rotation.
"""
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from qllm.disentangler import (  # noqa: E402
    PAIR_ANSATZE, _polar_structured, _scatter_blocks, build_circuit,
    circuit_param_count, disentangle, rope_pair_groups,
)


def main():
    torch.manual_seed(0)

    # --- 1. the pairing is a partition -------------------------------------
    for pairing in ("rope", "adjacent", "random"):
        groups, ties = rope_pair_groups(10, 64, True, pairing, 0)
        flat = [i for g in groups for i in g]
        assert len(flat) == 1024, f"{pairing}: covered {len(flat)} of 1024"
        assert len(set(flat)) == 1024, f"{pairing}: an index is paired twice"
        assert all(len(g) == 2 for g in groups), f"{pairing}: not all pairs"
        assert len(set(ties)) == 32, f"{pairing}: {len(set(ties))} tie classes, want 32"
    # RoPE pairs i with i + head_dim/2 inside a head, and nothing across heads
    groups, _t = rope_pair_groups(10, 64, True, "rope", 0)
    assert all(b - a == 32 and a // 64 == b // 64 for a, b in groups), "not the RoPE stride"

    # --- 2. the structured polar update is exactly optimal ------------------
    g4, t4 = rope_pair_groups(4, 4, True, "rope", 0)
    env = torch.randn(16, 16)
    best = _polar_structured(env, g4, t4)
    assert float((best @ best.T - torch.eye(16)).abs().max()) < 1e-5, "not orthogonal"
    off = best.clone()
    for a, b in g4:
        off[a, a] = off[a, b] = off[b, a] = off[b, b] = 0.0
    assert float(off.abs().max()) < 1e-6, "mass outside the blocks"
    score = float((best * env).sum())
    labels = sorted(set(t4))
    for _ in range(3000):                      # random feasible points must not win
        blocks = {}
        for lab in labels:
            th = float(torch.randn(1)) * 0.8
            r = torch.tensor([[math.cos(th), -math.sin(th)],
                              [math.sin(th), math.cos(th)]])
            if float(torch.rand(1)) < 0.5:     # reflections are feasible too
                r = r @ torch.tensor([[1.0, 0.0], [0.0, -1.0]])
            blocks[lab] = r
        cand = float((_scatter_blocks(4, g4, t4, blocks) * env).sum())
        assert cand <= score + 1e-4, f"polar beaten by {cand - score:.2e}"

    # --- 3. the null is matched on cost, and tying is what makes it cheap ---
    costs = {a: circuit_param_count(10, 0, 0, "manifold", a, 64) for a in PAIR_ANSATZE}
    assert len(set(costs.values())) == 1, f"pairings differ in cost: {costs}"
    assert costs["rope-pair"] == 32, f"tied cost {costs['rope-pair']}, want head_dim/2"
    untied = circuit_param_count(10, 0, 0, "manifold", "rope-pair", 64, False)
    assert untied == 512, f"untied cost {untied}, want dim/2"
    # the input side has no RoPE, so it must fall back to the brickwall
    v_side = circuit_param_count(10, 2, 4, "manifold", "rope-pair", 64, True, "in")
    plain = circuit_param_count(10, 2, 4, "manifold", "brickwall")
    assert v_side == plain, f"V side {v_side} is not the brickwall {plain}"

    # --- 4. identity init is a no-op ---------------------------------------
    gates = build_circuit(10, 0, 0, "identity", None, "rope-pair", 64)
    assert len(gates) == 1 and gates[0].groups is not None, "no structured gate built"
    assert float((gates[0].matrix - torch.eye(1024)).abs().max()) == 0.0, "not identity"

    W = torch.randn(64, 64)
    r = disentangle(W, gate_size=0, depth=0, target_chi=2, sweeps=1,
                    tensorization="qubit", ansatz="rope-pair", head_dim=8)
    assert r.quantum_params == 4, f"Q={r.quantum_params}, want head_dim/2 = 4"
    assert r.retained >= r.retained_classical - 1e-6, "the sweep lost to its own start"

    # --- 5. the gradient path honours the constraint ------------------------
    # It used to refuse these outright, because expm(skew(theta)) over the whole
    # register would have dropped the block structure and trained dim SO(2^k)
    # angles. The constrained parameterization makes them trainable at their own
    # price, which is what stage 2 needs.
    for ansatz in PAIR_ANSATZE:
        r = disentangle(W, gate_size=0, depth=0, target_chi=2, sweeps=1,
                        tensorization="qubit", ansatz=ansatz, head_dim=8,
                        optimizer="gradient", gd_steps=10)
        assert r.quantum_params == 4, f"{ansatz}: Q={r.quantum_params} after Adam"
        gm = r.u_gates[0].matrix
        assert float((gm @ gm.T - torch.eye(gm.shape[0])).abs().max()) < 1e-4
        leaked = gm.clone()
        for a, b in r.u_gates[0].groups:
            leaked[a, a] = leaked[a, b] = leaked[b, a] = leaked[b, b] = 0.0
        assert float(leaked.abs().max()) == 0.0, f"{ansatz}: Adam left its blocks"

    print(f"  pairing partitions the register: 1024/1024 covered, 32 tie classes")
    print(f"  structured polar optimal over 3000 random feasible points")
    print(f"  matched null: rope/adjacent/random all cost {costs['rope-pair']} "
          f"(untied {untied}, all-to-all 523776)")
    print(f"  identity init is a no-op; sweep never loses to its own start")
    print("\nROPE SMOKE PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
